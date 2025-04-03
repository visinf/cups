from functools import cache
import os

from utils import *
import sys
sys.path.append(os.path.join(os.getcwd(), 'external/depthg'))
from src.modules import OriginalLocalHiddenPositiveProjection, LocalHiddenPositiveProjection, FeaturePyramidNet, \
    DinoFeaturizer, DinoFeaturizerWithDepth, ClusterLookup, ContrastiveCRFLoss, ContrastiveCorrelationLoss, \
    DepthContrastiveCorrelationLoss, norm, sample
from data import create_pascal_label_colormap, create_cityscapes_colormap, ContrastiveSegDataset
from depth_decay_modules import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
import sys, time
import wandb

torch.multiprocessing.set_sharing_strategy('file_system')

"""def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""


def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg

        self.n_classes = n_classes
        print("Number of classes: ", n_classes)

        self.use_depth = cfg.use_depth
        self.depth_only_intra = cfg.use_depth_only_intra
        self.depth_feature_correlation_loss = cfg.depth_feat_correlation_loss
        try:
            if self.cfg.lhp:
                if self.cfg.experiment_name.__contains__("lhp_original"):
                    print("Using Original LHP Module")
                    self.lhp_module = OriginalLocalHiddenPositiveProjection(cfg)
                else:
                    self.lhp_module = LocalHiddenPositiveProjection(cfg)
        except:
            print("self.cfg.lhp not in cfg")

        self.arch = cfg.arch

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        if cfg.arch == "feature-pyramid":
            cut_model = load_model(cfg.model_type, data_dir).cuda()
            self.net = FeaturePyramidNet(cfg.granularity, cut_model, dim, cfg.continuous)
        elif cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        elif cfg.arch == "dino_depth":
            self.net = DinoFeaturizerWithDepth(dim, cfg, 384, (28, 28))
            self.use_depth = True
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.train_cluster_probe = ClusterLookup(dim, n_classes)

        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)

        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.crf_loss_fn = ContrastiveCRFLoss(
            cfg.crf_samples, cfg.alpha, cfg.beta, cfg.gamma, cfg.w1, cfg.w2, cfg.shift)

        self.contrastive_corr_loss_fn = ContrastiveCorrelationLoss(cfg)

        if self.depth_only_intra:
            self.contrastive_corr_loss_fn = DepthContrastiveCorrelationLoss(cfg)

        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

        self.validation_step_outputs = []

        self.max_performance = 0.0

        self.max_cluster_accuracy = 0.0
        self.max_cluster_miou = 0.0
        self.max_linear_accuracy = 0.0
        self.max_linear_miou = 0.0
        self.max_combined_miou = 0.0
        self.max_combined_unsupervised = 0.0
        
        # add cups meter wehen overclustering
        self.extra_clusters = cfg.extra_clusters
        if self.extra_clusters > 0:
            sys.path.append(os.getcwd())
            from cups.metrics import PanopticQualitySemanticMatching
            from cups.data.cityscapes import CITYSCAPES_THING_CLASSES, CITYSCAPES_STUFF_CLASSES
            self.panoptic_meter = PanopticQualitySemanticMatching(
                            things=CITYSCAPES_THING_CLASSES,
                            stuffs=CITYSCAPES_STUFF_CLASSES,
                            things_prototype=None,
                            stuffs_prototype=None,
                            num_clusters=n_classes+cfg.extra_clusters,
                            # cache_device='cpu',
                        )
        
        

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        x = self.net(x)[1]

        if self.cfg.lhp:
            x = self.lhp_module(x)

        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()

        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            label_pos = batch["label_pos"]

            if self.use_depth:
                depth = batch["depth"]
                depth_pos = batch["depth_pos"]

        input = (img, depth) if self.use_depth and self.arch.__contains__("depth") else img

        # Original image features and codes transformed by the segmentation head
        outputs = self.net(input)

        if self.arch.__contains__("depth"):
            feats, code, orig_feats, attn = outputs

        else:
            feats, code, attn = outputs

        if self.cfg.lhp:
            lhp_projection_code = self.lhp_module(code, depth, img, attn)

        input_pos = (img_pos, depth_pos) if self.use_depth and self.arch.__contains__("depth") else img_pos

        if self.cfg.correspondence_weight > 0:
            outputs_pos = self.net(input_pos)
            if self.arch.__contains__("depth"):
                feats_pos, code_pos, orig_feats_pos, attn_pos = outputs_pos
            else:
                feats_pos, code_pos, attn_pos = outputs_pos

            if self.cfg.lhp:
                lhp_projection_code_pos = self.lhp_module(code_pos, None)

        log_args = dict(sync_dist=False, rank_zero_only=True)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        # This is True since use_true_labels=False
        else:
            signal = feats
            signal_pos = feats_pos

        loss = 0

        should_log_hist = (self.cfg.hist_freq is not None) and \
                          (self.global_step % self.cfg.hist_freq == 0) and \
                          (self.global_step > 0)

        if self.cfg.use_salience:
            salience = batch["mask"].to(torch.float32).squeeze(1)
            salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1)
        else:
            salience = None
            salience_pos = None

        if self.cfg.correspondence_weight > 0:

            if self.depth_feature_correlation_loss:
                (
                    pos_intra_loss, pos_intra_cd,
                    pos_inter_loss, pos_inter_cd,
                    neg_inter_loss, neg_inter_cd,
                    depth_feat_loss, depth_feat_cd,
                ) = self.contrastive_corr_loss_fn(
                    signal, signal_pos,
                    salience, salience_pos,
                    code, code_pos,
                    depth=depth, depth_pos=depth_pos
                )

                if self.cfg.lhp:
                    (
                        lhp_pos_intra_loss, _pos_intra_cd,
                        lhp_pos_inter_loss, _pos_inter_cd,
                        lhp_neg_inter_loss, _neg_inter_cd,
                        lhp_depth_feat_loss, _depth_feat_cd,
                    ) = self.contrastive_corr_loss_fn(
                        signal, signal_pos,
                        salience, salience_pos,
                        lhp_projection_code, lhp_projection_code_pos,
                        depth=depth, depth_pos=depth_pos
                    )

            else:
                (
                    pos_intra_loss, pos_intra_cd,
                    pos_inter_loss, pos_inter_cd,
                    neg_inter_loss, neg_inter_cd,
                ) = self.contrastive_corr_loss_fn(
                    signal, signal_pos,
                    salience, salience_pos,
                    code, code_pos,
                    None, None
                )

                if self.cfg.lhp:
                    lhp_depth_feat_loss = 0.0
                    _depth_feat_cd = 0.0
                    (
                        lhp_pos_intra_loss, _pos_intra_cd,
                        lhp_pos_inter_loss, _pos_inter_cd,
                        lhp_neg_inter_loss, _neg_inter_cd,
                    ) = self.contrastive_corr_loss_fn(
                        signal, signal_pos,
                        salience, salience_pos,
                        lhp_projection_code, lhp_projection_code_pos,
                        depth=None, depth_pos=None
                    )


            if self.cfg.lhp:
                pass

            if should_log_hist:
                self.logger.experiment.add_histogram("intra_cd", pos_intra_cd, self.global_step)
                self.logger.experiment.add_histogram("inter_cd", pos_inter_cd, self.global_step)
                self.logger.experiment.add_histogram("neg_cd", neg_inter_cd, self.global_step)

            neg_inter_loss = neg_inter_loss.mean()
            pos_intra_loss = pos_intra_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()

            if self.cfg.lhp:
                lhp_neg_inter_loss = lhp_neg_inter_loss.mean()
                lhp_pos_intra_loss = lhp_pos_intra_loss.mean()
                lhp_pos_inter_loss = lhp_pos_inter_loss.mean()

            if self.depth_feature_correlation_loss:
                depth_feat_loss = depth_feat_loss.mean()

                self.log('loss/depth_feat', depth_feat_loss.mean(), **log_args)
                self.log('cd/depth_feat', depth_feat_cd.mean(), **log_args)

            self.log('loss/pos_intra', pos_intra_loss, **log_args)
            self.log('loss/pos_inter', pos_inter_loss, **log_args)
            self.log('loss/neg_inter', neg_inter_loss, **log_args)
            self.log('cd/pos_intra', pos_intra_cd.mean(), **log_args)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args)

            if self.depth_feature_correlation_loss:
                if self.cfg.lhp and self.cfg.lhp_weight_balance:
                    balance = self.cfg.lhp_weight
                else:
                    balance = 0.0

                loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                         self.cfg.pos_intra_weight * pos_intra_loss +
                         self.cfg.neg_inter_weight * neg_inter_loss +
                         self.cfg.depth_feat_weight * depth_feat_loss) * (self.cfg.correspondence_weight - balance)

                if self.cfg.experiment_name.__contains__("lhp_original"):
                    loss = 0.0
                    self.cfg.lhp_weight = 1.0

                if self.cfg.lhp:
                    loss += (self.cfg.pos_inter_weight * lhp_pos_inter_loss +
                             self.cfg.pos_intra_weight * lhp_pos_intra_loss +
                             self.cfg.neg_inter_weight * lhp_neg_inter_loss +
                             self.cfg.depth_feat_weight * self.cfg.lhp_depth_weight * lhp_depth_feat_loss) * self.cfg.lhp_weight


            else:
                loss += (self.cfg.pos_inter_weight * pos_inter_loss +
                         self.cfg.pos_intra_weight * pos_intra_loss +
                         self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight

        """
        Here we can activate different kinds of decays
        """
        # LEGACY before June 23 2023
        if self.cfg.depth_loss_decay and self.global_step % self.cfg.decay_every_steps == 0 and self.global_step > 0:
            self.cfg.depth_feat_weight = self.cfg.depth_feat_weight * self.cfg.depth_loss_decay_factor
            if not self.cfg.fix_depth_feat_shift:
                self.cfg.depth_feat_shift = self.cfg.depth_feat_shift * self.cfg.depth_loss_decay_factor

        if self.cfg.fps_until_step > 0 and self.global_step >= self.cfg.fps_until_step:
            self.contrastive_corr_loss_fn.cfg.depth_sampling = "none"
            self.contrastive_corr_loss_fn.cfg.feature_samples = self.cfg.post_fps_samples

        # Important: Until Friday June 23 2023 there was a bug here. FPS decay was applied at step=0 and not at step=cfg.fps_sample_decay_every_steps.
        # This led to the feature_samples being decayed right at the beginning of training.
        # Our SOTA ViT-S results therefore start with a feature sample of 9 instead of 11.

        # LEGACY:
        if self.cfg.fps_sample_decay and self.global_step % self.cfg.fps_sample_decay_every_steps == 0:
            self.contrastive_corr_loss_fn.cfg.feature_samples = int(
                self.contrastive_corr_loss_fn.cfg.feature_samples * self.cfg.fps_sample_decay_factor)

            if self.contrastive_corr_loss_fn.cfg.feature_samples < self.cfg.fps_min_samples:
                self.contrastive_corr_loss_fn.cfg.feature_samples = self.cfg.fps_min_samples

        # NEW:
        # if self.cfg.fps_sample_decay and self.global_step % self.cfg.fps_sample_decay_every_steps == 0 and self.global_step > 0:
        #    self.contrastive_corr_loss_fn.cfg.feature_samples = self.fps_sample_scheduler.return_update(step=self.global_step)

        # Log the current depth_feat_weight, depth_feat_shift and feature_samples
        self.log('cfg/depth_feat_weight', self.cfg.depth_feat_weight, **log_args)
        self.log('cfg/depth_feat_shift', self.cfg.depth_feat_shift, **log_args)
        self.log('cfg/feature_samples', self.contrastive_corr_loss_fn.cfg.feature_samples, **log_args)
        self.log('cfg/feature_samples_actual', self.contrastive_corr_loss_fn.cfg.feature_samples ** 2, **log_args)

        """
        Decay activation ends here
        """

        if self.cfg.rec_weight > 0:
            rec_feats = self.decoder(code)

            # This is the loss function from the original paper
            # In the config, rec_weight = 0.0, so this part is not used
            rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
            self.log('loss/rec', rec_loss, **log_args)
            loss += self.cfg.rec_weight * rec_loss

        if self.cfg.aug_alignment_weight > 0:
            orig_feats_aug, orig_code_aug = self.net(img_aug)
            downsampled_coord_aug = resize(
                coord_aug.permute(0, 3, 1, 2),
                orig_code_aug.shape[2]).permute(0, 2, 3, 1)
            aug_alignment = -torch.einsum(
                "bkhw,bkhw->bhw",
                norm(sample(code, downsampled_coord_aug)),
                norm(orig_code_aug)
            ).mean()
            self.log('loss/aug_alignment', aug_alignment, **log_args)
            loss += self.cfg.aug_alignment_weight * aug_alignment

        if self.cfg.crf_weight > 0:
            crf = self.crf_loss_fn(
                resize(img, 56),
                norm(resize(code, 56))
            ).mean()
            self.log('loss/crf', crf, **log_args)
            loss += self.cfg.crf_weight * crf

        # Flatten the label
        flat_label: np.ndarray = label.reshape(-1)
        # Mask out-of-bound labels
        mask: np.ndarray = (flat_label >= 0) & (flat_label < self.n_classes)

        detached_code = torch.clone(code.detach())

        # Compute the logits of the linear probe
        linear_logits = self.linear_probe(detached_code)
        # Upsample the logits to the size of the label
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        # Reshape the logits to make it compatible with the loss function
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        # Compute the linear probe loss
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()
        # Add the linear probe loss to the total loss
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        # Compute the cluster loss and the cluster probabilities
        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()

        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        if self.global_step % 2000 == 0 and self.global_step > 0:
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(),
            **self.cluster_metrics.compute()
        }
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, code = self.net(img)

            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)
            
            if self.extra_clusters > 0:
                preds = torch.stack((cluster_preds, torch.zeros_like(cluster_preds)), dim=-1)
                gts = torch.stack((label, torch.zeros_like(label)), dim=-1)
                self.panoptic_meter.update(preds, gts)

            self.validation_step_outputs.append({
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()})

            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        outputs = self.validation_step_outputs
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }

            if self.extra_clusters > 0:
                miou = self.panoptic_meter.compute()[6].item()*100
                self.panoptic_meter.reset()
                tb_metrics["test/cluster/mIoU"] = miou
                print('Replaced metric for overclustering')

            if tb_metrics["test/cluster/Accuracy"] > self.max_cluster_accuracy:
                self.max_cluster_accuracy = tb_metrics["test/cluster/Accuracy"]

            if tb_metrics["test/cluster/mIoU"] > self.max_cluster_miou:
                self.max_cluster_miou = tb_metrics["test/cluster/mIoU"]

            if tb_metrics["test/linear/Accuracy"] > self.max_linear_accuracy:
                self.max_linear_accuracy = tb_metrics["test/linear/Accuracy"]

            if tb_metrics["test/linear/mIoU"] > self.max_linear_miou:
                self.max_linear_miou = tb_metrics["test/linear/mIoU"]

            tb_metrics["test/cluster/MaxAccuracy"] = self.max_cluster_accuracy
            tb_metrics["test/cluster/MaxmIoU"] = self.max_cluster_miou
            tb_metrics["test/linear/MaxAccuracy"] = self.max_linear_accuracy
            tb_metrics["test/linear/MaxmIoU"] = self.max_linear_miou

            print(tb_metrics)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        return net_optim, linear_probe_optim, cluster_probe_optim


@hydra.main(config_path="configs", config_name="local_config.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    trial = None
    
# def my_app(trial=None) -> None:
#     import omegaconf
#     cfg = omegaconf.OmegaConf.load("external/depthg/src/configs/local_config.yml")
#     OmegaConf.set_struct(cfg, False)
#     print(OmegaConf.to_yaml(cfg))

    if os.path.exists("/data/stegodepth/"):
        cfg.output_root = "/data/outputs/"
        cfg.data_dir = "/data/stegodepth/"

        print(f"Data Path changed to {cfg.data_dir}")

    elif os.path.exists("/mnt/data/datasets"):
        cfg.output_root = "/mnt/data/outputs/"
        cfg.data_dir = "/mnt/data/datasets"

        print(f"Data Path changed to {cfg.data_dir}")

    elif os.path.exists("/Users/leonsick/image_datasets"):
        cfg.output_root = "/Users/leonsick/outputs/"
        cfg.data_dir = "/Users/leonsick/image_datasets/"

        print(f"Data Path changed to {cfg.data_dir}")

    elif os.path.exists("/mnt/hdd/leon/"):
        if cfg.pretrained_weights is not None:
            cfg.pretrained_weights = "/mnt/hdd/leon/checkpoints/backbone_ablation"

    data_dir = cfg.data_dir
    # data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    # This is what is used to be
    # eval_res = 320
    if cfg.model_type == "mae":
        eval_res = 224
    else:
        eval_res = 320

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()
    train_dataset = ContrastiveSegDataset(
        data_dir=os.path.join(os.path.dirname(data_dir[:-1]), 'depthg_processed_data'),
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True,
        return_depth=cfg.use_depth,
        depth_type=cfg.depth_type,
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    if cfg.dataset_name == "nyuv2":
        data_dir = os.path.join(data_dir, "nyuv2")

    if cfg.dataset_name == "cityscapes" or cfg.dataset_name == "cityscapes_19":
        data_dir = '/path_to_datasets/cityscapes'
    elif cfg.dataset_name == "kitti":
        data_dir = '/path_to_datasets/KITTI-raw'
    val_dataset = ContrastiveSegDataset(
        data_dir=data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(eval_res, False, val_loader_crop),
        target_transform=get_transform(eval_res, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )
        
    if trial is not None:
        cfg.depth_feat_shift = trial.suggest_float('depth_feat_shift', 0.0, 0.1, step=0.01)       #=0.03 
        cfg.depth_feat_weight = trial.suggest_float('depth_feat_weight', 0.0, 0.2, step=0.025)    #=0.09 
        cfg.depth_loss_decay_factor=0.8 
        cfg.neg_inter_shift = trial.suggest_float('neg_inter_shift', 0.05, 0.45, step=0.05)       #=0.26 
        cfg.neg_inter_weight = trial.suggest_float('neg_inter_weight', 0.4, 0.8, step=0.05)       #=0.5700000000000001 
        cfg.pos_inter_shift = trial.suggest_float('pos_inter_shift', 0.05, 0.45, step=0.05)       #=0.25 
        cfg.pos_inter_weight = trial.suggest_float('pos_inter_weight', 0.8, 1.2, step=0.05)       #=1.02 
        cfg.pos_intra_shift = trial.suggest_float('pos_intra_shift', 0.2, 0.6, step=0.05)         #=0.39 
        cfg.pos_intra_weight = trial.suggest_float('pos_intra_weight', 0.75, 1.15, step=0.05)     #=0.95

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    # if cfg.submitting_to_aml:
        # val_batch_size = 16
    # else:
    val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    config_dict = OmegaConf.to_container(
        cfg, resolve=True
    )
    print("Using config:")
    print(config_dict)
    if cfg.wandb_logging:
        wandb.init(project="stegodepth-src", name=name, config=config_dict, sync_tensorboard=True)

    cfg.submitting_to_aml=False
    if cfg.submitting_to_aml:
        pass
        """gpu_args = dict(gpus=1, val_check_interval=250)

        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")"""

    else:
        if torch.cuda.is_available() and cfg.gpus > 0:
            gpu_args = dict(devices=1, accelerator='gpu', val_check_interval=cfg.val_freq)
            if torch.cuda.device_count() > 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            gpu_args = dict(devices=1, accelerator='cpu', val_check_interval=cfg.val_freq)

        if gpu_args["val_check_interval"] > len(train_loader) // 4:
            gpu_args.pop("val_check_interval")

    if cfg.dataset_name == "potsdam":
        monitor = "test/cluster/Accuracy"
    else:
        monitor = "test/cluster/mIoU"

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                every_n_train_steps=5,
                save_top_k=1,
                save_last=True,
                monitor=monitor,
                mode="max",
            )
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)

    if cfg.wandb_logging:
        wandb.finish()
        
    return model.max_cluster_miou


if __name__ == "__main__":
    mode = ['standard', 'hyperparatuning'][0]
    prep_args()
    if mode == 'standard':
        my_app()
    elif mode == 'hyperparatuning':
        print('-- Hyperparametertuning --')
        import optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(my_app, n_trials=50)

        trial = study.best_trial

        print('Best mIoU: {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))
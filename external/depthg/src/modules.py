import torch
from typing import Tuple
import time, warnings, sys

from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type

        self.model = vits.__dict__[arch](
                patch_size=patch_size,
                num_classes=0)

        for p in self.model.parameters():
            p.requires_grad = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval().to(self.device)
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
                state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
                state_dict = state_dict["teacher"]
                # remove `module.` prefix
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                # remove `backbone.` prefix induced by multicrop wrapper
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

                # state_dict = {k.replace("projection_head", "mlp"): v for k, v in state_dict.items()}
                # state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}

                msg = self.model.load_state_dict(state_dict, strict=False)
                print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url=url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch.__contains__("small"):
            self.n_feats = 384
        else:
            self.n_feats = 768

        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg.projection_type
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            # Old:
            # if self.feat_type == "feat" and self.cfg.model_type.__contains__("reg"):
            if self.feat_type == "feat" and self.cfg.model_type.__contains__("v2"):
                # Also remove register tokens
                # Old:
                # image_feat = feat[:, 5:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
                image_feat = feat.reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK" and attn is not None:
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat
        if self.training:
            if self.cfg.dropout:
                return self.dropout(image_feat), code, attn
            else:
                return image_feat, code, attn
        else:
            if self.cfg.dropout:
                return self.dropout(image_feat), code
            else:
                return image_feat, code


class LocalHiddenPositiveProjection(nn.Module):
    def __init__(self, cfg):
        super(LocalHiddenPositiveProjection, self).__init__()
        self.dim = cfg.dim
        try:
            self.propagation_strategy = cfg.propagation_strategy
        except:
            print("Propagation strategy not specified in config. Using default depth propagation.")
            self.propagation_strategy = "depth"

        # self.projection_head = nn.Linear(cfg.dim, cfg.dim)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.dim, self.dim, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.dim, self.dim, (1, 1)))

        # From https://github.com/hynnsk/HP/blob/main/model/dino/DinoFeaturizer.py
        sz = cfg.res // cfg.dino_patch_size

        self.index_mask = torch.zeros((sz * sz, sz * sz), dtype=torch.float16)
        self.divide_num = torch.zeros((sz * sz), dtype=torch.long)
        for _im in range(sz * sz):
            if _im == 0:
                index_set = torch.tensor([_im, _im + 1, _im + sz, _im + (sz + 1)])
            elif _im == (sz - 1):
                index_set = torch.tensor([_im - 1, _im, _im + (sz - 1), _im + sz])
            elif _im == (sz * sz - sz):
                index_set = torch.tensor([_im - sz, _im - (sz - 1), _im, _im + 1])
            elif _im == (sz * sz - 1):
                index_set = torch.tensor([_im - (sz + 1), _im - sz, _im - 1, _im])

            elif ((1 <= _im) and (_im <= (sz - 2))):
                index_set = torch.tensor([_im - 1, _im, _im + 1, _im + (sz - 1), _im + sz, _im + (sz + 1)])
            elif (((sz * sz - sz + 1) <= _im) and (_im <= (sz * sz - 2))):
                index_set = torch.tensor([_im - (sz + 1), _im - sz, _im - (sz - 1), _im - 1, _im, _im + 1])
            elif (_im % sz == 0):
                index_set = torch.tensor([_im - sz, _im - (sz - 1), _im, _im + 1, _im + sz, _im + (sz + 1)])
            elif ((_im + 1) % sz == 0):
                index_set = torch.tensor([_im - (sz + 1), _im - sz, _im - 1, _im, _im + (sz - 1), _im + sz])
            else:
                index_set = torch.tensor(
                    [_im - (sz + 1), _im - sz, _im - (sz - 1), _im - 1, _im, _im + 1, _im + (sz - 1), _im + sz,
                     _im + (sz + 1)])
            self.index_mask[_im][index_set] = 1.
            self.divide_num = torch.zeros((sz * sz), dtype=torch.long)

        self.index_mask = self.index_mask.cuda()
        self.divide_num = self.divide_num.unsqueeze(1)
        self.divide_num = self.divide_num.cuda()

    def forward(self, code, depth, img=None, attn=None):
        if depth is None or attn is None:
            return self.projection_head(code)

        if self.propagation_strategy == "depth":
            return self.forward_depth(code, depth, img)
        elif self.propagation_strategy == "attn":
            return self.forward_attn(code, attn)
        else:
            raise ValueError("Unknown propagation strategy: {}".format(self.propagation_strategy))

    def forward_attn_lhp(self, code, attn=None):
        assert code.shape[0] == attn.shape[0], "Batch size of code and depth must be the same."
        batch_size = code.shape[0]

        attn = attn[:, :, 1:, 1:]
        attn = torch.mean(attn, dim=1)
        attn = attn.type(torch.float32)
        attn_max = torch.quantile(attn, 0.9, dim=2, keepdim=True)
        attn_min = torch.quantile(attn, 0.1, dim=2, keepdim=True)
        attn = torch.max(torch.min(attn, attn_max), attn_min)

        attn = attn.softmax(dim=-1)
        attn = attn * 28
        attn[attn < torch.mean(attn, dim=2, keepdim=True)] = 0.

        attn = attn * self.index_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        code_clone = code.clone()
        code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
        code_clone = code_clone.permute(0, 2, 1)

        code_3x3_all = []
        for bs in range(batch_size):
            code_3x3 = attn[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
            code_3x3 = torch.sum(code_3x3, dim=1)
            code_3x3 = code_3x3 / self.divide_num
            code_3x3_all.append(code_3x3)
        code_3x3_all = torch.stack(code_3x3_all)
        code_3x3_all = code_3x3_all.permute(0, 2, 1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        code_mixed = code_3x3_all.clone()

        return self.projection_head(code_mixed)

    def forward_attn(self, code, attn=None):
        assert code.shape[0] == attn.shape[0], "Batch size of code and depth must be the same."
        batch_size = code.shape[0]

        attn = attn[:, :, 1:, 1:]
        attn = torch.mean(attn, dim=1)

        # Attn has shape [32, 784, 784]

        # Min max normalization
        attn = (attn - torch.min(attn, dim=2, keepdim=True).values) / (
                torch.max(attn, dim=2, keepdim=True).values - torch.min(attn, dim=2, keepdim=True).values)

        attn = attn.type(torch.float32)

        attn_max = torch.quantile(attn, 0.99, dim=2, keepdim=True)

        attn_pre_thrs = attn.clone()

        attn[attn > attn_max] = 0.

        code_clone = code.clone()
        code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
        code_clone = code_clone.permute(0, 2, 1)

        code_3x3_all = []
        for bs in range(batch_size):
            code_3x3 = attn[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
            code_3x3 = torch.mean(code_3x3, dim=1)

            code_3x3_all.append(code_3x3)
        code_3x3_all = torch.stack(code_3x3_all)
        code_3x3_all = code_3x3_all.permute(0, 2, 1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        code_mixed = code_3x3_all.clone()

        return self.projection_head(code_mixed)

    def forward_depth(self, code, depth, img=None):

        assert code.shape[0] == depth.shape[0], "Batch size of code and depth must be the same."

        # print(f"code.shape: {code.shape}")

        code_clone = code.clone()
        code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)

        code_clone = code_clone.permute(0, 2, 1)
        # Shape of code_clone is now: [B, H*W, C], e.g. [16, 28*28, 384]

        # depth_downscaled = F.interpolate(depth, size=code_clone.shape[1:], mode='bilinear', align_corners=False)
        depth_downscaled = F.adaptive_avg_pool2d(depth, code.shape[-2:])
        # depth_downscale of shape [B, H*W, 1], e.g. [16, 28*28]

        # Convert to point cloud and calculate the distance between each point and every other point
        # Shape of point_cloud is [B, H*W, 3], e.g. [16, 28*28, 3]
        all_distances = []

        for i_depth in range(depth_downscaled.shape[0]):
            point_cloud = depth2points(depth_downscaled[i_depth], fov=90).view(3, -1).permute(1, 0)

            # Calculate the distance between each point and every other point
            spatial_dist = torch.cdist(point_cloud, point_cloud, p=2)
            all_distances.append(spatial_dist)

        all_distances = torch.stack(all_distances, dim=0)

        # Pre-process depth only have local neighbors
        actual_min = torch.min(all_distances, dim=2, keepdim=True).values
        actual_max = torch.max(all_distances, dim=2, keepdim=True).values

        # Normalize
        all_distances = ((all_distances - actual_min) / (actual_max - actual_min))
        all_distances_negative = 1 - all_distances
        all_distances_negative_full = all_distances_negative.clone()
        all_distances_ni = (1 - ((all_distances - actual_min) / (actual_max - actual_min))).float()

        # Pre-process depth only have local neighbors
        depth_min = torch.quantile(all_distances, 0.01, dim=2, keepdim=True)

        all_distances_ni[all_distances > depth_min] = 0.0
        all_distances_negative[all_distances > depth_min] = 0.0

        # lhp_map = all_distances_ni.clone()
        lhp_map = all_distances_negative.clone()

        batch_size = code_clone.shape[0]

        # 3 x 3 patch LHP maps
        code_3x3_all = []

        for bs in range(batch_size):
            code_3x3 = lhp_map[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
            code_3x3 = torch.mean(code_3x3, dim=1)

            code_3x3_all.append(code_3x3)
        code_3x3_all = torch.stack(code_3x3_all)
        code_3x3_all = code_3x3_all.permute(0, 2, 1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        # From HP run.py
        code_mixed = code_3x3_all.clone()

        projection_mixed = self.projection_head(code_mixed)

        return projection_mixed


class OriginalLocalHiddenPositiveProjection(nn.Module):
    def __init__(self, cfg):
        super(OriginalLocalHiddenPositiveProjection, self).__init__()
        self.dim = cfg.dim
        self.propagation_strategy = cfg.propagation_strategy

        # self.projection_head = nn.Linear(cfg.dim, cfg.dim)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.dim, self.dim, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.dim, self.dim, (1, 1)))

        # From https://github.com/hynnsk/HP/blob/main/model/dino/DinoFeaturizer.py
        sz = cfg.res // cfg.dino_patch_size

        self.index_mask = torch.zeros((sz * sz, sz * sz), dtype=torch.float16)
        self.divide_num = torch.zeros((sz * sz), dtype=torch.long)
        for _im in range(sz * sz):
            if _im == 0:
                index_set = torch.tensor([_im, _im + 1, _im + sz, _im + (sz + 1)])
            elif _im == (sz - 1):
                index_set = torch.tensor([_im - 1, _im, _im + (sz - 1), _im + sz])
            elif _im == (sz * sz - sz):
                index_set = torch.tensor([_im - sz, _im - (sz - 1), _im, _im + 1])
            elif _im == (sz * sz - 1):
                index_set = torch.tensor([_im - (sz + 1), _im - sz, _im - 1, _im])

            elif ((1 <= _im) and (_im <= (sz - 2))):
                index_set = torch.tensor([_im - 1, _im, _im + 1, _im + (sz - 1), _im + sz, _im + (sz + 1)])
            elif (((sz * sz - sz + 1) <= _im) and (_im <= (sz * sz - 2))):
                index_set = torch.tensor([_im - (sz + 1), _im - sz, _im - (sz - 1), _im - 1, _im, _im + 1])
            elif (_im % sz == 0):
                index_set = torch.tensor([_im - sz, _im - (sz - 1), _im, _im + 1, _im + sz, _im + (sz + 1)])
            elif ((_im + 1) % sz == 0):
                index_set = torch.tensor([_im - (sz + 1), _im - sz, _im - 1, _im, _im + (sz - 1), _im + sz])
            else:
                index_set = torch.tensor(
                    [_im - (sz + 1), _im - sz, _im - (sz - 1), _im - 1, _im, _im + 1, _im + (sz - 1), _im + sz,
                     _im + (sz + 1)])
            self.index_mask[_im][index_set] = 1.
            self.divide_num = torch.zeros((sz * sz), dtype=torch.long)

        self.index_mask = self.index_mask.cuda()
        self.divide_num = self.divide_num.unsqueeze(1)
        self.divide_num = self.divide_num.cuda()

    def forward(self, code, depth, img=None, attn=None):
        if depth is None or attn is None:
            return self.projection_head(code)

        if self.propagation_strategy == "depth":
            return self.forward_depth(code, depth, img)
        elif self.propagation_strategy == "attn":
            return self.forward_attn(code, attn)
        else:
            raise ValueError("Unknown propagation strategy: {}".format(self.propagation_strategy))

    def forward_attn(self, code, attn=None):
        assert code.shape[0] == attn.shape[0], "Batch size of code and depth must be the same."
        batch_size = code.shape[0]

        attn = attn[:, :, 1:, 1:]
        attn = torch.mean(attn, dim=1)
        attn = attn.type(torch.float32)
        attn_max = torch.quantile(attn, 0.9, dim=2, keepdim=True)
        attn_min = torch.quantile(attn, 0.1, dim=2, keepdim=True)
        attn = (attn - attn_min) / (attn_max - attn_min)

        # attn = attn.softmax(dim=-1)
        # attn = attn * 28
        attn[attn < torch.mean(attn, dim=2, keepdim=True)] = 0.

        attn = attn * self.index_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        code_clone = code.clone()
        code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
        code_clone = code_clone.permute(0, 2, 1)

        code_3x3_all = []
        for bs in range(batch_size):
            code_3x3 = attn[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
            code_3x3 = torch.sum(code_3x3, dim=1)
            code_3x3 = code_3x3 / self.divide_num
            code_3x3_all.append(code_3x3)
        code_3x3_all = torch.stack(code_3x3_all)
        code_3x3_all = code_3x3_all.permute(0, 2, 1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        code_mixed = code_3x3_all.clone()

        return self.projection_head(code_mixed)

    def forward_depth(self, code, depth, img=None):

        assert code.shape[0] == depth.shape[0], "Batch size of code and depth must be the same."
        batch_size = code.shape[0]

        depth_downscaled = F.adaptive_avg_pool2d(depth, code.shape[-2:])

        # Convert to point cloud and calculate the distance between each point and every other point
        # Shape of point_cloud is [B, H*W, 3], e.g. [16, 28*28, 3]
        all_distances = []

        for i_depth in range(depth_downscaled.shape[0]):
            point_cloud = depth2points(depth_downscaled[i_depth], fov=90).view(3, -1).permute(1, 0)

            # Calculate the distance between each point and every other point
            spatial_dist = torch.cdist(point_cloud, point_cloud, p=2)
            all_distances.append(spatial_dist)

        all_distances = torch.stack(all_distances, dim=0)

        # Pre-process depth only have local neighbors
        actual_min = torch.min(all_distances, dim=2, keepdim=True).values
        actual_max = torch.max(all_distances, dim=2, keepdim=True).values

        # Normalize
        all_distances = ((all_distances - actual_min) / (actual_max - actual_min))
        all_distances_negative = 1 - all_distances
        all_distances_ni = (1 - ((all_distances - actual_min) / (actual_max - actual_min))).float()

        # Pre-process depth only have local neighbors
        mean_depth = torch.mean(all_distances, dim=2, keepdim=True)

        all_distances_ni[all_distances > mean_depth] = 0.0
        all_distances_negative[all_distances > mean_depth] = 0.0

        lhp_map = all_distances_negative.clone()

        lhp_map = lhp_map * self.index_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        code_clone = code.clone()
        code_clone = code_clone.view(code_clone.size(0), code_clone.size(1), -1)
        code_clone = code_clone.permute(0, 2, 1)

        code_3x3_all = []
        for bs in range(batch_size):
            code_3x3 = lhp_map[bs].unsqueeze(-1) * code_clone[bs].unsqueeze(0)
            code_3x3 = torch.sum(code_3x3, dim=1)
            code_3x3 = code_3x3 / self.divide_num
            code_3x3_all.append(code_3x3)
        code_3x3_all = torch.stack(code_3x3_all)
        code_3x3_all = code_3x3_all.permute(0, 2, 1).view(code.size(0), code.size(1), code.size(2), code.size(3))

        code_mixed = code_3x3_all.clone()

        return self.projection_head(code_mixed)


class DinoFeaturizerWithDepth(DinoFeaturizer):
    def __init__(self, dim, cfg, embed_dim: int = 384, image_embedding_size: Tuple[int, int] = (28, 28),
                 depth_in_chans=1, activation=nn.GELU):
        super(DinoFeaturizerWithDepth, self).__init__(dim, cfg)

        self.depth_input_size = (8 * image_embedding_size[0], 8 * image_embedding_size[1])

        if self.n_feats == 384:
            self.depth_downscaling = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=2, stride=2),
                LayerNorm2d(64),
                activation(),
                nn.Conv2d(64, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.Conv2d(128, self.n_feats, kernel_size=2, stride=2),
            )
        else:
            self.depth_downscaling = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=2, stride=2),
                LayerNorm2d(64),
                activation(),
                nn.Conv2d(64, 128, kernel_size=2, stride=2),
                LayerNorm2d(128),
                activation(),
                nn.Conv2d(128, 256, kernel_size=2, stride=2),
                LayerNorm2d(256),
                activation(),
                nn.Conv2d(256, 512, kernel_size=2, stride=2),
                LayerNorm2d(512),
                activation(),
                nn.Conv2d(512, self.n_feats, kernel_size=2, stride=2),
            )

        self.cross_attn = torch.nn.MultiheadAttention(embed_dim=self.n_feats, num_heads=8, dropout=0.1)

        self.guidance = cfg.guidance

        assert self.guidance in ["cross_attn", "concat", "sum", "none"]

        self.no_depth_embed = nn.Embedding(1, self.n_feats)

    def forward_with_depth(self, img, depth, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        depth_feats = self.depth_downscaling(depth)

        batch_size = image_feat.size()[0]
        feat_h, feat_w = image_feat.size()[2], image_feat.size()[3]

        if self.training and self.guidance == "sum":
            image_depth_feat = image_feat + depth_feats
        elif self.training and self.guidance == "concat":
            pass
        elif self.training and self.guidance == "cross_attn":
            # Prepare feats for attn module
            image_feat = image_feat.reshape(batch_size, self.n_feats, -1).permute(2, 0, 1)
            depth_feats = depth_feats.reshape(batch_size, self.n_feats, -1).permute(2, 0, 1)

            # Apply attn
            image_depth_feat = self.cross_attn(depth_feats, image_feat, image_feat)[0]

            # Reshape back to original shape
            image_depth_feat = image_depth_feat.permute(1, 2, 0).reshape(batch_size, self.n_feats, feat_h, feat_w)
        elif self.guidance == "cross_attn":
            image_feat = image_feat.reshape(batch_size, self.n_feats, -1).permute(2, 0, 1)
            depth_feats = self.no_depth_embed.weight.reshape(1, 1, -1).expand(
                image_feat.size()[0], batch_size, -1
            )

            image_depth_feat = self.cross_attn(depth_feats, image_feat, image_feat)[0]

            # Reshape back to original shape
            image_depth_feat = image_depth_feat.permute(1, 2, 0).reshape(batch_size, self.n_feats, feat_h, feat_w)
        else:
            image_depth_feat = image_feat

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_depth_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_depth_feat))
        else:
            code = image_depth_feat

        if self.training:
            if self.cfg.dropout:
                return self.dropout(image_depth_feat), code, image_feat, attn
            else:
                return image_depth_feat, code, image_feat, attn
        else:
            if self.cfg.dropout:
                return self.dropout(image_depth_feat), code, attn
            else:
                return image_depth_feat, code, attn

    def forward(self, input, n=1, return_class_feat=False):
        if self.training:
            assert len(input) == 2, "Input should be a tuple of (image, depth)"
            img, depth = input
        else:
            img = input
            depth = torch.zeros((img.shape[0], 1, *self.depth_input_size)).to(img.device)
        return self.forward_with_depth(img, depth, n=n, return_class_feat=return_class_feat)


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ResizeAndClassify(nn.Module):

    def __init__(self, dim: int, size: int, n_classes: int):
        super(ResizeAndClassify, self).__init__()
        self.size = size
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(dim, n_classes, (1, 1)),
            torch.nn.LogSoftmax(1))

    def forward(self, x):
        return F.interpolate(self.predictor.forward(x), self.size, mode="bilinear", align_corners=False)


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        # This is the cluster loss from the paper
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


class FeaturePyramidNet(nn.Module):

    @staticmethod
    def _helper(x):
        # TODO remove this hard coded 56
        return F.interpolate(x, 56, mode="bilinear", align_corners=False).unsqueeze(-1)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def __init__(self, granularity, cut_model, dim, continuous):
        super(FeaturePyramidNet, self).__init__()
        self.layer_nums = [5, 6, 7]
        self.spatial_resolutions = [7, 14, 28, 56]
        self.feat_channels = [2048, 1024, 512, 3]
        self.extra_channels = [128, 64, 32, 32]
        self.granularity = granularity
        self.encoder = NetWithActivations(cut_model, self.layer_nums)
        self.dim = dim
        self.continuous = continuous
        self.n_feats = self.dim

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        assert granularity in {1, 2, 3, 4}
        self.cluster1 = self.make_clusterer(self.feat_channels[0])
        self.cluster1_nl = self.make_nonlinear_clusterer(self.feat_channels[0])

        if granularity >= 2:
            # self.conv1 = DoubleConv(self.feat_channels[0], self.extra_channels[0])
            # self.conv2 = DoubleConv(self.extra_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.conv2 = DoubleConv(self.feat_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.cluster2 = self.make_clusterer(self.extra_channels[1])
        if granularity >= 3:
            self.conv3 = DoubleConv(self.extra_channels[1] + self.feat_channels[2], self.extra_channels[2])
            self.cluster3 = self.make_clusterer(self.extra_channels[2])
        if granularity >= 4:
            self.conv4 = DoubleConv(self.extra_channels[2] + self.feat_channels[3], self.extra_channels[3])
            self.cluster4 = self.make_clusterer(self.extra_channels[3])

    def c(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        low_res_feats = feats[self.layer_nums[-1]]

        all_clusters = []

        # all_clusters.append(self.cluster1(low_res_feats) + self.cluster1_nl(low_res_feats))
        all_clusters.append(self.cluster1(low_res_feats))

        if self.granularity >= 2:
            # f1 = self.conv1(low_res_feats)
            # f1_up = self.up(f1)
            f1_up = self.up(low_res_feats)
            f2 = self.conv2(self.c(f1_up, feats[self.layer_nums[-2]]))
            all_clusters.append(self.cluster2(f2))
        if self.granularity >= 3:
            f2_up = self.up(f2)
            f3 = self.conv3(self.c(f2_up, feats[self.layer_nums[-3]]))
            all_clusters.append(self.cluster3(f3))
        if self.granularity >= 4:
            f3_up = self.up(f3)
            final_size = self.spatial_resolutions[-1]
            f4 = self.conv4(self.c(f3_up, F.interpolate(
                x, (final_size, final_size), mode="bilinear", align_corners=False)))
            all_clusters.append(self.cluster4(f4))

        avg_code = torch.cat(all_clusters, 4).mean(4)

        if self.continuous:
            clusters = avg_code
        else:
            clusters = torch.log_softmax(avg_code, 1)

        return low_res_feats, clusters


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    """
    n: batch
    c: feature dimension
    h: height 1
    w: width 1
    j: height 2
    i: width 2
    Each element is the cosine similarity of feat_vec 1 to feat_vec 2 at that location
    Example: coord f1 (1,2), coord f2 (10, 8). In batch 0, this would give you the cosine similarity by indexing at
    cos_sim(between f1 and f2) = tensor[0, 1, 2, 10, 8]
    """
    return torch.einsum("nchw,ncij->nhwij", a, b)


def depth_correlation(a, b):
    # c=1 since it is a depth map, as opposed to c=384 for a feature map
    return torch.einsum("nchw,ncij->nhwij", a, b)


# Note on depth correlation: With this computation, you will get a tensor that stores the difference in depth between
# all points in the two depth maps. If for a specific pair, this value is low, but high in the feature_correlation tensor,
# , this should result in large penalty.


def sample(t: torch.Tensor, coords: torch.Tensor):
    # Input coord_shape: [32, 11, 11, 2]
    # Permuted coord_shape: [32, 11, 11, 2]
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


def simple_depth_informed_sampling(t: torch.Tensor, depth: torch.Tensor, n_samples: int):
    # Downsample depth map to match size of feature map
    depth = F.adaptive_max_pool2d(depth, t.shape[-2:])

    depth = (depth * 10).round() / 10

    # Model the depth distribution of the depth map
    # This is done individually for each sample in the batch
    depths_list = []
    counts_list = []
    for i in range(depth.shape[0]):
        depths, counts = torch.unique(depth[i], return_counts=True)
        depths_list.append(depths)
        counts_list.append(counts)

    # Sample n_samples locations from the depth distribution
    # This is done individually for each sample in the batch
    samples = []
    for i in range(depth.shape[0]):
        counts_as_probs = counts_list[i].float() / counts_list[i].sum()
        sample = torch.multinomial(counts_as_probs, n_samples, replacement=True)
        sampled_depths = depths_list[i][sample]

        samples.append(sampled_depths)

    # Convert the samples to x,y coordinates from the individual depth maps
    # This is done individually for each sample in the batch
    coords = []
    for i in range(depth.shape[0]):
        sample_coords = []
        for sample_depth in samples[i]:
            # Get all x,y coordinates with the same depth as the sample depth
            same_depth_coords = torch.nonzero(depth[i].squeeze(0) == sample_depth)

            # Sample a random coordinate from the same_depth_coords
            try:
                coord = same_depth_coords[torch.randint(same_depth_coords.size(0), (1,))]
            except:
                print(f"sample_depth = {sample_depth}")
                print(f"depth = {depth[i][0]}")
                warnings.warn("No coordinates with the same depth as the sampled depth")
                coord = torch.Tensor([0.0, 0.0])

            # Convert coordinate to a relative coordinate wrt to the image size
            coord = (coord.float() + 0.5) / torch.tensor([t.shape[2], t.shape[3]], dtype=torch.float32,
                                                         device=coord.device)

            sample_coords.append(coord)

        # Convert the list of coordinates to a tensor
        coords.append(torch.stack(sample_coords))

    # Convert the list of tensors to a single tensor
    coords = torch.stack(coords)

    return coords


def delete(arr: torch.Tensor, ind: int, dim=None) -> torch.Tensor:
    # From https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L5054-L5243
    ndim = len(arr.size())
    if dim is None:
        if ndim != 1:
            arr = torch.ravel(arr)
        ndim = len(arr.size())
        dim = ndim - 1

    skip = [i for i in range(arr.size(dim)) if i != ind]
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)


def fps_gpu(points, n_samples):
    points_left = torch.arange(len(points))

    sample_inds = torch.zeros(n_samples, dtype=torch.int64, device=points.device)

    dists = torch.ones_like(points_left, device=points.device) * float('inf')

    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = torch.minimum(dist_to_last_added_point,
                                           dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = torch.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = delete(points_left, selected)

        # sample_inds = sample_inds.to(point_device)

    return points[sample_inds], sample_inds


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    point_device = points.device
    th_points = points
    points = np.array(points.cpu())

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return th_points[sample_inds], sample_inds


def depth2points(depth, fov=30, far=5.0):
    factor = 2.0 * torch.tan(torch.tensor([fov], dtype=depth.dtype, device=depth.device) / 2.0)
    aspect = depth.size(-2) / depth.size(-1)
    Y, X = torch.meshgrid(torch.arange(depth.size(-2), device=depth.device),
                          torch.arange(depth.size(-1), device=depth.device))
    Y = factor * depth * (Y - depth.size(-2) / 2.0) / depth.size(-2)
    X = factor * depth * (X - depth.size(-1) / 2.0) / depth.size(-1)
    coords = torch.stack([X, Y, -depth * far])
    return coords


def farthest_point_sampling_depth(t: torch.Tensor, depth: torch.Tensor, n_samples: int, include_feats: bool = False,
                                  gpu: bool = False, batched: bool = False):
    # Downsample depth map to match size of feature map
    # This used to be adaptive_max_pool2d, but that was changed to adaptive_avg_pool2d
    depth = F.adaptive_avg_pool2d(depth, t.shape[-2:])

    # Normalize depth values to between 0 and 1
    """if depth.max() > 1.0:
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)"""

    if batched:
        return batched_farthest_point_sampling_depth(t, depth, n_samples)

    all_coords = []

    for i in range(depth.shape[0]):

        point_cloud = depth2points(depth[i].squeeze(), fov=90).permute(1, 2, 0).reshape(-1, 3)
        feats = t[i].permute(1, 2, 0).reshape(-1, t.shape[1])

        # Sample from the point cloud
        sampled_points, sampled_inds = fps(point_cloud, n_samples ** 2)

        orig_depths = torch.zeros_like(depth[i].squeeze()).reshape(-1)
        orig_depths[sampled_inds] = 1.0

        coords = orig_depths.reshape(depth[i].shape[-2:]).nonzero().float()

        # Normalize coords to between 0 and 1
        coords[:, 0] = coords[:, 0] / depth.shape[-2]
        coords[:, 1] = coords[:, 1] / depth.shape[-1]

        coords = coords.reshape(n_samples, n_samples, 2)

        all_coords.append(coords)

    all_coords = torch.stack(all_coords, dim=0)

    return all_coords


def _apply_fps(tuple):
    depth, feats, n_samples = tuple
    point_cloud = depth2points(depth.squeeze(), fov=90).permute(1, 2, 0).reshape(-1, 3)
    feats = feats.permute(1, 2, 0).reshape(-1, feats.shape[1])

    sampled_points, sampled_inds = fps(point_cloud, n_samples ** 2)

    orig_depths = torch.zeros_like(depth.squeeze()).reshape(-1)
    orig_depths[sampled_inds] = 1.0

    coords = orig_depths.reshape(depth.shape[-2:]).nonzero().float()

    # Normalize coords to between 0 and 1
    coords[:, 0] = coords[:, 0] / depth.shape[-1]
    coords[:, 1] = coords[:, 1] / depth.shape[0]

    coords = coords.reshape(n_samples, n_samples, 2)

    return coords


def batched_farthest_point_sampling_depth(t: torch.Tensor, depth: torch.Tensor, n_samples: int):
    outputs = map(_apply_fps, zip(depth, t, [n_samples] * depth.shape[0]))
    outputs = list(outputs)
    outputs = torch.stack(outputs, dim=0)
    return outputs


def knn_for_coords(feats: torch.Tensor, coords: torch.Tensor, samples_per_coord: int):
    # Calculate the nearest neighbours of each sampled point
    # Return their indices
    new_coords = []

    assert samples_per_coord > 0, "Not enough samples per coord"

    # Get the nearest neighbours of each sampled point
    for i in range(feats.shape[0]):
        # Get anchor feats
        i_coords = torch.clone(coords[i].reshape(-1, 2))
        i_feats = feats[i].permute(1, 2, 0)

        # print(i_coords[:, 0]*i_feats.shape[0], i_coords[:, 1]*i_feats.shape[1])

        anchor_feats = i_feats[
            (i_coords[:, 0] * i_feats.shape[0]).to(torch.int64), (i_coords[:, 1] * i_feats.shape[1]).to(torch.int64)]

        i_nn_coords = []

        # Get neighbour feats
        for feat in anchor_feats:
            # Get the distance between the anchor feat and all other feats
            dists = torch.norm(feat - i_feats, dim=-1)

            # Set all 0s to inf
            dists[dists == 0.0] = float('inf')

            dists_flattened = dists.flatten()
            zeros_like_flattened = torch.zeros_like(dists_flattened)

            # Get the indices of the nearest neighbours
            _, nn_inds = torch.topk(dists_flattened, samples_per_coord + 1, largest=False)

            zeros_like_flattened[nn_inds] = 1
            nn_inds = zeros_like_flattened.reshape(dists.shape).nonzero()

            # Zero the already sampled points
            i_feats[nn_inds] = torch.zeros_like(i_feats[0])

            nn_inds = nn_inds.float()

            nn_inds[:, 0] = nn_inds[:, 0] / i_feats.shape[0]
            nn_inds[:, 1] = nn_inds[:, 1] / i_feats.shape[1]

            i_nn_coords.append(nn_inds)

        nn_coords_stacked = torch.stack(i_nn_coords, dim=0)

        new_coords.append(torch.cat((i_coords, nn_coords_stacked.reshape(-1, 2)), dim=0))

    return torch.stack(new_coords, dim=0)




def fps_depth_feats(points, feats, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points.cpu())
    feats = np.array(feats.cpu())

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        point_dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]
        feat_dist_to_last_added_point = (
                (feats[last_added] - feats[points_left]) ** 2).sum(-1)  # [P - i]

        # Normalize distances
        point_dist_to_last_added_point = point_dist_to_last_added_point / point_dist_to_last_added_point.max()
        feat_dist_to_last_added_point = feat_dist_to_last_added_point / feat_dist_to_last_added_point.max()

        # Combine distances
        dist_to_last_added_point = np.sum((point_dist_to_last_added_point, feat_dist_to_last_added_point), axis=0)

        # print(dist_to_last_added_point)

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds], sample_inds



@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


def create_graph_from_grid(feats: torch.Tensor, distance_measure: str = "cosine"):
    """
    This function takes a grid of features or a depth map as a torch Tensor and creates a graph from it.
    The graph is an undirected graph where all nodes are connected to their nearest neighbors.
    The distance between nodes is defined by the distance_measure parameter.
    grid: (N, H, W, 1) for depth or (N, H, W, D) for features
    distance_measure: "cosine" or ??
    """
    N, H, W = grid.size()

    # Connect each node to its nearest neighbors
    pass


class ContrastiveCorrelationLoss(nn.Module):
    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        # This is the loss function with 0-clamp and shift
        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def depth_feature_correlation(self, c1, c2, d1, d2, shift):
        # Interpolate c1, c2 to the size of d1
        cd = tensor_correlation(norm(c1), norm(c2))

        # Change later to upsample dd
        d1 = torch.nn.functional.interpolate(d1, size=c1.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.nn.functional.interpolate(d2, size=c2.shape[2:], mode='bilinear', align_corners=True)

        # Values already normalized
        dd = depth_correlation(norm(d1), norm(d2))

        # Interpolate cd to dimensions of dd
        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (dd - shift)
        else:
            loss = - cd.clamp(min_val) * (dd - shift)

        return loss, dd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_salience: torch.Tensor, orig_salience_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                depth: torch.Tensor, depth_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

        # use_salience is set to False by default
        if self.cfg.use_salience:
            coords1_nonzero = sample_nonzero_locations(orig_salience, coord_shape)
            coords2_nonzero = sample_nonzero_locations(orig_salience_pos, coord_shape)
            coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > .1).unsqueeze(-1).to(torch.float32)
            coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
            coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)

        elif self.cfg.depth_sampling == 'simple':
            coords1 = simple_depth_informed_sampling(orig_feats, depth, n_samples=self.cfg.feature_samples) * 2 - 1
            coords2 = simple_depth_informed_sampling(orig_feats_pos, depth_pos,
                                                     n_samples=self.cfg.feature_samples) * 2 - 1

        elif self.cfg.depth_sampling == "fps":
            coords1 = farthest_point_sampling_depth(orig_feats, depth, n_samples=self.cfg.feature_samples,
                                                    gpu=self.cfg.fps_gpu) * 2 - 1
            coords2 = farthest_point_sampling_depth(orig_feats_pos, depth_pos, n_samples=self.cfg.feature_samples,
                                                    gpu=self.cfg.fps_gpu) * 2 - 1

            assert coords1.shape == coords2.shape == torch.zeros(
                coord_shape).shape, f"{coords1.shape} != {coords2.shape} != {torch.zeros(coord_shape).shape}"

        elif self.cfg.depth_sampling == "fps_depth_feat":
            coords1 = farthest_point_sampling_depth(orig_feats, depth, n_samples=self.cfg.feature_samples,
                                                    include_feats=True) * 2 - 1
            coords2 = farthest_point_sampling_depth(orig_feats_pos, depth_pos,
                                                    n_samples=self.cfg.feature_samples, include_feats=True) * 2 - 1

        else:
            coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        if self.cfg.depth_feat_correlation_loss:
            depth_feat_loss, depth_feat_cd = self.depth_feature_correlation(
                code, code, depth, depth, self.cfg.depth_feat_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)

        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        if self.cfg.depth_feat_correlation_loss:
            return (pos_intra_loss.mean(),
                    pos_intra_cd,
                    pos_inter_loss.mean(),
                    pos_inter_cd,
                    neg_inter_loss,
                    neg_inter_cd,
                    depth_feat_loss.mean(),
                    depth_feat_cd)
        else:
            return (pos_intra_loss.mean(),
                    pos_intra_cd,
                    pos_inter_loss.mean(),
                    pos_inter_cd,
                    neg_inter_loss,
                    neg_inter_cd)


class DepthContrastiveCorrelationLoss(nn.Module):
    def __init__(self, cfg, ):
        super(DepthContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        # This is the loss function with 0-clamp and shift
        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_salience: torch.Tensor, orig_salience_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                depth_aug_feats: torch.Tensor, depth_aug_feats_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

        # use_salience is set to False by default
        if self.cfg.use_salience:
            coords1_nonzero = sample_nonzero_locations(orig_salience, coord_shape)
            coords2_nonzero = sample_nonzero_locations(orig_salience_pos, coord_shape)
            coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > .1).unsqueeze(-1).to(torch.float32)
            coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
            coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)
        else:
            coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        depth_aug_feats = sample(depth_aug_feats, coords1)
        depth_aug_feats_pos = sample(depth_aug_feats_pos, coords2)

        # Self-correlation loss
        pos_intra_loss, pos_intra_cd = self.helper(
            depth_aug_feats, depth_aug_feats, code, code, self.cfg.pos_intra_shift)

        # k-NN positive correlation loss
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        # random negative correlation loss
        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (pos_intra_loss.mean(),
                pos_intra_cd,
                pos_inter_loss.mean(),
                pos_inter_cd,
                neg_inter_loss,
                neg_inter_cd)


class IntraDepthFeatureDiversityLoss(nn.Module):
    def __init__(self):
        super(IntraDepthFeatureDiversityLoss, self).__init__()

    def forward(self, feats, depth):
        pass


class Decoder(nn.Module):
    def __init__(self, code_channels, feat_channels):
        super().__init__()
        self.linear = torch.nn.Conv2d(code_channels, feat_channels, (1, 1))
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, feat_channels, (1, 1)))

    def forward(self, x):
        return self.linear(x) + self.nonlinear(x)


class NetWithActivations(torch.nn.Module):
    def __init__(self, model, layer_nums):
        super(NetWithActivations, self).__init__()
        self.layers = nn.ModuleList(model.children())
        self.layer_nums = []
        for l in layer_nums:
            if l < 0:
                self.layer_nums.append(len(self.layers) + l)
            else:
                self.layer_nums.append(l)
        self.layer_nums = set(sorted(self.layer_nums))

    def forward(self, x):
        activations = {}
        for ln, l in enumerate(self.layers):
            x = l(x)
            if ln in self.layer_nums:
                activations[ln] = x
        return activations


class ContrastiveCRFLoss(nn.Module):

    def __init__(self, n_samples, alpha, beta, gamma, w1, w2, shift):
        super(ContrastiveCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift

    def forward(self, guidance, clusters):
        device = clusters.device
        assert (guidance.shape[0] == clusters.shape[0])
        assert (guidance.shape[2:] == clusters.shape[2:])
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat([
            torch.randint(0, h, size=[1, self.n_samples], device=device),
            torch.randint(0, w, size=[1, self.n_samples], device=device)], 0)

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]
        coord_diff = (coords.unsqueeze(-1) - coords.unsqueeze(1)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(2)).square().sum(1)

        sim_kernel = self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(- coord_diff / (2 * self.gamma)) - self.shift

        selected_clusters = clusters[:, :, coords[0, :], coords[1, :]]
        cluster_sims = torch.einsum("nka,nkb->nab", selected_clusters, selected_clusters)
        return -(cluster_sims * sim_kernel)

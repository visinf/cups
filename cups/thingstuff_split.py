import torch


# get statistics for thing/stuff split
class ThingStuffSplitter:
    def __init__(self, num_classes_all: int):
        # class distribution pixel wise in instance masks
        self.class_distrib_instances_pixel = torch.zeros(num_classes_all, dtype=torch.int64)
        # pseudo class distribution per instance mask
        self.class_distrib_instances_inst = torch.zeros(num_classes_all, dtype=torch.int64)
        # pseudo class distribution all pixels in dataset
        self.class_distribution = torch.zeros(num_classes_all, dtype=torch.int64)
        self.num_classes_all = num_classes_all

    def update(self, prediction: torch.Tensor):
        prediction = prediction.clone().detach().cpu()
        semantic_pred, instance_pred = prediction[..., 0], prediction[..., 1]
        self.class_distrib_instances_pixel += torch.bincount(
            semantic_pred[instance_pred != 0].cpu(), minlength=self.num_classes_all
        )
        is_per_instance_mask = [semantic_pred[instance_pred == n][0] for n in instance_pred.unique() if n != 0]
        self.class_distrib_instances_inst += torch.bincount(
            torch.Tensor(is_per_instance_mask).long().cpu(), minlength=self.num_classes_all
        )
        self.class_distribution += torch.bincount(semantic_pred.flatten(), minlength=self.num_classes_all)

    def compute(self):
        return self.class_distrib_instances_pixel, self.class_distrib_instances_inst, self.class_distribution

    def reset(self):
        self.class_distrib_instances_pixel = torch.zeros(self.num_classes_all, dtype=torch.int64)
        self.class_distrib_instances_inst = torch.zeros(self.num_classes_all, dtype=torch.int64)
        self.class_distribution = torch.zeros(self.num_classes_all, dtype=torch.int64)

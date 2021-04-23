from glob import glob

import torch
from torch.utils.data import Dataset


class CompositeDataset(Dataset):
    """
    Dataset containing composite images of either the ground truth scenes (class 0)
    or synthetic scenes (class 1)
    """

    def __init__(self, gt_dir, preds_dir):
        """
        gt_dir: path to gt scene composites
        preds_dir: path to synthetic scene composites
        """
        gt_files = glob(f"{gt_dir}/*.pth")
        pred_files = glob(f"{preds_dir}/*.pth")
        # small lists, store in memory and create a fixed label set
        self.files = list(gt_files)[:876] + list(pred_files)
        self.labels = [0] * len(gt_files) + [1] * len(pred_files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = torch.load(self.files[i])
        label = self.labels[i]

        return (img, label)

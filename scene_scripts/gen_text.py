"""
Render GT scenes from their sequences
"""
from collections import defaultdict
import argparse
from tqdm import tqdm
import json

from datasets.suncg_shift_seperate_dataset_deepsynth import SUNCG_Dataset


import torch
import torchtext
from torchvision.transforms import Compose
from transforms.scene import (
    SeqToTensor,
    Get_cat_shift_info,
    AddStartToken,
    Add_Descriptions,
    Add_Relations,
    Add_Glove_Embeddings,
)

from utils.config import read_config


def render(cfg):
    print("Processing GT seqs")
    t = Compose(
        [
            Get_cat_shift_info(cfg),
            Add_Relations(),
            Add_Descriptions(),
            AddStartToken(cfg),
            SeqToTensor(),
            Add_Glove_Embeddings(max_sentences=3),
        ]
    )
    dataset = SUNCG_Dataset(
        data_folder=cfg["data"]["data_path"],
        list_path=cfg["data"]["list_path"],
        transform=t,
    )

    for ndx, sample in enumerate(tqdm(dataset)):
        desc_emb = sample["desc_emb"]

        if ndx == len(dataset) - 2:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to config file")
    args = parser.parse_args()
    cfg = read_config(args.cfg_path)

    render(cfg)

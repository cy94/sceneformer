import os, os.path as osp
import yaml
import copy
import random
import pickle
import csv
import json
from pathlib import Path
import cv2

import numpy as np
import pandas as pd
import transforms3d
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from transforms.scene import SeqToTensor, Padding_joint
from datasets.filter import GlobalCategoryFilter
# from utils.room import get_corners


class SUNCG_Dataset(Dataset):
    def __init__(self, data_folder="datasets/suncg_bedroom/", list_path=None, transform=None):
        self.root = data_folder

        all_files = os.listdir(self.root)
        self.files = list(filter(lambda s: s.endswith('.pkl') and s.split('.')[0].isdigit(), all_files))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ndx):
        # ndx=1059 ,
        # ndx = 1089
        # ndx = 1096
        # ndx = 5
        path = osp.join(self.root, f'{ndx}.pkl')
        with open(path, "rb") as f:
            room_pkl = pickle.load(f)
        # get the floor, wall window door and the room object from the pickle contents

        (floor, wall, window, door, nodes), roomobj = room_pkl
        # (floor, wall,nodes), roomobj = room_pkl

        # make a copy of the object, dont change the original
        # TODO: make a copy in the transform, not here
        room = copy.deepcopy(roomobj)
        # apply transform
        floor[floor > 0.1] = 1
        floor[floor < 0.1] = 0

        # window = torch.from_numpy(window.astype(np.float32))
        # window[window > 0.1] = 1
        # window[window < 0.1] = 0
        #
        # door = torch.from_numpy(door.astype(np.float32))
        # door[door > 0.1] = 1
        # door[door < 0.1] = 0

        # multichannel image
        room_shape = torch.zeros(1, floor.shape[0],floor.shape[1])

        room_shape[0]=floor
        # room_shape[1]=window
        # room_shape[2]=door



        sample = {'floor': room_shape, 'room': room}

        if self.transform:
            sample = self.transform(sample)

        sample['file_path'] = path
        sample['pickle_file'] = f'{ndx}.pkl'
        sample['wall'] = wall
        return sample








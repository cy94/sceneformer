import random
import csv
import json
import copy
import pickle
from collections import Counter, defaultdict

from num2words import num2words
from nltk.tokenize import word_tokenize

import torch
import torchtext
import transforms3d

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
import pandas as pd

from datasets.filter import GlobalCategoryFilter
from datasets.filter import GlobalCategoryFilter
from utils.room import compute_rel
from utils.text_utils import get_article

class Add_Glove_Embeddings(object):
    def __init__(self, max_sentences=3, max_length=40):
        self.max_sentences = max_sentences
        self.glove = torchtext.vocab.GloVe(name="6B", dim=50, cache='/shared/data/.vector_cache')
        self.max_length = max_length

    def __call__(self, sample):
        sentence = ''.join(sample['description'][:self.max_sentences])
        tokens = list(word_tokenize(sentence))
        # pad to maximum length
        tokens += ['<pad>'] * (self.max_length - len(tokens))
        # embed words
        sample['desc_emb'] = torch.cat([self.glove[token].unsqueeze(0) for token in tokens])

        return sample

class Padding(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, seq):
        if self.cfg['model']['start_token'] is not None:
            seq['seq'] = np.hstack(([self.cfg['model']['start_token']], seq['seq']))
        if self.cfg['model']['stop_token'] is not None:
            seq['seq'] = np.hstack((seq['seq'], [self.cfg['model']['stop_token']]))
        if self.cfg['model']['pad_token'] is not None:
            n_pad = self.cfg['model']['max_seq_len'] - len(seq['seq'])
            assert n_pad >= 0
            seq['seq'] = np.pad(seq['seq'], (0, n_pad), mode='constant', constant_values=self.cfg['model']['pad_token'])

        return seq

class SeqToTensor(object):
    def __call__(self, sample):
        del sample['room']
        for i in sample:
            if i not in ('room_type', 'modelids', 'floor', 'house_id', 'room_id', \
                        'bbox', 'cat_name_seq', 'bboxes', 'file_path', 'relations', \
                            'description'):
                if i == 'desc_emb':
                    sample[i] = torch.FloatTensor(sample[i])
                else:
                    sample[i] = torch.LongTensor(sample[i])

        return sample

class Seq_to_Scene(object):
    def __init__(self):
        self.bedroom_filter = GlobalCategoryFilter.get_bedroom_filter()

    def __call__(self, seq):
        seq = seq[1: -1]
        dic = {}
        obj_dic = {}
        obj_num = -1
        if len(seq) % 4 != 0:
            return False
        for i, token in enumerate(seq):
            if token == 201:
                return False
            if i % 4 == 0:
                if token >= len(self.bedroom_filter):
                    return False
                obj_num += 1
                cat = self.bedroom_filter[token]
                obj_dic['class'] = cat
                dic[obj_num] = obj_dic.copy()
            if i % 4 == 1:
                obj_dic['loc'] = seq[i:i+3].tolist()
                dic[obj_num] = obj_dic.copy()
            pass
        return dic

class AddStartToken(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, sample):
        sample['cat_seq'] = np.hstack(([self.cfg['model']['cat']['start_token']], sample['cat_seq']))
        sample['x_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['x_loc_seq']))
        sample['y_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['y_loc_seq']))
        sample['z_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['z_loc_seq']))
        sample['orient_seq'] = np.hstack(([self.cfg['model']['orient']['start_token']], sample['orient_seq']))

        return sample

class Padding_joint(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, sample):
        sample['cat_seq'] = np.hstack(([self.cfg['model']['cat']['start_token']], sample['cat_seq']))
        sample['cat_seq'] = np.hstack((sample['cat_seq'], self.cfg['model']['cat']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['cat_seq'])
        assert n_pad >= 0
        sample['cat_seq'] = np.pad(sample['cat_seq'], (0, n_pad), mode='constant', constant_values=self.cfg['model']['cat']['pad_token'])

        sample['x_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['x_loc_seq']))
        sample['x_loc_seq'] = np.hstack((sample['x_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['x_loc_seq'] = np.pad(sample['x_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['y_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['y_loc_seq']))
        sample['y_loc_seq'] = np.hstack((sample['y_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['y_loc_seq'] = np.pad(sample['y_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['z_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['z_loc_seq']))
        sample['z_loc_seq'] = np.hstack((sample['z_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['z_loc_seq'] = np.pad(sample['z_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['orient_seq'] = np.hstack(([self.cfg['model']['orient']['start_token']], sample['orient_seq']))
        sample['orient_seq'] = np.hstack((sample['orient_seq'], self.cfg['model']['orient']['stop_token']))
        sample['orient_seq'] = np.pad(sample['orient_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['orient']['pad_token'])
        if 'dim_seq' in sample.keys():
            sample['dim_seq'] = np.hstack(([self.cfg['model']['dim']['start_token']], sample['dim_seq']))
            sample['dim_seq'] = np.hstack((sample['dim_seq'], self.cfg['model']['dim']['stop_token']))
            n_pad = self.cfg['model']['max_seq_len'] - len(sample['dim_seq'])
            assert n_pad >= 0
            sample['dim_seq'] = np.pad(sample['dim_seq'], (0, n_pad), mode='constant',
                                       constant_values=self.cfg['model']['dim']['pad_token'])
        return sample

class Padding_shift_loc_model(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, sample):

        sample['curr_cat_seq'] = copy.deepcopy(sample['cat_seq'])
        sample['curr_cat_seq'] = np.hstack(([self.cfg['model']['cat']['start_token']], sample['curr_cat_seq']))
        sample['curr_cat_seq'] = np.hstack((sample['curr_cat_seq'], self.cfg['model']['cat']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['curr_cat_seq'])
        assert n_pad >= 0
        sample['curr_cat_seq'] = np.pad(sample['curr_cat_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['cat']['pad_token'])
        sample['cat_seq'] = np.hstack((sample['cat_seq'], self.cfg['model']['cat']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['cat_seq'])
        assert n_pad >= 0
        sample['cat_seq'] = np.pad(sample['cat_seq'], (0, n_pad), mode='constant', constant_values=self.cfg['model']['cat']['pad_token'])



        sample['loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['loc_seq']))
        sample['loc_seq'] = np.hstack((sample['loc_seq'], self.cfg['model']['coor']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['loc_seq'])
        assert n_pad >= 0
        sample['loc_seq'] = np.pad(sample['loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['orient_seq'] = np.hstack((sample['orient_seq'], self.cfg['model']['orient']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['orient_seq'])
        assert n_pad >= 0
        sample['orient_seq'] = np.pad(sample['orient_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['orient']['pad_token'])

        return sample

class Augment_rotation(object):
    def __init__(self, degree_list):
        self.degree_list = degree_list
    
    def __call__(self, sample):
        room = sample['room']
        floor = sample['floor']
        degree = random.choice(self.degree_list)
        room, floor = self.augment_rot(room, degree, floor)
        sample['room'] = room
        sample['floor'] = floor
        return sample

    def augment_rot(self, room, degree, floor):
        # get transformation matrix
        r = R.from_euler('y', degree, degrees=True)
        ro = r.as_matrix()
        T = np.array([0, 0, 0])
        Z = [1, 1, 1]
        t_rot = transforms3d.affines.compose(T, ro, Z, S=None)

        def update_bbox(node):
            (xmin, zmin, ymin) = list(rotate(np.asarray([node.xmin, node.zmin, node.ymin, 1]))[0:3])
            (xmax, zmax, ymax) = list(rotate(np.asarray([node.xmax, node.zmax, node.ymax, 1]))[0:3])

            if xmin > xmax:
                xmin, xmax = xmax, xmin
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            node.bbox["min"] = (xmin, zmin, ymin)
            node.bbox["max"] = (xmax, zmax, ymax)
            (node.xmin, node.zmin, node.ymin) = node.bbox["min"]
            (node.xmax, node.zmax, node.ymax) = node.bbox["max"]

        def rotate(t):
            t = np.dot(t, t_rot)  
            return t

        room.transform = t_rot
        for node in room.nodes:
            node.transform = list(rotate(np.asarray(node.transform).reshape(4, 4)).flatten())
            update_bbox(node)
        update_bbox(room)

        def rotate_tensor(inputs, x):
            for i in range(inputs.shape[0]):
                inputs[i] = torch.from_numpy(ndimage.rotate(inputs[i], x, reshape=False))
            return inputs

        floor = rotate_tensor(floor, degree)
        return room, floor

class Augment_jitterring(object):
    def __init__(self,jitter_list ):
        self.jitter_list = jitter_list
    def __call__(self, sample):
        room = sample['room']

        jitter_x = random.choice(self.jitter_list)
        jitter_y = random.choice(self.jitter_list)
        jitter_z = random.choice(self.jitter_list)
        jitter = [jitter_x, jitter_y, jitter_z]
        for node in room.nodes:
            node.transform[-4:-1] += np.array(jitter)

        return sample

class Get_loc_shift_info(object):
    def __init__(self, cfg, window_door_first=True):
        self.root = cfg['data']['data_path']
        self.room_size_cap = [6.05, 4.05, 6.05]
        self.model_to_categories = {}
        model_cat_file = "datasets/ModelCategoryMapping.csv"
        self.window_door_first = window_door_first
        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[2], l[3], l[5]]

        with open(f'{self.root}/final_categories_frequency', 'r') as f:
            self.cat_freq = {}
            lines = f.readlines()
            for line in lines:
                self.cat_freq[line.split()[0]] = line.split()[1]

    def __call__(self, sample):
        room = sample['room']
        room_type = room.roomTypes
        objects = room.nodes
        objects_sorted = self.sort_filter_obj(objects, room_type)
        cat_seq = np.array([])

        loc_seq = np.array([])
        orient_seq = np.array([])
        dim_l_seq = np.array([])
        dim_w_seq = np.array([])
        dim_h_seq = np.array([])
        modelids, cat_name_seq, bboxes = [], [], []

        for obj in objects_sorted:
            cat = self.get_final_category(obj.modelId)
            # name = "table", "chair" ..
            cat_idx = np.where(np.array(list(self.cat_freq.keys())) == cat)[0]
            trans = np.array(obj.transform).reshape(4, 4).T
            # shift
            shift = - (np.array(room.bbox['min']) * 0.5 + np.array(room.bbox['max']) * 0.5 - np.array(
                self.room_size_cap) * 1.5 * 0.5)
            # get x, y, z coordinates of objects
            loc = transforms3d.affines.decompose44(trans)[0] + shift
            # get rotation degree
            rot_matrix = transforms3d.affines.decompose44(trans)[1]
            r = R.from_matrix(rot_matrix)
            orient = r.as_euler('yzx', degrees=True)[0]
            # scale
            loc, orient = self.scale(loc=loc, orient=orient)

            if self.window_door_first == True:
                if cat == 'window' or cat == 'door':
                    cat_seq = np.hstack((cat_idx, cat_seq))
                    loc_seq = np.hstack((loc, loc_seq))
                    orient_seq = np.hstack((orient, orient_seq))
                    cat_name_seq.insert(0, cat)
                    bboxes.insert(0, obj.bbox)
                    modelids.insert(0, obj.modelId)
                else:
                    cat_seq = np.hstack((cat_seq, cat_idx))
                    loc_seq = np.hstack((loc_seq, loc))
                    orient_seq = np.hstack((orient_seq, orient))
                    cat_name_seq.append(cat)
                    bboxes.append(obj.bbox)
                    modelids.append(obj.modelId)
            else:
                cat_seq = np.hstack((cat_seq, cat_idx))
                loc_seq = np.hstack((loc_seq, loc))
                orient_seq = np.hstack((orient_seq, orient))
                cat_name_seq.append(cat)
                bboxes.append(obj.bbox)
                modelids.append(obj.modelId)

        cat_seq = np.repeat(cat_seq, 3)
        orient_seq = np.repeat(orient_seq, 3)

        sample['cat_seq'] = cat_seq
        sample['loc_seq'] = loc_seq
        sample['orient_seq'] = orient_seq
        sample['modelids'] = modelids
        sample['bboxes'] = bboxes
        sample['cat_name_seq'] = cat_name_seq
        return sample

    def scale(self, loc, orient=None, dim=None):
        loc *= 200/(np.array(self.room_size_cap) * 1.5)
        if orient is not None:
            orient += 180
        if dim is not None:
            dim *= 100
        return loc, orient

    def sort_filter_obj(self, objects, room_type):
        index_freq_pairs = []
        for index in range(len(objects)):
            obj = objects[index]
            # using the fine cat, can be changed to coarse
            cat = self.get_final_category(obj.modelId)
            # get model freq based on room_type and model cat
            freq = self.cat_freq[cat]
            # set the index to frequency pairs
            index_freq_pairs.append((index, freq))
            index_freq_pairs.sort(key=lambda tup: tup[1], reverse=True)
        # sort objects based on freq
        sorted_objects = [objects[tup[0]] for tup in index_freq_pairs]
        return sorted_objects
    def get_final_category(self, model_id):
        """
        Final categories used in the generated dataset
        Minor tweaks from fine categories
        """
        model_id = model_id.replace("_mirror","")
        category = self.model_to_categories[model_id][0]
        if model_id == "199":
            category = "dressing_table_with_stool"
        if category == "nightstand":
            category = "stand"
        if category == "bookshelf":
            category = "shelving"
        return category

def dict_bbox_to_vec(dict_box):
    '''
    input: {'min': [1,2,3], 'max': [4,5,6]}
    output: [1,2,3,4,5,6]
    '''
    return dict_box['min'] + dict_box['max']

def clean_obj_name(name):
    return name.replace('_', ' ')

class Add_Descriptions(object):
    '''
    Add text descriptions to each scene
    sample['description'] = str is a sentence
    eg: 'The room contains a bed, a table and a chair. The chair is next to the window'
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        sentences = []
        # clean object names once
        obj_names = list(map(clean_obj_name, sample['cat_name_seq']))
        # objects that can be referred to
        refs = []
        # TODO: handle commas, use "and"
        # TODO: don't repeat, get counts and pluralize
        # describe the first 2 or 3 objects
        first_n = random.choice([2, 3])
        # first_n = len(obj_names)
        first_n_names = obj_names[:first_n] 
        first_n_counts = Counter(first_n_names)

        s = 'The room has '
        for ndx, name in enumerate(sorted(set(first_n_names), key=first_n_names.index)):
            if ndx == len(set(first_n_names)) - 1 and len(set(first_n_names)) >= 2:
                s += "and "
            if first_n_counts[name] > 1:
                s += f'{num2words(first_n_counts[name])} {name}s '
            else:
                s += f'{get_article(name)} {name} '
            if ndx == len(set(first_n_names)) - 1:
                s += ". "
            if ndx < len(set(first_n_names)) - 2:
                s += ', '
        sentences.append(s)
        refs = set(range(first_n))

        # for each object, the "position" of that object within its class
        # eg: sofa table table sofa
        #   -> 1    1    2      1
        # use this to get "first", "second"

        seen_counts = defaultdict(int)
        in_cls_pos = [0 for _ in obj_names]
        for ndx, name in enumerate(first_n_names):
            seen_counts[name] += 1
            in_cls_pos[ndx] = seen_counts[name]

        for ndx in range(1, len(obj_names)):
            # higher prob of describing the 2nd object
            prob_thresh = 0.3
                
            if random.random() > prob_thresh:
                # possible backward references for this object
                possible_relations = [r for r in sample['relations'] \
                                        if r[0] == ndx \
                                        and r[2] in refs \
                                        and r[3] < 1.5]
                if len(possible_relations) == 0:
                    continue
                # now future objects can refer to this object
                refs.add(ndx)

                # if we haven't seen this object already
                if in_cls_pos[ndx] == 0:
                    # update the number of objects of this class which have been seen
                    seen_counts[obj_names[ndx]] += 1
                    # update the in class position of this object = first, second ..
                    in_cls_pos[ndx] = seen_counts[obj_names[ndx]]

                # pick any one
                (n1, rel, n2, dist) = random.choice(possible_relations)
                o1 = obj_names[n1]
                o2 = obj_names[n2]

                # prepend "second", "third" for repeated objects
                if seen_counts[o1] > 1:
                    o1 = f'{num2words(in_cls_pos[n1], ordinal=True)} {o1}'
                if seen_counts[o2] > 1:
                    o2 = f'{num2words(in_cls_pos[n2], ordinal=True)} {o2}'

                # dont relate objects of the same kind
                if o1 == o2:
                    continue

                a1 = get_article(o1)

                if 'touching' in rel:
                    if ndx in (1, 2):
                        s = F'The {o1} is next to the {o2}'
                    else:
                        s = F'There is {a1} {o1} next to the {o2}'
                elif rel in ('left of', 'right of'):
                    if ndx in (1, 2):
                        s = f'The {o1} is to the {rel} the {o2}'
                    else:
                        s = f'There is {a1} {o1} to the {rel} the {o2}'
                elif rel in ('surrounding', 'inside', 'behind', 'in front of', 'on', 'above'):
                    if ndx in (1, 2):
                        s = F'The {o1} is {rel} the {o2}'
                    else:
                        s = F'There is {a1} {o1} {rel} the {o2}'
                s += ' . '
                sentences.append(s)

        # set back into the sample
        sample['description'] = sentences
        return sample

class Add_Relations(object):
    '''
    Add relations to sample['relations']
    '''
    def __init__(self):
        pass

    def __call__(self, sample):
        relations = []
        num_objs = len(sample['bboxes'])

        for ndx, this_box in enumerate(sample['bboxes']):
            # only backward relations
            choices = [other for other in range(num_objs) if other < ndx]
            for other_ndx in choices:
                box1 = dict_bbox_to_vec(this_box)
                box2 = dict_bbox_to_vec(sample['bboxes'][other_ndx])

                relation_str, distance = compute_rel(box1, box2)
                if relation_str is not None:
                    relation = (ndx, relation_str, other_ndx, distance)
                    relations.append(relation)
            
        sample['relations'] = relations

        return sample

class Get_cat_shift_info(object):
    def __init__(self, cfg, window_door_first=True, inference = False):
        self.root = cfg['data']['data_path']
        self.room_size_cap = [6.05, 4.05, 6.05]
        self.model_to_categories = {}
        # self.shuffle = cfg['model']['cat']['shuffle']
        model_cat_file = "datasets/ModelCategoryMapping.csv"
        self.window_door_first = window_door_first
        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[2], l[3], l[5]]

        with open(f'{self.root}/final_categories_frequency', 'r') as f:
            # self.cat_freq = json.load(f)
            self.cat_freq = {}
            lines = f.readlines()
            for line in lines:
                self.cat_freq[line.split()[0]] =line.split()[1]

        with open(f'{self.root}/model_dims.pkl', 'rb') as f:
            self.model_dims = pickle.load(f)
        self.inference = inference
    def __call__(self, sample):
        room = sample['room']
        sample['house_id'] = room.house_id
        sample['room_id'] = room.modelId
        sample['bbox'] = room.bbox
        room_type = room.roomTypes
        objects = room.nodes
        # sort objects by frequency
        objects_sorted = self.sort_filter_obj(objects, room_type)
        # init empty sequences
        cat_seq, x_loc_seq, y_loc_seq, z_loc_seq, orient_seq = np.array([]), \
            np.array([]), np.array([]), np.array([]), np.array([])
        dim_seq = np.array([])
        modelids, cat_name_seq, bboxes = [], [], []

        for obj in objects_sorted:
            cat = self.get_final_category(obj.modelId)
            cat_idx = np.where(np.array(list(self.cat_freq.keys())) == cat)[0]
            trans = np.array(obj.transform).reshape(4, 4).T
            # shift
            shift = - (np.array(room.bbox['min']) * 0.5 + np.array(room.bbox['max']) * 0.5 - np.array(
                self.room_size_cap) * 1.5 * 0.5)
            # get x, y, z coordinates of objects
            loc = transforms3d.affines.decompose44(trans)[0] + shift
            # get rotation degree
            rot_matrix = transforms3d.affines.decompose44(trans)[1]
            r = R.from_matrix(rot_matrix)
            orient = r.as_euler('yzx', degrees=True)[0]
            modelId = obj.modelId
            dim = copy.deepcopy(self.model_dims[modelId][:3])
            # scale
            loc, orient, dim = self.scale(loc=loc, orient=orient, dim=dim)
            x, y, z = loc[0], loc[1], loc[2]



            if self.window_door_first == True:
                if cat == 'window' or cat == 'door':
                    x_loc_seq = np.hstack((x, x_loc_seq))
                    y_loc_seq = np.hstack((y, y_loc_seq))
                    z_loc_seq = np.hstack((z, z_loc_seq))
                    cat_seq = np.hstack((cat_idx, cat_seq))
                    orient_seq = np.hstack((orient, orient_seq))
                    cat_name_seq.insert(0, cat)
                    bboxes.insert(0, obj.bbox)
                    modelids.insert(0, obj.modelId)
                    dim_seq = np.hstack((dim, dim_seq))
                else:
                    x_loc_seq = np.hstack((x_loc_seq, x))
                    y_loc_seq = np.hstack((y_loc_seq, y))
                    z_loc_seq = np.hstack((z_loc_seq, z))
                    cat_seq = np.hstack((cat_seq, cat_idx))
                    orient_seq = np.hstack((orient_seq, orient))
                    cat_name_seq.append(cat)
                    bboxes.append(obj.bbox)
                    modelids.append(obj.modelId)
                    dim_seq = np.hstack((dim_seq, dim))
            else:
                x_loc_seq = np.hstack((x_loc_seq, x))
                y_loc_seq = np.hstack((y_loc_seq, y))
                z_loc_seq = np.hstack((z_loc_seq, z))
                cat_seq = np.hstack((cat_seq, cat_idx))
                orient_seq = np.hstack((orient_seq, orient))
                cat_name_seq.append(cat)
                bboxes.append(obj.bbox)
                modelids.append(obj.modelId)
                dim_seq = np.hstack((dim_seq, dim))

        # insert back into the sample
        sample['cat_seq'] = cat_seq
        sample['cat_name_seq'] = cat_name_seq
        sample['x_loc_seq'] = x_loc_seq
        sample['y_loc_seq'] = y_loc_seq
        sample['z_loc_seq'] = z_loc_seq
        sample['orient_seq'] = orient_seq
        sample['modelids'] = modelids
        sample['bboxes'] = bboxes
        if self.inference:
            sample['dim_seq'] = dim_seq
        return sample

    def scale(self, loc, orient=None, dim=None):
        loc *= 200/(np.array(self.room_size_cap) * 1.5)
        if orient is not None:
            orient += 180
        if dim is not None:
            dim *= 100
        return loc, orient, dim


    def sort_filter_obj(self, objects, room_type):
        index_freq_pairs = []
        for index in range(len(objects)):
            obj = objects[index]
            # using the fine cat, can be changed to coarse
            cat = self.get_final_category(obj.modelId)
            # get model freq based on room_type and model cat
            freq = self.cat_freq[cat]
            # set the index to frequency pairs
            index_freq_pairs.append((index, freq))
            index_freq_pairs.sort(key=lambda tup: tup[1], reverse=True)
        # sort objects based on freq
        sorted_objects = [objects[tup[0]] for tup in index_freq_pairs]
        return sorted_objects


    def get_final_category(self, model_id):
        """
        Final categories used in the generated dataset
        Minor tweaks from fine categories
        """
        model_id = model_id.replace("_mirror", "")
        category = self.model_to_categories[model_id][0]
        if model_id == "199":
            category = "dressing_table_with_stool"
        if category == "nightstand":
            category = "stand"
        if category == "bookshelf":
            category = "shelving"
        return category

class Get_dim_shift_info(object):
    def __init__(self, cfg, window_door_first=True):
        self.root = cfg['data']['data_path']
        self.room_size_cap = [6.05, 4.05, 6.05]
        self.model_to_categories = {}
        self.window_door_first = window_door_first
        model_cat_file = "datasets/ModelCategoryMapping.csv"
        with open(model_cat_file, "r") as f:
            categories = csv.reader(f)
            for l in categories:
                self.model_to_categories[l[1]] = [l[2], l[3], l[5]]
        with open(f'{self.root}/final_categories_frequency', 'r') as f:
            # self.cat_freq = json.load(f)
            self.cat_freq = {}
            lines = f.readlines()
            for line in lines:
                self.cat_freq[line.split()[0]] = line.split()[1]
        with open(f'{self.root}/model_dims.pkl', 'rb') as f:
            self.model_dims = pickle.load(f)

    def __call__(self, sample):
        room = sample['room']
        room_type = room.roomTypes
        objects = room.nodes
        objects_sorted = self.sort_filter_obj(objects, room_type)
        cat_seq = np.array([])
        x_loc_seq = np.array([])
        y_loc_seq = np.array([])
        z_loc_seq = np.array([])
        orient_seq = np.array([])
        dim_seq = np.array([])
        modelids = []
        for obj in objects_sorted:
            cat = self.get_final_category(obj.modelId)

            cat_idx = np.where(np.array(list(self.cat_freq.keys())) == cat)[0]
            trans = np.array(obj.transform).reshape(4, 4).T
            # shift
            shift = - (np.array(room.bbox['min']) * 0.5 + np.array(room.bbox['max']) * 0.5 - np.array(
                self.room_size_cap) * 1.5 * 0.5)
            # get x, y, z coordinates of objects
            loc = transforms3d.affines.decompose44(trans)[0] + shift
            # get rotation degree
            rot_matrix = transforms3d.affines.decompose44(trans)[1]
            r = R.from_matrix(rot_matrix)
            orient = r.as_euler('yzx', degrees=True)[0]

            modelId = obj.modelId
            dim = copy.deepcopy(self.model_dims[modelId][:3])


            # scale
            loc, orient, dim = self.scale(loc=loc, orient=orient, dim=dim)

            x, y, z = loc[0], loc[1], loc[2]

            if self.window_door_first == True:
                if cat == 'window' or cat == 'door':
                    x_loc_seq = np.hstack((x, x_loc_seq))
                    y_loc_seq = np.hstack((y, y_loc_seq))
                    z_loc_seq = np.hstack((z, z_loc_seq))
                    cat_seq = np.hstack((cat_idx, cat_seq))
                    dim_seq = np.hstack((dim, dim_seq))
                    orient_seq = np.hstack((orient, orient_seq))
                    modelids.insert(0, obj.modelId)
                else:
                    x_loc_seq = np.hstack((x_loc_seq, x))
                    y_loc_seq = np.hstack((y_loc_seq, y))
                    z_loc_seq = np.hstack((z_loc_seq, z))
                    cat_seq = np.hstack((cat_seq, cat_idx))
                    orient_seq = np.hstack((orient_seq, orient))
                    dim_seq = np.hstack((dim_seq, dim))
                    modelids.append(obj.modelId)

            else:
                x_loc_seq = np.hstack((x_loc_seq, x))
                y_loc_seq = np.hstack((y_loc_seq, y))
                z_loc_seq = np.hstack((z_loc_seq, z))
                cat_seq = np.hstack((cat_seq, cat_idx))
                orient_seq = np.hstack((orient_seq, orient))
                dim_seq = np.hstack((dim_seq, dim))
                modelids.append(obj.modelId)

        cat_seq = np.repeat(cat_seq, 3)
        x_loc_seq = np.repeat(x_loc_seq, 3)
        y_loc_seq = np.repeat(y_loc_seq, 3)
        z_loc_seq = np.repeat(z_loc_seq, 3)
        orient_seq = np.repeat(orient_seq, 3)

        sample['cat_seq'] = cat_seq
        sample['x_loc_seq'] = x_loc_seq
        sample['y_loc_seq'] = y_loc_seq
        sample['z_loc_seq'] = z_loc_seq
        sample['orient_seq'] = orient_seq
        sample['modelids'] = modelids
        sample['dim_seq'] = dim_seq
        return sample

    def scale(self, loc, orient, dim):
        loc *= 200 / (np.array(self.room_size_cap) * 1.5)
        if orient is not None:
            orient += 180
            # orient *= 100
        if dim is not None:
            dim *= 100
        return loc, orient, dim

    def sort_filter_obj(self, objects, room_type):
        index_freq_pairs = []
        for index in range(len(objects)):
            obj = objects[index]
            # using the fine cat, can be changed to coarse
            cat = self.get_final_category(obj.modelId)
            # get model freq based on room_type and model cat
            freq = self.cat_freq[cat]
            # set the index to frequency pairs
            index_freq_pairs.append((index, freq))
            index_freq_pairs.sort(key=lambda tup: tup[1], reverse=True)
        # sort objects based on freq
        sorted_objects = [objects[tup[0]] for tup in index_freq_pairs]
        return sorted_objects

    def get_final_category(self, model_id):
        """
        Final categories used in the generated dataset
        Minor tweaks from fine categories
        """
        model_id = model_id.replace("_mirror", "")
        category = self.model_to_categories[model_id][0]
        if model_id == "199":
            category = "dressing_table_with_stool"
        if category == "nightstand":
            category = "stand"
        if category == "bookshelf":
            category = "shelving"
        return category

class Padding_shift_cat_model(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, sample):

        sample['cat_seq'] = np.hstack(([self.cfg['model']['cat']['start_token']], sample['cat_seq']))
        sample['cat_seq'] = np.hstack((sample['cat_seq'], self.cfg['model']['cat']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['cat_seq'])
        assert n_pad >= 0
        sample['cat_seq'] = np.pad(sample['cat_seq'], (0, n_pad), mode='constant', constant_values=self.cfg['model']['cat']['pad_token'])

        sample['loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['loc_seq']))
        sample['loc_seq'] = np.hstack((sample['loc_seq'], self.cfg['model']['coor']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['loc_seq'])
        assert n_pad >= 0
        sample['loc_seq'] = np.pad(sample['loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['orient_seq'] = np.hstack(([self.cfg['model']['orient']['start_token']], sample['orient_seq']))
        sample['orient_seq'] = np.hstack((sample['orient_seq'], self.cfg['model']['orient']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['orient_seq'])
        assert n_pad >= 0
        sample['orient_seq'] = np.pad(sample['orient_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['orient']['pad_token'])

        return sample

class Padding_shift_ori_model(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, sample):
        sample['cat_seq'] = np.hstack((sample['cat_seq'], self.cfg['model']['cat']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['cat_seq'])
        assert n_pad >= 0
        sample['cat_seq'] = np.pad(sample['cat_seq'], (0, n_pad), mode='constant', constant_values=self.cfg['model']['cat']['pad_token'])

        sample['x_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['x_loc_seq']))
        sample['x_loc_seq'] = np.hstack((sample['x_loc_seq'], self.cfg['model']['coor']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['x_loc_seq'])
        assert n_pad >= 0
        sample['x_loc_seq'] = np.pad(sample['x_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['y_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['y_loc_seq']))
        sample['y_loc_seq'] = np.hstack((sample['y_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['y_loc_seq'] = np.pad(sample['y_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['z_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['z_loc_seq']))
        sample['z_loc_seq'] = np.hstack((sample['z_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['z_loc_seq'] = np.pad(sample['z_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['orient_seq'] = np.hstack(([self.cfg['model']['orient']['start_token']], sample['orient_seq']))
        sample['orient_seq'] = np.hstack((sample['orient_seq'], self.cfg['model']['orient']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['orient_seq'])
        assert n_pad >= 0
        sample['orient_seq'] = np.pad(sample['orient_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['orient']['pad_token'])
        # sample['dim_seq'] = np.hstack(([self.cfg['model']['dim']['start_token']], sample['dim_seq']))
        # sample['dim_seq'] = np.hstack((sample['dim_seq'], self.cfg['model']['dim']['stop_token']))
        # n_pad = self.cfg['model']['max_seq_len'] - len(sample['dim_seq'])
        # assert n_pad >= 0
        # sample['dim_seq'] = np.pad(sample['dim_seq'], (0, n_pad), mode='constant',
        #                            constant_values=self.cfg['model']['dim']['pad_token'])

        return sample

class Padding_shift_loc_joint_model(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, sample):
        sample['cat_seq'] = np.hstack((sample['cat_seq'], self.cfg['model']['cat']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['cat_seq'])
        assert n_pad >= 0
        sample['cat_seq'] = np.pad(sample['cat_seq'], (0, n_pad), mode='constant', constant_values=self.cfg['model']['cat']['pad_token'])

        sample['x_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['x_loc_seq']))
        sample['x_loc_seq'] = np.hstack((sample['x_loc_seq'], self.cfg['model']['coor']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['x_loc_seq'])
        assert n_pad >= 0
        sample['x_loc_seq'] = np.pad(sample['x_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['y_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['y_loc_seq']))
        sample['y_loc_seq'] = np.hstack((sample['y_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['y_loc_seq'] = np.pad(sample['y_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['z_loc_seq'] = np.hstack(([self.cfg['model']['coor']['start_token']], sample['z_loc_seq']))
        sample['z_loc_seq'] = np.hstack((sample['z_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['z_loc_seq'] = np.pad(sample['z_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['orient_seq'] = np.hstack(([self.cfg['model']['orient']['start_token']], sample['orient_seq']))
        sample['orient_seq'] = np.hstack((sample['orient_seq'], self.cfg['model']['orient']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['orient_seq'])
        assert n_pad >= 0
        sample['orient_seq'] = np.pad(sample['orient_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['orient']['pad_token'])

        return sample

class Padding_shift_dim_model(object):
    def __init__(self, cfg):
        self.cfg = cfg
    def __call__(self, sample):
        sample['cat_seq'] = np.hstack((sample['cat_seq'], self.cfg['model']['cat']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['cat_seq'])
        assert n_pad >= 0
        sample['cat_seq'] = np.pad(sample['cat_seq'], (0, n_pad), mode='constant', constant_values=self.cfg['model']['cat']['pad_token'])


        sample['x_loc_seq'] = np.hstack((sample['x_loc_seq'], self.cfg['model']['coor']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['x_loc_seq'])
        assert n_pad >= 0
        sample['x_loc_seq'] = np.pad(sample['x_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['y_loc_seq'] = np.hstack((sample['y_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['y_loc_seq'] = np.pad(sample['y_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['z_loc_seq'] = np.hstack((sample['z_loc_seq'], self.cfg['model']['coor']['stop_token']))
        sample['z_loc_seq'] = np.pad(sample['z_loc_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['coor']['pad_token'])

        sample['orient_seq'] = np.hstack((sample['orient_seq'], self.cfg['model']['orient']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['orient_seq'])
        assert n_pad >= 0
        sample['orient_seq'] = np.pad(sample['orient_seq'], (0, n_pad), mode='constant',
                                   constant_values=self.cfg['model']['orient']['pad_token'])

        sample['dim_seq'] = np.hstack(([self.cfg['model']['dim']['start_token']], sample['dim_seq']))
        sample['dim_seq'] = np.hstack((sample['dim_seq'], self.cfg['model']['dim']['stop_token']))
        n_pad = self.cfg['model']['max_seq_len'] - len(sample['dim_seq'])
        assert n_pad >= 0
        sample['dim_seq'] = np.pad(sample['dim_seq'], (0, n_pad), mode='constant',
                                     constant_values=self.cfg['model']['dim']['pad_token'])

        return sample

class Sentence_to_indices(object):

    def __init__(self, max_sentences=3, max_length=40, cfg=None):

        self.max_sentences = max_sentences
        self.max_length = max_length
        self.data_folder = cfg['data']['data_path']
        with open(f'{self.data_folder}/voc_dic.pkl', 'rb') as f:
            self.dic = pickle.load(f)

    def __call__(self, sample):
        """transform the description into indices"""
        sentence = ''.join(sample['description'][:self.max_sentences])
        tokens = list(word_tokenize(sentence))
        # pad to maximum length
        tokens += ['<pad>'] * (self.max_length - len(tokens))
        indices = []
        for symbol in tokens:
            if symbol not in self.dic.keys():
                with open(f'{self.data_folder}/voc_dic.pkl', 'rb') as f:
                    self.dic = pickle.load(f)
                    self.dic[symbol] = len(self.dic)
                with open(f'{self.data_folder}/voc_dic.pkl', 'wb') as f:
                    pickle.dump(self.dic, f, pickle.HIGHEST_PROTOCOL)
            indices.append(self.dic[symbol])
        sample['text_indices'] = indices
        return sample
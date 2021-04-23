from separate_models.scene_shift_cat import scene_transformer as cat_model_pre
from separate_models.scene_shift_ori_col import scene_transformer as ori_model_pre
from separate_models.scene_shift_loc_col import scene_transformer as loc_model_pre
from separate_models.scene_shift_dim import scene_transformer as dim_model_pre
import json
import csv
import os
import pickle
import copy
import torch
import os.path as osp
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d
import trimesh
from tqdm import tqdm
import yaml
from transforms.scene import (
    SeqToTensor,
    Get_cat_shift_info,
    Padding_joint,
    Augment_rotation
)
from torchvision.transforms import Compose
from torch.utils.data import Subset
from datasets.suncg_shift_seperate_dataset_deepsynth import SUNCG_Dataset
import logging
from data.house import Room, Node
from data.top_down import TopDownView
from data.dataset import toimage
from data.rendered import RenderedScene




logging.getLogger("trimesh").setLevel(logging.ERROR)

compare = True

mode = 'gen'
room_size_cap = [6.05, 4.05, 6.05]
collision_threshold = 1 #0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
room_type = 'bedroom'
adding_wall = True
run = 'floor_1'





if room_type == 'living':
    path_cat = 'lightning_logs/version_401/hparams.yaml'
    path_loc = "lightning_logs/version_395/hparams.yaml"
    path_ori = 'lightning_logs/version_403/hparams.yaml'
    path_dim = "lightning_logs/version_352/hparams.yaml"
    model_type = 'no_window_door'
elif room_type == 'bedroom':
    path_cat = 'lightning_logs/version_606/hparams.yaml'
    path_loc = "lightning_logs/version_609/hparams.yaml"
    path_ori = 'lightning_logs/version_611/hparams.yaml'
    path_dim = "lightning_logs/version_583/hparams.yaml"
    model_type = 'with_window_door'

with open(path_dim) as f:
    cfg = yaml.safe_load(f)

root = cfg["data"]["data_path"]

out_path = cfg["test"]["log_dir"]
if compare == True:
    out_path = f'/home/xinpeng/dlcv-ss20/compare_{room_type}/'

if not os.path.exists(out_path):
    os.makedirs(out_path)

IMG_OUT_DIR = f"/home/xinpeng/dlcv-ss20/compare_{room_type}/topdown_{run}/"
os.makedirs(osp.join(IMG_OUT_DIR, "gt"), exist_ok=True)
os.makedirs(osp.join(IMG_OUT_DIR, "preds"), exist_ok=True)

RENDER_OUT_DIR = f"/home/xinpeng/dlcv-ss20/compare_{room_type}/composite_{run}/"
os.makedirs(osp.join(RENDER_OUT_DIR, "gt"), exist_ok=True)
os.makedirs(osp.join(RENDER_OUT_DIR, "preds"), exist_ok=True)



os.makedirs(osp.join(out_path, f'scene_{run}', "gt"), exist_ok=True)
os.makedirs(osp.join(out_path, f'scene_{run}', "preds"), exist_ok=True)

os.makedirs(osp.join(out_path, f'seq_{run}', "gt"), exist_ok=True)
os.makedirs(osp.join(out_path, f'seq_{run}', "preds"), exist_ok=True)


def model_prepare(model, path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    model = model(cfg)
    model_path = osp.join(cfg["test"]["model_file"])
    state_dict = torch.load(model_path,map_location=torch.device('cpu'))["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

cat_model = model_prepare(cat_model_pre, path_cat)
loc_model = model_prepare(loc_model_pre, path_loc)
ori_model = model_prepare(ori_model_pre, path_ori)
dim_model = model_prepare(dim_model_pre, path_dim)

def update(gen_seq, next_token):
    gen_seq.append(next_token)
    return gen_seq

def update_repeat(repeat_gen_seq, next_token):
    for i in range(3):
        repeat_gen_seq.append(next_token)
    return repeat_gen_seq


def get_model_cat(cat):
    if cat == "stand":
        cat = ["nightstand", "stand"]
    if cat == "shelving":
        cat = ["bookshelf", "shelving"]
    return cat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def get_intersection_area(bbox_0, bbox_1):
    """
    compute the intersection area of 2 bbox
    :param bbox_0:
    :param bbox_1:
    :return: interArea
    """
    x_min = max(bbox_0[0, 0], bbox_1[0, 0])
    y_min = max(bbox_0[0, 1], bbox_1[0, 1])
    z_min = max(bbox_0[0, 2], bbox_1[0, 2])

    x_max = min(bbox_0[1, 0], bbox_1[1, 0])
    y_max = min(bbox_0[1, 1], bbox_1[1, 1])
    z_max = min(bbox_0[1, 2], bbox_1[1, 2])

    interArea = abs(
        max((x_max - x_min, 0)) * max((y_max - y_min), 0) * max(((z_max - z_min), 0))
    )

    return interArea

def get_final_category(model_id):
    """
    Final categories used in the generated dataset
    Minor tweaks from fine categories
    """
    model_id = model_id.replace("_mirror", "")
    category = model_to_categories[model_id][0]
    if model_id == "199":
        category = "dressing_table_with_stool"
    if category == "nightstand":
        category = "stand"
    if category == "bookshelf":
        category = "shelving"
    return category

def vals_to_node(model_id, transform, mesh):
    """
    Create a Node object from the given values
    """
    # get the bbox of this object
    bbox = {}
    bbox["min"], bbox["max"] = mesh.bounds

    transform_flat = transform.T.reshape((-1)).tolist()
    transform_flat[-4:-1] -= 0.5 * 0.5* np.array([6.05, 4.05, 6.05])
    bbox["min"] = bbox["min"]- 0.5 * 0.5 * np.array([6.05, 4.05, 6.05])
    bbox["max"] = bbox["max"] - 0.5 * 0.5 * np.array([6.05, 4.05, 6.05])
    # all the values required in the Node
    node_dict = {
        "parent": None,
        "child": [],
        "bbox": bbox,
        # column major format
        "transform": transform_flat,
        "modelId": model_id,
        "type": "Object",
        "valid": 1,
    }
    # level is not used
    node = Node(node_dict, level=None)
    return node
class MyObject(object):
    pass

def create_room(nodes):
    """
    nodes: list of Node objects contained in this room

    return: Room object
    """
    # create a fake level object
    level = MyObject()
    level.house = MyObject()
    level.house.id = "0004d52d1aeeb8ae6de39d6bd993e992"

    # create a Node to represent the room
    room_dict = {
        "bbox": {"min": [0, 0, 0], "max": room_size_cap},
        "child": [],
        "house_id": "empty",
        "id": "1",
        "modelId": "fr_0rm_2",
        "nodeIndices": list(range(len(nodes))),
        "nodes": [],
        "roomTypes": ["Bedroom"],
        "type": "Room",
    }

    room_node = Node(room_dict, level=None)
    # now create a Room object with the same info
    r = Room(room=room_node, nodes=nodes, level=level)
    return r


def add_wall(scene, house_id, room_id, bbox):

    # read wall mesh
    path_root = f"/shared/data/suncg/room/{house_id}/"
    floor_id = f"{room_id}f.obj"
    wall_id = f"{room_id}w.obj"
    ceiling_id = f"{room_id}c.obj"


    T = -(
        np.array(bbox["min"]) * 0.5
        + np.array(bbox["max"]) * 0.5
        - np.array(room_size_cap) * 1.5 * 0.5
    )

    orient = 0
    r = R.from_euler("y", orient, degrees=True)  # .cpu(), degrees=True)

    Rotation = r.as_matrix()
    Z = [1, 1, 1]
    transform = transforms3d.affines.compose(T, Rotation, Z, S=None)

    floor_file = osp.join(path_root, floor_id)
    floor_mesh = trimesh.load_mesh(floor_file)
    wall_file = osp.join(path_root, wall_id)
    wall_mesh = trimesh.load_mesh(wall_file)


    # transform the wall
    wall_mesh.apply_transform(transform)
    scene.add_geometry(wall_mesh)
    for i in scene.geometry.items():
        bbox_dic[i[0]] = i[1].bounds
    floor_mesh.apply_transform(transform)
    scene.add_geometry(floor_mesh)


    ceiling_file = osp.join(path_root, ceiling_id)
    if os.path.isfile(ceiling_file):
        ceiling_mesh = trimesh.load_mesh(ceiling_file)
        ceiling_mesh.apply_transform(transform)
        scene.add_geometry(ceiling_mesh)

    return scene, bbox_dic


# def wall_collision_detection()
with open(f"{root}/final_categories_frequency", "r") as f:
    cat_freq = {}
    lines = f.readlines()
    for line in lines:
        cat_freq[line.split()[0]] = line.split()[1]


with open(f"{root}/model_frequency", "r") as f:
    lines = f.readlines()
    models = np.array([line.split()[0] for line in lines])
    model_freq = [int(l[:-1].split()[1]) for l in lines]
with open(f"{root}/model_dims.pkl", "rb") as f:
    model_dims = pickle.load(f)

cat_list = list(cat_freq.keys())
model_cat_file = "datasets/ModelCategoryMapping.csv"
model_to_categories = {}
with open(model_cat_file, "r") as f:
    categories = csv.reader(f)
    for l in categories:
        model_to_categories[l[1]] = [l[2], l[3], l[5]]
def add_object_collision_check(
    scene, cat, ori, x, y, z, dim_l, dim_h, dim_w, ndx, checker, bbox_dic, room_bbox, obj_nodes, method='iou'
):
    def filter_models(possible_models):
        filtered_models = copy.deepcopy(possible_models)
        freq_all = []
        for i in possible_models:
            if np.where(models == i)[0] > 0:
                index = np.where(models == i)[0][0]
                freq = model_freq[index]
                freq_all.append(freq)
        total_freq = sum(freq_all)

        # filter out model that's not the freq list
        for i in possible_models:
            if len(np.where(models == i)[0]) == 0:
                filtered_models = filtered_models[filtered_models != i]

        # filter out model that has low freq
        for i in possible_models:
            if np.where(models == i)[0] > 0:
                index = np.where(models == i)[0][0]
                freq = model_freq[index]
                if freq / total_freq <= 0.01:
                    filtered_models = filtered_models[filtered_models != i]

        return filtered_models

    model_cat_file = "datasets/ModelCategoryMapping.csv"
    df = pd.read_csv(model_cat_file)
    obj_path_root = "/shared/data/suncg/object"  
    cat_name = cat_list[cat]
    T = np.array([x * 6.05 * 1.5 / 200, y * 4.05 * 1.5 / 200, z * 6.05 * 1.5 / 200])
    orient = ori - 180
    r = R.from_euler("y", orient.cpu(), degrees=True)  # .cpu(), degrees=True)

    Rotation = r.as_matrix()
    Z = [1, 1, 1]
    transform = transforms3d.affines.compose(T, Rotation, Z, S=None)

    # given the index, get the cat
    category = cat_list[cat]
    dim = [dim_l, dim_h, dim_w]

    # some models with different cat is regarded as the same cat during training, here it is reversed, one cat could
    # link to different model categories
    model_cat = get_model_cat(category)

    # get all possible models based on the cat
    if type(model_cat) == list:
        possible_models = pd.concat(
            [
                df[df["fine_grained_class"] == model_cat[0]]["model_id"],
                df[df["fine_grained_class"] == model_cat[1]]["model_id"],
            ]
        )
    else:
        possible_models = df[df["fine_grained_class"] == model_cat]["model_id"]

    # filter out models with low freq
    models_filtered = filter_models(possible_models)

    # compute the l2 score with all relevant model
    stored_l2 = {}
    for model in models_filtered:
        if model in model_dims.keys():
            model_dim = model_dims[model] * 100
            l2 = (
                (dim[0] - model_dim[0]) ** 2
                + (dim[1] - model_dim[1]) ** 2
                + (dim[2] - model_dim[2]) ** 2
            )
            stored_l2[model] = l2

    # # sort the model based on l2 score
    model_list = sorted(stored_l2, key=lambda k: stored_l2[k])

    # # sort model based on freq
    # index_map = {v: i for i, v in enumerate(models)}
    # sorted_list = sorted(stored_l2.items(), key=lambda pair: index_map[pair[0]])

    # filter out models with too large l2 score
    filtered_list = []
    for i in model_list:
        # if stored_l2[i] < 40:
            filtered_list.append(i)

    # check collision for the model in filtered_list
    collision = False
    for i in range(len(filtered_list)):
        model_id = filtered_list[i]
        file = f"{model_id}/{model_id}.obj"
        obj_path = osp.join(obj_path_root, file)
        mesh = trimesh.load_mesh(obj_path)
        mesh.apply_transform(transform)
        mesh_trimesh = as_mesh(mesh)
        if category == 'window' or category == 'door' or mode == 'GT':
            collision = False
        elif method == 'trimesh':
            raise NotImplementedError
        elif method == 'iou':
            for bbox in bbox_dic:
                # compute the intersection area of two bbox
                interArea = get_intersection_area(mesh_trimesh.bounds, bbox_dic[bbox])
                if interArea > collision_threshold:
                    collision = True
                    break
            if not collision:
                bbox_dic[ndx] = mesh_trimesh.bounds
        # if no collision, insert the model into scene
        if not collision:
            scene.add_geometry(geometry=mesh)
            # create a "Node" object
            node = vals_to_node(model_id, transform, mesh)
            obj_nodes.append(node)
            break
    return scene, collision, bbox_dic, obj_nodes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t = Compose(
    # [   Augment_rotation(cfg['train']['aug']['rotation_list']),
    [
        Get_cat_shift_info(cfg, window_door_first=True, inference = True),
        Padding_joint(cfg),
        SeqToTensor(),
    ]
)

trainval_set = SUNCG_Dataset(
    data_folder=cfg["data"]["data_path"],
    list_path=cfg["data"]["list_path"],
    transform=t,
)
total_len = len(trainval_set) - 2
train_len = int(0.8 * total_len)
train_set = Subset(trainval_set, range(train_len))
val_set = Subset(trainval_set, range(train_len, total_len))

out_number = 200

for out_scene_ndx in tqdm(range(400,1000)):
    # out_scene_ndx += 400
    # out_scene_ndx = 125
    sample = trainval_set[out_scene_ndx]
    cat_GT = copy.deepcopy(sample["cat_seq"][1:])
    ori_GT = copy.deepcopy(sample["orient_seq"][1:])
    x_GT = copy.deepcopy(sample["x_loc_seq"][1:])
    y_GT = copy.deepcopy(sample["y_loc_seq"][1:])
    z_GT = copy.deepcopy(sample["z_loc_seq"][1:])
    dim_GT = copy.deepcopy(sample["dim_seq"][1:])

    room_shape = sample["floor"]

    cat_gen_seq = [cfg["model"]["cat"]["start_token"]]
    x_gen_seq = [cfg["model"]["coor"]["start_token"]]
    y_gen_seq = [cfg["model"]["coor"]["start_token"]]
    z_gen_seq = [cfg["model"]["coor"]["start_token"]]
    ori_gen_seq = [cfg["model"]["orient"]["start_token"]]

    # seq for loc model and dim model
    loc_gen_seq = [cfg["model"]["coor"]["start_token"]]
    dim_gen_seq = [cfg["model"]["dim"]["start_token"]]
    cat_repeat_gen_seq = []
    ori_repeat_gen_seq = []
    x_repeat_gen_seq = []
    y_repeat_gen_seq = []
    z_repeat_gen_seq = []

    decoder_seq_len = cfg["model"]["max_seq_len"]
    generate_seq_len = decoder_seq_len - 1

    # create empty scene
    scene = trimesh.scene.scene.Scene()
    # create CollisionManager
    collision_check = trimesh.collision.CollisionManager()
    bbox_dic = {}
    out_ndx = 0


    if adding_wall == True:
    # add wall mesh into scene
        scene, bbox_dic = add_wall(
            scene, sample["house_id"], sample["room_id"], sample["bbox"]
        )
    resample_count = 0
    print_ = True
    file_name = sample['pickle_file']
    obj_nodes = []
    for i in range(33):

        cat_count = 0
        cat_GT_curr = cat_list[cat_GT[out_ndx]] if cat_GT[out_ndx] < cfg['model']['cat']['pad_token'] else None
        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            # get the door and windows from GT
            cat = cat_GT[out_ndx]
        else:
            # don't generate anymore door windows
            while True:
                cat = cat_model.decode_multi_model(
                    out_ndx,
                    cat_gen_seq,
                    x_gen_seq,
                    y_gen_seq,
                    z_gen_seq,
                    ori_gen_seq,
                    room_shape=room_shape,
                )
                cat_count += 1
                if cat != cfg['model']['cat']['pad_token']-1 and cat != cfg['model']['cat']['pad_token']-2 and model_type=='with_window_door':
                    break
                elif model_type == 'no_window_door':
                    break
                if cat_count > 500:
                    cat = 999
                    print_ = False
                    break

        if cat == 999:
            break
        cat_gen_seq_old = copy.deepcopy(cat_gen_seq)
        cat_repeat_gen_seq_old = copy.deepcopy(cat_repeat_gen_seq)
        cat_gen_seq = update(cat_gen_seq, cat)
        cat_repeat_gen_seq = update_repeat(cat_repeat_gen_seq, cat)

        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            ori = ori_GT[out_ndx]
        else:
            ori = ori_model.decode_multi_model(
                out_ndx,
                cat_gen_seq[1:],
                x_gen_seq,
                y_gen_seq,
                z_gen_seq,
                ori_gen_seq,
                room_shape=room_shape,
        )

        if ori == 999:
            break
        ori_gen_seq_old = copy.deepcopy(ori_gen_seq)
        ori_repeat_gen_seq_old = copy.deepcopy(ori_repeat_gen_seq)
        ori_gen_seq = update(ori_gen_seq, ori)
        ori_repeat_gen_seq = update_repeat(ori_repeat_gen_seq, ori)

        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            x = x_GT[out_ndx]
        else:
            x = loc_model.decode_multi_model(
                3 * out_ndx,
                cat_repeat_gen_seq,
                loc_gen_seq,
                ori_repeat_gen_seq,
                room_shape=room_shape,
            )
        if x == 999:
            break
        loc_gen_seq_old = copy.deepcopy(loc_gen_seq)
        x_gen_seq_old = copy.deepcopy(x_gen_seq)
        x_repeat_gen_seq_old = copy.deepcopy(x_repeat_gen_seq)
        loc_gen_seq = update(loc_gen_seq, x)
        x_gen_seq = update(x_gen_seq, x)
        x_repeat_gen_seq = update_repeat(x_repeat_gen_seq, x)

        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            y = y_GT[out_ndx]
        else:
            y = loc_model.decode_multi_model(
                3 * out_ndx + 1,
                cat_repeat_gen_seq,
                loc_gen_seq,
                ori_repeat_gen_seq,
                room_shape=room_shape,
            )
        if y == 999:
            break
        y_gen_seq_old = copy.deepcopy(y_gen_seq)
        y_repeat_gen_seq_old = copy.deepcopy(y_repeat_gen_seq)
        loc_gen_seq = update(loc_gen_seq, y)
        y_gen_seq = update(y_gen_seq, y)
        y_repeat_gen_seq = update_repeat(y_repeat_gen_seq, y)

        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            z = z_GT[out_ndx]
        else:

            z = loc_model.decode_multi_model(
                3 * out_ndx + 2,
                cat_repeat_gen_seq,
                loc_gen_seq,
                ori_repeat_gen_seq,
                room_shape=room_shape,
            )
        if z == 999:
            break
        z_gen_seq_old = copy.deepcopy(z_gen_seq)
        z_repeat_gen_seq_old = copy.deepcopy(z_repeat_gen_seq)
        loc_gen_seq = update(loc_gen_seq, z)
        z_gen_seq = update(z_gen_seq, z)
        z_repeat_gen_seq = update_repeat(z_repeat_gen_seq, z)

        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            dim_l = dim_GT[3*out_ndx]
        else:
            dim_l = dim_model.decode_multi_model(
                3 * out_ndx,
                cat_repeat_gen_seq,
                x_repeat_gen_seq,
                y_repeat_gen_seq,
                z_repeat_gen_seq,
                ori_repeat_gen_seq,
                dim_gen_seq,
                room_shape=room_shape,
            )

        dim_gen_seq_old = copy.deepcopy(dim_gen_seq)
        dim_gen_seq = update(dim_gen_seq, dim_l)

        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            dim_h = dim_GT[3 * out_ndx + 1]
        else:
            dim_h = dim_model.decode_multi_model(
                3 * out_ndx + 1,
                cat_repeat_gen_seq,
                x_repeat_gen_seq,
                y_repeat_gen_seq,
                z_repeat_gen_seq,
                ori_repeat_gen_seq,
                dim_gen_seq,
                room_shape=room_shape,
            )

        dim_gen_seq = update(dim_gen_seq, dim_h)

        if cat_GT_curr == 'window' or cat_GT_curr == 'door' or mode == 'GT':
            dim_w = dim_GT[3 * out_ndx + 2]
        else:
            dim_w = dim_model.decode_multi_model(
                3 * out_ndx + 2,
                cat_repeat_gen_seq,
                x_repeat_gen_seq,
                y_repeat_gen_seq,
                z_repeat_gen_seq,
                ori_repeat_gen_seq,
                dim_gen_seq,
                room_shape=room_shape,
            )
        dim_gen_seq = update(dim_gen_seq, dim_w)

        # check if the new generated object collides with the scene. If not, add the mesh into the scene
        scene, collision, bbox_dic, obj_nodes = add_object_collision_check(
            scene,
            cat,
            ori,
            x,
            y,
            z,
            dim_l,
            dim_h,
            dim_w,
            out_ndx,
            collision_check,
            bbox_dic,
            sample["bbox"],
            obj_nodes=obj_nodes
        )

        # if re-selecting the model and re-sampling  doesn't work, remove the current object
        if collision == True:
            cat_gen_seq = copy.deepcopy(cat_gen_seq_old)
            cat_repeat_gen_seq = copy.deepcopy(cat_repeat_gen_seq_old)
            ori_gen_seq = copy.deepcopy(ori_gen_seq_old)
            ori_repeat_gen_seq = copy.deepcopy(ori_repeat_gen_seq_old)
            loc_gen_seq = copy.deepcopy(loc_gen_seq_old)
            x_gen_seq = copy.deepcopy(x_gen_seq_old)
            x_repeat_gen_seq = copy.deepcopy(x_repeat_gen_seq_old)
            y_gen_seq = copy.deepcopy(y_gen_seq_old)
            y_repeat_gen_seq = copy.deepcopy(y_repeat_gen_seq_old)
            z_gen_seq = copy.deepcopy(z_gen_seq_old)
            z_repeat_gen_seq = copy.deepcopy(z_repeat_gen_seq_old)
            dim_gen_seq = copy.deepcopy(dim_gen_seq_old)
            # add a resample count, if resample 5 times fails, discard the scene
            resample_count += 1
            if resample_count >= 50:
                print_ = False
                break


        else:
            out_ndx += 1
        if mode == 'GT' and cat_GT[out_ndx] == cfg['model']['cat']['stop_token']:
            break
        # print(len(cat_gen_seq))


    # now create a Room object with all the nodes
    room = create_room(obj_nodes)
    renderer = TopDownView(size=512)
    img, data = renderer.render(room)

    _, _, nodes = data

    data = (sample['floor'].squeeze(0)*0.1209, sample['wall']*0.711, nodes)


    floor_img = sample['floor'].squeeze(0).numpy() * 0.711
    floor_rgb = toimage(floor_img, cmin=0, cmax=1)
    floor_out_path = osp.join(IMG_OUT_DIR, 'preds',f"{out_scene_ndx}_floor.jpg")

    img += sample['floor'].squeeze(0).numpy()*0.1209 + sample['wall'].numpy()*0.711
    img_rgb = toimage(img, cmin=0, cmax=1)
    img_out_path = osp.join(IMG_OUT_DIR, 'preds',f"{out_scene_ndx}.jpg")

    # if using old model, read the windows and doors
    if model_type == 'no_window_door':
        # read the pickle file in the new datafoler
        path = osp.join(cfg['data']['data_path'], file_name)
        with open(path, "rb") as f:
            room_pkl = pickle.load(f)
        (floor, wall, window, door, nodes), room = room_pkl
        bbox = room.bbox
        # seach for window and door
        for i in room.nodes:
            cat = get_final_category(i.modelId)
            if cat == 'window' or cat == 'door':
                # add the window door into the scene:
                model_id = i.modelId.replace("_mirror", "")
                obj_path_root = "/shared/data/suncg/object"
                file = f"{model_id}/{model_id}.obj"
                obj_path = osp.join(obj_path_root, file)
                mesh = trimesh.load_mesh(obj_path)

                transform = np.array(i.transform).reshape((4, 4)).T
                T = -(
                        np.array(bbox["min"]) * 0.5
                        + np.array(bbox["max"]) * 0.5
                        - np.array(room_size_cap) * 1.5 * 0.5
                )
                transform[0:3, 3] += T
                mesh.apply_transform(transform)
                scene.add_geometry(geometry=mesh)

    output = {}
    output["cat_seq"] = cat_gen_seq[1:]
    output["x_loc_seq"] = x_gen_seq[1:]
    output["y_loc_seq"] = y_gen_seq[1:]
    output["z_loc_seq"] = z_gen_seq[1:]
    output["dim_seq"] = dim_gen_seq[1:]
    output["orient_seq"] = ori_gen_seq[1:]
    output['house_id'] = sample["house_id"]
    output['room_id'] = sample["room_id"]
    output['cat_name_seq'] = sample['cat_name_seq']

    if print_:
        pickle_file = osp.join(out_path, f'seq_{run}','preds', f"{out_scene_ndx}.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

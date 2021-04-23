from datasets.ply_dataset import PLYDataset
from datasets.shapenet_dataset import ShapeNetDataset


def get_dataset(cfg, transform=None):
    dataset_name = cfg["name"]

    if dataset_name == "ply":
        dataset = PLYDataset(cfg["data_path"], cfg["list_path"], transform=transform)
    elif dataset_name == "shapenet":
        dataset = ShapeNetDataset(
            cfg["data_path"], categories=cfg["categories"], transform=transform
        )
    else:
        raise NotImplementedError

    return dataset

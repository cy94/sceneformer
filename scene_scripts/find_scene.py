from datasets.suncg_shift_seperate_dataset_deepsynth import SUNCG_Dataset
from torchvision.transforms import Compose
from transforms.scene import (
    SeqToTensor,
    Get_cat_shift_info,
    Padding_joint,
)
from pytorch_lightning import Trainer, seed_everything
import yaml
from tqdm import tqdm
seed_everything(1)
target_house_id = "81d5da079703cd75e1c0d33a21215299"

path_dim = "tests/configs/separate/scene_shift_cat_config.yaml"
with open(path_dim) as f:
    cfg = yaml.safe_load(f)


t = Compose(
    [
        Get_cat_shift_info(cfg, window_door_first=True),
        Padding_joint(cfg),
        SeqToTensor(),
    ]
)


dataset = SUNCG_Dataset(
    data_folder=cfg["data"]["data_path"],
    list_path=None,
    transform=t,
)

for id, sample in tqdm(enumerate(dataset)):
    sample_house_id = sample['house_id']
    if sample_house_id == target_house_id:
        print(id)
        print('True')
# sample = dataset[137]
pass
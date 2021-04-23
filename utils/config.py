import yaml


def read_config(path):
    """
    path: path to config yaml file
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg

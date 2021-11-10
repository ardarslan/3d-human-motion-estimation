import yaml


def read_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

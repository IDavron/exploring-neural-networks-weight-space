from pathlib import Path
import yaml
from dotenv import dotenv_values

def get_default_dirs():
    ROOT_DIR = Path(__file__).parent.parent.parent

    default_dirs = {
        "root_dir": ROOT_DIR,
        "datasets_dir": f"{ROOT_DIR}/data",
        "models_dir": f"{ROOT_DIR}/models",
        "configs_dir": f"{ROOT_DIR}/configs/weight_classifiers",

        "env_path": f"{ROOT_DIR}/.env",
    }
    return default_dirs

def load_env():
    default_dirs = get_default_dirs()
    env = dotenv_values(default_dirs["env_path"])
    return env

def load_config(config_name):
    default_dirs = get_default_dirs()
    config_path = Path(default_dirs['configs_dir']) / config_name
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

def get_config_path(config_name):
    default_dirs = get_default_dirs()
    config_path = Path(default_dirs['configs_dir']) / config_name
    return config_path
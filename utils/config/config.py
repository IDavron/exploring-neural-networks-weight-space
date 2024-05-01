from pathlib import Path
import yaml
import os
from dotenv import dotenv_values

def get_default_dirs():
    ROOT_DIR = Path(__file__).parent.parent.parent

    default_dirs = {
        "ROOT_DIR": ROOT_DIR,
        "DATASETS_DIR": f"{ROOT_DIR}/data",
        "MODELS_DIR": f"{ROOT_DIR}/models",
        "CONFIGS_DIR": f"{ROOT_DIR}/utils/config/experiments",

        "ENV_PATH": f"{ROOT_DIR}/.env",
    }
    return default_dirs

def load_config(config_name, include_env=False):
    default_dirs = get_default_dirs()
    config_path = os.path.join(default_dirs['CONFIGS_DIR'], config_name)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if include_env:
        env = dotenv_values(default_dirs["ENV_PATH"])
        config.update(env)

    return config
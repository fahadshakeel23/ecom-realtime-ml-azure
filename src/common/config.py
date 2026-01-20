from pathlib import Path
import os
import yaml

def load_config():
    env = os.getenv("APP_ENV", "dev")
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / f"{env}.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)
    

def load_yaml(path: str) -> dict:
    p = Path(path)
    with open(p, "r") as f:
        return yaml.safe_load(f)
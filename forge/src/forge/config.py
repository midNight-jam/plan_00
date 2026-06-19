import yaml
from pathlib import Path

ROOOT_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = ROOOT_DIR / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

settings = load_config()
import os
from pathlib import Path
from dotenv import load_dotenv


dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)


class Config:
    SPRITE_CORE_DIR = Path(__file__).parent
    SPRITE_DIR = SPRITE_CORE_DIR.parent.parent.parent
    OUTPUTS_DIR = SPRITE_DIR / "outputs"
    WANDB_DIR = OUTPUTS_DIR / "wandb_logs"
    CASA_PRIVATE_KEY = os.getenv("CASA_PRIVATE_KEY")
    CASA_SSH_HOST = os.getenv("CASA_SSH_HOST")
    CASA_SSH_USERNAME = os.getenv("CASA_SSH_USERNAME")
    REMOTE_DIR = os.getenv("REMOTE_DIR")
    DATA_DIR = os.getenv("DATA_DIR")
    ORIG_DATA_DIR = os.getenv("ORIG_DATA_DIR")
    MAPBOX_TOKEN = os.getenv("MAP_BOX_TOKEN")

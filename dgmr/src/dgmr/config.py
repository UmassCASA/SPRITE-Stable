import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SPRITE_DIR = os.path.dirname(ROOT_DIR)
    OUTPUTS_DIR = os.path.join(SPRITE_DIR, "outputs")
    CASA_PRIVATE_KEY = os.getenv("CASA_PRIVATE_KEY")
    CASA_SSH_HOST = os.getenv("CASA_SSH_HOST")
    CASA_SSH_USERNAME = os.getenv("CASA_SSH_USERNAME")
    REMOTE_DIR = os.getenv("REMOTE_DIR")
    DATA_DIR = os.getenv("DATA_DIR")
    ORIG_DATA_DIR = os.getenv("ORIG_DATA_DIR")
    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR")
    MAPBOX_TOKEN = os.getenv("MAP_BOX_TOKEN")

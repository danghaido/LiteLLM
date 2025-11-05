import os
from dotenv import load_dotenv

from pathlib import Path

import yaml
from loguru import logger

load_dotenv()
env = os.getenv("APP_ENV", "dev")


class DictToObject:
    def __init__(self, dictionary: dict) -> None:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return str(self.__dict__)


class Config:
    """OCR Configuration class that loads OCR settings from a YAML config file."""

    def __init__(self) -> None:
        pass

    def load_config(self, config_file: str) -> DictToObject | None:
        """Load the YAML configuration file and interpolate environment variables."""
        try:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
                return DictToObject(config)
        except FileNotFoundError:
            logger.error(
                f"Error: The configuration file '{config_file}' was not found."
            )
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error: There was an issue reading the YAML file: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None


CONFIG_PATH = f"configs/{env}.yaml"

CONFIG = Config().load_config(config_file=str(CONFIG_PATH))

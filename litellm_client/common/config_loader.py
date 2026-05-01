import os
import re

import yaml
from dotenv import load_dotenv
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

    _ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

    def _resolve_env(self, value):
        """Resolve ${ENV} or ${ENV:default} placeholders recursively."""
        if isinstance(value, dict):
            return {k: self._resolve_env(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._resolve_env(v) for v in value]

        if isinstance(value, str):

            def replacer(match):
                env_name = match.group(1)
                default_value = match.group(2)
                return os.getenv(env_name, default_value if default_value is not None else "")

            return self._ENV_PATTERN.sub(replacer, value)

        return value

    def load_config(self, config_file: str) -> DictToObject | None:
        """Load the YAML configuration file and interpolate environment variables."""
        try:
            with open(config_file, "r") as file:
                config = yaml.safe_load(file) or {}
                config = self._resolve_env(config)

                # Optional: push values from YAML `env:` into runtime environment.
                env_vars = config.get("env", {})
                if isinstance(env_vars, dict):
                    for key, value in env_vars.items():
                        if isinstance(value, str):
                            os.environ[key] = value

                return DictToObject(config)
        except FileNotFoundError:
            logger.error(f"Error: The configuration file '{config_file}' was not found.")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error: There was an issue reading the YAML file: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None


CONFIG_PATH = f"configs/{env}.yaml"

CONFIG = Config().load_config(config_file=str(CONFIG_PATH))

import yaml
import logging
import os


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "app_logger") -> logging.Logger:
    """Setup basic logger"""

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def get_env_variable(var_name: str, default=None):
    """Fetch environment variables safely"""
    return os.getenv(var_name, default)

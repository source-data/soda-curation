from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    This function reads a YAML configuration file and returns its contents as a dictionary.
    It supports multi-document YAML files, merging all documents into a single dictionary.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary containing all key-value pairs from the YAML file.

    Raises:
        FileNotFoundError: If the configuration file is not found at the specified path.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    try:
        with open(config_path, "r") as config_file:
            documents = yaml.safe_load_all(config_file)
            config = {}
            for doc in documents:
                config.update(doc)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")

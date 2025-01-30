"""Configuration loading with environment variable support."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

def load_environment() -> None:
    """
    Load environment variables from .env file.
    Handles test and production environments.
    """
    # Load environment variables from .env file
    env_path = Path(".env")
    if not env_path.exists():
        raise FileNotFoundError(
            ".env file not found. Please copy .env.example to .env and fill in your keys"
        )
    
    load_dotenv(env_path)
    
    # Validate required environment variables
    required_vars = [
        "OPENAI_API_KEY_TEST",
        "OPENAI_API_KEY_PROD",
        "ENVIRONMENT"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def get_api_keys() -> Dict[str, str]:
    """
    Get API keys for current environment.
    
    Returns:
        Dict with API keys for current environment
    """
    environment = os.getenv("ENVIRONMENT", "test")
    env_suffix = "_PROD" if environment == "prod" else "_TEST"
    
    return {
        "openai": os.getenv(f"OPENAI_API_KEY{env_suffix}"),
        "anthropic": os.getenv(f"ANTHROPIC_API_KEY{env_suffix}")
    }

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML and merge with environment variables.
    
    Args:
        config_path: Path to configuration YAML
        
    Returns:
        Complete configuration dictionary
    """
    # First load environment variables
    load_environment()
    environment = os.getenv("ENVIRONMENT", "test")
    api_keys = get_api_keys()
    
    # Load YAML configuration
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            
        # Get environment-specific configuration
        env_config = config.get(environment, config.get("default", {}))
        
        # Add API keys to provider configurations
        if "openai" in env_config:
            env_config["openai"]["api_key"] = api_keys["openai"]
        if "anthropic" in env_config:
            env_config["anthropic"]["api_key"] = api_keys["anthropic"]
            
        return env_config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")

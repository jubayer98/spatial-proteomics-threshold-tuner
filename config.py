import yaml
from pathlib import Path

def load_config(config_path=None):
    """Load configuration from a YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}
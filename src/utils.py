from pathlib import Path
import yaml

def load_params(path: str | Path) -> dict:
    """Reads a YAML file and returns its contents as a dictionary.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

"""
Configuration persistence utility.

Saves experiment hyperparameters as a human-readable JSON file so that
every run directory is self-describing.
"""

import json
import os
from typing import Any, Dict


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Write a configuration dictionary to a JSON file.

    Args:
        config: Flat dict of hyperparameters (must be JSON-serialisable).
        path:   Destination file path.  Parent directories are created
                automatically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[config] Saved to {path}")


def args_to_dict(args) -> Dict[str, Any]:
    """
    Convert an argparse.Namespace to a plain dict, converting any
    non-JSON-serialisable values to strings.

    Args:
        args: argparse.Namespace object.

    Returns:
        Dict suitable for passing to save_config().
    """
    result = {}
    for k, v in vars(args).items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return result

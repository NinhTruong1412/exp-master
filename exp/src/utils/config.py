"""
Configuration module for loading and managing YAML-based experiment configs.

Provides dot-accessible nested namespace configuration with merging, validation,
and serialization capabilities.
"""

import yaml
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Optional, Union


class Config:
    """
    Dot-accessible configuration class that loads from YAML and supports nested access.

    Converts nested dictionaries into a dot-accessible namespace object, allowing
    both dict-style and attribute-style access (e.g., config.model.lr or config['model']['lr']).

    Attributes:
        Each attribute corresponds to a key in the original dictionary.
        Nested dicts are recursively converted to Config objects.

    Examples:
        >>> config = Config.from_yaml('config.yaml')
        >>> print(config.model.lr)  # attribute access
        >>> config.merge({'model': {'lr': 1e-4}})  # override values
        >>> config.save('new_config.yaml')  # save modified config
    """

    def __init__(self, d: Optional[Dict[str, Any]] = None):
        """
        Initialize Config from a dictionary.

        Args:
            d: Dictionary to convert to Config. Nested dicts become Config objects.
               Defaults to empty dict if None.
        """
        d = d or {}
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config back to nested dictionary.

        Returns:
            Dictionary representation of the config with nested Config objects
            converted back to dicts.
        """
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the config with a default fallback.

        Args:
            key: Attribute name to retrieve.
            default: Value to return if key doesn't exist.

        Returns:
            The value at key, or default if key doesn't exist.
        """
        return getattr(self, key, default)

    def validate_required(self, required_fields: list) -> None:
        """
        Validate that required fields exist in the config.

        Args:
            required_fields: List of field names that must exist.

        Raises:
            ValueError: If any required field is missing.

        Examples:
            >>> config.validate_required(['model', 'optimizer', 'data'])
        """
        missing = [field for field in required_fields if not hasattr(self, field)]
        if missing:
            raise ValueError(
                f"Missing required config fields: {missing}. "
                f"Available: {list(self.__dict__.keys())}"
            )

    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """
        Merge override dictionary into this config (in-place).

        Deeply merges nested dictionaries. Scalar values override existing ones.
        Can be used for CLI argument overrides.

        Args:
            overrides: Dictionary of values to merge. Can contain nested dicts.

        Returns:
            Self for method chaining.

        Examples:
            >>> config.merge({'model': {'lr': 1e-4}, 'batch_size': 32})
            >>> config = config.merge({'model': {'lr': 1e-4}})
        """
        for k, v in overrides.items():
            if isinstance(v, dict) and hasattr(self, k) and isinstance(getattr(self, k), Config):
                # Recursively merge nested dicts
                getattr(self, k).merge(v)
            else:
                # Override or set new value
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
        return self

    @staticmethod
    def from_yaml(path: Union[str, Path]) -> "Config":
        """
        Load config from a YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            Config object loaded from the YAML file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML is malformed.

        Examples:
            >>> config = Config.from_yaml('config.yaml')
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, 'r') as f:
                d = yaml.safe_load(f) or {}
            return Config(d)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}") from e

    def save(self, path: Union[str, Path]) -> None:
        """
        Save config to a YAML file.

        Creates parent directories if they don't exist.

        Args:
            path: Path where to save the YAML config file.

        Examples:
            >>> config.save('config_output.yaml')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )

    def __repr__(self) -> str:
        """String representation of Config."""
        return f"Config({self.to_dict()})"

    def __str__(self) -> str:
        """Pretty string representation."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        """Support dict-style access: config['key']."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dict-style assignment: config['key'] = value."""
        if isinstance(value, dict):
            setattr(self, key, Config(value))
        else:
            setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: 'key' in config."""
        return hasattr(self, key)

    def keys(self):
        """Return config keys."""
        return self.__dict__.keys()

    def values(self):
        """Return config values."""
        return self.__dict__.values()

    def items(self):
        """Return config items for dict-like iteration."""
        return self.__dict__.items()

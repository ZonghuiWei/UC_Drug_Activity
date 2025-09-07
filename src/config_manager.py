import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

@dataclass
class ConfigurationManager:
    """
    Enterprise-grade configuration management system with hierarchical loading,
    environment variable overrides, and validation capabilities.
    """
    config_dir: Path = field(default_factory=lambda: Path("config"))
    _config_cache: Dict[str, Any] = field(default_factory=dict, init=False)
    _watchers: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        self.config_dir = Path(self.config_dir)
        self._setup_logging()
        self._validate_config_directory()
    
    def _setup_logging(self) -> None:
        """Initialize configuration manager logging."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _validate_config_directory(self) -> None:
        """Validate that config directory exists and is readable."""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
        if not self.config_dir.is_dir():
            raise NotADirectoryError(f"Config path is not a directory: {self.config_dir}")
    
    def load_configuration(self, config_name: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON files with caching and validation.
        
        Args:
            config_name: Name of config file (without extension)
            force_reload: Force reload from disk even if cached
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        cache_key = f"{config_name}_config"
        
        if not force_reload and cache_key in self._config_cache:
            self.logger.debug(f"Using cached configuration for {config_name}")
            return self._config_cache[cache_key]
        
        config_data = self._load_config_file(config_name)
        config_data = self._apply_environment_overrides(config_data, config_name)
        config_data = self._validate_configuration_schema(config_data, config_name)
        
        self._config_cache[cache_key] = config_data
        self.logger.info(f"Successfully loaded configuration: {config_name}")
        
        return config_data
    
    def _load_config_file(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        yaml_path = self.config_dir / f"{config_name}.yaml"
        json_path = self.config_dir / f"{config_name}.json"
        
        if yaml_path.exists():
            with open(yaml_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            available_configs = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.json"))
            raise FileNotFoundError(
                f"Configuration file not found: {config_name} (.yaml or .json). "
                f"Available configs: {[p.stem for p in available_configs]}"
            )
    
    def _apply_environment_overrides(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        env_prefix = f"{config_name.upper()}_"
        overrides_applied = 0
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                config_key_path = env_key[len(env_prefix):].lower().split('_')
                self._set_nested_value(config, config_key_path, self._parse_env_value(env_value))
                overrides_applied += 1
        
        if overrides_applied > 0:
            self.logger.debug(f"Applied {overrides_applied} environment overrides to {config_name}")
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], key_path: list, value: Any) -> None:
        """Set nested configuration value using dot notation path."""
        current = config
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[key_path[-1]] = value
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Parse environment variable value to appropriate type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    
    def _validate_configuration_schema(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Validate configuration against predefined schema."""
        validators = {
            'settings': self._validate_settings_config,
            'runtime': self._validate_runtime_config,
        }
        
        validator = validators.get(config_name)
        if validator:
            validator(config)
        
        return config
    
    def _validate_settings_config(self, config: Dict[str, Any]) -> None:
        """Validate main settings configuration."""
        required_sections = ['app', 'model', 'training', 'data', 'paths']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _validate_runtime_config(self, config: Dict[str, Any]) -> None:
        """Validate runtime configuration."""
        if 'runtime' not in config:
            raise ValueError("Runtime configuration must contain 'runtime' section")
    
    def get_config_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get specific configuration value using dot notation.
        
        Args:
            config_name: Name of configuration
            key_path: Dot-separated path to value (e.g., 'model.base_model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_configuration(config_name)
        keys = key_path.split('.')
        
        current = config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def update_config_value(self, config_name: str, key_path: str, value: Any) -> None:
        """Update configuration value and invalidate cache."""
        config = self.load_configuration(config_name)
        keys = key_path.split('.')
        self._set_nested_value(config, keys, value)
        
        # Invalidate cache to force reload
        cache_key = f"{config_name}_config"
        if cache_key in self._config_cache:
            del self._config_cache[cache_key]
    
    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._config_cache.clear()
        self.logger.debug("Configuration cache cleared")

# Global configuration manager instance
config_manager = ConfigurationManager()

def get_config(config_name: str = 'settings') -> Dict[str, Any]:
    """Convenience function to get configuration."""
    return config_manager.load_configuration(config_name)

def get_config_value(key_path: str, config_name: str = 'settings', default: Any = None) -> Any:
    """Convenience function to get specific config value."""
    return config_manager.get_config_value(config_name, key_path, default)
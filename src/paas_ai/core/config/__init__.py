"""
Configuration management for PaaS AI.

Provides hierarchical configuration loading with support for:
- YAML configuration files
- Environment variable substitution
- Predefined and custom profiles
- Profile-based configuration system
- Graceful fallbacks
"""

from .schemas import (
    # Configuration models
    Config,
    ConfigFile,
    APIConfig, 
    AgentConfig,
    MultiAgentConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrieverConfig,
    LoaderConfig,
    SplitterConfig,
    ContentValidatorConfig,
    ResourceConfig,
    
    # Enums
    EmbeddingType,
    VectorStoreType,
    RetrieverType,
    LoaderType,
    SplitterType,
    ContentValidatorType,
    ResourceType,
    
    # Predefined profiles
    DEFAULT_CONFIG_PROFILES
)
from .loader import (
    load_config, 
    load_config_file,
    save_config_file,
    create_sample_config, 
    list_available_profiles,
    get_current_profile,
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigFileEmptyError,
    InvalidYAMLError,
    InvalidConfigurationError,
    MissingEnvironmentVariableError
)

__all__ = [
    # Configuration models
    'Config',
    'ConfigFile',
    'APIConfig',
    'AgentConfig',
    'MultiAgentConfig',
    'EmbeddingConfig',
    'VectorStoreConfig',
    'RetrieverConfig',
    'LoaderConfig',
    'SplitterConfig',
    'ContentValidatorConfig',
    'ResourceConfig',
    
    # Enums
    'EmbeddingType',
    'VectorStoreType',
    'RetrieverType',
    'LoaderType',
    'SplitterType',
    'ContentValidatorType',
    'ResourceType',
    
    # Predefined profiles
    'DEFAULT_CONFIG_PROFILES',
    
    # Loader functions
    'load_config', 
    'load_config_file',
    'save_config_file',
    'create_sample_config', 
    'list_available_profiles',
    'get_current_profile',
    
    # Error types
    'ConfigurationError',
    'ConfigFileNotFoundError',
    'ConfigFileEmptyError',
    'InvalidYAMLError',
    'InvalidConfigurationError',
    'MissingEnvironmentVariableError'
]

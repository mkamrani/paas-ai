"""
Configuration loader for PaaS AI.

Loads configuration from YAML files with profile support:
1. Load environment variables from .env files
2. PAAS_AI_CONFIG environment variable (custom config file path)  
3. PAAS_AI_PROFILE environment variable (override current profile)
4. ~/.paas-ai/config.yaml (default location)
5. DEFAULT_CONFIG_PROFILES fallback (hardcoded defaults)
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .schemas import DEFAULT_CONFIG_PROFILES, Config, ConfigFile


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""

    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a configuration file does not exist."""

    pass


class ConfigFileEmptyError(ConfigurationError):
    """Raised when a configuration file exists but is empty."""

    pass


class InvalidYAMLError(ConfigurationError):
    """Raised when a configuration file contains invalid YAML."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration content is invalid."""

    pass


class MissingEnvironmentVariableError(ConfigurationError):
    """Raised when an environment variable referenced in config is missing."""

    pass


def _load_dotenv_files():
    """Load environment variables from .env files."""
    if not DOTENV_AVAILABLE:
        return

    # Look for .env files in order of precedence
    env_files = [
        ".env.local",  # Local overrides (highest priority)
        ".env",  # Main environment file
        ".env.default",  # Default values (lowest priority)
    ]

    # Start from the current directory and work upwards
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        for env_file in reversed(
            env_files
        ):  # Load in reverse order so higher priority files override
            env_path = parent / env_file
            if env_path.exists():
                load_dotenv(env_path, override=False)  # Don't override existing env vars


def _substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively substitute environment variables in config values."""
    if isinstance(config_dict, dict):
        return {k: _substitute_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [_substitute_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        # Replace ${VAR_NAME} with environment variable value
        def replace_env_var(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise MissingEnvironmentVariableError(
                    f"Environment variable '{var_name}' not found"
                )
            return env_value

        return re.sub(r"\$\{([^}]+)\}", replace_env_var, config_dict)
    else:
        return config_dict


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dict[str, Any]: Parsed configuration dictionary

    Raises:
        ConfigFileNotFoundError: If file doesn't exist
        ConfigFileEmptyError: If file is empty
        InvalidYAMLError: If file contains invalid YAML
        MissingEnvironmentVariableError: If referenced environment variable is missing
        ConfigurationError: For other configuration-related errors
    """
    if not config_path.exists():
        raise ConfigFileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        if not raw_config:
            raise ConfigFileEmptyError(f"Config file is empty: {config_path}")

        # Substitute environment variables
        config = _substitute_env_vars(raw_config)
        return config

    except yaml.YAMLError as e:
        raise InvalidYAMLError(f"Invalid YAML in config file {config_path}: {e}")
    except (MissingEnvironmentVariableError, ConfigurationError):
        # Re-raise specific configuration errors
        raise
    except Exception as e:
        raise ConfigurationError(f"Failed to load config file {config_path}: {e}")


def _resolve_profile(
    config_file: ConfigFile, profile_override: Optional[str] = None
) -> Tuple[Config, str]:
    """
    Resolve configuration profile to Config object.

    Args:
        config_file: Configuration file object
        profile_override: Optional profile name to override current

    Returns:
        Tuple[Config, str]: Resolved configuration object and active profile name

    Raises:
        InvalidConfigurationError: If profile is invalid or missing required fields
    """
    # Determine which profile to use
    profile_name = profile_override or config_file.current

    # Check if it's a predefined profile
    if profile_name in DEFAULT_CONFIG_PROFILES:
        return DEFAULT_CONFIG_PROFILES[profile_name], profile_name

    # Check if it's a custom profile in the config file
    if profile_name in config_file.profiles:
        return config_file.profiles[profile_name], profile_name

    # Profile not found
    available_profiles = list(DEFAULT_CONFIG_PROFILES.keys()) + list(config_file.profiles.keys())
    raise InvalidConfigurationError(
        f"Unknown profile '{profile_name}'. Available profiles: {available_profiles}"
    )


def load_config() -> Config:
    """
    Load RAG configuration with hierarchical fallback and profile support.

    Loading order:
    1. Load environment variables from .env files
    2. PAAS_AI_CONFIG environment variable (custom config file path)
    3. PAAS_AI_PROFILE environment variable (override current profile)
    4. ~/.paas-ai/config.yaml (default location)
    5. DEFAULT_CONFIG_PROFILES['default'] (fallback)

    Returns:
        Config: Loaded configuration object

    Raises:
        ConfigFileNotFoundError: If specified config file doesn't exist
        ConfigFileEmptyError: If config file is empty
        InvalidYAMLError: If config file contains invalid YAML
        InvalidConfigurationError: If configuration is invalid
        MissingEnvironmentVariableError: If referenced environment variable is missing
        ConfigurationError: For other configuration-related errors
    """
    # Load environment variables from .env files first
    _load_dotenv_files()
    # Check for profile override
    profile_override = os.getenv("PAAS_AI_PROFILE")

    # 1. Check PAAS_AI_CONFIG environment variable
    custom_config_path = os.getenv("PAAS_AI_CONFIG")
    if custom_config_path:
        config_path = Path(custom_config_path).expanduser().resolve()
        try:
            config_dict = _load_yaml_config(config_path)
            config_file = ConfigFile(**config_dict)
            config, _ = _resolve_profile(config_file, profile_override)
            return config

        except (ConfigFileNotFoundError, ConfigFileEmptyError):
            # Silently fall back to default if custom config doesn't exist or is empty
            pass
        except ConfigurationError as e:
            # Preserve the specific error type but add context about the file
            e.args = (f"Error in custom config file {config_path}: {e}",)
            raise

    # 2. Check default location ~/.paas-ai/config.yaml
    default_config_path = Path.home() / ".paas-ai" / "config.yaml"
    try:
        config_dict = _load_yaml_config(default_config_path)
        config_file = ConfigFile(**config_dict)
        config, _ = _resolve_profile(config_file, profile_override)
        return config

    except (ConfigFileNotFoundError, ConfigFileEmptyError):
        # Silently fall back to default if default config doesn't exist or is empty
        pass
    except ConfigurationError as e:
        # For other errors, add context about the file
        e.args = (f"Error in default config file {default_config_path}: {e}",)
        raise

    # 3. Fallback to default configuration (with profile override if specified)
    if profile_override and profile_override in DEFAULT_CONFIG_PROFILES:
        return DEFAULT_CONFIG_PROFILES[profile_override]

    # Use 'local' as default instead of 'default' to avoid API key requirements
    return DEFAULT_CONFIG_PROFILES["local"]


def load_config_file(config_path: Optional[Path] = None) -> Tuple[ConfigFile, Path]:
    """
    Load configuration file object with profile support.

    Args:
        config_path: Optional specific config file path

    Returns:
        Tuple[ConfigFile, Path]: Configuration file object and its path

    Raises:
        ConfigFileNotFoundError: If config file doesn't exist
        ConfigurationError: For other configuration-related errors
    """
    # Determine config file path
    if config_path:
        file_path = config_path.expanduser().resolve()
    else:
        custom_config_path = os.getenv("PAAS_AI_CONFIG")
        if custom_config_path:
            file_path = Path(custom_config_path).expanduser().resolve()
        else:
            file_path = Path.home() / ".paas-ai" / "config.yaml"

    # Load and parse config file
    config_dict = _load_yaml_config(file_path)
    config_file = ConfigFile(**config_dict)

    return config_file, file_path


def _convert_enums_to_strings(obj: Any) -> Any:
    """Convert enum values to strings recursively for YAML serialization."""
    if isinstance(obj, dict):
        return {_convert_enums_to_strings(k): _convert_enums_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_enums_to_strings(item) for item in obj]
    elif hasattr(obj, "value"):  # Enum check
        return obj.value
    else:
        return obj


def save_config_file(config_file: ConfigFile, config_path: Optional[Path] = None) -> Path:
    """
    Save configuration file.

    Args:
        config_file: Configuration file object to save
        config_path: Optional specific path to save to

    Returns:
        Path: Path where config was saved
    """
    if config_path is None:
        custom_config_path = os.getenv("PAAS_AI_CONFIG")
        if custom_config_path:
            config_path = Path(custom_config_path).expanduser().resolve()
        else:
            config_path = Path.home() / ".paas-ai" / "config.yaml"

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and handle enum serialization properly
    config_dict = config_file.model_dump(exclude_none=True)
    config_dict = _convert_enums_to_strings(config_dict)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)

    return config_path


def create_sample_config(config_path: Optional[Path] = None) -> Path:
    """
    Create a sample configuration file that includes all required sections.

    Args:
        config_path: Path to create the config file. Defaults to ~/.paas-ai/config.yaml

    Returns:
        Path: Path to the created config file
    """
    if config_path is None:
        config_path = Path.home() / ".paas-ai" / "config.yaml"

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    sample_config = {
        "current": "my-custom-profile",
        "profiles": {
            "my-custom-profile": {
                # Embedding configuration
                "embedding": {
                    "type": "sentence_transformers",
                    "model_name": "all-MiniLM-L6-v2",
                    "params": {},
                },
                # Vector store configuration
                "vectorstore": {
                    "type": "chroma",
                    "persist_directory": "./rag_data/custom",
                    "collection_name": "custom-knowledge",
                    "params": {},
                },
                # Retriever configuration
                "retriever": {"type": "similarity", "search_kwargs": {"k": 5}, "params": {}},
                # LLM configuration (REQUIRED)
                "llm": {
                    "provider": "openai",
                    "model_name": "gpt-3.5-turbo",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                    "api_key_env_var": "OPENAI_API_KEY",
                    "params": {},
                },
                # Citation configuration (REQUIRED)
                "citation": {
                    "enabled": True,
                    "verbosity": "minimal",
                    "format": "inline",
                    "resource_overrides": {},
                    "include_quotes": True,
                    "max_quote_length": 150,
                    "include_confidence": False,
                    "generate_deep_links": True,
                    "strategies": {
                        "dsl": "technical_citation",
                        "contextual": "web_citation",
                        "guidelines": "policy_citation",
                        "domain_rules": "rule_citation",
                    },
                    "base_urls": {},
                },
                # Multi-agent configuration (REQUIRED)
                "multi_agent": {
                    "enabled": True,
                    "mode": "supervisor",
                    "default_agent": "designer",
                    "track_tokens": False,
                    "token_callback": "console",
                    "verbose": False,
                    "persistence": {
                        "enabled": True,
                        "checkpointer_type": "memory",
                        "sqlite_path": None,
                        "postgres_url": None,
                        "max_history_length": 20,
                        "cleanup_interval": 3600,
                    },
                },
                # Agent configurations (REQUIRED)
                "agents": {
                    "designer": {"model": "gpt-4o-mini", "temperature": 0.1},
                    "paas_manifest_generator": {"model": "gpt-4o-mini", "temperature": 0.0},
                },
                # Additional configuration fields
                "batch_size": 32,
                "validate_urls": True,
                "max_parallel": 5,
                "timeout": 30,
                "log_level": "INFO",
            }
        },
    }

    with open(config_path, "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False, indent=2)

    return config_path


def list_available_profiles(include_custom: bool = True) -> Dict[str, str]:
    """
    List all available profiles (built-in and optionally custom).

    Args:
        include_custom: Whether to include custom profiles from config file

    Returns:
        Dict[str, str]: Profile name to description mapping
    """
    # Get all built-in profiles from DEFAULT_CONFIG_PROFILES
    profiles = {}
    for name, config in DEFAULT_CONFIG_PROFILES.items():
        if name == "default":
            profiles[name] = "OpenAI embeddings + Chroma vector store"
        elif name == "local":
            profiles[name] = "SentenceTransformers + Chroma (no API keys required)"
        elif name == "production":
            profiles[name] = "OpenAI large embeddings + Pinecone + ensemble retrieval"
        elif name == "verbose":
            profiles[name] = "Verbose mode with token tracking enabled"
        else:
            profiles[name] = f"Built-in profile: {name}"

    if include_custom:
        try:
            config_file, _ = load_config_file()
            for name in config_file.profiles.keys():
                profiles[name] = f"Custom profile: {name}"
        except (ConfigFileNotFoundError, ConfigurationError):
            # No custom profiles available
            pass

    return profiles


def get_current_profile() -> str:
    """
    Get the name of the currently active profile.

    Returns:
        str: Current profile name
    """
    # Check for environment override first
    profile_override = os.getenv("PAAS_AI_PROFILE")
    if profile_override:
        return profile_override

    try:
        config_file, _ = load_config_file()
        return config_file.current
    except (ConfigFileNotFoundError, ConfigurationError):
        return "default"

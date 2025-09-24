"""
Configuration management commands for PaaS AI.

Provides CLI commands for managing all types of configurations including
RAG, API, agent, and system settings with profile support.
"""

import os
from pathlib import Path
from typing import Optional

import click

from paas_ai.core.config import (
    Config,
    ConfigFile,
    ConfigFileNotFoundError,
    ConfigurationError,
    EmbeddingConfig,
    EmbeddingType,
    RetrieverConfig,
    RetrieverType,
    VectorStoreConfig,
    VectorStoreType,
    create_sample_config,
    get_current_profile,
    list_available_profiles,
    load_config,
    load_config_file,
    save_config_file,
)
from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.cli.config")


@click.group()
def config():
    """
    ⚙️ Manage PaaS AI configuration and profiles.

    Configure RAG, API, agent, and system settings using profile-based configuration.
    Built-in profiles: local, default, production
    """
    pass


@config.command("show")
@click.option("--path", is_flag=True, help="Show config file path")
@click.option("--profiles", is_flag=True, help="Show available profiles")
@click.option("--profile", help="Show specific profile configuration")
def show_config(path: bool, profiles: bool, profile: Optional[str]):
    """Show current configuration or profile information."""
    try:
        if path:
            # Show config file paths
            custom_path = os.getenv("PAAS_AI_CONFIG")
            default_path = Path.home() / ".paas-ai" / "config.yaml"
            profile_override = os.getenv("PAAS_AI_PROFILE")

            logger.info("Configuration file locations:")
            if custom_path:
                logger.info(f"  Custom (PAAS_AI_CONFIG): {custom_path}")
                if Path(custom_path).exists():
                    logger.success("    ✅ File exists")
                else:
                    logger.warning("    ⚠️  File not found")

            logger.info(f"  Default: {default_path}")
            if default_path.exists():
                logger.success("    ✅ File exists")
            else:
                logger.info("    ℹ️  File not found (using fallback)")

            if profile_override:
                logger.info(f"  Profile override (PAAS_AI_PROFILE): {profile_override}")

            return

        if profiles:
            # Show available profiles
            available_profiles = list_available_profiles(include_custom=True)
            current = get_current_profile()

            logger.info("Available Profiles:")
            logger.info("=" * 30)

            for name, description in available_profiles.items():
                marker = "👉 " if name == current else "   "
                logger.info(f"{marker}{name}: {description}")

            logger.info(f"\nCurrent active profile: {current}")
            return

        if profile:
            # Show specific profile configuration
            try:
                config_file, _ = load_config_file()

                # Check if it's a built-in profile
                from paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES

                if profile in DEFAULT_CONFIG_PROFILES:
                    config_obj = DEFAULT_CONFIG_PROFILES[profile]
                    logger.info(f"Built-in Profile: {profile}")
                elif profile in config_file.profiles:
                    config_obj = config_file.profiles[profile]
                    logger.info(f"Custom Profile: {profile}")
                else:
                    logger.error(f"Profile '{profile}' not found")
                    return

                logger.info("-" * 30)
                logger.info(f"Embedding Type: {config_obj.embedding.type}")
                logger.info(f"Embedding Model: {config_obj.embedding.model_name}")
                logger.info(f"Vector Store: {config_obj.vectorstore.type}")
                logger.info(f"Collection: {config_obj.vectorstore.collection_name}")
                if config_obj.vectorstore.persist_directory:
                    logger.info(f"Persist Directory: {config_obj.vectorstore.persist_directory}")
                logger.info(f"Retriever: {config_obj.retriever.type}")
                logger.info(f"Batch Size: {config_obj.batch_size}")
                logger.info(f"Validate URLs: {config_obj.validate_urls}")

            except ConfigFileNotFoundError:
                logger.error(
                    "No configuration file found. Use 'paas-ai config init' to create one."
                )
                return
            except Exception as e:
                logger.error(f"Error loading profile '{profile}': {e}")
                return

        # Show current active configuration
        try:
            config = load_config()
            current = get_current_profile()

            logger.info(f"Current Active Configuration (Profile: {current}):")
            logger.info("-" * 40)
            logger.info(f"Embedding Type: {config.embedding.type}")
            logger.info(f"Embedding Model: {config.embedding.model_name}")
            logger.info(f"Vector Store: {config.vectorstore.type}")
            logger.info(f"Collection: {config.vectorstore.collection_name}")
            if config.vectorstore.persist_directory:
                logger.info(f"Persist Directory: {config.vectorstore.persist_directory}")
            logger.info(f"Retriever: {config.retriever.type}")
            logger.info(f"Batch Size: {config.batch_size}")
            logger.info(f"Validate URLs: {config.validate_urls}")
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")

    except Exception as e:
        logger.error(f"Failed to show configuration: {e}")
        return False


@config.command("init")
@click.option(
    "--path",
    type=click.Path(path_type=Path),
    help="Custom path for config file (default: ~/.paas-ai/config.yaml)",
)
@click.option(
    "--profile",
    type=click.Choice(["local", "default", "production"]),
    default="local",
    help="Default active profile to use",
)
@click.option("--force", is_flag=True, help="Overwrite existing config file")
def init_config(path: Optional[Path], profile: str, force: bool):
    """Initialize configuration file with profile support."""
    try:
        # Determine config path
        if path:
            config_path = path.expanduser().resolve()
        else:
            config_path = Path.home() / ".paas-ai" / "config.yaml"

        # Check if file exists
        if config_path.exists() and not force:
            logger.error(f"Configuration file already exists: {config_path}")
            logger.info("Use --force to overwrite or choose a different path")
            return False

        # Create sample config
        created_path = create_sample_config(config_path)

        # Update current profile in the created config if not default
        if profile != "local":
            try:
                config_file, _ = load_config_file(created_path)
                config_file.current = profile
                save_config_file(config_file, created_path)
            except Exception as e:
                logger.warning(f"Could not set profile to {profile}: {e}")

        logger.success(f"Configuration file created: {created_path}")
        logger.info(f"Current active profile: {profile}")
        logger.info("\nTo use this config:")
        if path:
            logger.info(f"  export PAAS_AI_CONFIG={created_path}")
        logger.info("  paas-ai config show")
        logger.info("\n📝 Manage profiles:")
        logger.info("  paas-ai config profiles                    # List all profiles")
        logger.info("  paas-ai config set-current <profile>       # Switch active profile")
        logger.info("  paas-ai config add-profile <name> ...      # Add custom profile")

    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        return False


@config.command("profiles")
def list_profiles():
    """List available configuration profiles."""
    try:
        profiles = list_available_profiles(include_custom=True)
        current = get_current_profile()

        logger.info("Available Configuration Profiles:")
        logger.info("=" * 40)

        # Separate built-in and custom profiles
        builtin_profiles = {
            k: v for k, v in profiles.items() if not v.startswith("Custom profile:")
        }
        custom_profiles = {k: v for k, v in profiles.items() if v.startswith("Custom profile:")}

        logger.info("Built-in Profiles:")
        for name, description in builtin_profiles.items():
            marker = "👉 " if name == current else "   "
            logger.info(f"{marker}📋 {name}")
            logger.info(f"      {description}")

        if custom_profiles:
            logger.info("\nCustom Profiles:")
            for name, description in custom_profiles.items():
                marker = "👉 " if name == current else "   "
                logger.info(f"{marker}🔧 {name}")
                logger.info(f"      {description}")

        logger.info(f"\n🎯 Current active profile: {current}")

    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")


@config.command("set-current")
@click.argument("profile_name")
def set_current_profile(profile_name: str):
    """Set the current active profile."""
    try:
        # Check if profile exists
        available_profiles = list_available_profiles(include_custom=True)
        if profile_name not in available_profiles:
            logger.error(f"Profile '{profile_name}' not found")
            logger.info("Available profiles:")
            for name in available_profiles.keys():
                logger.info(f"  - {name}")
            return False

        # Load or create config file
        try:
            config_file, config_path = load_config_file()
        except ConfigFileNotFoundError:
            # Create a new config file
            logger.info("No configuration file found. Creating new one...")
            config_path = Path.home() / ".paas-ai" / "config.yaml"
            config_file = ConfigFile(current=profile_name, profiles={})

        # Update current profile
        old_profile = config_file.current
        config_file.current = profile_name

        # Save config file
        save_config_file(config_file, config_path)

        logger.success(f"Active profile changed from '{old_profile}' to '{profile_name}'")
        logger.info(f"Configuration saved to: {config_path}")

        # Show brief info about the new profile
        logger.info(f"\nProfile '{profile_name}': {available_profiles[profile_name]}")

    except Exception as e:
        logger.error(f"Failed to set current profile: {e}")
        return False


@config.command("add-profile")
@click.argument("profile_name")
@click.option(
    "--embedding-type",
    type=click.Choice(["openai", "sentence_transformers", "cohere"]),
    default="openai",
    help="Embedding type",
)
@click.option("--embedding-model", default="text-embedding-3-small", help="Embedding model name")
@click.option(
    "--vectorstore-type",
    type=click.Choice(["chroma", "faiss", "pinecone"]),
    default="chroma",
    help="Vector store type",
)
@click.option("--vectorstore-dir", help="Vector store persist directory")
@click.option("--collection-name", help="Vector store collection name")
@click.option(
    "--retriever-type",
    type=click.Choice(["similarity", "mmr", "ensemble"]),
    default="similarity",
    help="Retriever type",
)
@click.option("--batch-size", type=int, default=32, help="Processing batch size")
@click.option("--activate", is_flag=True, help="Set as current active profile after creation")
def add_profile(
    profile_name: str,
    embedding_type: str,
    embedding_model: str,
    vectorstore_type: str,
    vectorstore_dir: Optional[str],
    collection_name: Optional[str],
    retriever_type: str,
    batch_size: int,
    activate: bool,
):
    """Add a new custom profile."""
    try:
        # Check if profile name conflicts with built-in profiles
        from paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES

        if profile_name in DEFAULT_CONFIG_PROFILES:
            logger.error(
                f"Cannot create profile '{profile_name}': name conflicts with built-in profile"
            )
            logger.info(f"Built-in profiles: {list(DEFAULT_CONFIG_PROFILES.keys())}")
            return False

        # Load or create config file
        try:
            config_file, config_path = load_config_file()
        except ConfigFileNotFoundError:
            logger.info("No configuration file found. Creating new one...")
            config_path = Path.home() / ".paas-ai" / "config.yaml"
            config_file = ConfigFile(current="default", profiles={})

        # Set defaults for optional fields
        if not collection_name:
            collection_name = f"{profile_name}_knowledge"
        if not vectorstore_dir and vectorstore_type == "chroma":
            vectorstore_dir = f"./rag_data/{profile_name}"

        # Create new profile configuration
        new_profile = Config(
            embedding=EmbeddingConfig(
                type=EmbeddingType(embedding_type), model_name=embedding_model
            ),
            vectorstore=VectorStoreConfig(
                type=VectorStoreType(vectorstore_type),
                persist_directory=vectorstore_dir,
                collection_name=collection_name,
            ),
            retriever=RetrieverConfig(type=RetrieverType(retriever_type), search_kwargs={"k": 5}),
            batch_size=batch_size,
            validate_urls=True,
        )

        # Add profile to config file
        config_file.profiles[profile_name] = new_profile

        # Activate if requested
        if activate:
            old_profile = config_file.current
            config_file.current = profile_name
            logger.info(f"Setting '{profile_name}' as active profile (was '{old_profile}')")

        # Save config file
        save_config_file(config_file, config_path)

        logger.success(f"Profile '{profile_name}' created successfully")
        logger.info(f"Configuration saved to: {config_path}")

        # Show profile details
        logger.info(f"\nProfile '{profile_name}' details:")
        logger.info(f"  Embedding: {embedding_type} ({embedding_model})")
        logger.info(f"  Vector Store: {vectorstore_type}")
        if vectorstore_dir:
            logger.info(f"  Persist Directory: {vectorstore_dir}")
        logger.info(f"  Collection: {collection_name}")
        logger.info(f"  Retriever: {retriever_type}")
        logger.info(f"  Batch Size: {batch_size}")

    except Exception as e:
        logger.error(f"Failed to add profile '{profile_name}': {e}")
        return False


@config.command("remove-profile")
@click.argument("profile_name")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def remove_profile(profile_name: str, confirm: bool):
    """Remove a custom profile."""
    try:
        # Check if profile name is built-in
        from paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES

        if profile_name in DEFAULT_CONFIG_PROFILES:
            logger.error(f"Cannot remove built-in profile '{profile_name}'")
            return False

        # Load config file
        try:
            config_file, config_path = load_config_file()
        except ConfigFileNotFoundError:
            logger.error("No configuration file found")
            return False

        # Check if profile exists
        if profile_name not in config_file.profiles:
            logger.error(f"Custom profile '{profile_name}' not found")
            available = list(config_file.profiles.keys())
            if available:
                logger.info(f"Available custom profiles: {available}")
            return False

        # Confirm deletion
        if not confirm:
            response = click.confirm(f"Are you sure you want to remove profile '{profile_name}'?")
            if not response:
                logger.info("Profile removal cancelled")
                return

        # Check if it's the current profile
        if config_file.current == profile_name:
            logger.warning(f"Profile '{profile_name}' is currently active")
            new_current = "default"
            config_file.current = new_current
            logger.info(f"Switching active profile to '{new_current}'")

        # Remove profile
        del config_file.profiles[profile_name]

        # Save config file
        save_config_file(config_file, config_path)

        logger.success(f"Profile '{profile_name}' removed successfully")
        logger.info(f"Configuration saved to: {config_path}")

    except Exception as e:
        logger.error(f"Failed to remove profile '{profile_name}': {e}")
        return False


@config.command("validate")
@click.option(
    "--config-path", type=click.Path(path_type=Path), help="Validate specific config file"
)
@click.option("--profile", help="Validate specific profile")
def validate_config(config_path: Optional[Path], profile: Optional[str]):
    """Validate configuration file or specific profile."""
    try:
        if config_path:
            # Set temporary environment variable to test specific config
            original_env = os.getenv("PAAS_AI_CONFIG")
            os.environ["PAAS_AI_CONFIG"] = str(config_path)
            try:
                config_file, _ = load_config_file(config_path)
                logger.success(f"✅ Configuration file is valid: {config_path}")
                logger.info(f"   Current profile: {config_file.current}")
                logger.info(f"   Custom profiles: {list(config_file.profiles.keys())}")

                if profile:
                    if profile in config_file.profiles:
                        config_obj = config_file.profiles[profile]
                        logger.info(
                            f"   Profile '{profile}': {config_obj.embedding.type} embeddings"
                        )
                    else:
                        logger.warning(f"   Profile '{profile}' not found in config file")

            finally:
                # Restore original environment
                if original_env:
                    os.environ["PAAS_AI_CONFIG"] = original_env
                else:
                    os.environ.pop("PAAS_AI_CONFIG", None)
        else:
            if profile:
                # Validate specific profile
                config_file, _ = load_config_file()

                from paas_ai.core.config.schemas import DEFAULT_CONFIG_PROFILES

                if profile in DEFAULT_CONFIG_PROFILES:
                    config_obj = DEFAULT_CONFIG_PROFILES[profile]
                    logger.success(f"✅ Built-in profile '{profile}' is valid")
                elif profile in config_file.profiles:
                    config_obj = config_file.profiles[profile]
                    logger.success(f"✅ Custom profile '{profile}' is valid")
                else:
                    logger.error(f"❌ Profile '{profile}' not found")
                    return False

                logger.info(f"   Profile type: {config_obj.embedding.type} embeddings")
                logger.info(f"   Vector store: {config_obj.vectorstore.type}")
            else:
                # Validate current configuration
                config = load_config()
                current = get_current_profile()
                logger.success("✅ Current configuration is valid")
                logger.info(f"   Active profile: {current}")
                logger.info(f"   Profile type: {config.embedding.type} embeddings")
                logger.info(f"   Vector store: {config.vectorstore.type}")

    except ConfigurationError as e:
        logger.error(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False


@config.command("edit")
def edit_config():
    """Open configuration file in default editor."""
    try:
        # Find config file
        custom_path = os.getenv("PAAS_AI_CONFIG")
        if custom_path:
            config_path = Path(custom_path)
        else:
            config_path = Path.home() / ".paas-ai" / "config.yaml"

        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Create one with: paas-ai config init")
            return False

        # Open in editor
        import subprocess

        editor = os.getenv("EDITOR", "nano")  # Default to nano if no EDITOR set

        logger.info(f"Opening {config_path} in {editor}...")
        subprocess.run([editor, str(config_path)])

        logger.info("Validating edited configuration...")
        # Validate the edited config
        load_config()
        logger.success("✅ Configuration is valid")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to edit configuration: {e}")
        return False

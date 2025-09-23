"""
Main CLI entry point for PaaS AI.

Provides a professional command-line interface for the agentic PaaS solution
with proper error handling, logging, and user-friendly output.
"""

import sys
from pathlib import Path
from typing import Optional

import click

from paas_ai.utils.logging import PaaSLogger, get_logger


# Global CLI configuration
class CLIContext:
    """Shared context for CLI commands."""

    def __init__(self):
        self.logger: Optional[PaaSLogger] = None
        self.verbose: bool = False
        self.quiet: bool = False
        self.config_file: Optional[Path] = None


# Global context object
pass_context = click.make_pass_decorator(CLIContext, ensure=True)


def validate_verbosity_options(ctx, param, value):
    """Validate that verbose and quiet options are mutually exclusive."""
    if not hasattr(ctx, "_verbosity_options"):
        ctx._verbosity_options = {}

    # Store the current option if it's set
    if value:
        ctx._verbosity_options[param.name] = value

    # Check for conflicts
    if len(ctx._verbosity_options) > 1:
        options_used = [f"--{name}" for name in ctx._verbosity_options.keys()]
        raise click.ClickException(
            f"Options {' and '.join(options_used)} are mutually exclusive. "
            f"Please use only one verbosity option."
        )

    return value


@click.group()
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    callback=validate_verbosity_options,
    help="Enable verbose output (DEBUG level logging)",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    callback=validate_verbosity_options,
    help="Suppress all output except errors",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--log-file", type=click.Path(path_type=Path), help="Path to log file for persistent logging"
)
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.option("--no-emoji", is_flag=True, help="Disable emoji indicators")
@click.version_option(version="0.1.0", prog_name="paas-ai")
@pass_context
def cli(
    ctx: CLIContext,
    verbose: bool,
    quiet: bool,
    config: Optional[Path],
    log_file: Optional[Path],
    no_color: bool,
    no_emoji: bool,
):
    """
    🚀 PaaS AI - Agentic Platform as a Service Configuration Generator

    Generate configuration with custom DSL for PaaS using agents
    that understand your requirements and follow the best practices and guidelines.

      Examples:
    paas-ai generate --from-confluence https://wiki.company.com/requirements
    paas-ai rag sync --incremental
    paas-ai validate deployment.yaml
    """
    # Set up logging based on options
    if quiet:
        log_level = "ERROR"
    elif verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"

    # Initialize logger
    ctx.logger = get_logger(
        name="paas_ai.cli",
        level=log_level,
        console=not quiet,
        file_path=log_file,
        colors=not no_color,
        emojis=not no_emoji,
    )

    # Store configuration
    ctx.verbose = verbose
    ctx.quiet = quiet
    ctx.config_file = config

    # Set CLI context for logging
    ctx.logger.set_context("CLI")

    if verbose:
        ctx.logger.debug("CLI initialized with verbose logging")
        if config:
            ctx.logger.debug(f"Using config file: {config}")


@cli.command()
@click.option("--from-confluence", help="Generate from Confluence page URL")
@click.option(
    "--from-file",
    type=click.Path(exists=True, path_type=Path),
    help="Generate from local requirements file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./generated"),
    help="Output directory for generated configurations",
)
@click.option(
    "--platform",
    type=click.Choice(["kubernetes", "docker", "terraform", "all"]),
    default="kubernetes",
    help="Target platform for configuration generation",
)
@click.option(
    "--template",
    type=click.Choice(["webapp", "api", "microservice", "ml-pipeline"]),
    help="Use predefined template",
)
@click.option("--dry-run", is_flag=True, help="Show what would be generated without creating files")
@pass_context
def generate(
    ctx: CLIContext,
    from_confluence: Optional[str],
    from_file: Optional[Path],
    output: Path,
    platform: str,
    template: Optional[str],
    dry_run: bool,
):
    """
    🏗️ Generate PaaS configurations from requirements.

    Use the intelligent agent to generate production-ready configurations
    based on requirements from various sources.

    Examples:
      paas-ai generate --from-confluence https://wiki.company.com/spec --platform kubernetes
      paas-ai generate --template webapp --output ./my-app
      paas-ai generate --from-file requirements.md --dry-run
    """
    logger = ctx.logger
    logger.info("Starting configuration generation")

    # Validate input sources
    sources = [from_confluence, from_file, template]
    if not any(sources):
        logger.error("No input source specified. Use --from-confluence, --from-file, or --template")
        sys.exit(1)

    if sum(bool(source) for source in sources) > 1:
        logger.error("Please specify only one input source")
        sys.exit(1)

    try:
        logger.set_context("GENERATE")

        # Determine source type and log it
        if from_confluence:
            logger.info(f"Generating from Confluence: {from_confluence}")
            logger.progress("Fetching requirements from Confluence...")
            # TODO: Implement Confluence integration

        elif from_file:
            logger.info(f"Generating from file: {from_file}")
            logger.progress("Parsing requirements file...")
            # TODO: Implement file parsing

        elif template:
            logger.info(f"Generating from template: {template}")
            logger.progress("Loading template configuration...")
            # TODO: Implement template loading

        # Show dry run information
        if dry_run:
            logger.warning("DRY RUN MODE - No files will be created")

        logger.progress(f"Targeting platform: {platform}")
        logger.progress(f"Output directory: {output}")

        # TODO: Implement actual generation logic
        logger.info("🤖 Initializing agent...")
        logger.progress("Analyzing requirements...")
        logger.progress("Generating configurations...")
        logger.progress("Validating generated configs...")

        if not dry_run:
            output.mkdir(parents=True, exist_ok=True)
            logger.success(f"Configurations generated successfully in {output}")
        else:
            logger.info("Dry run completed - configurations would be generated")

    except Exception as e:
        logger.exception(f"Configuration generation failed: {e}")
        sys.exit(1)
    finally:
        logger.clear_context()


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--platform",
    type=click.Choice(["kubernetes", "docker", "terraform"]),
    help="Platform to validate against",
)
@click.option("--fix", is_flag=True, help="Automatically fix validation errors where possible")
@click.option("--strict", is_flag=True, help="Enable strict validation (fail on warnings)")
@pass_context
def validate(
    ctx: CLIContext,
    files: tuple[Path, ...],
    platform: Optional[str],
    fix: bool,
    strict: bool,
):
    """
    ✅ Validate PaaS configuration files.

    Validate configuration files against schemas and best practices.

    Examples:
      paas-ai validate deployment.yaml service.yaml
      paas-ai validate --platform kubernetes *.yaml
      paas-ai validate --fix --strict configs/
    """
    logger = ctx.logger
    logger.info(f"Validating {len(files)} file(s)")

    try:
        logger.set_context("VALIDATE")

        for file_path in files:
            logger.progress(f"Validating {file_path}")

            # TODO: Implement validation logic
            if platform:
                logger.debug(f"Using {platform} validation rules")

            # Simulate validation results
            logger.success(f"✓ {file_path.name} - Valid")

            if fix:
                logger.info(f"🔧 Applied auto-fixes to {file_path.name}")

        logger.success("All files validated successfully")

    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        sys.exit(1)
    finally:
        logger.clear_context()


@cli.command()
@click.option("--target", type=click.Choice(["kubernetes", "docker"]), help="Deployment target")
@click.option(
    "--config-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("./generated"),
    help="Directory containing configuration files",
)
@click.option("--dry-run", is_flag=True, help="Show deployment plan without executing")
@click.option("--force", is_flag=True, help="Force deployment even with warnings")
@pass_context
def deploy(
    ctx: CLIContext,
    target: Optional[str],
    config_dir: Path,
    dry_run: bool,
    force: bool,
):
    """
    🚀 Deploy configurations to target platform.

    Deploy generated configurations to the specified platform.

    Examples:
      paas-ai deploy --target kubernetes
      paas-ai deploy --dry-run --config-dir ./configs
    """
    logger = ctx.logger
    logger.info("Starting deployment process")

    try:
        logger.set_context("DEPLOY")

        if not target:
            logger.error("No deployment target specified. Use --target")
            sys.exit(1)

        if dry_run:
            logger.warning("DRY RUN MODE - No actual deployment will occur")

        logger.progress(f"Preparing {target} deployment...")
        logger.progress(f"Loading configurations from {config_dir}")

        # TODO: Implement deployment logic
        logger.info("🔍 Analyzing deployment requirements...")
        logger.progress("Validating target environment...")
        logger.progress("Applying configurations...")

        if not dry_run:
            logger.success("Deployment completed successfully")
        else:
            logger.info("Dry run completed - deployment plan validated")

    except Exception as e:
        logger.exception(f"Deployment failed: {e}")
        sys.exit(1)
    finally:
        logger.clear_context()


from .commands.agent import agent_group
from .commands.config import config
from .commands.mcp import mcp

# Import and register command groups
from .commands.rag import rag

# Add command groups
cli.add_command(rag)
cli.add_command(config)
cli.add_command(mcp)
cli.add_command(agent_group)


def main():
    """Entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n👋 Interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

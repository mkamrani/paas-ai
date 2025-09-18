"""
Logging utility for with colors, levels, and emojis.

Provides structured logging for both CLI and API components with:
- Color-coded output for different log levels
- Emoji indicators for visual clarity
- JSON structured logging for production
- Context-aware formatting with thread-safe context management

Usage:
    # Common usage (recommended)
    from paas_ai.utils.logging import get_logger
    logger = get_logger(__name__)
    
    # Convenience logger (todo: remove if not needed)
    from paas_ai.utils.logging import logger
"""

import logging
import sys
import json
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from contextvars import ContextVar

import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform color support
colorama.init(autoreset=True)

# Thread-safe context variable for logging context
_log_context: ContextVar[Optional[str]] = ContextVar('log_context', default=None)

# Custom log level mappings
CUSTOM_LOG_LEVELS = {
    "SUCCESS": 25,
    "PROGRESS": 15
}

# Standard log level mappings
STANDARD_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# All log level mappings combined
ALL_LOG_LEVELS = {**STANDARD_LOG_LEVELS, **CUSTOM_LOG_LEVELS}


class LogLevel(Enum):
    """Log levels with associated colors and emojis."""
    
    DEBUG = ("DEBUG", Fore.CYAN, "🔍")
    INFO = ("INFO", Fore.GREEN, "ℹ️")
    WARNING = ("WARNING", Fore.YELLOW, "⚠️")
    ERROR = ("ERROR", Fore.RED, "❌")
    CRITICAL = ("CRITICAL", Fore.MAGENTA + Style.BRIGHT, "💥")
    SUCCESS = ("SUCCESS", Fore.GREEN + Style.BRIGHT, "✅")
    PROGRESS = ("PROGRESS", Fore.BLUE, "⏳")


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors and emojis to log messages."""
    
    def __init__(self, use_colors: bool = True, use_emojis: bool = True):
        self.use_colors = use_colors
        self.use_emojis = use_emojis
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Get log level info
        level_name =  record.levelname if not hasattr(record, 'custom_level') else record.custom_level.upper()

        record_level = LogLevel.INFO
        if self.use_colors or self.use_emojis:
            for level in LogLevel:
                if level.value[0] == level_name:
                    record_level = level
                    break

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        
        # Add emoji, color, level_name and timestamp to the message
        if self.use_emojis:
            emoji = f"{record_level.value[2]} "
        else:
            emoji = ""

        if self.use_colors:
            color = record_level.value[1]
            level_part = f"{color}{level_name:<8}{Style.RESET_ALL}"
            time_part = f"{Fore.BLACK + Style.BRIGHT}{timestamp}{Style.RESET_ALL}"
        else:
            level_part = f"{level_name:<8}"
            time_part = timestamp        
        
        # Add context if available
        context = ""
        if hasattr(record, 'context') and record.context:
            context_str = f"[{record.context}]"
            if self.use_colors:
                context = f" {Fore.BLACK + Style.BRIGHT}{context_str}{Style.RESET_ALL}"
            else:
                context = f" {context_str}"
        
        # Combine all parts
        message = f"{emoji}{time_part} {level_part}{context} {record.getMessage()}"
        
        # Add exception info if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_entry["context"] = record.context
        
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info with better structure for log aggregation
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_entry["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": self._format_traceback_lines(exc_traceback)
            }
        
        return json.dumps(log_entry)
    
    def _format_traceback_lines(self, tb) -> List[str]:
        """Format traceback as an array of lines for better log aggregation, preserving indentation."""
        if tb is None:
            return []
        
        try:
            # Get the full traceback as a list of lines
            tb_lines = traceback.format_tb(tb)
            # Flatten the list and preserve indentation while removing trailing newlines
            formatted_lines = []
            for line in tb_lines:
                # Remove trailing newlines but preserve leading whitespace/indentation
                cleaned_line = line.rstrip('\n\r')
                if cleaned_line:  # Only add non-empty lines
                    formatted_lines.append(cleaned_line)
            return formatted_lines
        except Exception:
            # Fallback to string representation if formatting fails
            return [str(tb)]


class PaaSLogger:
    """
    Enhanced logger for PaaS AI with context support and custom levels.
    
    Features:
    - Color-coded console output
    - JSON structured logging for files
    - Thread-safe context-aware logging using contextvars
    - Custom log levels (SUCCESS, PROGRESS)
    - Easy configuration for different environments
    """
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        console_output: bool = True,
        file_output: Optional[Path] = None,
        json_format: bool = False,
        use_colors: bool = True,
        use_emojis: bool = True,
    ):
        self.logger = logging.getLogger(name)
        
        # Add custom levels to logging module first
        self._add_custom_levels()
        
        # Map log level string to numeric value
        log_level = ALL_LOG_LEVELS.get(level.upper())
        if log_level is None:
            raise ValueError(f"Invalid log level: {level}. Valid levels are: {list(ALL_LOG_LEVELS.keys())}")
        
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Disable propagation to parent loggers to avoid duplicate messages
        self.logger.propagate = False
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            if json_format:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(
                    ColoredFormatter(use_colors=use_colors, use_emojis=use_emojis)
                )
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            file_handler = logging.FileHandler(file_output)
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
    
    def _add_custom_levels(self):
        """Add custom log levels to the logging module."""
        # Add SUCCESS level
        logging.addLevelName(25, "SUCCESS")
        # Add PROGRESS level  
        logging.addLevelName(15, "PROGRESS")
    
    def set_context(self, context: str):
        """Set context for subsequent log messages in the current context."""
        _log_context.set(context)
    
    def clear_context(self):
        """Clear the current context."""
        _log_context.set(None)
    
    def get_context(self) -> Optional[str]:
        """Get the current context."""
        return _log_context.get()
    
    def _log_with_context(self, level: int, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log with context and extra fields."""
        record_extra = dict(extra) if extra else {}
        
        # Get context from contextvar (thread-safe)
        context = _log_context.get()
        if context:
            record_extra['context'] = context
        
        # Add custom level for SUCCESS and PROGRESS
        if level == 25:  # SUCCESS
            record_extra['custom_level'] = 'success'
        elif level == 15:  # PROGRESS
            record_extra['custom_level'] = 'progress'
        
        self.logger.log(level, message, extra=record_extra, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        self._log_with_context(25, message, **kwargs)
    
    def progress(self, message: str, **kwargs):
        """Log progress message."""
        self._log_with_context(15, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._log_with_context(logging.ERROR, message, exc_info=True, **kwargs)


def get_logger(
    name: str,
    level: str = "INFO",
    console: bool = True,
    file_path: Optional[Union[str, Path]] = None,
    json_format: bool = False,
    colors: bool = True,
    emojis: bool = True,
) -> PaaSLogger:
    """
    Get a configured PaaS logger instance.
    
    This is the recommended way to create loggers in library code, as it allows
    consumers to have full control over logging configuration.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, SUCCESS, PROGRESS)
        console: Enable console output
        file_path: Optional file path for logging
        json_format: Use JSON format (for production)
        colors: Enable colored output
        emojis: Enable emoji indicators
    
    Returns:
        Configured PaaSLogger instance
        
    Example:
        >>> from paas_ai.utils.logging import get_logger
        >>> logger = get_logger(__name__, level="DEBUG")
        >>> logger.info("This is a test message")
    """
    file_output = Path(file_path) if file_path else None
    
    return PaaSLogger(
        name=name,
        level=level,
        console_output=console,
        file_output=file_output,
        json_format=json_format,
        use_colors=colors,
        use_emojis=emojis,
    )


# Convenience logger for application usage
logger = get_logger("paas_ai")

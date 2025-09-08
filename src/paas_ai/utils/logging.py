"""
Professional logging utility for PaaS AI with colors, levels, and emojis.

Provides structured logging for both CLI and API components with:
- Color-coded output for different log levels
- Emoji indicators for visual clarity
- JSON structured logging for production
- Context-aware formatting
"""

import logging
import sys
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Union
from pathlib import Path

import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform color support
colorama.init(autoreset=True)


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
        level_name = record.levelname
        emoji = ""
        color = ""
        
        # Map standard levels to our enhanced levels
        for level in LogLevel:
            if level.value[0] == level_name:
                if self.use_emojis:
                    emoji = f"{level.value[2]} "
                if self.use_colors:
                    color = level.value[1]
                break
        
        # Handle custom SUCCESS and PROGRESS levels
        if hasattr(record, 'custom_level'):
            custom_level = record.custom_level
            for level in LogLevel:
                if level.name.lower() == custom_level.lower():
                    if self.use_emojis:
                        emoji = f"{level.value[2]} "
                    if self.use_colors:
                        color = level.value[1]
                    level_name = level.value[0]
                    break
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        
        # Build the message
        if self.use_colors:
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
        
        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class PaaSLogger:
    """
    Enhanced logger for PaaS AI with context support and custom levels.
    
    Features:
    - Color-coded console output
    - JSON structured logging for files
    - Context-aware logging
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
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add custom levels
        self._add_custom_levels()
        
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
        
        self.context = None
    
    def _add_custom_levels(self):
        """Add custom log levels."""
        # Add SUCCESS level
        logging.addLevelName(25, "SUCCESS")
        # Add PROGRESS level  
        logging.addLevelName(15, "PROGRESS")
    
    def set_context(self, context: str):
        """Set context for subsequent log messages."""
        self.context = context
    
    def clear_context(self):
        """Clear the current context."""
        self.context = None
    
    def _log_with_context(self, level: int, message: str, extra: Optional[Dict] = None, **kwargs):
        """Log with context and extra fields."""
        record_extra = extra or {}
        if self.context:
            record_extra['context'] = self.context
        
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
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console: Enable console output
        file_path: Optional file path for logging
        json_format: Use JSON format (for production)
        colors: Enable colored output
        emojis: Enable emoji indicators
    
    Returns:
        Configured PaaSLogger instance
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


# Default logger for the application
logger = get_logger("paas_ai") 
"""
Unit tests for the logging utility.

Tests all components of the logging system including:
- LogLevel enum
- ColoredFormatter
- JSONFormatter
- PaaSLogger class
- get_logger function
- Context management
- Custom log levels
"""

import json
import sys
import logging
import pytest
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextvars import ContextVar

from paas_ai.utils.logging import (
    LogLevel,
    ColoredFormatter,
    JSONFormatter,
    PaaSLogger,
    get_logger,
    logger,
    CUSTOM_LOG_LEVELS,
    STANDARD_LOG_LEVELS,
    ALL_LOG_LEVELS,
    _log_context,
)


class TestLogLevel:
    """Test the LogLevel enum."""
    
    def test_log_level_values(self):
        """Test that LogLevel enum has correct values."""
        assert LogLevel.DEBUG.value == ("DEBUG", "\x1b[36m", "🔍")
        assert LogLevel.INFO.value == ("INFO", "\x1b[32m", "ℹ️")
        assert LogLevel.WARNING.value == ("WARNING", "\x1b[33m", "⚠️")
        assert LogLevel.ERROR.value == ("ERROR", "\x1b[31m", "❌")
        assert LogLevel.CRITICAL.value == ("CRITICAL", "\x1b[35m\x1b[1m", "💥")
        assert LogLevel.SUCCESS.value == ("SUCCESS", "\x1b[32m\x1b[1m", "✅")
        assert LogLevel.PROGRESS.value == ("PROGRESS", "\x1b[34m", "⏳")
    
    def test_log_level_enumeration(self):
        """Test that all log levels can be enumerated."""
        levels = [level.value[0] for level in LogLevel]
        expected_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SUCCESS", "PROGRESS"]
        assert set(levels) == set(expected_levels)


class TestColoredFormatter:
    """Test the ColoredFormatter class."""
    
    def test_init_defaults(self):
        """Test ColoredFormatter initialization with defaults."""
        formatter = ColoredFormatter()
        assert formatter.use_colors is True
        assert formatter.use_emojis is True
    
    def test_init_custom(self):
        """Test ColoredFormatter initialization with custom settings."""
        formatter = ColoredFormatter(use_colors=False, use_emojis=False)
        assert formatter.use_colors is False
        assert formatter.use_emojis is False
    
    def test_format_with_colors_and_emojis(self):
        """Test formatting with colors and emojis enabled."""
        formatter = ColoredFormatter(use_colors=True, use_emojis=True)
        
        # Create a real log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should contain emoji and color codes
        assert "ℹ️" in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
        assert "\x1b[" in formatted  # ANSI color codes
    
    def test_format_without_colors_and_emojis(self):
        """Test formatting with colors and emojis disabled."""
        formatter = ColoredFormatter(use_colors=False, use_emojis=False)
        
        # Create a real log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should not contain emoji or color codes
        assert "ℹ️" not in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted
        assert "\x1b[" not in formatted  # No ANSI color codes
    
    def test_format_with_context(self):
        """Test formatting with context information."""
        formatter = ColoredFormatter()
        
        # Create a real log record with context
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.context = "test-context"
        
        formatted = formatter.format(record)
        
        assert "[test-context]" in formatted
        assert "Test message" in formatted
    
    def test_format_with_exception(self):
        """Test formatting with exception information."""
        formatter = ColoredFormatter()
        
        # Create a real log record with exception
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error",
            args=(),
            exc_info=None
        )
        
        # Create a real exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            record.exc_info = sys.exc_info()
        
        formatted = formatter.format(record)
        
        assert "Test error" in formatted
        assert "ValueError" in formatted
    
    def test_format_custom_levels(self):
        """Test formatting with custom log levels."""
        formatter = ColoredFormatter()
        
        # Test SUCCESS level
        record = logging.LogRecord(
            name="test",
            level=25,  # SUCCESS level
            pathname="test.py",
            lineno=1,
            msg="Success message",
            args=(),
            exc_info=None
        )
        record.levelname = "SUCCESS"
        record.custom_level = "success"
        
        formatted = formatter.format(record)
        
        assert "✅" in formatted
        assert "SUCCESS" in formatted
        assert "Success message" in formatted
        
        # Test PROGRESS level
        record = logging.LogRecord(
            name="test",
            level=15,  # PROGRESS level
            pathname="test.py",
            lineno=1,
            msg="Progress message",
            args=(),
            exc_info=None
        )
        record.levelname = "PROGRESS"
        record.custom_level = "progress"
        
        formatted = formatter.format(record)
        
        assert "⏳" in formatted
        assert "PROGRESS" in formatted
        assert "Progress message" in formatted


class TestJSONFormatter:
    """Test the JSONFormatter class."""
    
    def test_format_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        
        # Create a real log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["level"] == "INFO"
        assert log_entry["message"] == "Test message"
        assert log_entry["module"] == "test_module"
        assert log_entry["function"] == "test_function"
        assert log_entry["line"] == 42
        assert "timestamp" in log_entry
    
    def test_format_with_context(self):
        """Test JSON formatting with context."""
        formatter = JSONFormatter()
        
        # Create a real log record with context
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.context = "test-context"
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["context"] == "test-context"
    
    def test_format_with_extra_fields(self):
        """Test JSON formatting with extra fields."""
        formatter = JSONFormatter()
        
        # Create a real log record with extra fields
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.extra_fields = {"user_id": "123", "request_id": "abc"}
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["user_id"] == "123"
        assert log_entry["request_id"] == "abc"
    
    def test_format_with_exception(self):
        """Test JSON formatting with exception information."""
        formatter = JSONFormatter()
        
        # Create a real log record with exception
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Test error",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        # Create a real exception for testing
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            record.exc_info = (type(e), e, e.__traceback__)
        
        formatted = formatter.format(record)
        log_entry = json.loads(formatted)
        
        assert log_entry["level"] == "ERROR"
        assert log_entry["message"] == "Test error"
        assert "exception" in log_entry
        assert log_entry["exception"]["type"] == "ValueError"
        assert log_entry["exception"]["message"] == "Test exception"
        assert isinstance(log_entry["exception"]["traceback"], list)
    
    def test_format_traceback_lines(self):
        """Test the _format_traceback_lines method."""
        formatter = JSONFormatter()
        
        # Test with None traceback
        result = formatter._format_traceback_lines(None)
        assert result == []
        
        # Test with real traceback
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            tb = e.__traceback__
            result = formatter._format_traceback_lines(tb)
            assert isinstance(result, list)
            assert len(result) > 0
            # Check that lines don't have trailing newlines
            for line in result:
                assert not line.endswith('\n')
                assert not line.endswith('\r')
        
        # Test with invalid traceback (should return string representation)
        result = formatter._format_traceback_lines("invalid_tb")
        assert result == ["invalid_tb"]


class TestPaaSLogger:
    """Test the PaaSLogger class."""
    
    def test_init_defaults(self):
        """Test PaaSLogger initialization with defaults."""
        logger = PaaSLogger("test_logger")
        
        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.INFO
        assert len(logger.logger.handlers) == 1  # Console handler
        assert not logger.logger.propagate
    
    def test_init_custom_level(self):
        """Test PaaSLogger initialization with custom level."""
        logger = PaaSLogger("test_logger", level="DEBUG")
        assert logger.logger.level == logging.DEBUG
        
        logger = PaaSLogger("test_logger", level="SUCCESS")
        assert logger.logger.level == 25  # Custom SUCCESS level
    
    def test_init_invalid_level(self):
        """Test PaaSLogger initialization with invalid level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            PaaSLogger("test_logger", level="INVALID")
    
    def test_init_with_file_output(self):
        """Test PaaSLogger initialization with file output."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            logger = PaaSLogger("test_logger", file_output=tmp_path)
            assert len(logger.logger.handlers) == 2  # Console + file handler
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_init_no_console_output(self):
        """Test PaaSLogger initialization without console output."""
        logger = PaaSLogger("test_logger", console_output=False)
        assert len(logger.logger.handlers) == 0
    
    def test_init_json_format(self):
        """Test PaaSLogger initialization with JSON format."""
        logger = PaaSLogger("test_logger", json_format=True)
        # Should have console handler with JSON formatter
        assert len(logger.logger.handlers) == 1
        assert isinstance(logger.logger.handlers[0].formatter, JSONFormatter)
    
    def test_add_custom_levels(self):
        """Test that custom levels are added to logging module."""
        # Reset logging module state
        logging._levelToName.pop(25, None)
        logging._levelToName.pop(15, None)
        logging._nameToLevel.pop("SUCCESS", None)
        logging._nameToLevel.pop("PROGRESS", None)
        
        logger = PaaSLogger("test_logger")
        
        # Check that custom levels are registered
        assert logging.getLevelName(25) == "SUCCESS"
        assert logging.getLevelName(15) == "PROGRESS"
        assert logging._nameToLevel["SUCCESS"] == 25
        assert logging._nameToLevel["PROGRESS"] == 15
    
    def test_context_management(self):
        """Test context management methods."""
        logger = PaaSLogger("test_logger")
        
        # Test setting context
        logger.set_context("test-context")
        assert logger.get_context() == "test-context"
        
        # Test clearing context
        logger.clear_context()
        assert logger.get_context() is None
    
    def test_log_with_context(self):
        """Test logging with context."""
        logger = PaaSLogger("test_logger")
        logger.set_context("test-context")
        
        # Mock the logger.log method
        logger.logger.log = Mock()
        
        logger._log_with_context(logging.INFO, "Test message")
        
        # Verify that context was passed in extra
        logger.logger.log.assert_called_once()
        call_args = logger.logger.log.call_args
        assert call_args[0][0] == logging.INFO
        assert call_args[0][1] == "Test message"
        assert call_args[1]["extra"]["context"] == "test-context"
    
    def test_log_without_context(self):
        """Test logging without context."""
        logger = PaaSLogger("test_logger")
        logger.clear_context()
        
        # Mock the logger.log method
        logger.logger.log = Mock()
        
        logger._log_with_context(logging.INFO, "Test message")
        
        # Verify that no context was passed
        call_args = logger.logger.log.call_args
        assert "context" not in call_args[1]["extra"]
    
    def test_log_with_extra_fields(self):
        """Test logging with extra fields."""
        logger = PaaSLogger("test_logger")
        
        # Mock the logger.log method
        logger.logger.log = Mock()
        
        extra = {"user_id": "123", "request_id": "abc"}
        logger._log_with_context(logging.INFO, "Test message", extra=extra)
        
        # Verify that extra fields were passed
        call_args = logger.logger.log.call_args
        assert call_args[1]["extra"]["user_id"] == "123"
        assert call_args[1]["extra"]["request_id"] == "abc"
    
    def test_custom_level_logging(self):
        """Test logging with custom levels."""
        logger = PaaSLogger("test_logger")
        
        # Mock the logger.log method
        logger.logger.log = Mock()
        
        # Test SUCCESS level
        logger._log_with_context(25, "Success message")
        call_args = logger.logger.log.call_args
        assert call_args[1]["extra"]["custom_level"] == "success"
        
        # Test PROGRESS level
        logger._log_with_context(15, "Progress message")
        call_args = logger.logger.log.call_args
        assert call_args[1]["extra"]["custom_level"] == "progress"
    
    def test_standard_logging_methods(self):
        """Test all standard logging methods."""
        logger = PaaSLogger("test_logger")
        
        # Mock the _log_with_context method
        logger._log_with_context = Mock()
        
        # Test all logging methods
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        logger.success("Success message")
        logger.progress("Progress message")
        
        # Verify all methods were called with correct levels
        expected_calls = [
            (logging.DEBUG, "Debug message"),
            (logging.INFO, "Info message"),
            (logging.WARNING, "Warning message"),
            (logging.ERROR, "Error message"),
            (logging.CRITICAL, "Critical message"),
            (25, "Success message"),
            (15, "Progress message"),
        ]
        
        assert logger._log_with_context.call_count == 7
        for i, (level, message) in enumerate(expected_calls):
            call_args = logger._log_with_context.call_args_list[i]
            assert call_args[0][0] == level
            assert call_args[0][1] == message
    
    def test_exception_logging(self):
        """Test exception logging method."""
        logger = PaaSLogger("test_logger")
        
        # Mock the _log_with_context method
        logger._log_with_context = Mock()
        
        logger.exception("Exception message")
        
        # Verify exception was logged with exc_info=True
        call_args = logger._log_with_context.call_args
        assert call_args[0][0] == logging.ERROR
        assert call_args[0][1] == "Exception message"
        assert call_args[1]["exc_info"] is True


class TestGetLogger:
    """Test the get_logger function."""
    
    def test_get_logger_defaults(self):
        """Test get_logger with default parameters."""
        logger = get_logger("test_logger")
        
        assert isinstance(logger, PaaSLogger)
        assert logger.logger.name == "test_logger"
        assert logger.logger.level == logging.INFO
        assert len(logger.logger.handlers) == 1  # Console handler
    
    def test_get_logger_custom_parameters(self):
        """Test get_logger with custom parameters."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            logger = get_logger(
                name="test_logger",
                level="DEBUG",
                console=False,
                file_path=tmp_path,
                json_format=True,
                colors=False,
                emojis=False
            )
            
            assert logger.logger.level == logging.DEBUG
            assert len(logger.logger.handlers) == 1  # File handler only
            assert isinstance(logger.logger.handlers[0].formatter, JSONFormatter)
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_get_logger_with_string_path(self):
        """Test get_logger with string file path."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            logger = get_logger("test_logger", file_path=tmp_path)
            assert len(logger.logger.handlers) == 2  # Console + file handler
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestLoggingConstants:
    """Test logging constants and mappings."""
    
    def test_custom_log_levels(self):
        """Test custom log level mappings."""
        assert CUSTOM_LOG_LEVELS["SUCCESS"] == 25
        assert CUSTOM_LOG_LEVELS["PROGRESS"] == 15
    
    def test_standard_log_levels(self):
        """Test standard log level mappings."""
        assert STANDARD_LOG_LEVELS["DEBUG"] == logging.DEBUG
        assert STANDARD_LOG_LEVELS["INFO"] == logging.INFO
        assert STANDARD_LOG_LEVELS["WARNING"] == logging.WARNING
        assert STANDARD_LOG_LEVELS["ERROR"] == logging.ERROR
        assert STANDARD_LOG_LEVELS["CRITICAL"] == logging.CRITICAL
    
    def test_all_log_levels(self):
        """Test combined log level mappings."""
        assert len(ALL_LOG_LEVELS) == len(CUSTOM_LOG_LEVELS) + len(STANDARD_LOG_LEVELS)
        assert "SUCCESS" in ALL_LOG_LEVELS
        assert "PROGRESS" in ALL_LOG_LEVELS
        assert "DEBUG" in ALL_LOG_LEVELS
        assert "INFO" in ALL_LOG_LEVELS


class TestConvenienceLogger:
    """Test the convenience logger instance."""
    
    def test_convenience_logger_exists(self):
        """Test that the convenience logger exists and is properly configured."""
        assert isinstance(logger, PaaSLogger)
        assert logger.logger.name == "paas_ai"


class TestContextVar:
    """Test the context variable functionality."""
    
    def test_context_var_initialization(self):
        """Test that the context variable is properly initialized."""
        assert isinstance(_log_context, ContextVar)
        assert _log_context.name == "log_context"
        assert _log_context.get() is None
    
    def test_context_var_set_get(self):
        """Test setting and getting context variable values."""
        # Test setting context
        _log_context.set("test-context")
        assert _log_context.get() == "test-context"
        
        # Test clearing context
        _log_context.set(None)
        assert _log_context.get() is None


class TestIntegration:
    """Integration tests for the logging system."""
    
    def test_full_logging_workflow(self):
        """Test a complete logging workflow."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Create logger with file output
            logger = get_logger("integration_test", file_path=tmp_path)
            
            # Set context
            logger.set_context("integration-test")
            
            # Log various messages
            logger.info("Info message")
            logger.success("Success message")
            logger.progress("Progress message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            # Clear context and log without context
            logger.clear_context()
            logger.info("Message without context")
            
            # Check that file was written to
            assert tmp_path.exists()
            assert tmp_path.stat().st_size > 0
            
            # Read and verify log entries
            with open(tmp_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) >= 5  # At least 5 log entries (PROGRESS level 15 is below INFO level 20)
            
            # Verify JSON format
            for line in lines:
                log_entry = json.loads(line.strip())
                assert "timestamp" in log_entry
                assert "level" in log_entry
                assert "message" in log_entry
                assert "module" in log_entry
                assert "function" in log_entry
                assert "line" in log_entry
        
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_exception_handling_integration(self):
        """Test exception handling in integration scenario."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            logger = get_logger("exception_test", file_path=tmp_path)
            
            # Log an exception
            try:
                raise ValueError("Integration test exception")
            except ValueError:
                logger.exception("Caught exception in integration test")
            
            # Verify exception was logged
            with open(tmp_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 1
            log_entry = json.loads(lines[0].strip())
            
            assert log_entry["level"] == "ERROR"
            assert log_entry["message"] == "Caught exception in integration test"
            assert "exception" in log_entry
            assert log_entry["exception"]["type"] == "ValueError"
            assert log_entry["exception"]["message"] == "Integration test exception"
            assert isinstance(log_entry["exception"]["traceback"], list)
            assert len(log_entry["exception"]["traceback"]) > 0
        
        finally:
            tmp_path.unlink(missing_ok=True)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_message(self):
        """Test logging with empty message."""
        logger = PaaSLogger("test_logger")
        logger._log_with_context = Mock()
        
        logger.info("")
        
        call_args = logger._log_with_context.call_args
        assert call_args[0][1] == ""
    
    def test_very_long_message(self):
        """Test logging with very long message."""
        logger = PaaSLogger("test_logger")
        logger._log_with_context = Mock()
        
        long_message = "x" * 10000
        logger.info(long_message)
        
        call_args = logger._log_with_context.call_args
        assert call_args[0][1] == long_message
    
    def test_special_characters_in_message(self):
        """Test logging with special characters."""
        logger = PaaSLogger("test_logger")
        logger._log_with_context = Mock()
        
        special_message = "Test with special chars: \n\t\r\"'\\"
        logger.info(special_message)
        
        call_args = logger._log_with_context.call_args
        assert call_args[0][1] == special_message
    
    def test_unicode_message(self):
        """Test logging with unicode message."""
        logger = PaaSLogger("test_logger")
        logger._log_with_context = Mock()
        
        unicode_message = "Test with unicode: 🚀 中文 العربية"
        logger.info(unicode_message)
        
        call_args = logger._log_with_context.call_args
        assert call_args[0][1] == unicode_message
    
    def test_none_context(self):
        """Test setting None as context."""
        logger = PaaSLogger("test_logger")
        
        logger.set_context(None)
        assert logger.get_context() is None
        
        logger._log_with_context = Mock()
        logger.info("Test message")
        
        # Should not include context in extra
        call_args = logger._log_with_context.call_args
        # Check if extra exists and context is not in it
        if "extra" in call_args[1]:
            assert "context" not in call_args[1]["extra"]
        else:
            # If no extra dict, that's also fine
            assert True
    
    def test_empty_string_context(self):
        """Test setting empty string as context."""
        logger = PaaSLogger("test_logger")
        
        logger.set_context("")
        assert logger.get_context() == ""
        
        # Mock the logger.log method instead of _log_with_context
        logger.logger.log = Mock()
        logger.info("Test message")
        
        # Verify that empty context was NOT passed in extra (empty strings are falsy)
        logger.logger.log.assert_called_once()
        call_args = logger.logger.log.call_args
        assert "context" not in call_args[1]["extra"]

"""
Unit tests for markdown splitter strategy.

Tests all components of the MarkdownSplitterStrategy including:
- MarkdownSplitterStrategy class
- Configuration validation
- Splitter creation
- Error handling and edge cases
- Headers parameter handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.splitters.markdown import MarkdownSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestMarkdownSplitterStrategy:
    """Test the MarkdownSplitterStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = MarkdownSplitterStrategy()
        assert strategy is not None
        assert isinstance(strategy, MarkdownSplitterStrategy)
    
    def test_create_splitter_basic(self):
        """Test creating splitter with basic configuration."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_custom_headers(self):
        """Test creating splitter with custom headers."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4")
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_additional_params(self):
        """Test creating splitter with additional parameters."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [("#", "Header 1"), ("##", "Header 2")],
                "strip_headers": True
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
                strip_headers=True
            )
            assert result == mock_splitter
    
    def test_create_splitter_empty_params(self):
        """Test creating splitter with empty params."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]
            )
            assert result == mock_splitter
    
    def test_validate_config_valid_headers(self):
        """Test configuration validation with valid headers."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ]
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_headers_not_list(self):
        """Test configuration validation with headers that are not a list."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": "not-a-list"}
        )
        
        with pytest.raises(ValueError, match="headers_to_split_on must be a list of tuples"):
            strategy.validate_config(config)
    
    def test_validate_config_headers_not_tuples(self):
        """Test configuration validation with headers that are not tuples."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [["#", "Header 1"], "not-a-tuple"]}
        )
        
        with pytest.raises(ValueError, match="each header must be a tuple of \\(markdown_header, name\\)"):
            strategy.validate_config(config)
    
    def test_validate_config_headers_wrong_length(self):
        """Test configuration validation with headers of wrong length."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [("#", "Header 1", "extra"), ("##", "Header 2")]}
        )
        
        with pytest.raises(ValueError, match="each header must be a tuple of \\(markdown_header, name\\)"):
            strategy.validate_config(config)
    
    def test_validate_config_headers_non_string_elements(self):
        """Test configuration validation with headers containing non-string elements."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [("#", 123), ("##", "Header 2")]}
        )
        
        with pytest.raises(ValueError, match="header tuple must contain two strings"):
            strategy.validate_config(config)
    
    def test_validate_config_empty_headers_list(self):
        """Test configuration validation with empty headers list."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": []}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_none_headers(self):
        """Test configuration validation with None headers."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": None}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_no_headers_param(self):
        """Test configuration validation without headers parameter."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestMarkdownSplitterStrategyEdgeCases:
    """Test edge cases for MarkdownSplitterStrategy."""
    
    def test_create_splitter_with_special_characters_in_headers(self):
        """Test creating splitter with special characters in headers."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header with spaces"),
                    ("##", "Header-with-dashes"),
                    ("###", "Header_with_underscores")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("#", "Header with spaces"),
                    ("##", "Header-with-dashes"),
                    ("###", "Header_with_underscores")
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_unicode_headers(self):
        """Test creating splitter with unicode characters in headers."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header with émojis 🚀"),
                    ("##", "中文标题"),
                    ("###", "العنوان العربي")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("#", "Header with émojis 🚀"),
                    ("##", "中文标题"),
                    ("###", "العنوان العربي")
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = MarkdownSplitterStrategy()
        original_params = {
            "headers_to_split_on": [("#", "Header 1"), ("##", "Header 2")],
            "strip_headers": True
        }
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_splitter_with_many_headers(self):
        """Test creating splitter with many headers."""
        strategy = MarkdownSplitterStrategy()
        headers = [(f"{'#' * i}", f"Header {i}") for i in range(1, 7)]
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": headers}
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=headers
            )
            assert result == mock_splitter
    
    def test_validate_config_with_very_long_header_names(self):
        """Test configuration validation with very long header names."""
        strategy = MarkdownSplitterStrategy()
        long_name = "a" * 1000
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", long_name),
                    ("##", "Normal Header")
                ]
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_long_markdown_headers(self):
        """Test configuration validation with very long markdown headers."""
        strategy = MarkdownSplitterStrategy()
        long_header = "#" * 100
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    (long_header, "Header 1"),
                    ("##", "Header 2")
                ]
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_mixed_valid_and_invalid_headers(self):
        """Test configuration validation with mix of valid and invalid headers."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Valid Header"),
                    ("##", 123),  # Invalid - not a string
                    ("###", "Another Valid Header")
                ]
            }
        )
        
        with pytest.raises(ValueError, match="header tuple must contain two strings"):
            strategy.validate_config(config)


class TestMarkdownSplitterStrategyIntegration:
    """Integration tests for MarkdownSplitterStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = MarkdownSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once()
    
    def test_workflow_with_different_header_configurations(self):
        """Test workflow with different header configurations."""
        strategy = MarkdownSplitterStrategy()
        
        # Test with different header configurations
        configs = [
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={
                    "headers_to_split_on": [
                        ("#", "Header 1"),
                        ("##", "Header 2"),
                        ("###", "Header 3"),
                        ("####", "Header 4")
                    ]
                }
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={
                    "headers_to_split_on": [
                        ("#", "Main Header"),
                        ("##", "Sub Header")
                    ]
                }
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={
                    "headers_to_split_on": [
                        ("#", "Single Header")
                    ]
                }
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={}  # Default headers
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            for config in configs:
                # Validate config
                strategy.validate_config(config)
                
                # Create splitter
                result = strategy.create_splitter(config)
                
                assert result == mock_splitter
            
            # Should have been called 4 times
            assert mock_splitter_class.call_count == 4
    
    def test_workflow_with_mixed_valid_and_invalid_configs(self):
        """Test workflow with mix of valid and invalid configurations."""
        strategy = MarkdownSplitterStrategy()
        
        # Valid config
        valid_config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2")
                ]
            }
        )
        
        # Invalid config (non-string header name)
        invalid_config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", 123)  # Invalid
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_splitter(valid_config)
            assert result == mock_splitter
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="header tuple must contain two strings"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_additional_parameters(self):
        """Test workflow with additional parameters."""
        strategy = MarkdownSplitterStrategy()
        
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2")
                ],
                "strip_headers": True,
                "add_start_index": False
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2")
                ],
                strip_headers=True,
                add_start_index=False
            )
    
    def test_workflow_with_header_validation_errors(self):
        """Test workflow with header validation errors."""
        strategy = MarkdownSplitterStrategy()
        
        # Test with various invalid header configurations
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": "not-a-list"}
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [["#", "Header 1"], "not-a-tuple"]}
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [("#", "Header 1", "extra"), ("##", "Header 2")]}
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [("#", 123), ("##", "Header 2")]}
            )
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                strategy.validate_config(config)

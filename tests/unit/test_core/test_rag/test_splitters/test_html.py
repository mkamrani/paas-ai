"""
Unit tests for HTML splitter strategy.

Tests all components of the HTMLSplitterStrategy including:
- HTMLSplitterStrategy class
- Configuration validation
- Splitter creation
- Error handling and edge cases
- Headers parameter handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.splitters.html import HTMLSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestHTMLSplitterStrategy:
    """Test the HTMLSplitterStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = HTMLSplitterStrategy()
        assert strategy is not None
        assert isinstance(strategy, HTMLSplitterStrategy)
    
    def test_create_splitter_basic(self):
        """Test creating splitter with basic configuration."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_custom_headers(self):
        """Test creating splitter with custom headers."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                    ("h4", "Header 4"),
                    ("h5", "Header 5"),
                    ("h6", "Header 6")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                    ("h4", "Header 4"),
                    ("h5", "Header 5"),
                    ("h6", "Header 6")
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_additional_params(self):
        """Test creating splitter with additional parameters."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [("h1", "Header 1"), ("h2", "Header 2")],
                "strip_headers": True
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")],
                strip_headers=True
            )
            assert result == mock_splitter
    
    def test_create_splitter_empty_params(self):
        """Test creating splitter with empty params."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3"),
                ]
            )
            assert result == mock_splitter
    
    def test_validate_config_valid_headers(self):
        """Test configuration validation with valid headers."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3")
                ]
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_headers_not_list(self):
        """Test configuration validation with headers that are not a list."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": "not-a-list"}
        )
        
        with pytest.raises(ValueError, match="headers_to_split_on must be a list of tuples"):
            strategy.validate_config(config)
    
    def test_validate_config_headers_not_tuples(self):
        """Test configuration validation with headers that are not tuples."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [["h1", "Header 1"], "not-a-tuple"]}
        )
        
        with pytest.raises(ValueError, match="each header must be a tuple of \\(html_tag, name\\)"):
            strategy.validate_config(config)
    
    def test_validate_config_headers_wrong_length(self):
        """Test configuration validation with headers of wrong length."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [("h1", "Header 1", "extra"), ("h2", "Header 2")]}
        )
        
        with pytest.raises(ValueError, match="each header must be a tuple of \\(html_tag, name\\)"):
            strategy.validate_config(config)
    
    def test_validate_config_headers_non_string_elements(self):
        """Test configuration validation with headers containing non-string elements."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [("h1", 123), ("h2", "Header 2")]}
        )
        
        with pytest.raises(ValueError, match="header tuple must contain two strings"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_html_tag(self):
        """Test configuration validation with invalid HTML tag."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [("div", "Header 1"), ("h2", "Header 2")]}
        )
        
        with pytest.raises(ValueError, match="HTML header tag must be h1-h6, got: div"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_html_tag_format(self):
        """Test configuration validation with invalid HTML tag format."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [("h10", "Header 1"), ("h2", "Header 2")]}
        )
        
        with pytest.raises(ValueError, match="HTML header tag must be h1-h6, got: h10"):
            strategy.validate_config(config)
    
    def test_validate_config_case_insensitive_html_tags(self):
        """Test configuration validation with case insensitive HTML tags."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": [("H1", "Header 1"), ("H2", "Header 2")]}
        )
        
        # Should not raise any exception (case insensitive)
        strategy.validate_config(config)
    
    def test_validate_config_empty_headers_list(self):
        """Test configuration validation with empty headers list."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": []}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_none_headers(self):
        """Test configuration validation with None headers."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": None}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_no_headers_param(self):
        """Test configuration validation without headers parameter."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestHTMLSplitterStrategyEdgeCases:
    """Test edge cases for HTMLSplitterStrategy."""
    
    def test_create_splitter_with_special_characters_in_headers(self):
        """Test creating splitter with special characters in headers."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header with spaces"),
                    ("h2", "Header-with-dashes"),
                    ("h3", "Header_with_underscores")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("h1", "Header with spaces"),
                    ("h2", "Header-with-dashes"),
                    ("h3", "Header_with_underscores")
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_unicode_headers(self):
        """Test creating splitter with unicode characters in headers."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header with émojis 🚀"),
                    ("h2", "中文标题"),
                    ("h3", "العنوان العربي")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("h1", "Header with émojis 🚀"),
                    ("h2", "中文标题"),
                    ("h3", "العنوان العربي")
                ]
            )
            assert result == mock_splitter
    
    def test_create_splitter_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = HTMLSplitterStrategy()
        original_params = {
            "headers_to_split_on": [("h1", "Header 1"), ("h2", "Header 2")],
            "strip_headers": True
        }
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_splitter_with_all_html_header_levels(self):
        """Test creating splitter with all HTML header levels."""
        strategy = HTMLSplitterStrategy()
        headers = [(f"h{i}", f"Header {i}") for i in range(1, 7)]
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={"headers_to_split_on": headers}
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=headers
            )
            assert result == mock_splitter
    
    def test_validate_config_with_very_long_header_names(self):
        """Test configuration validation with very long header names."""
        strategy = HTMLSplitterStrategy()
        long_name = "a" * 1000
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", long_name),
                    ("h2", "Normal Header")
                ]
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_mixed_valid_and_invalid_headers(self):
        """Test configuration validation with mix of valid and invalid headers."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Valid Header"),
                    ("div", "Invalid Header"),  # Invalid - not h1-h6
                    ("h2", "Another Valid Header")
                ]
            }
        )
        
        with pytest.raises(ValueError, match="HTML header tag must be h1-h6, got: div"):
            strategy.validate_config(config)
    
    def test_validate_config_with_whitespace_in_html_tags(self):
        """Test configuration validation with whitespace in HTML tags."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    (" h1 ", "Header 1"),  # With whitespace
                    ("h2", "Header 2")
                ]
            }
        )
        
        with pytest.raises(ValueError, match="HTML header tag must be h1-h6, got:  h1 "):
            strategy.validate_config(config)
    
    def test_validate_config_with_numeric_html_tags(self):
        """Test configuration validation with numeric HTML tags."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h7", "Header 7")  # Invalid - h7 doesn't exist
                ]
            }
        )
        
        with pytest.raises(ValueError, match="HTML header tag must be h1-h6, got: h7"):
            strategy.validate_config(config)


class TestHTMLSplitterStrategyIntegration:
    """Integration tests for HTMLSplitterStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = HTMLSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
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
        strategy = HTMLSplitterStrategy()
        
        # Test with different header configurations
        configs = [
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={
                    "headers_to_split_on": [
                        ("h1", "Header 1"),
                        ("h2", "Header 2"),
                        ("h3", "Header 3"),
                        ("h4", "Header 4")
                    ]
                }
            ),
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={
                    "headers_to_split_on": [
                        ("h1", "Main Header"),
                        ("h2", "Sub Header")
                    ]
                }
            ),
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={
                    "headers_to_split_on": [
                        ("h1", "Single Header")
                    ]
                }
            ),
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={}  # Default headers
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
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
        strategy = HTMLSplitterStrategy()
        
        # Valid config
        valid_config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("h2", "Header 2")
                ]
            }
        )
        
        # Invalid config (non-HTML header tag)
        invalid_config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("div", "Invalid Header")  # Invalid
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_splitter(valid_config)
            assert result == mock_splitter
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="HTML header tag must be h1-h6, got: div"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_additional_parameters(self):
        """Test workflow with additional parameters."""
        strategy = HTMLSplitterStrategy()
        
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("h2", "Header 2")
                ],
                "strip_headers": True,
                "add_start_index": False
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2")
                ],
                strip_headers=True,
                add_start_index=False
            )
    
    def test_workflow_with_header_validation_errors(self):
        """Test workflow with header validation errors."""
        strategy = HTMLSplitterStrategy()
        
        # Test with various invalid header configurations
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": "not-a-list"}
            ),
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [["h1", "Header 1"], "not-a-tuple"]}
            ),
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [("h1", "Header 1", "extra"), ("h2", "Header 2")]}
            ),
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [("h1", 123), ("h2", "Header 2")]}
            ),
            SplitterConfig(
                type=SplitterType.HTML,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [("div", "Header 1"), ("h2", "Header 2")]}
            )
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                strategy.validate_config(config)

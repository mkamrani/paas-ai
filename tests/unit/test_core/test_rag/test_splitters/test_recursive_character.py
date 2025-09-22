"""
Unit tests for recursive character splitter strategy.

Tests all components of the RecursiveCharacterSplitterStrategy including:
- RecursiveCharacterSplitterStrategy class
- Configuration validation
- Splitter creation
- Error handling and edge cases
- Separators parameter handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.splitters.recursive_character import RecursiveCharacterSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestRecursiveCharacterSplitterStrategy:
    """Test the RecursiveCharacterSplitterStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = RecursiveCharacterSplitterStrategy()
        assert strategy is not None
        assert isinstance(strategy, RecursiveCharacterSplitterStrategy)
    
    def test_create_splitter_basic(self):
        """Test creating splitter with basic configuration."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=None
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_custom_separators(self):
        """Test creating splitter with custom separators."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\n\n", "\n", " ", ""]}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_additional_params(self):
        """Test creating splitter with additional parameters."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1500,
            chunk_overlap=300,
            params={
                "separators": ["\n\n", "\n"],
                "length_function": len,
                "is_separator_regex": False
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n"],
                length_function=len,
                is_separator_regex=False
            )
            assert result == mock_splitter
    
    def test_create_splitter_empty_params(self):
        """Test creating splitter with empty params."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=None
            )
            assert result == mock_splitter
    
    def test_validate_config_valid_chunk_size(self):
        """Test configuration validation with valid chunk size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_zero_chunk_size(self):
        """Test configuration validation with zero chunk size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=0,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_size(self):
        """Test configuration validation with negative chunk size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=-100,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_overlap(self):
        """Test configuration validation with negative chunk overlap."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=-50,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_equals_chunk_size(self):
        """Test configuration validation when chunk_overlap equals chunk_size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=1000,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_greater_than_chunk_size(self):
        """Test configuration validation when chunk_overlap is greater than chunk_size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=1500,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_valid_separators(self):
        """Test configuration validation with valid separators."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\n\n", "\n", " ", ""]}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_invalid_separators_not_list(self):
        """Test configuration validation with separators that are not a list."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": "not-a-list"}
        )
        
        with pytest.raises(ValueError, match="separators must be a list of strings"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_separators_non_string_elements(self):
        """Test configuration validation with separators containing non-string elements."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\n\n", 123, " "]}
        )
        
        with pytest.raises(ValueError, match="all separators must be strings"):
            strategy.validate_config(config)
    
    def test_validate_config_empty_separators_list(self):
        """Test configuration validation with empty separators list."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": []}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_none_separators(self):
        """Test configuration validation with None separators."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": None}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestRecursiveCharacterSplitterStrategyEdgeCases:
    """Test edge cases for RecursiveCharacterSplitterStrategy."""
    
    def test_create_splitter_with_special_characters_in_separators(self):
        """Test creating splitter with special characters in separators."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\t\n\r", "\f\v", "\x00"]}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\t\n\r", "\f\v", "\x00"]
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_regex_separators(self):
        """Test creating splitter with regex separators."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "separators": [r"\n\s*\n", r"\s+", r""],
                "is_separator_regex": True
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=[r"\n\s*\n", r"\s+", r""],
                is_separator_regex=True
            )
            assert result == mock_splitter
    
    def test_create_splitter_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = RecursiveCharacterSplitterStrategy()
        original_params = {"separators": ["\n\n", "\n"], "length_function": len}
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_splitter_with_mixed_separator_types(self):
        """Test creating splitter with mixed separator types (should fail validation)."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\n\n", 123, None, " "]}
        )
        
        with pytest.raises(ValueError, match="all separators must be strings"):
            strategy.validate_config(config)
    
    def test_create_splitter_with_unicode_separators(self):
        """Test creating splitter with unicode separators."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\u2029", "\u2028", " "]}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\u2029", "\u2028", " "]
            )
            assert result == mock_splitter
    
    def test_validate_config_with_very_large_chunk_size(self):
        """Test configuration validation with very large chunk size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000000,
            chunk_overlap=100000,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_small_chunk_size(self):
        """Test configuration validation with very small chunk size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1,
            chunk_overlap=0,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_chunk_overlap_almost_equal_to_chunk_size(self):
        """Test configuration validation with chunk_overlap almost equal to chunk_size."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=999,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_long_separators(self):
        """Test configuration validation with very long separators."""
        strategy = RecursiveCharacterSplitterStrategy()
        long_separator = "a" * 1000
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": [long_separator, "\n\n", " "]}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestRecursiveCharacterSplitterStrategyIntegration:
    """Integration tests for RecursiveCharacterSplitterStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = RecursiveCharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\n\n", "\n", " "]}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once()
    
    def test_workflow_with_different_separator_configurations(self):
        """Test workflow with different separator configurations."""
        strategy = RecursiveCharacterSplitterStrategy()
        
        # Test with different separator configurations
        configs = [
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separators": ["\n\n", "\n", " ", ""]}
            ),
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separators": ["\n", " "]}
            ),
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separators": [" "]}
            ),
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={}  # Default separators
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
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
        strategy = RecursiveCharacterSplitterStrategy()
        
        # Valid config
        valid_config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\n\n", "\n", " "]}
        )
        
        # Invalid config (negative chunk size)
        invalid_config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=-100,
            chunk_overlap=200,
            params={"separators": ["\n\n", "\n", " "]}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_splitter(valid_config)
            assert result == mock_splitter
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_custom_length_function(self):
        """Test workflow with custom length function."""
        strategy = RecursiveCharacterSplitterStrategy()
        
        def custom_length_function(text):
            return len(text.split())
        
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=100,
            chunk_overlap=20,
            params={
                "separators": ["\n\n", "\n", " "],
                "length_function": custom_length_function
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once_with(
                chunk_size=100,
                chunk_overlap=20,
                separators=["\n\n", "\n", " "],
                length_function=custom_length_function
            )
    
    def test_workflow_with_separator_validation_errors(self):
        """Test workflow with separator validation errors."""
        strategy = RecursiveCharacterSplitterStrategy()
        
        # Test with invalid separators
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separators": "not-a-list"}
            ),
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separators": ["\n\n", 123, " "]}
            )
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                strategy.validate_config(config)

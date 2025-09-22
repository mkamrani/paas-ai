"""
Unit tests for token splitter strategy.

Tests all components of the TokenSplitterStrategy including:
- TokenSplitterStrategy class
- Configuration validation
- Splitter creation
- Error handling and edge cases
- Encoding and model parameter handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.splitters.token import TokenSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestTokenSplitterStrategy:
    """Test the TokenSplitterStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = TokenSplitterStrategy()
        assert strategy is not None
        assert isinstance(strategy, TokenSplitterStrategy)
    
    def test_create_splitter_basic(self):
        """Test creating splitter with basic configuration."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_encoding_name(self):
        """Test creating splitter with encoding name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": "cl100k_base"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                encoding_name="cl100k_base"
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_model_name(self):
        """Test creating splitter with model name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"model_name": "gpt-3.5-turbo"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                model_name="gpt-3.5-turbo"
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_additional_params(self):
        """Test creating splitter with additional parameters."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1500,
            chunk_overlap=300,
            params={
                "encoding_name": "cl100k_base",
                "model_name": "gpt-4",
                "allowed_special": ["<|endoftext|>"],
                "disallowed_special": ["<|startoftext|>"]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1500,
                chunk_overlap=300,
                encoding_name="cl100k_base",
                model_name="gpt-4",
                allowed_special=["<|endoftext|>"],
                disallowed_special=["<|startoftext|>"]
            )
            assert result == mock_splitter
    
    def test_create_splitter_empty_params(self):
        """Test creating splitter with empty params."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200
            )
            assert result == mock_splitter
    
    def test_validate_config_valid_chunk_size(self):
        """Test configuration validation with valid chunk size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_zero_chunk_size(self):
        """Test configuration validation with zero chunk size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=0,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_size(self):
        """Test configuration validation with negative chunk size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=-100,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_overlap(self):
        """Test configuration validation with negative chunk overlap."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=-50,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_equals_chunk_size(self):
        """Test configuration validation when chunk_overlap equals chunk_size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=1000,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_greater_than_chunk_size(self):
        """Test configuration validation when chunk_overlap is greater than chunk_size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=1500,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_valid_encoding_name(self):
        """Test configuration validation with valid encoding name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": "cl100k_base"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            mock_tiktoken.get_encoding.return_value = Mock()
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_invalid_encoding_name(self):
        """Test configuration validation with invalid encoding name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": "invalid_encoding"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            mock_tiktoken.get_encoding.side_effect = Exception("Invalid encoding")
            
            with pytest.raises(ValueError, match="Invalid encoding: invalid_encoding"):
                strategy.validate_config(config)
    
    def test_validate_config_none_encoding_name(self):
        """Test configuration validation with None encoding name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": None}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_no_encoding_name_param(self):
        """Test configuration validation without encoding_name parameter."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_valid_model_name(self):
        """Test configuration validation with valid model name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"model_name": "gpt-3.5-turbo"}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_invalid_model_name_type(self):
        """Test configuration validation with invalid model name type."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"model_name": 123}
        )
        
        with pytest.raises(ValueError, match="model_name must be a string"):
            strategy.validate_config(config)
    
    def test_validate_config_none_model_name(self):
        """Test configuration validation with None model name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"model_name": None}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_no_model_name_param(self):
        """Test configuration validation without model_name parameter."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestTokenSplitterStrategyEdgeCases:
    """Test edge cases for TokenSplitterStrategy."""
    
    def test_create_splitter_with_special_characters_in_encoding(self):
        """Test creating splitter with special characters in encoding name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": "cl100k_base-v2"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                encoding_name="cl100k_base-v2"
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_unicode_model_name(self):
        """Test creating splitter with unicode characters in model name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"model_name": "gpt-3.5-turbo-测试"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                model_name="gpt-3.5-turbo-测试"
            )
            assert result == mock_splitter
    
    def test_create_splitter_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = TokenSplitterStrategy()
        original_params = {
            "encoding_name": "cl100k_base",
            "model_name": "gpt-3.5-turbo",
            "allowed_special": ["<|endoftext|>"]
        }
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_splitter_with_very_large_chunk_size(self):
        """Test creating splitter with very large chunk size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000000,
                chunk_overlap=200
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_very_small_chunk_size(self):
        """Test creating splitter with very small chunk size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1,
            chunk_overlap=0,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1,
                chunk_overlap=0
            )
            assert result == mock_splitter
    
    def test_validate_config_with_very_large_chunk_size(self):
        """Test configuration validation with very large chunk size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000000,
            chunk_overlap=100000,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_small_chunk_size(self):
        """Test configuration validation with very small chunk size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1,
            chunk_overlap=0,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_chunk_overlap_almost_equal_to_chunk_size(self):
        """Test configuration validation with chunk_overlap almost equal to chunk_size."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=999,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_empty_encoding_name(self):
        """Test configuration validation with empty encoding name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": ""}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            mock_tiktoken.get_encoding.side_effect = Exception("Invalid encoding")
            
            with pytest.raises(ValueError, match="Invalid encoding: "):
                strategy.validate_config(config)
    
    def test_validate_config_with_empty_model_name(self):
        """Test configuration validation with empty model name."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"model_name": ""}
        )
        
        # Should not raise any exception (empty string is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_very_long_model_name(self):
        """Test configuration validation with very long model name."""
        strategy = TokenSplitterStrategy()
        long_model_name = "a" * 1000
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"model_name": long_model_name}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestTokenSplitterStrategyIntegration:
    """Integration tests for TokenSplitterStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = TokenSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": "cl100k_base", "model_name": "gpt-3.5-turbo"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            mock_tiktoken.get_encoding.return_value = Mock()
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once()
    
    def test_workflow_with_different_parameter_configurations(self):
        """Test workflow with different parameter configurations."""
        strategy = TokenSplitterStrategy()
        
        # Test with different parameter configurations
        configs = [
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"encoding_name": "cl100k_base"}
            ),
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=1500,
                chunk_overlap=300,
                params={"model_name": "gpt-4"}
            ),
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=500,
                chunk_overlap=100,
                params={
                    "encoding_name": "cl100k_base",
                    "model_name": "gpt-3.5-turbo",
                    "allowed_special": ["<|endoftext|>"]
                }
            ),
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=1000,
                chunk_overlap=200,
                params={}  # No additional params
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            mock_tiktoken.get_encoding.return_value = Mock()
            
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
        strategy = TokenSplitterStrategy()
        
        # Valid config
        valid_config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": "cl100k_base"}
        )
        
        # Invalid config (negative chunk size)
        invalid_config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=-100,
            chunk_overlap=200,
            params={"encoding_name": "cl100k_base"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            mock_tiktoken.get_encoding.return_value = Mock()
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_splitter(valid_config)
            assert result == mock_splitter
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_encoding_validation_errors(self):
        """Test workflow with encoding validation errors."""
        strategy = TokenSplitterStrategy()
        
        # Test with various invalid encoding configurations
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"encoding_name": "invalid_encoding"}
            ),
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"encoding_name": ""}
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            mock_tiktoken.get_encoding.side_effect = Exception("Invalid encoding")
            
            for config in invalid_configs:
                with pytest.raises(ValueError, match="Invalid encoding"):
                    strategy.validate_config(config)
    
    def test_workflow_with_model_name_validation_errors(self):
        """Test workflow with model name validation errors."""
        strategy = TokenSplitterStrategy()
        
        # Test with various invalid model name configurations
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"model_name": 123}
            ),
            SplitterConfig(
                type=SplitterType.TOKEN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"model_name": []}
            )
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError, match="model_name must be a string"):
                strategy.validate_config(config)
    
    def test_workflow_with_complex_parameter_combinations(self):
        """Test workflow with complex parameter combinations."""
        strategy = TokenSplitterStrategy()
        
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=2000,
            chunk_overlap=400,
            params={
                "encoding_name": "cl100k_base",
                "model_name": "gpt-4",
                "allowed_special": ["<|endoftext|>", "<|startoftext|>"],
                "disallowed_special": ["<|fim|>"],
                "custom_param": "custom_value"
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            mock_tiktoken.get_encoding.return_value = Mock()
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once_with(
                chunk_size=2000,
                chunk_overlap=400,
                encoding_name="cl100k_base",
                model_name="gpt-4",
                allowed_special=["<|endoftext|>", "<|startoftext|>"],
                disallowed_special=["<|fim|>"],
                custom_param="custom_value"
            )

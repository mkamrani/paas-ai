"""
Unit tests for code splitter strategy.

Tests all components of the CodeSplitterStrategy including:
- CodeSplitterStrategy class
- Configuration validation
- Splitter creation
- Error handling and edge cases
- Language parameter handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_text_splitters import Language

from src.paas_ai.core.rag.splitters.code import CodeSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestCodeSplitterStrategy:
    """Test the CodeSplitterStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = CodeSplitterStrategy()
        assert strategy is not None
        assert isinstance(strategy, CodeSplitterStrategy)
    
    def test_create_splitter_basic(self):
        """Test creating splitter with basic configuration."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.from_language.assert_called_once_with(
                language=Language.PYTHON,  # Default language
                chunk_size=1000,
                chunk_overlap=200
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_custom_language(self):
        """Test creating splitter with custom language."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "javascript"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            mock_language.return_value = Mock()
            
            result = strategy.create_splitter(config)
            
            mock_language.assert_called_once_with("javascript")
            mock_splitter_class.from_language.assert_called_once_with(
                language=mock_language.return_value,
                chunk_size=1000,
                chunk_overlap=200
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_language_enum(self):
        """Test creating splitter with Language enum."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": Language.JS}  # Use actual Language enum
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.from_language.assert_called_once_with(
                language=Language.JS,
                chunk_size=1000,
                chunk_overlap=200
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_additional_params(self):
        """Test creating splitter with additional parameters."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1500,
            chunk_overlap=300,
            params={
                "language": "python",
                "separators": ["\n\n", "\n", " "],
                "length_function": len
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            mock_language.return_value = Mock()
            
            result = strategy.create_splitter(config)
            
            mock_language.assert_called_once_with("python")
            mock_splitter_class.from_language.assert_called_once_with(
                language=mock_language.return_value,
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", " "],
                length_function=len
            )
            assert result == mock_splitter
    
    def test_create_splitter_empty_params(self):
        """Test creating splitter with empty params."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.from_language.assert_called_once_with(
                language=Language.PYTHON,  # Default language
                chunk_size=1000,
                chunk_overlap=200
            )
            assert result == mock_splitter
    
    def test_validate_config_valid_chunk_size(self):
        """Test configuration validation with valid chunk size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_zero_chunk_size(self):
        """Test configuration validation with zero chunk size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=0,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_size(self):
        """Test configuration validation with negative chunk size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=-100,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_overlap(self):
        """Test configuration validation with negative chunk overlap."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=-50,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_equals_chunk_size(self):
        """Test configuration validation when chunk_overlap equals chunk_size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=1000,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_greater_than_chunk_size(self):
        """Test configuration validation when chunk_overlap is greater than chunk_size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=1500,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_valid_language_string(self):
        """Test configuration validation with valid language string."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "python"}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_valid_language_enum(self):
        """Test configuration validation with valid Language enum."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": Language.JS}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_invalid_language_string(self):
        """Test configuration validation with invalid language string."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "invalid_language"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            mock_language.side_effect = ValueError("Invalid language")
            
            with pytest.raises(ValueError, match="Unsupported language: invalid_language"):
                strategy.validate_config(config)
    
    def test_validate_config_none_language(self):
        """Test configuration validation with None language."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": None}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_no_language_param(self):
        """Test configuration validation without language parameter."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestCodeSplitterStrategyEdgeCases:
    """Test edge cases for CodeSplitterStrategy."""
    
    def test_create_splitter_with_case_insensitive_language(self):
        """Test creating splitter with case insensitive language."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "PYTHON"}  # Uppercase
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            mock_language.return_value = Mock()
            
            result = strategy.create_splitter(config)
            
            mock_language.assert_called_once_with("python")  # Should be lowercase
            assert result == mock_splitter
    
    def test_create_splitter_with_whitespace_in_language(self):
        """Test creating splitter with whitespace in language."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": " python "}  # With whitespace
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            mock_language.return_value = Mock()
            
            result = strategy.create_splitter(config)
            
            mock_language.assert_called_once_with(" python ")  # Should preserve whitespace
            assert result == mock_splitter
    
    def test_create_splitter_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = CodeSplitterStrategy()
        original_params = {
            "language": "python",
            "separators": ["\n\n", "\n"],
            "length_function": len
        }
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            mock_language.return_value = Mock()
            
            result = strategy.create_splitter(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_splitter_with_unsupported_language(self):
        """Test creating splitter with unsupported language."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "unsupported_lang"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            mock_language.side_effect = ValueError("Invalid language")
            
            with pytest.raises(ValueError, match="Unsupported language: unsupported_lang"):
                strategy.create_splitter(config)
    
    def test_create_splitter_with_non_string_non_enum_language(self):
        """Test creating splitter with non-string, non-enum language."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": 123}  # Not a string or enum
        )
        
        with pytest.raises(ValueError, match="Language 123 is not supported"):
            strategy.create_splitter(config)
    
    def test_validate_config_with_very_large_chunk_size(self):
        """Test configuration validation with very large chunk size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000000,
            chunk_overlap=100000,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_small_chunk_size(self):
        """Test configuration validation with very small chunk size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1,
            chunk_overlap=0,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_chunk_overlap_almost_equal_to_chunk_size(self):
        """Test configuration validation with chunk_overlap almost equal to chunk_size."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=999,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_unicode_language(self):
        """Test configuration validation with unicode language name."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "python测试"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.Language') as mock_language:
            mock_language.side_effect = ValueError("Invalid language")
            
            with pytest.raises(ValueError, match="Unsupported language: python测试"):
                strategy.validate_config(config)


class TestCodeSplitterStrategyIntegration:
    """Integration tests for CodeSplitterStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = CodeSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "python"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.from_language.assert_called_once()
    
    def test_workflow_with_different_language_configurations(self):
        """Test workflow with different language configurations."""
        strategy = CodeSplitterStrategy()
        
        # Test with different language configurations
        configs = [
            SplitterConfig(
                type=SplitterType.CODE,
                chunk_size=1000,
                chunk_overlap=200,
                params={"language": "python"}
            ),
            SplitterConfig(
                type=SplitterType.CODE,
                chunk_size=1000,
                chunk_overlap=200,
                params={"language": "js"}
            ),
            SplitterConfig(
                type=SplitterType.CODE,
                chunk_size=1000,
                chunk_overlap=200,
                params={"language": "java"}
            ),
            SplitterConfig(
                type=SplitterType.CODE,
                chunk_size=1000,
                chunk_overlap=200,
                params={}  # Default language
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            for config in configs:
                # Validate config
                strategy.validate_config(config)
                
                # Create splitter
                result = strategy.create_splitter(config)
                
                assert result == mock_splitter
            
            # Should have been called 4 times
            assert mock_splitter_class.from_language.call_count == 4
    
    def test_workflow_with_mixed_valid_and_invalid_configs(self):
        """Test workflow with mix of valid and invalid configurations."""
        strategy = CodeSplitterStrategy()
        
        # Valid config
        valid_config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "python"}
        )
        
        # Invalid config (negative chunk size)
        invalid_config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=-100,
            chunk_overlap=200,
            params={"language": "python"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_splitter(valid_config)
            assert result == mock_splitter
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_language_validation_errors(self):
        """Test workflow with language validation errors."""
        strategy = CodeSplitterStrategy()
        
        # Test with various invalid language configurations
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.CODE,
                chunk_size=1000,
                chunk_overlap=200,
                params={"language": "unsupported_lang"}
            ),
            SplitterConfig(
                type=SplitterType.CODE,
                chunk_size=1000,
                chunk_overlap=200,
                params={"language": 123}
            )
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                strategy.validate_config(config)
    
    def test_workflow_with_complex_parameter_combinations(self):
        """Test workflow with complex parameter combinations."""
        strategy = CodeSplitterStrategy()
        
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1500,
            chunk_overlap=300,
            params={
                "language": "python",
                "separators": ["\n\n", "\n", " "],
                "length_function": len,
                "custom_param": "custom_value"
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.from_language.assert_called_once_with(
                language=Language.PYTHON,
                chunk_size=1500,
                chunk_overlap=300,
                separators=["\n\n", "\n", " "],
                length_function=len,
                custom_param="custom_value"
            )

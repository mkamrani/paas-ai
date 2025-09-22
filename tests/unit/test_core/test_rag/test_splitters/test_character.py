"""
Unit tests for character splitter strategy.

Tests all components of the CharacterSplitterStrategy including:
- CharacterSplitterStrategy class
- Configuration validation
- Splitter creation
- Error handling and edge cases
- Parameter handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.splitters.character import CharacterSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestCharacterSplitterStrategy:
    """Test the CharacterSplitterStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = CharacterSplitterStrategy()
        assert strategy is not None
        assert isinstance(strategy, CharacterSplitterStrategy)
    
    def test_create_splitter_basic(self):
        """Test creating splitter with basic configuration."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separator='\n\n'
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_custom_separator(self):
        """Test creating splitter with custom separator."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separator='\n'
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_additional_params(self):
        """Test creating splitter with additional parameters."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1500,
            chunk_overlap=300,
            params={
                "separator": " ",
                "length_function": len,
                "is_separator_regex": False
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1500,
                chunk_overlap=300,
                separator=' ',
                length_function=len,
                is_separator_regex=False
            )
            assert result == mock_splitter
    
    def test_create_splitter_empty_params(self):
        """Test creating splitter with empty params."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separator='\n\n'
            )
            assert result == mock_splitter
    
    def test_validate_config_valid_chunk_size(self):
        """Test configuration validation with valid chunk size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_zero_chunk_size(self):
        """Test configuration validation with zero chunk size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=0,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_size(self):
        """Test configuration validation with negative chunk size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=-100,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_overlap(self):
        """Test configuration validation with negative chunk overlap."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=-50,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_equals_chunk_size(self):
        """Test configuration validation when chunk_overlap equals chunk_size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=1000,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_chunk_overlap_greater_than_chunk_size(self):
        """Test configuration validation when chunk_overlap is greater than chunk_size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=1500,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            strategy.validate_config(config)
    
    def test_validate_config_zero_chunk_overlap(self):
        """Test configuration validation with zero chunk overlap."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=0,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_large_chunk_size(self):
        """Test configuration validation with large chunk size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=100000,
            chunk_overlap=10000,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestCharacterSplitterStrategyEdgeCases:
    """Test edge cases for CharacterSplitterStrategy."""
    
    def test_create_splitter_with_special_characters_in_separator(self):
        """Test creating splitter with special characters in separator."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\t\n\r"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separator='\t\n\r'
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_regex_separator(self):
        """Test creating splitter with regex separator."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "separator": r"\s+",
                "is_separator_regex": True
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separator=r'\s+',
                is_separator_regex=True
            )
            assert result == mock_splitter
    
    def test_create_splitter_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = CharacterSplitterStrategy()
        original_params = {"separator": "\n", "length_function": len}
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_splitter_with_none_separator(self):
        """Test creating splitter with None separator."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": None}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separator=None
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_empty_string_separator(self):
        """Test creating splitter with empty string separator."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": ""}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separator=""
            )
            assert result == mock_splitter
    
    def test_validate_config_with_very_large_chunk_size(self):
        """Test configuration validation with very large chunk size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000000,
            chunk_overlap=100000,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_small_chunk_size(self):
        """Test configuration validation with very small chunk size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1,
            chunk_overlap=0,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_chunk_overlap_almost_equal_to_chunk_size(self):
        """Test configuration validation with chunk_overlap almost equal to chunk_size."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=999,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestCharacterSplitterStrategyIntegration:
    """Integration tests for CharacterSplitterStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = CharacterSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once()
    
    def test_workflow_with_different_separators(self):
        """Test workflow with different separator configurations."""
        strategy = CharacterSplitterStrategy()
        
        # Test with different separator configurations
        configs = [
            SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separator": "\n\n"}
            ),
            SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separator": "\n"}
            ),
            SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separator": " "}
            ),
            SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={}  # Default separator
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
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
        strategy = CharacterSplitterStrategy()
        
        # Valid config
        valid_config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        # Invalid config (negative chunk size)
        invalid_config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=-100,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
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
        strategy = CharacterSplitterStrategy()
        
        def custom_length_function(text):
            return len(text.split())
        
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=100,
            chunk_overlap=20,
            params={
                "separator": " ",
                "length_function": custom_length_function
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
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
                separator=' ',
                length_function=custom_length_function
            )

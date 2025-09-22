"""
Unit tests for JSON splitter strategy.

Tests all components of the JSONSplitterStrategy including:
- JSONSplitterStrategy class
- Configuration validation
- Splitter creation
- Error handling and edge cases
- Parameter handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.splitters.json import JSONSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestJSONSplitterStrategy:
    """Test the JSONSplitterStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = JSONSplitterStrategy()
        assert strategy is not None
        assert isinstance(strategy, JSONSplitterStrategy)
    
    def test_create_splitter_basic(self):
        """Test creating splitter with basic configuration."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=1000
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_additional_params(self):
        """Test creating splitter with additional parameters."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1500,
            chunk_overlap=300,
            params={
                "max_depth": 3,
                "min_chunk_size": 100
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=1500,
                max_depth=3,
                min_chunk_size=100
            )
            assert result == mock_splitter
    
    def test_create_splitter_empty_params(self):
        """Test creating splitter with empty params."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=1000
            )
            assert result == mock_splitter
    
    def test_validate_config_valid_chunk_size(self):
        """Test configuration validation with valid chunk size."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_zero_chunk_size(self):
        """Test configuration validation with zero chunk size."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=0,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_negative_chunk_size(self):
        """Test configuration validation with negative chunk size."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=-100,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            strategy.validate_config(config)
    
    def test_validate_config_valid_max_depth(self):
        """Test configuration validation with valid max_depth."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": 3}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_zero_max_depth(self):
        """Test configuration validation with zero max_depth."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": 0}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_negative_max_depth(self):
        """Test configuration validation with negative max_depth."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": -1}
        )
        
        with pytest.raises(ValueError, match="max_depth must be a non-negative integer"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_max_depth_type(self):
        """Test configuration validation with invalid max_depth type."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": "not-an-integer"}
        )
        
        with pytest.raises(ValueError, match="max_depth must be a non-negative integer"):
            strategy.validate_config(config)
    
    def test_validate_config_float_max_depth(self):
        """Test configuration validation with float max_depth."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": 3.5}
        )
        
        with pytest.raises(ValueError, match="max_depth must be a non-negative integer"):
            strategy.validate_config(config)
    
    def test_validate_config_none_max_depth(self):
        """Test configuration validation with None max_depth."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": None}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_no_max_depth_param(self):
        """Test configuration validation without max_depth parameter."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestJSONSplitterStrategyEdgeCases:
    """Test edge cases for JSONSplitterStrategy."""
    
    def test_create_splitter_with_complex_params(self):
        """Test creating splitter with complex parameters."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=2000,
            chunk_overlap=400,
            params={
                "max_depth": 5,
                "min_chunk_size": 50,
                "separator": ".",
                "keep_separator": True
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=2000,
                max_depth=5,
                min_chunk_size=50,
                separator=".",
                keep_separator=True
            )
            assert result == mock_splitter
    
    def test_create_splitter_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = JSONSplitterStrategy()
        original_params = {
            "max_depth": 3,
            "min_chunk_size": 100,
            "separator": "."
        }
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_splitter_with_very_large_chunk_size(self):
        """Test creating splitter with very large chunk size."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=1000000
            )
            assert result == mock_splitter
    
    def test_create_splitter_with_very_small_chunk_size(self):
        """Test creating splitter with very small chunk size."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1,
            chunk_overlap=0,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = strategy.create_splitter(config)
            
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=1
            )
            assert result == mock_splitter
    
    def test_validate_config_with_very_large_max_depth(self):
        """Test configuration validation with very large max_depth."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": 1000}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_small_chunk_size(self):
        """Test configuration validation with very small chunk size."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1,
            chunk_overlap=0,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_mixed_valid_and_invalid_params(self):
        """Test configuration validation with mix of valid and invalid parameters."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "max_depth": 3,  # Valid
                "invalid_param": "should be ignored"  # Should be ignored
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestJSONSplitterStrategyIntegration:
    """Integration tests for JSONSplitterStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = JSONSplitterStrategy()
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": 3}
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once()
    
    def test_workflow_with_different_parameter_configurations(self):
        """Test workflow with different parameter configurations."""
        strategy = JSONSplitterStrategy()
        
        # Test with different parameter configurations
        configs = [
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=1000,
                chunk_overlap=200,
                params={"max_depth": 3}
            ),
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=1500,
                chunk_overlap=300,
                params={
                    "max_depth": 5,
                    "min_chunk_size": 100
                }
            ),
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=500,
                chunk_overlap=100,
                params={
                    "max_depth": 2,
                    "separator": ".",
                    "keep_separator": True
                }
            ),
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=1000,
                chunk_overlap=200,
                params={}  # No additional params
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
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
        strategy = JSONSplitterStrategy()
        
        # Valid config
        valid_config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": 3}
        )
        
        # Invalid config (negative chunk size)
        invalid_config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=-100,
            chunk_overlap=200,
            params={"max_depth": 3}
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_splitter(valid_config)
            assert result == mock_splitter
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_max_depth_validation_errors(self):
        """Test workflow with max_depth validation errors."""
        strategy = JSONSplitterStrategy()
        
        # Test with various invalid max_depth configurations
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=1000,
                chunk_overlap=200,
                params={"max_depth": -1}
            ),
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=1000,
                chunk_overlap=200,
                params={"max_depth": "not-an-integer"}
            ),
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=1000,
                chunk_overlap=200,
                params={"max_depth": 3.5}
            )
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError, match="max_depth must be a non-negative integer"):
                strategy.validate_config(config)
    
    def test_workflow_with_chunk_size_validation_errors(self):
        """Test workflow with chunk_size validation errors."""
        strategy = JSONSplitterStrategy()
        
        # Test with various invalid chunk_size configurations
        invalid_configs = [
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=0,
                chunk_overlap=200,
                params={}
            ),
            SplitterConfig(
                type=SplitterType.JSON,
                chunk_size=-100,
                chunk_overlap=200,
                params={}
            )
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                strategy.validate_config(config)
    
    def test_workflow_with_complex_parameter_combinations(self):
        """Test workflow with complex parameter combinations."""
        strategy = JSONSplitterStrategy()
        
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=2000,
            chunk_overlap=400,
            params={
                "max_depth": 4,
                "min_chunk_size": 200,
                "separator": ".",
                "keep_separator": True,
                "custom_param": "custom_value"
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Validate config
            strategy.validate_config(config)
            
            # Create splitter
            result = strategy.create_splitter(config)
            
            assert result == mock_splitter
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=2000,
                max_depth=4,
                min_chunk_size=200,
                separator=".",
                keep_separator=True,
                custom_param="custom_value"
            )

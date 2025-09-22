"""
Unit tests for ensemble retriever strategy.

Tests all components of the EnsembleRetrieverStrategy including:
- EnsembleRetrieverStrategy class
- Configuration validation
- Retriever creation
- Error handling and edge cases
- Ensemble-specific parameter validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.ensemble import EnsembleRetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


class TestEnsembleRetrieverStrategy:
    """Test the EnsembleRetrieverStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = EnsembleRetrieverStrategy()
        assert strategy is not None
        assert isinstance(strategy, EnsembleRetrieverStrategy)
    
    def test_create_retriever_basic(self):
        """Test creating retriever with basic configuration."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore)
                
                # Should create two retrievers (similarity and mmr)
                assert mock_as_retriever.call_count == 2
                
                # Check similarity retriever call
                similarity_call = mock_as_retriever.call_args_list[0]
                assert similarity_call[1]['search_type'] == "similarity"
                assert similarity_call[1]['search_kwargs'] == {"k": 5}
                
                # Check mmr retriever call
                mmr_call = mock_as_retriever.call_args_list[1]
                assert mmr_call[1]['search_type'] == "mmr"
                assert mmr_call[1]['search_kwargs'] == {"k": 5}
                
                # Check ensemble creation
                mock_ensemble_class.assert_called_once()
                ensemble_call = mock_ensemble_class.call_args
                assert ensemble_call[1]['retrievers'] == [mock_retriever, mock_retriever]
                assert ensemble_call[1]['weights'] == [0.5, 0.5]
                
                assert result == mock_ensemble
    
    def test_create_retriever_with_default_weights(self):
        """Test creating retriever with default weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={}  # No weights specified
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore)
                
                # Check ensemble creation with default weights
                mock_ensemble_class.assert_called_once()
                ensemble_call = mock_ensemble_class.call_args
                assert ensemble_call[1]['weights'] == [0.5, 0.5]
                
                assert result == mock_ensemble
    
    def test_create_retriever_with_custom_weights(self):
        """Test creating retriever with custom weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 10},
            params={"weights": [0.7, 0.3]}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore)
                
                # Check ensemble creation with custom weights
                mock_ensemble_class.assert_called_once()
                ensemble_call = mock_ensemble_class.call_args
                assert ensemble_call[1]['weights'] == [0.7, 0.3]
                
                assert result == mock_ensemble
    
    def test_create_retriever_with_additional_params(self):
        """Test creating retriever with additional parameters."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={
                "weights": [0.6, 0.4],
                "custom_param": "value",
                "another_param": 42
            }
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore)
                
                # Check ensemble creation with additional params (excluding weights)
                mock_ensemble_class.assert_called_once()
                ensemble_call = mock_ensemble_class.call_args
                assert ensemble_call[1]['weights'] == [0.6, 0.4]
                assert ensemble_call[1]['custom_param'] == "value"
                assert ensemble_call[1]['another_param'] == 42
                
                assert result == mock_ensemble
    
    def test_create_retriever_with_llm(self):
        """Test creating retriever with LLM parameter (should be ignored)."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore, llm)
                
                # LLM should be ignored for ensemble retriever
                assert result == mock_ensemble
    
    def test_validate_config_with_valid_parameters(self):
        """Test configuration validation with valid parameters."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_valid_k_values(self):
        """Test configuration validation with valid k values."""
        strategy = EnsembleRetrieverStrategy()
        valid_k_values = [1, 5, 10, 100, 1000]
        
        for k in valid_k_values:
            config = RetrieverConfig(
                type=RetrieverType.ENSEMBLE,
                search_kwargs={"k": k},
                params={"weights": [0.5, 0.5]}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_valid_weights(self):
        """Test configuration validation with valid weights."""
        strategy = EnsembleRetrieverStrategy()
        valid_weights = [
            [0.5, 0.5],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.0, 1.0],
            [1.0, 0.0]
        ]
        
        for weights in valid_weights:
            config = RetrieverConfig(
                type=RetrieverType.ENSEMBLE,
                search_kwargs={"k": 5},
                params={"weights": weights}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_default_weights(self):
        """Test configuration validation with default weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={}  # No weights specified
        )
        
        # Should not raise any exception (defaults are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_zero(self):
        """Test configuration validation with k=0."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 0},
            params={"weights": [0.5, 0.5]}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_negative(self):
        """Test configuration validation with negative k."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": -5},
            params={"weights": [0.5, 0.5]}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_weights_wrong_length(self):
        """Test configuration validation with wrong number of weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5]}  # Only one weight, should be two
        )
        
        with pytest.raises(ValueError, match="params\\['weights'\\] must be a list of 2 numbers"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_weights_three_weights(self):
        """Test configuration validation with three weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.3, 0.3, 0.4]}  # Three weights, should be two
        )
        
        with pytest.raises(ValueError, match="params\\['weights'\\] must be a list of 2 numbers"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_weights_not_list(self):
        """Test configuration validation with weights not being a list."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": "not-a-list"}
        )
        
        with pytest.raises(ValueError, match="params\\['weights'\\] must be a list of 2 numbers"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_weights_negative(self):
        """Test configuration validation with negative weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [-0.1, 1.1]}
        )
        
        with pytest.raises(ValueError, match="All weights must be numbers between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_weights_greater_than_one(self):
        """Test configuration validation with weights > 1."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 1.5]}
        )
        
        with pytest.raises(ValueError, match="All weights must be numbers between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_weights_not_numbers(self):
        """Test configuration validation with non-numeric weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": ["0.5", "0.5"]}
        )
        
        with pytest.raises(ValueError, match="All weights must be numbers between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_weights_not_summing_to_one(self):
        """Test configuration validation with weights not summing to 1."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.3, 0.3]}  # Sums to 0.6, not 1.0
        )
        
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            strategy.validate_config(config)
    
    def test_validate_config_with_weights_summing_to_one_with_floating_point_error(self):
        """Test configuration validation with weights summing to 1 with floating point error."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.1, 0.9]}  # Should be close enough to 1.0
        )
        
        # Should not raise any exception (close enough to 1.0)
        strategy.validate_config(config)


class TestEnsembleRetrieverStrategyEdgeCases:
    """Test edge cases for EnsembleRetrieverStrategy."""
    
    def test_create_retriever_with_special_characters_in_params(self):
        """Test creating retriever with special characters in parameters."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={
                "weights": [0.5, 0.5],
                "custom_param": "value with spaces",
                "unicode_param": "测试参数",
                "special_chars": "!@#$%^&*()",
                "nested_dict": {"key": "value"},
                "list_param": [1, 2, 3]
            }
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore)
                
                # Check ensemble creation with special characters
                mock_ensemble_class.assert_called_once()
                ensemble_call = mock_ensemble_class.call_args
                assert ensemble_call[1]['weights'] == [0.5, 0.5]
                assert ensemble_call[1]['custom_param'] == "value with spaces"
                assert ensemble_call[1]['unicode_param'] == "测试参数"
                assert ensemble_call[1]['special_chars'] == "!@#$%^&*()"
                assert ensemble_call[1]['nested_dict'] == {"key": "value"}
                assert ensemble_call[1]['list_param'] == [1, 2, 3]
                
                assert result == mock_ensemble
    
    def test_create_retriever_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = EnsembleRetrieverStrategy()
        original_params = {"weights": [0.5, 0.5], "custom_param": "value"}
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params=original_params
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore)
                
                # Original params should not be modified
                assert config.params == original_params
                assert config.params is not original_params  # Should be a copy
    
    def test_create_retriever_search_kwargs_copy_behavior(self):
        """Test that search_kwargs are copied and not modified."""
        strategy = EnsembleRetrieverStrategy()
        original_search_kwargs = {"k": 5}
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs=original_search_kwargs,
            params={"weights": [0.5, 0.5]}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore)
                
                # Original search_kwargs should not be modified
                assert config.search_kwargs == original_search_kwargs
                assert config.search_kwargs is not original_search_kwargs  # Should be a copy
    
    def test_validate_config_with_very_large_k(self):
        """Test configuration validation with very large k value."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 1000000},
            params={"weights": [0.5, 0.5]}
        )
        
        # Should not raise any exception (large k is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_float_weights(self):
        """Test configuration validation with float weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        
        # Should not raise any exception (float weights are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_integer_weights(self):
        """Test configuration validation with integer weights."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [1, 0]}  # Integer weights
        )
        
        # Should not raise any exception (integer weights are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_missing_search_kwargs(self):
        """Test configuration validation with missing search_kwargs."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={},  # Empty search_kwargs
            params={"weights": [0.5, 0.5]}
        )
        
        # Should not raise any exception (default k=4 is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_none_params(self):
        """Test configuration validation with None params."""
        strategy = EnsembleRetrieverStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={}
        )
        
        # Should not raise error with empty params (default weights are used)
        strategy.validate_config(config)


class TestEnsembleRetrieverStrategyIntegration:
    """Integration tests for EnsembleRetrieverStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5], "custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_ensemble
                mock_ensemble_class.assert_called_once()
    
    def test_workflow_with_different_weight_combinations(self):
        """Test workflow with different weight combinations."""
        strategy = EnsembleRetrieverStrategy()
        vectorstore = Mock()
        
        # Test with different weight combinations
        weight_combinations = [
            [0.5, 0.5],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.0, 1.0],
            [1.0, 0.0]
        ]
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                for weights in weight_combinations:
                    config = RetrieverConfig(
                        type=RetrieverType.ENSEMBLE,
                        search_kwargs={"k": 5},
                        params={"weights": weights}
                    )
                    
                    # Validate config
                    strategy.validate_config(config)
                    
                    # Create retriever
                    result = strategy.create_retriever(config, vectorstore)
                    
                    assert result == mock_ensemble
                
                # Should have been called for each weight combination
                assert mock_ensemble_class.call_count == len(weight_combinations)
    
    def test_workflow_with_different_k_values(self):
        """Test workflow with different k values."""
        strategy = EnsembleRetrieverStrategy()
        vectorstore = Mock()
        
        # Test with different k values
        k_values = [1, 5, 10, 50, 100]
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                for k in k_values:
                    config = RetrieverConfig(
                        type=RetrieverType.ENSEMBLE,
                        search_kwargs={"k": k},
                        params={"weights": [0.5, 0.5]}
                    )
                    
                    # Validate config
                    strategy.validate_config(config)
                    
                    # Create retriever
                    result = strategy.create_retriever(config, vectorstore)
                    
                    assert result == mock_ensemble
                
                # Should have been called for each k value
                assert mock_ensemble_class.call_count == len(k_values)
    
    def test_workflow_with_mixed_valid_and_invalid_configs(self):
        """Test workflow with mix of valid and invalid configurations."""
        strategy = EnsembleRetrieverStrategy()
        vectorstore = Mock()
        
        # Valid config
        valid_config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        
        # Invalid config (weights don't sum to 1)
        invalid_config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.3, 0.3]}
        )
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                # Valid config should work
                strategy.validate_config(valid_config)
                result = strategy.create_retriever(valid_config, vectorstore)
                assert result == mock_ensemble
                
                # Invalid config should raise error
                with pytest.raises(ValueError, match="Weights must sum to 1.0"):
                    strategy.validate_config(invalid_config)
    
    def test_workflow_with_vectorstore_errors(self):
        """Test workflow with vectorstore errors."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        vectorstore = Mock()
        
        # Mock vectorstore to raise an error
        with patch.object(vectorstore, 'as_retriever', side_effect=Exception("VectorStore error")):
            with pytest.raises(Exception, match="VectorStore error"):
                strategy.create_retriever(config, vectorstore)
    
    def test_workflow_with_none_vectorstore(self):
        """Test workflow with None vectorstore."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        
        # Should handle None vectorstore gracefully
        with pytest.raises(AttributeError):
            strategy.create_retriever(config, None)
    
    def test_workflow_with_ensemble_retriever_errors(self):
        """Test workflow with EnsembleRetriever errors."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5]}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever', side_effect=Exception("EnsembleRetriever error")):
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                with pytest.raises(Exception, match="EnsembleRetriever error"):
                    strategy.create_retriever(config, vectorstore)
    
    def test_workflow_with_custom_vectorstore_implementation(self):
        """Test workflow with custom vectorstore implementation."""
        strategy = EnsembleRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.5, 0.5], "custom_param": "value"}
        )
        
        # Custom vectorstore with as_retriever method
        class CustomVectorStore:
            def __init__(self):
                self.calls = []
            
            def as_retriever(self, **kwargs):
                self.calls.append(kwargs)
                return Mock()
        
        vectorstore = CustomVectorStore()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            # Validate config
            strategy.validate_config(config)
            
            # Create retriever
            result = strategy.create_retriever(config, vectorstore)
            
            assert result == mock_ensemble
            assert len(vectorstore.calls) == 2  # Should create two retrievers
            
            # Check similarity retriever call
            similarity_call = vectorstore.calls[0]
            assert similarity_call["search_type"] == "similarity"
            assert similarity_call["search_kwargs"] == {"k": 5}
            
            # Check mmr retriever call
            mmr_call = vectorstore.calls[1]
            assert mmr_call["search_type"] == "mmr"
            assert mmr_call["search_kwargs"] == {"k": 5}

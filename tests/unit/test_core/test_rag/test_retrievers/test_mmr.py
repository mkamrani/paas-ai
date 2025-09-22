"""
Unit tests for MMR (Maximum Marginal Relevance) retriever strategy.

Tests all components of the MMRRetrieverStrategy including:
- MMRRetrieverStrategy class
- Configuration validation
- Retriever creation
- Error handling and edge cases
- MMR-specific parameter validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.mmr import MMRRetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


class TestMMRRetrieverStrategy:
    """Test the MMRRetrieverStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = MMRRetrieverStrategy()
        assert strategy is not None
        assert isinstance(strategy, MMRRetrieverStrategy)
    
    def test_create_retriever_basic(self):
        """Test creating retriever with basic configuration."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_additional_params(self):
        """Test creating retriever with additional parameters."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7},
            params={"custom_param": "value", "another_param": 42}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="mmr",
                search_kwargs={"k": 10, "fetch_k": 30, "lambda_mult": 0.7},
                custom_param="value",
                another_param=42
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_defaults(self):
        """Test creating retriever with default values."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5},  # Only k specified, others should use defaults
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            # Should use default values for fetch_k and lambda_mult
            mock_as_retriever.assert_called_once_with(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_llm(self):
        """Test creating retriever with LLM parameter (should be ignored)."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore, llm)
            
            # LLM should be ignored for MMR retriever
            mock_as_retriever.assert_called_once_with(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
            )
            assert result == mock_retriever
    
    def test_validate_config_with_valid_parameters(self):
        """Test configuration validation with valid parameters."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_valid_k_values(self):
        """Test configuration validation with valid k values."""
        strategy = MMRRetrieverStrategy()
        valid_k_values = [1, 5, 10, 100, 1000]
        
        for k in valid_k_values:
            config = RetrieverConfig(
                type=RetrieverType.MMR,
                search_kwargs={"k": k, "fetch_k": k + 10, "lambda_mult": 0.5},
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_valid_fetch_k_values(self):
        """Test configuration validation with valid fetch_k values."""
        strategy = MMRRetrieverStrategy()
        valid_fetch_k_values = [10, 20, 50, 100, 1000]
        
        for fetch_k in valid_fetch_k_values:
            config = RetrieverConfig(
                type=RetrieverType.MMR,
                search_kwargs={"k": 5, "fetch_k": fetch_k, "lambda_mult": 0.5},
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_valid_lambda_mult_values(self):
        """Test configuration validation with valid lambda_mult values."""
        strategy = MMRRetrieverStrategy()
        valid_lambda_mult_values = [0.0, 0.1, 0.5, 0.7, 1.0]
        
        for lambda_mult in valid_lambda_mult_values:
            config = RetrieverConfig(
                type=RetrieverType.MMR,
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": lambda_mult},
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_default_values(self):
        """Test configuration validation with default values."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5},  # Only k specified
            params={}
        )
        
        # Should not raise any exception (defaults are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_zero(self):
        """Test configuration validation with k=0."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 0, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_negative(self):
        """Test configuration validation with negative k."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": -5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_fetch_k_zero(self):
        """Test configuration validation with fetch_k=0."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 0, "lambda_mult": 0.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['fetch_k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_fetch_k_negative(self):
        """Test configuration validation with negative fetch_k."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": -10, "lambda_mult": 0.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['fetch_k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_fetch_k_less_than_k(self):
        """Test configuration validation with fetch_k < k."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 10, "fetch_k": 5, "lambda_mult": 0.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['fetch_k'\\] must be >= search_kwargs\\['k'\\]"):
            strategy.validate_config(config)
    
    def test_validate_config_with_fetch_k_equal_to_k(self):
        """Test configuration validation with fetch_k = k."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 5, "lambda_mult": 0.5},
            params={}
        )
        
        # Should not raise any exception (fetch_k = k is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_invalid_lambda_mult_negative(self):
        """Test configuration validation with negative lambda_mult."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": -0.1},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['lambda_mult'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_lambda_mult_greater_than_one(self):
        """Test configuration validation with lambda_mult > 1."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 1.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['lambda_mult'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_lambda_mult_string(self):
        """Test configuration validation with string lambda_mult."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": "0.5"},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['lambda_mult'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_lambda_mult_none(self):
        """Test configuration validation with None lambda_mult."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": None},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['lambda_mult'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)


class TestMMRRetrieverStrategyEdgeCases:
    """Test edge cases for MMRRetrieverStrategy."""
    
    def test_create_retriever_with_special_characters_in_params(self):
        """Test creating retriever with special characters in parameters."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={
                "custom_param": "value with spaces",
                "unicode_param": "测试参数",
                "special_chars": "!@#$%^&*()",
                "nested_dict": {"key": "value"},
                "list_param": [1, 2, 3]
            }
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
                custom_param="value with spaces",
                unicode_param="测试参数",
                special_chars="!@#$%^&*()",
                nested_dict={"key": "value"},
                list_param=[1, 2, 3]
            )
            assert result == mock_retriever
    
    def test_create_retriever_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = MMRRetrieverStrategy()
        original_params = {"custom_param": "value", "another_param": 42}
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params=original_params
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_retriever_search_kwargs_copy_behavior(self):
        """Test that search_kwargs are copied and not modified."""
        strategy = MMRRetrieverStrategy()
        original_search_kwargs = {"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs=original_search_kwargs,
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            # Original search_kwargs should not be modified
            assert config.search_kwargs == original_search_kwargs
            assert config.search_kwargs is not original_search_kwargs  # Should be a copy
    
    def test_validate_config_with_very_large_values(self):
        """Test configuration validation with very large values."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 1000000, "fetch_k": 2000000, "lambda_mult": 0.5},
            params={}
        )
        
        # Should not raise any exception (large values are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_float_k_and_fetch_k(self):
        """Test configuration validation with float k and fetch_k."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5.0, "fetch_k": 20.0, "lambda_mult": 0.5},
            params={}
        )
        
        # Should raise error (k and fetch_k must be integers)
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_integer_lambda_mult(self):
        """Test configuration validation with integer lambda_mult."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 1},
            params={}
        )
        
        # Should not raise any exception (integer 1 is valid for lambda_mult)
        strategy.validate_config(config)
    
    def test_validate_config_with_missing_search_kwargs(self):
        """Test configuration validation with missing search_kwargs."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={},  # Empty search_kwargs
            params={}
        )
        
        # Should not raise any exception (defaults are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_none_params(self):
        """Test configuration validation with None params."""
        strategy = MMRRetrieverStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        
        # Should not raise error with empty params
        strategy.validate_config(config)


class TestMMRRetrieverStrategyIntegration:
    """Integration tests for MMRRetrieverStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            # Validate config
            strategy.validate_config(config)
            
            # Create retriever
            result = strategy.create_retriever(config, vectorstore)
            
            assert result == mock_retriever
            mock_as_retriever.assert_called_once()
    
    def test_workflow_with_different_parameter_combinations(self):
        """Test workflow with different parameter combinations."""
        strategy = MMRRetrieverStrategy()
        vectorstore = Mock()
        
        # Test with different parameter combinations
        test_cases = [
            {"k": 1, "fetch_k": 10, "lambda_mult": 0.0},
            {"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            {"k": 10, "fetch_k": 50, "lambda_mult": 1.0},
            {"k": 3, "fetch_k": 15, "lambda_mult": 0.3},
        ]
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            for params in test_cases:
                config = RetrieverConfig(
                    type=RetrieverType.MMR,
                    search_kwargs=params,
                    params={}
                )
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_retriever
            
            # Should have been called for each test case
            assert mock_as_retriever.call_count == len(test_cases)
    
    def test_workflow_with_mixed_valid_and_invalid_configs(self):
        """Test workflow with mix of valid and invalid configurations."""
        strategy = MMRRetrieverStrategy()
        vectorstore = Mock()
        
        # Valid config
        valid_config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        
        # Invalid config (fetch_k < k)
        invalid_config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 10, "fetch_k": 5, "lambda_mult": 0.5},
            params={}
        )
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_retriever(valid_config, vectorstore)
            assert result == mock_retriever
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="search_kwargs\\['fetch_k'\\] must be >= search_kwargs\\['k'\\]"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_vectorstore_errors(self):
        """Test workflow with vectorstore errors."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        vectorstore = Mock()
        
        # Mock vectorstore to raise an error
        with patch.object(vectorstore, 'as_retriever', side_effect=Exception("VectorStore error")):
            with pytest.raises(Exception, match="VectorStore error"):
                strategy.create_retriever(config, vectorstore)
    
    def test_workflow_with_none_vectorstore(self):
        """Test workflow with None vectorstore."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={}
        )
        
        # Should handle None vectorstore gracefully
        with pytest.raises(AttributeError):
            strategy.create_retriever(config, None)
    
    def test_workflow_with_custom_vectorstore_implementation(self):
        """Test workflow with custom vectorstore implementation."""
        strategy = MMRRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={"custom_param": "value"}
        )
        
        # Custom vectorstore with as_retriever method
        class CustomVectorStore:
            def __init__(self):
                self.calls = []
            
            def as_retriever(self, **kwargs):
                self.calls.append(kwargs)
                return Mock()
        
        vectorstore = CustomVectorStore()
        
        # Validate config
        strategy.validate_config(config)
        
        # Create retriever
        result = strategy.create_retriever(config, vectorstore)
        
        assert result is not None
        assert len(vectorstore.calls) == 1
        call_kwargs = vectorstore.calls[0]
        assert call_kwargs["search_type"] == "mmr"
        assert call_kwargs["search_kwargs"] == {"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
        assert call_kwargs["custom_param"] == "value"
    
    def test_workflow_with_edge_case_lambda_mult_values(self):
        """Test workflow with edge case lambda_mult values."""
        strategy = MMRRetrieverStrategy()
        vectorstore = Mock()
        
        # Test edge case lambda_mult values
        edge_cases = [0.0, 1.0, 0.001, 0.999]
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            for lambda_mult in edge_cases:
                config = RetrieverConfig(
                    type=RetrieverType.MMR,
                    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": lambda_mult},
                    params={}
                )
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_retriever
            
            # Should have been called for each edge case
            assert mock_as_retriever.call_count == len(edge_cases)

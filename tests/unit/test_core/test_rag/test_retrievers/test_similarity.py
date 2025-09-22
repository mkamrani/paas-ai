"""
Unit tests for similarity retriever strategy.

Tests all components of the SimilarityRetrieverStrategy including:
- SimilarityRetrieverStrategy class
- Configuration validation
- Retriever creation
- Error handling and edge cases
- Search kwargs validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.similarity import SimilarityRetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


class TestSimilarityRetrieverStrategy:
    """Test the SimilarityRetrieverStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = SimilarityRetrieverStrategy()
        assert strategy is not None
        assert isinstance(strategy, SimilarityRetrieverStrategy)
    
    def test_create_retriever_basic(self):
        """Test creating retriever with basic configuration."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_additional_params(self):
        """Test creating retriever with additional parameters."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 10, "score_threshold": 0.8},
            params={"custom_param": "value", "another_param": 42}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 10, "score_threshold": 0.8},
                custom_param="value",
                another_param=42
            )
            assert result == mock_retriever
    
    def test_create_retriever_empty_params(self):
        """Test creating retriever with empty params."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_llm(self):
        """Test creating retriever with LLM parameter (should be ignored)."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore, llm)
            
            # LLM should be ignored for similarity retriever
            mock_as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            assert result == mock_retriever
    
    def test_validate_config_with_valid_k(self):
        """Test configuration validation with valid k value."""
        strategy = SimilarityRetrieverStrategy()
        valid_k_values = [1, 5, 10, 100, 1000]
        
        for k in valid_k_values:
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": k},
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_default_k(self):
        """Test configuration validation with default k value."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={},  # No k specified, should use default
            params={}
        )
        
        # Should not raise any exception (default k=4 is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_zero(self):
        """Test configuration validation with k=0."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 0},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_negative(self):
        """Test configuration validation with negative k."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": -5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_float(self):
        """Test configuration validation with float k."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_string(self):
        """Test configuration validation with string k."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": "5"},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_none(self):
        """Test configuration validation with None k."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": None},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_additional_search_kwargs(self):
        """Test configuration validation with additional search kwargs."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={
                "k": 5,
                "score_threshold": 0.8,
                "filter": {"category": "test"},
                "include_metadata": True
            },
            params={}
        )
        
        # Should not raise any exception (additional kwargs are allowed)
        strategy.validate_config(config)


class TestSimilarityRetrieverStrategyEdgeCases:
    """Test edge cases for SimilarityRetrieverStrategy."""
    
    def test_create_retriever_with_special_characters_in_params(self):
        """Test creating retriever with special characters in parameters."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
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
                search_type="similarity",
                search_kwargs={"k": 5},
                custom_param="value with spaces",
                unicode_param="测试参数",
                special_chars="!@#$%^&*()",
                nested_dict={"key": "value"},
                list_param=[1, 2, 3]
            )
            assert result == mock_retriever
    
    def test_create_retriever_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = SimilarityRetrieverStrategy()
        original_params = {"custom_param": "value", "another_param": 42}
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
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
        strategy = SimilarityRetrieverStrategy()
        original_search_kwargs = {"k": 5, "score_threshold": 0.8}
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
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
    
    def test_validate_config_with_very_large_k(self):
        """Test configuration validation with very large k value."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 1000000},
            params={}
        )
        
        # Should not raise any exception (large k is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_missing_search_kwargs(self):
        """Test configuration validation with missing search_kwargs."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={},  # Empty search_kwargs
            params={}
        )
        
        # Should not raise any exception (default k=4 is used)
        strategy.validate_config(config)
    
    def test_validate_config_with_none_params(self):
        """Test configuration validation with None params."""
        strategy = SimilarityRetrieverStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        
        # Should not raise error with empty params
        strategy.validate_config(config)
    
    def test_validate_config_with_missing_params_attribute(self):
        """Test configuration validation with config that has no params attribute."""
        strategy = SimilarityRetrieverStrategy()
        
        # Create a config object without params attribute
        class ConfigWithoutParams:
            def __init__(self):
                self.type = RetrieverType.SIMILARITY
                self.search_kwargs = {"k": 5}
                # No params attribute
        
        config = ConfigWithoutParams()
        
        # Should not raise error - the strategy doesn't access params for validation
        strategy.validate_config(config)
    
    def test_validate_config_with_non_dict_params(self):
        """Test configuration validation with non-dict params."""
        strategy = SimilarityRetrieverStrategy()
        # Pydantic doesn't allow non-dict params, so we test with a custom config object
        class ConfigWithNonDictParams:
            def __init__(self):
                self.type = RetrieverType.SIMILARITY
                self.search_kwargs = {"k": 5}
                self.params = "not-a-dict"
        
        config = ConfigWithNonDictParams()
        
        # Should not raise error - the strategy doesn't access params for validation
        strategy.validate_config(config)


class TestSimilarityRetrieverStrategyIntegration:
    """Integration tests for SimilarityRetrieverStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
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
    
    def test_workflow_with_different_k_values(self):
        """Test workflow with different k values."""
        strategy = SimilarityRetrieverStrategy()
        vectorstore = Mock()
        
        # Test with different k values
        k_values = [1, 5, 10, 50, 100]
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            for k in k_values:
                config = RetrieverConfig(
                    type=RetrieverType.SIMILARITY,
                    search_kwargs={"k": k},
                    params={}
                )
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_retriever
            
            # Should have been called for each k value
            assert mock_as_retriever.call_count == len(k_values)
    
    def test_workflow_with_mixed_valid_and_invalid_configs(self):
        """Test workflow with mix of valid and invalid configurations."""
        strategy = SimilarityRetrieverStrategy()
        vectorstore = Mock()
        
        # Valid config
        valid_config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        
        # Invalid config (k=0)
        invalid_config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 0},
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
            with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_vectorstore_errors(self):
        """Test workflow with vectorstore errors."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        # Mock vectorstore to raise an error
        with patch.object(vectorstore, 'as_retriever', side_effect=Exception("VectorStore error")):
            with pytest.raises(Exception, match="VectorStore error"):
                strategy.create_retriever(config, vectorstore)
    
    def test_workflow_with_none_vectorstore(self):
        """Test workflow with None vectorstore."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        
        # Should handle None vectorstore gracefully
        with pytest.raises(AttributeError):
            strategy.create_retriever(config, None)
    
    def test_workflow_with_custom_vectorstore_implementation(self):
        """Test workflow with custom vectorstore implementation."""
        strategy = SimilarityRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
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
        assert call_kwargs["search_type"] == "similarity"
        assert call_kwargs["search_kwargs"] == {"k": 5}
        assert call_kwargs["custom_param"] == "value"

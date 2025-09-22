"""
Unit tests for similarity score threshold retriever strategy.

Tests all components of the SimilarityScoreThresholdRetrieverStrategy including:
- SimilarityScoreThresholdRetrieverStrategy class
- Configuration validation
- Retriever creation
- Error handling and edge cases
- Score threshold validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.similarity_score_threshold import SimilarityScoreThresholdRetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


class TestSimilarityScoreThresholdRetrieverStrategy:
    """Test the SimilarityScoreThresholdRetrieverStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        assert strategy is not None
        assert isinstance(strategy, SimilarityScoreThresholdRetrieverStrategy)
    
    def test_create_retriever_basic(self):
        """Test creating retriever with basic configuration."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.7}
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_additional_params(self):
        """Test creating retriever with additional parameters."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 10, "score_threshold": 0.8},
            params={"custom_param": "value", "another_param": 42}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_as_retriever.assert_called_once_with(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 10, "score_threshold": 0.8},
                custom_param="value",
                another_param=42
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_defaults(self):
        """Test creating retriever with default values."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5},  # Only k specified, score_threshold should use default
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore)
            
            # Should use default value for score_threshold
            mock_as_retriever.assert_called_once_with(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.0}
            )
            assert result == mock_retriever
    
    def test_create_retriever_with_llm(self):
        """Test creating retriever with LLM parameter (should be ignored)."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = strategy.create_retriever(config, vectorstore, llm)
            
            # LLM should be ignored for similarity score threshold retriever
            mock_as_retriever.assert_called_once_with(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.7}
            )
            assert result == mock_retriever
    
    def test_validate_config_with_valid_parameters(self):
        """Test configuration validation with valid parameters."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_valid_k_values(self):
        """Test configuration validation with valid k values."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        valid_k_values = [1, 5, 10, 100, 1000]
        
        for k in valid_k_values:
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
                search_kwargs={"k": k, "score_threshold": 0.7},
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_valid_score_threshold_values(self):
        """Test configuration validation with valid score_threshold values."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        valid_score_threshold_values = [0.0, 0.1, 0.5, 0.7, 0.9, 1.0]
        
        for score_threshold in valid_score_threshold_values:
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
                search_kwargs={"k": 5, "score_threshold": score_threshold},
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_default_values(self):
        """Test configuration validation with default values."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5},  # Only k specified
            params={}
        )
        
        # Should not raise any exception (defaults are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_zero(self):
        """Test configuration validation with k=0."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 0, "score_threshold": 0.7},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_negative(self):
        """Test configuration validation with negative k."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": -5, "score_threshold": 0.7},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_score_threshold_negative(self):
        """Test configuration validation with negative score_threshold."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": -0.1},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['score_threshold'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_score_threshold_greater_than_one(self):
        """Test configuration validation with score_threshold > 1."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 1.5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['score_threshold'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_score_threshold_string(self):
        """Test configuration validation with string score_threshold."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": "0.7"},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['score_threshold'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_score_threshold_none(self):
        """Test configuration validation with None score_threshold."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": None},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['score_threshold'\\] must be a number between 0 and 1"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_string(self):
        """Test configuration validation with string k."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": "5", "score_threshold": 0.7},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_none(self):
        """Test configuration validation with None k."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": None, "score_threshold": 0.7},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_additional_search_kwargs(self):
        """Test configuration validation with additional search kwargs."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7,
                "filter": {"category": "test"},
                "include_metadata": True
            },
            params={}
        )
        
        # Should not raise any exception (additional kwargs are allowed)
        strategy.validate_config(config)


class TestSimilarityScoreThresholdRetrieverStrategyEdgeCases:
    """Test edge cases for SimilarityScoreThresholdRetrieverStrategy."""
    
    def test_create_retriever_with_special_characters_in_params(self):
        """Test creating retriever with special characters in parameters."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
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
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.7},
                custom_param="value with spaces",
                unicode_param="测试参数",
                special_chars="!@#$%^&*()",
                nested_dict={"key": "value"},
                list_param=[1, 2, 3]
            )
            assert result == mock_retriever
    
    def test_create_retriever_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        original_params = {"custom_param": "value", "another_param": 42}
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
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
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        original_search_kwargs = {"k": 5, "score_threshold": 0.7}
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
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
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 1000000, "score_threshold": 0.7},
            params={}
        )
        
        # Should not raise any exception (large k is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_float_k(self):
        """Test configuration validation with float k."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5.0, "score_threshold": 0.7},
            params={}
        )
        
        # Should raise error (k must be integer)
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_integer_score_threshold(self):
        """Test configuration validation with integer score_threshold."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 1},
            params={}
        )
        
        # Should not raise any exception (integer 1 is valid for score_threshold)
        strategy.validate_config(config)
    
    def test_validate_config_with_missing_search_kwargs(self):
        """Test configuration validation with missing search_kwargs."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={},  # Empty search_kwargs
            params={}
        )
        
        # Should not raise any exception (defaults are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_none_params(self):
        """Test configuration validation with None params."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={}
        )
        
        # Should not raise error with empty params
        strategy.validate_config(config)


class TestSimilarityScoreThresholdRetrieverStrategyIntegration:
    """Integration tests for SimilarityScoreThresholdRetrieverStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
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
    
    def test_workflow_with_different_score_threshold_values(self):
        """Test workflow with different score_threshold values."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        vectorstore = Mock()
        
        # Test with different score_threshold values
        score_threshold_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            for score_threshold in score_threshold_values:
                config = RetrieverConfig(
                    type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
                    search_kwargs={"k": 5, "score_threshold": score_threshold},
                    params={}
                )
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_retriever
            
            # Should have been called for each score_threshold value
            assert mock_as_retriever.call_count == len(score_threshold_values)
    
    def test_workflow_with_different_k_values(self):
        """Test workflow with different k values."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        vectorstore = Mock()
        
        # Test with different k values
        k_values = [1, 5, 10, 50, 100]
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            for k in k_values:
                config = RetrieverConfig(
                    type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
                    search_kwargs={"k": k, "score_threshold": 0.7},
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
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        vectorstore = Mock()
        
        # Valid config
        valid_config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={}
        )
        
        # Invalid config (score_threshold > 1)
        invalid_config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 1.5},
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
            with pytest.raises(ValueError, match="search_kwargs\\['score_threshold'\\] must be a number between 0 and 1"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_vectorstore_errors(self):
        """Test workflow with vectorstore errors."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={}
        )
        vectorstore = Mock()
        
        # Mock vectorstore to raise an error
        with patch.object(vectorstore, 'as_retriever', side_effect=Exception("VectorStore error")):
            with pytest.raises(Exception, match="VectorStore error"):
                strategy.create_retriever(config, vectorstore)
    
    def test_workflow_with_none_vectorstore(self):
        """Test workflow with None vectorstore."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={}
        )
        
        # Should handle None vectorstore gracefully
        with pytest.raises(AttributeError):
            strategy.create_retriever(config, None)
    
    def test_workflow_with_custom_vectorstore_implementation(self):
        """Test workflow with custom vectorstore implementation."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
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
        assert call_kwargs["search_type"] == "similarity_score_threshold"
        assert call_kwargs["search_kwargs"] == {"k": 5, "score_threshold": 0.7}
        assert call_kwargs["custom_param"] == "value"
    
    def test_workflow_with_edge_case_score_threshold_values(self):
        """Test workflow with edge case score_threshold values."""
        strategy = SimilarityScoreThresholdRetrieverStrategy()
        vectorstore = Mock()
        
        # Test edge case score_threshold values
        edge_cases = [0.0, 1.0, 0.001, 0.999]
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            for score_threshold in edge_cases:
                config = RetrieverConfig(
                    type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
                    search_kwargs={"k": 5, "score_threshold": score_threshold},
                    params={}
                )
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_retriever
            
            # Should have been called for each edge case
            assert mock_as_retriever.call_count == len(edge_cases)

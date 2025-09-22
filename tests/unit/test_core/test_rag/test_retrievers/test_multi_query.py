"""
Unit tests for multi-query retriever strategy.

Tests all components of the MultiQueryRetrieverStrategy including:
- MultiQueryRetrieverStrategy class
- Configuration validation
- Retriever creation
- Error handling and edge cases
- LLM requirement validation
- Multi-query specific parameter validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.multi_query import MultiQueryRetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


class TestMultiQueryRetrieverStrategy:
    """Test the MultiQueryRetrieverStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = MultiQueryRetrieverStrategy()
        assert strategy is not None
        assert isinstance(strategy, MultiQueryRetrieverStrategy)
    
    def test_create_retriever_basic(self):
        """Test creating retriever with basic configuration."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore, llm)
                
                # Should create base retriever
                mock_as_retriever.assert_called_once_with(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                
                # Should create MultiQueryRetriever from LLM
                mock_multi_query_class.from_llm.assert_called_once_with(
                    retriever=mock_retriever,
                    llm=llm
                )
                
                assert result == mock_multi_query
    
    def test_create_retriever_with_additional_params(self):
        """Test creating retriever with additional parameters."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 10},
            params={
                "query_count": 3,
                "prompt": "Custom prompt for multi-query",
                "custom_param": "value"
            }
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore, llm)
                
                # Should create MultiQueryRetriever with additional params
                mock_multi_query_class.from_llm.assert_called_once_with(
                    retriever=mock_retriever,
                    llm=llm,
                    query_count=3,
                    prompt="Custom prompt for multi-query",
                    custom_param="value"
                )
                
                assert result == mock_multi_query
    
    def test_create_retriever_without_llm(self):
        """Test creating retriever without LLM (should raise error)."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        with pytest.raises(ValueError, match="LLM is required for MultiQueryRetriever"):
            strategy.create_retriever(config, vectorstore)
    
    def test_create_retriever_with_none_llm(self):
        """Test creating retriever with None LLM (should raise error)."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        with pytest.raises(ValueError, match="LLM is required for MultiQueryRetriever"):
            strategy.create_retriever(config, vectorstore, None)
    
    def test_validate_config_with_valid_parameters(self):
        """Test configuration validation with valid parameters."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_valid_k_values(self):
        """Test configuration validation with valid k values."""
        strategy = MultiQueryRetrieverStrategy()
        valid_k_values = [1, 5, 10, 100, 1000]
        
        for k in valid_k_values:
            config = RetrieverConfig(
                type=RetrieverType.MULTI_QUERY,
                search_kwargs={"k": k},
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_valid_query_count(self):
        """Test configuration validation with valid query_count."""
        strategy = MultiQueryRetrieverStrategy()
        valid_query_counts = [1, 2, 3, 5, 10]
        
        for query_count in valid_query_counts:
            config = RetrieverConfig(
                type=RetrieverType.MULTI_QUERY,
                search_kwargs={"k": 5},
                params={"query_count": query_count}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_valid_prompt(self):
        """Test configuration validation with valid prompt."""
        strategy = MultiQueryRetrieverStrategy()
        valid_prompts = [
            "Generate multiple queries for: {question}",
            "Create variations of this question: {question}",
            "Multi-query prompt: {question}",
            "Test prompt with {question} placeholder"
        ]
        
        for prompt in valid_prompts:
            config = RetrieverConfig(
                type=RetrieverType.MULTI_QUERY,
                search_kwargs={"k": 5},
                params={"prompt": prompt}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_default_values(self):
        """Test configuration validation with default values."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}  # No optional params specified
        )
        
        # Should not raise any exception (defaults are valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_zero(self):
        """Test configuration validation with k=0."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 0},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_k_negative(self):
        """Test configuration validation with negative k."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": -5},
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_query_count_zero(self):
        """Test configuration validation with query_count=0."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": 0}
        )
        
        with pytest.raises(ValueError, match="params\\['query_count'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_query_count_negative(self):
        """Test configuration validation with negative query_count."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": -1}
        )
        
        with pytest.raises(ValueError, match="params\\['query_count'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_query_count_string(self):
        """Test configuration validation with string query_count."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": "3"}
        )
        
        with pytest.raises(ValueError, match="params\\['query_count'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_query_count_none(self):
        """Test configuration validation with None query_count."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": None}
        )
        
        with pytest.raises(ValueError, match="params\\['query_count'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_prompt_empty(self):
        """Test configuration validation with empty prompt."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"prompt": ""}
        )
        
        with pytest.raises(ValueError, match="params\\['prompt'\\] must be a non-empty string"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_prompt_whitespace_only(self):
        """Test configuration validation with whitespace-only prompt."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"prompt": "   "}
        )
        
        with pytest.raises(ValueError, match="params\\['prompt'\\] must be a non-empty string"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_prompt_none(self):
        """Test configuration validation with None prompt."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"prompt": None}
        )
        
        with pytest.raises(ValueError, match="params\\['prompt'\\] must be a non-empty string"):
            strategy.validate_config(config)
    
    def test_validate_config_with_invalid_prompt_not_string(self):
        """Test configuration validation with non-string prompt."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"prompt": 123}
        )
        
        with pytest.raises(ValueError, match="params\\['prompt'\\] must be a non-empty string"):
            strategy.validate_config(config)


class TestMultiQueryRetrieverStrategyEdgeCases:
    """Test edge cases for MultiQueryRetrieverStrategy."""
    
    def test_create_retriever_with_special_characters_in_params(self):
        """Test creating retriever with special characters in parameters."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={
                "query_count": 3,
                "prompt": "Generate queries for: {question} with special chars: !@#$%^&*()",
                "custom_param": "value with spaces",
                "unicode_param": "测试参数",
                "nested_dict": {"key": "value"},
                "list_param": [1, 2, 3]
            }
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore, llm)
                
                # Check MultiQueryRetriever creation with special characters
                mock_multi_query_class.from_llm.assert_called_once()
                call_kwargs = mock_multi_query_class.from_llm.call_args[1]
                assert call_kwargs["retriever"] == mock_retriever
                assert call_kwargs["llm"] == llm
                assert call_kwargs["query_count"] == 3
                assert call_kwargs["prompt"] == "Generate queries for: {question} with special chars: !@#$%^&*()"
                assert call_kwargs["custom_param"] == "value with spaces"
                assert call_kwargs["unicode_param"] == "测试参数"
                assert call_kwargs["nested_dict"] == {"key": "value"}
                assert call_kwargs["list_param"] == [1, 2, 3]
                
                assert result == mock_multi_query
    
    def test_create_retriever_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = MultiQueryRetrieverStrategy()
        original_params = {"query_count": 3, "prompt": "Test prompt", "custom_param": "value"}
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params=original_params
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore, llm)
                
                # Original params should not be modified
                assert config.params == original_params
                assert config.params is not original_params  # Should be a copy
    
    def test_create_retriever_search_kwargs_copy_behavior(self):
        """Test that search_kwargs are copied and not modified."""
        strategy = MultiQueryRetrieverStrategy()
        original_search_kwargs = {"k": 5}
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs=original_search_kwargs,
            params={}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                result = strategy.create_retriever(config, vectorstore, llm)
                
                # Original search_kwargs should not be modified
                assert config.search_kwargs == original_search_kwargs
                assert config.search_kwargs is not original_search_kwargs  # Should be a copy
    
    def test_validate_config_with_very_large_k(self):
        """Test configuration validation with very large k value."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 1000000},
            params={}
        )
        
        # Should not raise any exception (large k is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_very_large_query_count(self):
        """Test configuration validation with very large query_count."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": 1000}
        )
        
        # Should not raise any exception (large query_count is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_very_long_prompt(self):
        """Test configuration validation with very long prompt."""
        strategy = MultiQueryRetrieverStrategy()
        long_prompt = "Generate multiple queries for: {question}. " * 100
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"prompt": long_prompt}
        )
        
        # Should not raise any exception (long prompt is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_float_k(self):
        """Test configuration validation with float k."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5.0},
            params={}
        )
        
        # Should raise error (k must be integer)
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_float_query_count(self):
        """Test configuration validation with float query_count."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": 3.0}
        )
        
        # Should raise error (query_count must be integer)
        with pytest.raises(ValueError, match="params\\['query_count'\\] must be a positive integer"):
            strategy.validate_config(config)
    
    def test_validate_config_with_missing_search_kwargs(self):
        """Test configuration validation with missing search_kwargs."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={},  # Empty search_kwargs
            params={}
        )
        
        # Should not raise any exception (default k=4 is valid)
        strategy.validate_config(config)
    
    def test_validate_config_with_none_params(self):
        """Test configuration validation with None params."""
        strategy = MultiQueryRetrieverStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        
        # Should not raise error with empty params
        strategy.validate_config(config)


class TestMultiQueryRetrieverStrategyIntegration:
    """Integration tests for MultiQueryRetrieverStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": 3, "prompt": "Generate queries for: {question}"}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore, llm)
                
                assert result == mock_multi_query
                mock_multi_query_class.from_llm.assert_called_once()
    
    def test_workflow_with_different_query_counts(self):
        """Test workflow with different query_count values."""
        strategy = MultiQueryRetrieverStrategy()
        vectorstore = Mock()
        llm = Mock()
        
        # Test with different query_count values
        query_counts = [1, 2, 3, 5, 10]
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                for query_count in query_counts:
                    config = RetrieverConfig(
                        type=RetrieverType.MULTI_QUERY,
                        search_kwargs={"k": 5},
                        params={"query_count": query_count}
                    )
                    
                    # Validate config
                    strategy.validate_config(config)
                    
                    # Create retriever
                    result = strategy.create_retriever(config, vectorstore, llm)
                    
                    assert result == mock_multi_query
                
                # Should have been called for each query_count value
                assert mock_multi_query_class.from_llm.call_count == len(query_counts)
    
    def test_workflow_with_different_prompts(self):
        """Test workflow with different prompt values."""
        strategy = MultiQueryRetrieverStrategy()
        vectorstore = Mock()
        llm = Mock()
        
        # Test with different prompts
        prompts = [
            "Generate multiple queries for: {question}",
            "Create variations of: {question}",
            "Multi-query prompt: {question}",
            "Test prompt with {question} placeholder"
        ]
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                for prompt in prompts:
                    config = RetrieverConfig(
                        type=RetrieverType.MULTI_QUERY,
                        search_kwargs={"k": 5},
                        params={"prompt": prompt}
                    )
                    
                    # Validate config
                    strategy.validate_config(config)
                    
                    # Create retriever
                    result = strategy.create_retriever(config, vectorstore, llm)
                    
                    assert result == mock_multi_query
                
                # Should have been called for each prompt
                assert mock_multi_query_class.from_llm.call_count == len(prompts)
    
    def test_workflow_with_mixed_valid_and_invalid_configs(self):
        """Test workflow with mix of valid and invalid configurations."""
        strategy = MultiQueryRetrieverStrategy()
        vectorstore = Mock()
        llm = Mock()
        
        # Valid config
        valid_config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": 3, "prompt": "Generate queries for: {question}"}
        )
        
        # Invalid config (empty prompt)
        invalid_config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": 3, "prompt": ""}
        )
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                # Valid config should work
                strategy.validate_config(valid_config)
                result = strategy.create_retriever(valid_config, vectorstore, llm)
                assert result == mock_multi_query
                
                # Invalid config should raise error
                with pytest.raises(ValueError, match="params\\['prompt'\\] must be a non-empty string"):
                    strategy.validate_config(invalid_config)
    
    def test_workflow_with_vectorstore_errors(self):
        """Test workflow with vectorstore errors."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        llm = Mock()
        
        # Mock vectorstore to raise an error
        with patch.object(vectorstore, 'as_retriever', side_effect=Exception("VectorStore error")):
            with pytest.raises(Exception, match="VectorStore error"):
                strategy.create_retriever(config, vectorstore, llm)
    
    def test_workflow_with_none_vectorstore(self):
        """Test workflow with None vectorstore."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        llm = Mock()
        
        # Should handle None vectorstore gracefully
        with pytest.raises(AttributeError):
            strategy.create_retriever(config, None, llm)
    
    def test_workflow_with_multi_query_retriever_errors(self):
        """Test workflow with MultiQueryRetriever errors."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever.from_llm', side_effect=Exception("MultiQueryRetriever error")):
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                with pytest.raises(Exception, match="MultiQueryRetriever error"):
                    strategy.create_retriever(config, vectorstore, llm)
    
    def test_workflow_with_custom_vectorstore_implementation(self):
        """Test workflow with custom vectorstore implementation."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={"query_count": 3, "prompt": "Generate queries for: {question}"}
        )
        llm = Mock()
        
        # Custom vectorstore with as_retriever method
        class CustomVectorStore:
            def __init__(self):
                self.calls = []
            
            def as_retriever(self, **kwargs):
                self.calls.append(kwargs)
                return Mock()
        
        vectorstore = CustomVectorStore()
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            # Validate config
            strategy.validate_config(config)
            
            # Create retriever
            result = strategy.create_retriever(config, vectorstore, llm)
            
            assert result == mock_multi_query
            assert len(vectorstore.calls) == 1  # Should create one base retriever
            
            # Check base retriever call
            base_retriever_call = vectorstore.calls[0]
            assert base_retriever_call["search_type"] == "similarity"
            assert base_retriever_call["search_kwargs"] == {"k": 5}
            
            # Check MultiQueryRetriever creation
            mock_multi_query_class.from_llm.assert_called_once()
            call_kwargs = mock_multi_query_class.from_llm.call_args[1]
            assert call_kwargs["llm"] == llm
            assert call_kwargs["query_count"] == 3
            assert call_kwargs["prompt"] == "Generate queries for: {question}"
    
    def test_workflow_with_different_llm_types(self):
        """Test workflow with different LLM types."""
        strategy = MultiQueryRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.MULTI_QUERY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        # Test with different LLM types
        llm_types = [Mock(), Mock(), Mock()]
        
        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
            mock_multi_query = Mock()
            mock_multi_query_class.from_llm.return_value = mock_multi_query
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                for llm in llm_types:
                    result = strategy.create_retriever(config, vectorstore, llm)
                    assert result == mock_multi_query
                
                # Should have been called for each LLM
                assert mock_multi_query_class.from_llm.call_count == len(llm_types)

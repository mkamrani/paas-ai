"""
Integration tests for the retrievers system.

Tests the complete retrievers system integration including:
- End-to-end workflows
- Cross-component interactions
- Real retriever creation (with mocking)
- Error propagation
- Configuration validation across components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.factory import RetrieverFactory
from src.paas_ai.core.rag.retrievers.registry import RetrieverRegistry
from src.paas_ai.core.rag.retrievers.similarity import SimilarityRetrieverStrategy
from src.paas_ai.core.rag.retrievers.mmr import MMRRetrieverStrategy
from src.paas_ai.core.rag.retrievers.similarity_score_threshold import SimilarityScoreThresholdRetrieverStrategy
from src.paas_ai.core.rag.retrievers.ensemble import EnsembleRetrieverStrategy
from src.paas_ai.core.rag.retrievers.multi_query import MultiQueryRetrieverStrategy
from src.paas_ai.core.rag.retrievers.parent_document import ParentDocumentRetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    # Store original strategies
    original_strategies = RetrieverRegistry._strategies.copy()
    
    yield
    
    # Restore original strategies
    RetrieverRegistry._strategies = original_strategies


class TestRetrieversSystemIntegration:
    """Integration tests for the complete retrievers system."""
    
    def test_similarity_workflow_integration(self):
        """Test complete similarity retriever workflow integration."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            # Test through factory
            result = RetrieverFactory.create_retriever(config, vectorstore)
            
            assert result == mock_retriever
            mock_as_retriever.assert_called_once_with(
                search_type="similarity",
                search_kwargs={"k": 5},
                custom_param="value"
            )
    
    def test_mmr_workflow_integration(self):
        """Test complete MMR retriever workflow integration."""
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            # Test through factory
            result = RetrieverFactory.create_retriever(config, vectorstore)
            
            assert result == mock_retriever
            mock_as_retriever.assert_called_once_with(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
                custom_param="value"
            )
    
    def test_similarity_score_threshold_workflow_integration(self):
        """Test complete similarity score threshold retriever workflow integration."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            search_kwargs={"k": 5, "score_threshold": 0.7},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            # Test through factory
            result = RetrieverFactory.create_retriever(config, vectorstore)
            
            assert result == mock_retriever
            mock_as_retriever.assert_called_once_with(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 5, "score_threshold": 0.7},
                custom_param="value"
            )
    
    def test_ensemble_workflow_integration(self):
        """Test complete ensemble retriever workflow integration."""
        config = RetrieverConfig(
            type=RetrieverType.ENSEMBLE,
            search_kwargs={"k": 5},
            params={"weights": [0.6, 0.4], "custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble_class.return_value = mock_ensemble
            
            with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                mock_retriever = Mock()
                mock_as_retriever.return_value = mock_retriever
                
                # Test through factory
                result = RetrieverFactory.create_retriever(config, vectorstore)
                
                assert result == mock_ensemble
                mock_ensemble_class.assert_called_once()
                call_kwargs = mock_ensemble_class.call_args[1]
                assert call_kwargs["weights"] == [0.6, 0.4]
                assert call_kwargs["custom_param"] == "value"
    
    def test_multi_query_workflow_integration(self):
        """Test complete multi-query retriever workflow integration."""
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
                
                # Test through factory
                result = RetrieverFactory.create_retriever(config, vectorstore, llm)
                
                assert result == mock_multi_query
                mock_multi_query_class.from_llm.assert_called_once()
                call_kwargs = mock_multi_query_class.from_llm.call_args[1]
                assert call_kwargs["llm"] == llm
                assert call_kwargs["query_count"] == 3
                assert call_kwargs["prompt"] == "Generate queries for: {question}"
    
    def test_parent_document_workflow_integration(self):
        """Test complete parent document retriever workflow integration."""
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter, "custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            # Test through factory
            result = RetrieverFactory.create_retriever(config, vectorstore)
            
            assert result == mock_parent_doc
            mock_parent_doc_class.assert_called_once()
            call_kwargs = mock_parent_doc_class.call_args[1]
            assert call_kwargs["vectorstore"] == vectorstore
            assert call_kwargs["child_splitter"] == mock_child_splitter
            assert call_kwargs["custom_param"] == "value"
    
    def test_registry_factory_integration(self):
        """Test integration between registry and factory."""
        # Test that factory uses registry correctly
        supported_types = RetrieverFactory.list_supported_types()
        registry_types = RetrieverRegistry.list_supported_types()
        
        assert supported_types == registry_types
        
        # Test that factory can create retrievers for all registry types
        for retriever_type in registry_types:
            if retriever_type is None:
                continue
                
            config = RetrieverConfig(
                type=retriever_type,
                search_kwargs={"k": 5},
                params={}
            )
            vectorstore = Mock()
            
            # Should not raise "No strategy registered" error
            try:
                with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
                    mock_retriever = Mock()
                    mock_as_retriever.return_value = mock_retriever
                    
                    # Mock specific retriever classes for complex types
                    if retriever_type == RetrieverType.ENSEMBLE:
                        with patch('src.paas_ai.core.rag.retrievers.ensemble.EnsembleRetriever') as mock_ensemble_class:
                            mock_ensemble_class.return_value = Mock()
                            result = RetrieverFactory.create_retriever(config, vectorstore)
                            assert result is not None
                    elif retriever_type == RetrieverType.MULTI_QUERY:
                        with patch('src.paas_ai.core.rag.retrievers.multi_query.MultiQueryRetriever') as mock_multi_query_class:
                            mock_multi_query_class.from_llm.return_value = Mock()
                            result = RetrieverFactory.create_retriever(config, vectorstore, Mock())
                            assert result is not None
                    elif retriever_type == RetrieverType.PARENT_DOCUMENT:
                        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
                            mock_parent_doc_class.return_value = Mock()
                            # Create mock child splitter
                            mock_child_splitter = Mock()
                            mock_child_splitter.split_documents = Mock()
                            config.params["child_splitter"] = mock_child_splitter
                            result = RetrieverFactory.create_retriever(config, vectorstore)
                            assert result is not None
                    else:
                        result = RetrieverFactory.create_retriever(config, vectorstore)
                        assert result is not None
            except ValueError as e:
                if "No strategy registered" in str(e):
                    pytest.fail(f"Strategy not registered for {retriever_type}")
    
    def test_strategy_registration_workflow(self):
        """Test complete strategy registration workflow."""
        # Store original strategy
        original_strategy = RetrieverRegistry._strategies.get(RetrieverType.SIMILARITY)
        
        try:
            # Track calls at class level
            validation_calls = []
            creation_calls = []
            
            # Register a custom strategy
            class CustomRetrieverStrategy:
                def validate_config(self, config):
                    validation_calls.append(config)
                
                def create_retriever(self, config, vectorstore, llm=None):
                    creation_calls.append((config, vectorstore, llm))
                    return Mock()
            
            # Register through factory
            RetrieverFactory.register_strategy(RetrieverType.SIMILARITY, CustomRetrieverStrategy)
            
            # Verify it's registered
            strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy, CustomRetrieverStrategy)
            
            # Test workflow
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": 5},
                params={}
            )
            vectorstore = Mock()
            
            result = RetrieverFactory.create_retriever(config, vectorstore)
            assert result is not None
            assert len(validation_calls) == 1
            assert len(creation_calls) == 1
        finally:
            # Restore original strategy
            if original_strategy:
                RetrieverRegistry._strategies[RetrieverType.SIMILARITY] = original_strategy
    
    def test_error_propagation_integration(self):
        """Test error propagation through the system."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 0},  # Invalid k value
            params={}
        )
        vectorstore = Mock()
        
        # Should propagate validation error from strategy through factory
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            RetrieverFactory.create_retriever(config, vectorstore)
    
    def test_configuration_validation_integration(self):
        """Test configuration validation across components."""
        # Test valid configuration
        valid_config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = RetrieverFactory.create_retriever(valid_config, vectorstore)
            assert result == mock_retriever
        
        # Test invalid configuration
        invalid_config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 0},  # Invalid k value
            params={}
        )
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            RetrieverFactory.create_retriever(invalid_config, vectorstore)
    
    def test_multiple_retriever_types_integration(self):
        """Test integration with multiple retriever types."""
        configs = [
            RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": 5},
                params={}
            ),
            RetrieverConfig(
                type=RetrieverType.MMR,
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
                params={}
            ),
            RetrieverConfig(
                type=RetrieverType.SIMILARITY_SCORE_THRESHOLD,
                search_kwargs={"k": 5, "score_threshold": 0.7},
                params={}
            )
        ]
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            # Test each config
            for config in configs:
                result = RetrieverFactory.create_retriever(config, vectorstore)
                assert result == mock_retriever
            
            # Should have been called for each config
            assert mock_as_retriever.call_count == len(configs)


class TestRetrieversSystemErrorHandling:
    """Test error handling across the retrievers system."""
    
    def test_missing_strategy_error_handling(self):
        """Test error handling for missing strategies."""
        # Try to get a strategy that doesn't exist
        with pytest.raises(ValueError, match="No strategy registered for retriever type"):
            RetrieverRegistry.get_strategy("nonexistent_type")
    
    def test_invalid_config_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with invalid k value
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 0},  # Invalid k value
            params={}
        )
        vectorstore = Mock()
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            RetrieverFactory.create_retriever(config, vectorstore)
    
    def test_strategy_creation_error_handling(self):
        """Test error handling for strategy creation errors."""
        # Mock a strategy that raises an error during creation
        class ErrorStrategy:
            def validate_config(self, config):
                pass
            
            def create_retriever(self, config, vectorstore, llm=None):
                raise RuntimeError("Strategy creation error")
        
        # Register the error strategy
        RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, ErrorStrategy)
        
        try:
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": 5},
                params={}
            )
            vectorstore = Mock()
            
            with pytest.raises(RuntimeError, match="Strategy creation error"):
                RetrieverFactory.create_retriever(config, vectorstore)
        finally:
            # Restore original strategy
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, SimilarityRetrieverStrategy)
    
    def test_validation_error_handling(self):
        """Test error handling for validation errors."""
        # Test with invalid model name
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 0},  # Invalid k value
            params={}
        )
        vectorstore = Mock()
        
        with pytest.raises(ValueError, match="search_kwargs\\['k'\\] must be a positive integer"):
            RetrieverFactory.create_retriever(config, vectorstore)
    
    def test_import_error_handling(self):
        """Test error handling for import errors."""
        # Mock import error by patching the registry's get_strategy method
        with patch.object(RetrieverRegistry, 'get_strategy', side_effect=ImportError("Module not found")):
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": 5},
                params={}
            )
            vectorstore = Mock()
            
            with pytest.raises(ImportError, match="Module not found"):
                RetrieverFactory.create_retriever(config, vectorstore)


class TestRetrieversSystemPerformance:
    """Test performance characteristics of the retrievers system."""
    
    def test_strategy_instance_reuse(self):
        """Test that strategy instances are created fresh each time."""
        # Get the same strategy multiple times
        strategy1 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
        strategy2 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
        
        # Should be different instances
        assert strategy1 is not strategy2
        assert type(strategy1) == type(strategy2)
    
    def test_factory_method_performance(self):
        """Test that factory methods are efficient."""
        import time
        
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            # Time multiple calls
            start_time = time.time()
            for _ in range(100):
                result = RetrieverFactory.create_retriever(config, vectorstore)
                assert result == mock_retriever
            end_time = time.time()
            
            # Should be reasonably fast (less than 1 second for 100 calls)
            assert (end_time - start_time) < 1.0
    
    def test_registry_lookup_performance(self):
        """Test that registry lookups are efficient."""
        import time
        
        # Time multiple lookups
        start_time = time.time()
        for _ in range(1000):
            strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert strategy is not None
        end_time = time.time()
        
        # Should be very fast (less than 0.1 seconds for 1000 lookups)
        assert (end_time - start_time) < 0.1


class TestRetrieversSystemCompatibility:
    """Test compatibility and interoperability of the retrievers system."""
    
    def test_config_compatibility(self):
        """Test that configurations work across different components."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        
        # Test that config works with registry
        strategy = RetrieverRegistry.get_strategy(config.type)
        assert strategy is not None
        
        # Test that config works with factory
        with patch.object(vectorstore, 'as_retriever') as mock_as_retriever:
            mock_retriever = Mock()
            mock_as_retriever.return_value = mock_retriever
            
            result = RetrieverFactory.create_retriever(config, vectorstore)
            assert result == mock_retriever
    
    def test_strategy_interface_compatibility(self):
        """Test that all strategies implement the required interface."""
        for retriever_type in RetrieverRegistry.list_supported_types():
            if retriever_type is None:
                continue
                
            strategy = RetrieverRegistry.get_strategy(retriever_type)
            
            # Check required methods
            assert hasattr(strategy, 'create_retriever')
            assert hasattr(strategy, 'validate_config')
            assert callable(strategy.create_retriever)
            assert callable(strategy.validate_config)
    
    def test_factory_registry_compatibility(self):
        """Test that factory and registry are compatible."""
        # Test that factory delegates to registry correctly
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.validate_config = Mock()
            mock_strategy.create_retriever = Mock(return_value=Mock())
            mock_get_strategy.return_value = mock_strategy
            
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": 5},
                params={}
            )
            vectorstore = Mock()
            
            result = RetrieverFactory.create_retriever(config, vectorstore)
            assert result is not None
            mock_get_strategy.assert_called_once_with(RetrieverType.SIMILARITY)
    
    def test_cross_component_error_handling(self):
        """Test error handling across components."""
        # Test that errors from registry propagate to factory
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = ValueError("Registry error")
            
            config = RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": 5},
                params={}
            )
            vectorstore = Mock()
            
            with pytest.raises(ValueError, match="Registry error"):
                RetrieverFactory.create_retriever(config, vectorstore)

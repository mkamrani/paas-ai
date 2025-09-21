"""
Integration tests for the embeddings system.

Tests the complete embeddings system integration including:
- End-to-end workflows
- Cross-component interactions
- Real embedding creation (with mocking)
- Error propagation
- Configuration validation across components
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.embeddings.factory import EmbeddingsFactory
from src.paas_ai.core.rag.embeddings.registry import EmbeddingRegistry
from src.paas_ai.core.rag.embeddings.openai import OpenAIEmbeddingStrategy
from src.paas_ai.core.rag.embeddings.sentence_transformers import SentenceTransformersEmbeddingStrategy
from src.paas_ai.core.rag.config import EmbeddingConfig, EmbeddingType


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    # Reset to defaults before each test
    EmbeddingRegistry.reset_to_defaults()
    
    yield
    
    # Reset to defaults after each test
    EmbeddingRegistry.reset_to_defaults()


class TestEmbeddingsSystemIntegration:
    """Integration tests for the complete embeddings system."""
    
    def test_openai_workflow_integration(self):
        """Test complete OpenAI workflow integration."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Test through factory
            result = EmbeddingsFactory.create_embeddings(config)
            
            assert result is not None
            # The mock should be called with the correct parameters
            mock_embeddings_class.assert_called_once()
            call_args = mock_embeddings_class.call_args
            assert call_args[1]['model'] == "text-embedding-3-small"
            assert call_args[1]['api_key'] == "test-key"
    
    def test_sentence_transformers_workflow_integration(self):
        """Test complete SentenceTransformers workflow integration."""
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params={"model_kwargs": {"device": "cpu"}}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Test through factory
            result = EmbeddingsFactory.create_embeddings(config)
            
            assert result is not None
            mock_embeddings_class.assert_called_once_with(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
    
    def test_registry_factory_integration(self):
        """Test integration between registry and factory."""
        # Test that factory uses registry correctly
        supported_types = EmbeddingsFactory.list_supported_types()
        registry_types = EmbeddingRegistry.list_supported_types()
        
        assert supported_types == registry_types
        
        # Test that factory can create embeddings for all registry types
        for embedding_type in registry_types:
            if embedding_type is None:
                continue
                
            config = EmbeddingConfig(
                type=embedding_type,
                model_name="test-model",
                params={}
            )
            
            # Should not raise "No strategy registered" error
            try:
                with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
                    mock_strategy = Mock()
                    mock_strategy.validate_config = Mock()
                    mock_strategy.create_embeddings = Mock(return_value=Mock())
                    mock_get_strategy.return_value = mock_strategy
                    
                    result = EmbeddingsFactory.create_embeddings(config)
                    assert result is not None
            except ValueError as e:
                if "No strategy registered" in str(e):
                    pytest.fail(f"Strategy not registered for {embedding_type}")
    
    def test_strategy_registration_workflow(self):
        """Test complete strategy registration workflow."""
        # Store original strategy
        original_strategy = EmbeddingRegistry._strategies.get(EmbeddingType.OPENAI)
        
        try:
            # Track calls at class level
            validation_calls = []
            creation_calls = []
            
            # Register a custom strategy
            class CustomEmbeddingStrategy:
                def validate_config(self, config):
                    validation_calls.append(config)
                
                def create_embeddings(self, config):
                    creation_calls.append(config)
                    return Mock()
            
            # Register through factory
            EmbeddingsFactory.register_strategy(EmbeddingType.OPENAI, CustomEmbeddingStrategy)
            
            # Verify it's registered
            strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy, CustomEmbeddingStrategy)
            
            # Test workflow
            config = EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="test-model",
                params={}
            )
            
            result = EmbeddingsFactory.create_embeddings(config)
            assert result is not None
            assert len(validation_calls) == 1
            assert len(creation_calls) == 1
        finally:
            # Restore original strategy
            if original_strategy:
                EmbeddingRegistry._strategies[EmbeddingType.OPENAI] = original_strategy
    
    def test_error_propagation_integration(self):
        """Test error propagation through the system."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="invalid-model",
            params={"api_key": "test-key"}
        )
        
        # Should propagate validation error from strategy through factory
        with pytest.raises(ValueError, match="Unsupported OpenAI model"):
            EmbeddingsFactory.create_embeddings(config)
    
    def test_configuration_validation_integration(self):
        """Test configuration validation across components."""
        # Test valid configuration
        valid_config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = EmbeddingsFactory.create_embeddings(valid_config)
            assert result is not None
        
        # Test invalid configuration
        invalid_config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="invalid-model",
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match="Unsupported OpenAI model"):
            EmbeddingsFactory.create_embeddings(invalid_config)
    
    def test_multiple_embedding_types_integration(self):
        """Test integration with multiple embedding types."""
        configs = [
            EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="text-embedding-3-small",
                params={"api_key": "test-key"}
            ),
            EmbeddingConfig(
                type=EmbeddingType.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2",
                params={}
            )
        ]
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_openai_class, \
             patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_hf_class:
            
            mock_openai_embeddings = Mock()
            mock_hf_embeddings = Mock()
            mock_openai_class.return_value = mock_openai_embeddings
            mock_hf_class.return_value = mock_hf_embeddings
            
            # Test OpenAI config
            result1 = EmbeddingsFactory.create_embeddings(configs[0])
            assert result1 is not None
            
            # Test SentenceTransformers config
            result2 = EmbeddingsFactory.create_embeddings(configs[1])
            assert result2 is not None
            
            # Verify both were called
            mock_openai_class.assert_called_once()
            mock_hf_class.assert_called_once()
    
    def test_environment_variable_integration(self):
        """Test integration with environment variables."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={}  # No API key in params
        )
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
                mock_embeddings = Mock()
                mock_embeddings_class.return_value = mock_embeddings
                
                result = EmbeddingsFactory.create_embeddings(config)
                
                assert result is not None
                mock_embeddings_class.assert_called_once_with(
                    model="text-embedding-3-small"
                )
    
    def test_warning_integration(self):
        """Test integration with warning system."""
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="unusual-model-name",
            params={}
        )
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
                mock_embeddings = Mock()
                mock_embeddings_class.return_value = mock_embeddings
                
                result = EmbeddingsFactory.create_embeddings(config)
                
                assert result == mock_embeddings
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
                assert "doesn't match common SentenceTransformers patterns" in str(w[0].message)


class TestEmbeddingsSystemErrorHandling:
    """Test error handling across the embeddings system."""
    
    def test_missing_strategy_error_handling(self):
        """Test error handling for missing strategies."""
        # Try to get a strategy that doesn't exist
        with pytest.raises(ValueError, match="No strategy registered for embedding type"):
            EmbeddingRegistry.get_strategy("nonexistent_type")
    
    def test_invalid_config_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with missing API key
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={}
        )
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI embeddings require OPENAI_API_KEY"):
                EmbeddingsFactory.create_embeddings(config)
    
    def test_strategy_creation_error_handling(self):
        """Test error handling for strategy creation errors."""
        # Mock a strategy that raises an error during creation
        class ErrorStrategy:
            def validate_config(self, config):
                pass
            
            def create_embeddings(self, config):
                raise RuntimeError("Strategy creation error")
        
        # Register the error strategy
        EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, ErrorStrategy)
        
        try:
            config = EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="test-model",
                params={}
            )
            
            with pytest.raises(RuntimeError, match="Strategy creation error"):
                EmbeddingsFactory.create_embeddings(config)
        finally:
            # Restore original strategy
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, OpenAIEmbeddingStrategy)
    
    def test_validation_error_handling(self):
        """Test error handling for validation errors."""
        # Test with invalid model name
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="invalid-model",
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match="Unsupported OpenAI model"):
            EmbeddingsFactory.create_embeddings(config)
    
    def test_import_error_handling(self):
        """Test error handling for import errors."""
        # Mock import error
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings', side_effect=ImportError("Module not found")):
            config = EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="text-embedding-3-small",
                params={"api_key": "test-key"}
            )
            
            with pytest.raises(ImportError, match="Module not found"):
                EmbeddingsFactory.create_embeddings(config)


class TestEmbeddingsSystemPerformance:
    """Test performance characteristics of the embeddings system."""
    
    def test_strategy_instance_reuse(self):
        """Test that strategy instances are created fresh each time."""
        # Get the same strategy multiple times
        strategy1 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
        strategy2 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
        
        # Should be different instances
        assert strategy1 is not strategy2
        assert type(strategy1) == type(strategy2)
    
    def test_factory_method_performance(self):
        """Test that factory methods are efficient."""
        import time
        
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Time multiple calls
            start_time = time.time()
            for _ in range(100):
                result = EmbeddingsFactory.create_embeddings(config)
                assert result == mock_embeddings
            end_time = time.time()
            
            # Should be reasonably fast (less than 1 second for 100 calls)
            assert (end_time - start_time) < 1.0
    
    def test_registry_lookup_performance(self):
        """Test that registry lookups are efficient."""
        import time
        
        # Time multiple lookups
        start_time = time.time()
        for _ in range(1000):
            strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert strategy is not None
        end_time = time.time()
        
        # Should be very fast (less than 0.1 seconds for 1000 lookups)
        assert (end_time - start_time) < 0.1


class TestEmbeddingsSystemCompatibility:
    """Test compatibility and interoperability of the embeddings system."""
    
    def test_config_compatibility(self):
        """Test that configurations work across different components."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        # Test that config works with registry
        strategy = EmbeddingRegistry.get_strategy(config.type)
        assert strategy is not None
        
        # Test that config works with factory
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = EmbeddingsFactory.create_embeddings(config)
            assert result == mock_embeddings
    
    def test_strategy_interface_compatibility(self):
        """Test that all strategies implement the required interface."""
        for embedding_type in EmbeddingRegistry.list_supported_types():
            if embedding_type is None:
                continue
                
            strategy = EmbeddingRegistry.get_strategy(embedding_type)
            
            # Check required methods
            assert hasattr(strategy, 'create_embeddings')
            assert hasattr(strategy, 'validate_config')
            assert callable(strategy.create_embeddings)
            assert callable(strategy.validate_config)
    
    def test_factory_registry_compatibility(self):
        """Test that factory and registry are compatible."""
        # Test that factory delegates to registry correctly
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.validate_config = Mock()
            mock_strategy.create_embeddings = Mock(return_value=Mock())
            mock_get_strategy.return_value = mock_strategy
            
            config = EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="test-model",
                params={}
            )
            
            result = EmbeddingsFactory.create_embeddings(config)
            assert result is not None
            mock_get_strategy.assert_called_once_with(EmbeddingType.OPENAI)
    
    def test_cross_component_error_handling(self):
        """Test error handling across components."""
        # Test that errors from registry propagate to factory
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = ValueError("Registry error")
            
            config = EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="test-model",
                params={}
            )
            
            with pytest.raises(ValueError, match="Registry error"):
                EmbeddingsFactory.create_embeddings(config)

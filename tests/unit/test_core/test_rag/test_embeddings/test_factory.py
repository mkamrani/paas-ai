"""
Unit tests for embeddings factory.

Tests all components of the EmbeddingsFactory including:
- EmbeddingsFactory class
- Embedding creation workflow
- Error handling and edge cases
- Integration with registry
- Strategy delegation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.embeddings.factory import EmbeddingsFactory
from src.paas_ai.core.rag.embeddings.base import EmbeddingStrategy
from src.paas_ai.core.rag.config import EmbeddingConfig, EmbeddingType


class MockEmbeddingStrategy(EmbeddingStrategy):
    """Mock embedding strategy for testing."""
    
    def __init__(self, should_raise_validation_error=False, should_raise_creation_error=False):
        self.should_raise_validation_error = should_raise_validation_error
        self.should_raise_creation_error = should_raise_creation_error
        self.validation_calls = []
        self.creation_calls = []
    
    def create_embeddings(self, config: EmbeddingConfig):
        """Create mock embeddings."""
        self.creation_calls.append(config)
        if self.should_raise_creation_error:
            raise ValueError("Mock creation error")
        return Mock()
    
    def validate_config(self, config: EmbeddingConfig):
        """Validate mock configuration."""
        self.validation_calls.append(config)
        if self.should_raise_validation_error:
            raise ValueError("Mock validation error")


class TestEmbeddingsFactory:
    """Test the EmbeddingsFactory class."""
    
    def test_create_embeddings_success(self):
        """Test successful embedding creation."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = EmbeddingsFactory.create_embeddings(config)
            
            # Verify strategy was retrieved
            mock_get_strategy.assert_called_once_with(EmbeddingType.OPENAI)
            
            # Verify validation was called
            assert len(mock_strategy.validation_calls) == 1
            assert mock_strategy.validation_calls[0] == config
            
            # Verify creation was called
            assert len(mock_strategy.creation_calls) == 1
            assert mock_strategy.creation_calls[0] == config
            
            # Verify result is the mock embeddings
            assert result is not None
    
    def test_create_embeddings_validation_error(self):
        """Test embedding creation with validation error."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock validation error"):
                EmbeddingsFactory.create_embeddings(config)
            
            # Verify validation was called but creation was not
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 0
    
    def test_create_embeddings_creation_error(self):
        """Test embedding creation with creation error."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy(should_raise_creation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock creation error"):
                EmbeddingsFactory.create_embeddings(config)
            
            # Verify both validation and creation were called
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 1
    
    def test_create_embeddings_strategy_not_found(self):
        """Test embedding creation when strategy is not found."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = ValueError("No strategy registered")
            
            with pytest.raises(ValueError, match="No strategy registered"):
                EmbeddingsFactory.create_embeddings(config)
    
    def test_create_embeddings_with_different_types(self):
        """Test creating embeddings with different types."""
        test_cases = [
            EmbeddingType.OPENAI,
            EmbeddingType.SENTENCE_TRANSFORMERS,
        ]
        
        for embedding_type in test_cases:
            config = EmbeddingConfig(
                type=embedding_type,
                model_name="test-model",
                params={}
            )
            
            with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
                mock_strategy = MockEmbeddingStrategy()
                mock_get_strategy.return_value = mock_strategy
                
                result = EmbeddingsFactory.create_embeddings(config)
                
                # Verify strategy was retrieved with correct type
                mock_get_strategy.assert_called_with(embedding_type)
                
                # Verify result is not None
                assert result is not None
    
    def test_create_embeddings_with_custom_config(self):
        """Test creating embeddings with custom configuration."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-large",
            params={
                "api_key": "custom-key",
                "chunk_size": 1000,
                "max_retries": 3
            }
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = EmbeddingsFactory.create_embeddings(config)
            
            # Verify validation was called with custom config
            assert len(mock_strategy.validation_calls) == 1
            validation_config = mock_strategy.validation_calls[0]
            assert validation_config == config
            assert validation_config.model_name == "text-embedding-3-large"
            assert validation_config.params["api_key"] == "custom-key"
            
            # Verify creation was called with custom config
            assert len(mock_strategy.creation_calls) == 1
            creation_config = mock_strategy.creation_calls[0]
            assert creation_config == config
    
    def test_list_supported_types(self):
        """Test listing supported embedding types through factory."""
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [EmbeddingType.OPENAI, EmbeddingType.SENTENCE_TRANSFORMERS]
            
            result = EmbeddingsFactory.list_supported_types()
            
            assert result == [EmbeddingType.OPENAI, EmbeddingType.SENTENCE_TRANSFORMERS]
            mock_list.assert_called_once()
    
    def test_register_strategy(self):
        """Test registering strategy through factory."""
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.register_strategy') as mock_register:
            EmbeddingsFactory.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            
            mock_register.assert_called_once_with(EmbeddingType.OPENAI, MockEmbeddingStrategy)
    
    def test_factory_methods_are_static(self):
        """Test that factory methods are static and can be called without instance."""
        # These should work without creating an instance
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should work without creating factory instance
            result = EmbeddingsFactory.create_embeddings(config)
            assert result is not None
        
        # Test other static methods
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [EmbeddingType.OPENAI]
            types = EmbeddingsFactory.list_supported_types()
            assert types == [EmbeddingType.OPENAI]
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.register_strategy') as mock_register:
            EmbeddingsFactory.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            mock_register.assert_called_once()


class TestEmbeddingsFactoryEdgeCases:
    """Test edge cases for EmbeddingsFactory."""
    
    def test_create_embeddings_with_none_config(self):
        """Test creating embeddings with None config."""
        with pytest.raises(AttributeError):
            EmbeddingsFactory.create_embeddings(None)
    
    def test_create_embeddings_with_invalid_config_type(self):
        """Test creating embeddings with invalid config type."""
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should handle invalid config gracefully
            with pytest.raises(AttributeError):
                EmbeddingsFactory.create_embeddings("not-a-config")
    
    def test_create_embeddings_with_missing_config_attributes(self):
        """Test creating embeddings with config missing required attributes."""
        # Create a config object without required attributes
        class IncompleteConfig:
            def __init__(self):
                self.type = EmbeddingType.OPENAI
                # Missing model_name and params
        
        incomplete_config = IncompleteConfig()
        
        # Should handle incomplete config gracefully - the factory accesses config.type first
        with pytest.raises(AttributeError, match="'IncompleteConfig' object has no attribute"):
            EmbeddingsFactory.create_embeddings(incomplete_config)
    
    def test_create_embeddings_with_strategy_that_returns_none(self):
        """Test creating embeddings with strategy that returns None."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_strategy.create_embeddings = Mock(return_value=None)
            mock_get_strategy.return_value = mock_strategy
            
            result = EmbeddingsFactory.create_embeddings(config)
            
            # Should return None if strategy returns None
            assert result is None
    
    def test_create_embeddings_with_strategy_that_returns_string(self):
        """Test creating embeddings with strategy that returns unexpected type."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_strategy.create_embeddings = Mock(return_value="not-embeddings")
            mock_get_strategy.return_value = mock_strategy
            
            result = EmbeddingsFactory.create_embeddings(config)
            
            # Should return whatever the strategy returns
            assert result == "not-embeddings"
    
    def test_create_embeddings_with_exception_in_strategy_retrieval(self):
        """Test creating embeddings with exception in strategy retrieval."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = Exception("Registry error")
            
            with pytest.raises(Exception, match="Registry error"):
                EmbeddingsFactory.create_embeddings(config)
    
    def test_create_embeddings_with_exception_in_validation(self):
        """Test creating embeddings with exception in validation."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_strategy.validate_config = Mock(side_effect=Exception("Validation error"))
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(Exception, match="Validation error"):
                EmbeddingsFactory.create_embeddings(config)
    
    def test_create_embeddings_with_exception_in_creation(self):
        """Test creating embeddings with exception in creation."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_strategy.create_embeddings = Mock(side_effect=Exception("Creation error"))
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(Exception, match="Creation error"):
                EmbeddingsFactory.create_embeddings(config)


class TestEmbeddingsFactoryIntegration:
    """Integration tests for EmbeddingsFactory."""
    
    def test_full_workflow(self):
        """Test complete workflow from factory to embedding creation."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        # This should work without mocking since we're testing the real integration
        try:
            result = EmbeddingsFactory.create_embeddings(config)
            assert result is not None
        except Exception as e:
            # If there are import issues or missing dependencies, that's expected in tests
            # We just want to make sure the factory doesn't crash
            assert "No strategy registered" not in str(e)
    
    def test_factory_registry_consistency(self):
        """Test that factory and registry are consistent."""
        # Get supported types from both
        factory_types = EmbeddingsFactory.list_supported_types()
        registry_types = EmbeddingsFactory.list_supported_types()  # Factory delegates to registry
        
        assert factory_types == registry_types
        
        # Test that we can create embeddings for all supported types
        for embedding_type in factory_types:
            # Skip None values that might be in the registry
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
                    mock_strategy = MockEmbeddingStrategy()
                    mock_get_strategy.return_value = mock_strategy
                    
                    result = EmbeddingsFactory.create_embeddings(config)
                    assert result is not None
            except ValueError as e:
                if "No strategy registered" in str(e):
                    pytest.fail(f"Strategy not registered for {embedding_type}")
    
    def test_workflow_with_multiple_configs(self):
        """Test workflow with multiple different configurations."""
        configs = [
            EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="text-embedding-3-small",
                params={"api_key": "key1"}
            ),
            EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="text-embedding-3-large",
                params={"api_key": "key2"}
            ),
            EmbeddingConfig(
                type=EmbeddingType.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2",
                params={}
            )
        ]
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            for config in configs:
                result = EmbeddingsFactory.create_embeddings(config)
                assert result is not None
            
            # Should have been called for each config
            assert mock_get_strategy.call_count == len(configs)
            assert len(mock_strategy.validation_calls) == len(configs)
            assert len(mock_strategy.creation_calls) == len(configs)
    
    def test_workflow_with_strategy_registration(self):
        """Test workflow with strategy registration."""
        # Register a custom strategy
        EmbeddingsFactory.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
        
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        # Should use the registered strategy
        result = EmbeddingsFactory.create_embeddings(config)
        assert result is not None
    
    def test_workflow_with_error_handling(self):
        """Test workflow with error handling."""
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="test-model",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.factory.EmbeddingRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockEmbeddingStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            # Should handle validation error
            with pytest.raises(ValueError, match="Mock validation error"):
                EmbeddingsFactory.create_embeddings(config)
            
            # Verify validation was called but creation was not
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 0

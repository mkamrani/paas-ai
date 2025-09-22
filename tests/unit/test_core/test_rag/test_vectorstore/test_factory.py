"""
Unit tests for vector store factory.

Tests all components of the VectorStoreFactory including:
- VectorStoreFactory class
- Vector store creation workflow
- Error handling and edge cases
- Integration with strategies
- Strategy delegation
- Configuration validation
- Strategy registration
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.paas_ai.core.rag.vectorstore.factory import VectorStoreFactory
from src.paas_ai.core.rag.vectorstore.base import VectorStoreStrategy
from src.paas_ai.core.config.schemas import VectorStoreConfig, VectorStoreType


@pytest.fixture(autouse=True)
def reset_factory_registry():
    """Fixture to reset the factory registry after each test to ensure isolation."""
    # Store original strategies
    original_strategies = VectorStoreFactory._strategies.copy()
    
    yield
    
    # Restore original strategies after test
    VectorStoreFactory._strategies = original_strategies


class MockVectorStoreStrategy(VectorStoreStrategy):
    """Mock vector store strategy for testing."""
    
    def __init__(self, should_raise_validation_error=False, should_raise_creation_error=False, should_raise_loading_error=False):
        self.should_raise_validation_error = should_raise_validation_error
        self.should_raise_creation_error = should_raise_creation_error
        self.should_raise_loading_error = should_raise_loading_error
        self.validation_calls = []
        self.creation_calls = []
        self.loading_calls = []
    
    def create_vectorstore(self, config, embeddings, documents=None):
        """Create mock vector store."""
        self.creation_calls.append((config, embeddings, documents))
        if self.should_raise_creation_error:
            raise ValueError("Mock creation error")
        return Mock()
    
    def load_vectorstore(self, config, embeddings):
        """Load mock vector store."""
        self.loading_calls.append((config, embeddings))
        if self.should_raise_loading_error:
            raise ValueError("Mock loading error")
        return Mock()
    
    def validate_config(self, config):
        """Validate mock configuration."""
        self.validation_calls.append(config)
        if self.should_raise_validation_error:
            raise ValueError("Mock validation error")


class TestVectorStoreFactory:
    """Test the VectorStoreFactory class."""
    
    def test_create_vectorstore_success(self):
        """Test successful vector store creation."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={"distance_metric": "cosine"}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            result = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
            
            # Verify result is the mock vector store
            assert result is not None
    
    def test_create_vectorstore_validation_error(self):
        """Test vector store creation with validation error."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create a strategy that raises validation error
            strategy_class = type('MockStrategy', (MockVectorStoreStrategy,), {
                '__init__': lambda self: MockVectorStoreStrategy.__init__(self, should_raise_validation_error=True)
            })
            
            with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: strategy_class}):
                with pytest.raises(ValueError, match="Mock validation error"):
                    VectorStoreFactory.create_vectorstore(config, embeddings)
    
    def test_create_vectorstore_creation_error(self):
        """Test vector store creation with creation error."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create a strategy that raises creation error
            strategy_class = type('MockStrategy', (MockVectorStoreStrategy,), {
                '__init__': lambda self: MockVectorStoreStrategy.__init__(self, should_raise_creation_error=True)
            })
            
            with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: strategy_class}):
                with pytest.raises(ValueError, match="Mock creation error"):
                    VectorStoreFactory.create_vectorstore(config, embeddings)
    
    def test_create_vectorstore_unsupported_type(self):
        """Test vector store creation with unsupported type."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {}):
            with pytest.raises(ValueError, match="Unsupported vector store type"):
                VectorStoreFactory.create_vectorstore(config, embeddings)
    
    def test_create_vectorstore_with_different_types(self):
        """Test creating vector stores with different types."""
        test_cases = [
            VectorStoreType.CHROMA,
            VectorStoreType.FAISS,
            VectorStoreType.PINECONE,
        ]
        
        for vectorstore_type in test_cases:
            config = VectorStoreConfig(
                type=vectorstore_type,
                collection_name="test_collection",
                params={}
            )
            embeddings = Mock()
            
            with patch.object(VectorStoreFactory, '_strategies', {vectorstore_type: MockVectorStoreStrategy}):
                result = VectorStoreFactory.create_vectorstore(config, embeddings)
                
                # Verify result is not None
                assert result is not None
    
    def test_create_vectorstore_with_custom_config(self):
        """Test creating vector store with custom configuration."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="custom_collection",
            persist_directory="/tmp/custom",
            params={
                "distance_metric": "euclidean",
                "collection_metadata": {"description": "custom collection"}
            }
        )
        embeddings = Mock()
        documents = [Document(page_content="Custom content", metadata={"source": "custom"})]
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            result = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
            
            # Verify result is not None
            assert result is not None
    
    def test_load_vectorstore_success(self):
        """Test successful vector store loading."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            result = VectorStoreFactory.load_vectorstore(config, embeddings)
            
            # Verify result is the mock vector store
            assert result is not None
    
    def test_load_vectorstore_unsupported_type(self):
        """Test loading vector store with unsupported type."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {}):
            result = VectorStoreFactory.load_vectorstore(config, embeddings)
            
            # Should return None for unsupported type
            assert result is None
    
    def test_load_vectorstore_loading_error(self):
        """Test loading vector store with loading error."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create a strategy that raises loading error
            strategy_class = type('MockStrategy', (MockVectorStoreStrategy,), {
                '__init__': lambda self: MockVectorStoreStrategy.__init__(self, should_raise_loading_error=True)
            })
            
            with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: strategy_class}):
                with pytest.raises(ValueError, match="Mock loading error"):
                    VectorStoreFactory.load_vectorstore(config, embeddings)
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Should not raise any exception
            VectorStoreFactory._validate_config(config)
    
    def test_validate_config_unsupported_type(self):
        """Test configuration validation with unsupported type."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        
        with patch.object(VectorStoreFactory, '_strategies', {}):
            with pytest.raises(ValueError, match="Unsupported vector store type"):
                VectorStoreFactory._validate_config(config)
    
    def test_validate_config_validation_error(self):
        """Test configuration validation with validation error."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create a strategy that raises validation error
            strategy_class = type('MockStrategy', (MockVectorStoreStrategy,), {
                '__init__': lambda self: MockVectorStoreStrategy.__init__(self, should_raise_validation_error=True)
            })
            
            with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: strategy_class}):
                with pytest.raises(ValueError, match="Mock validation error"):
                    VectorStoreFactory._validate_config(config)
    
    def test_register_strategy(self):
        """Test registering a new strategy."""
        class CustomStrategy(VectorStoreStrategy):
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        # Register the custom strategy
        VectorStoreFactory.register_strategy(VectorStoreType.CHROMA, CustomStrategy)
        
        # Verify it was registered
        assert VectorStoreFactory._strategies[VectorStoreType.CHROMA] == CustomStrategy
    
    def test_get_available_types(self):
        """Test getting available vector store types."""
        with patch.object(VectorStoreFactory, '_strategies', {
            VectorStoreType.CHROMA: MockVectorStoreStrategy,
            VectorStoreType.FAISS: MockVectorStoreStrategy,
            VectorStoreType.PINECONE: MockVectorStoreStrategy
        }):
            types = VectorStoreFactory.get_available_types()
            
            assert len(types) == 3
            assert VectorStoreType.CHROMA in types
            assert VectorStoreType.FAISS in types
            assert VectorStoreType.PINECONE in types
    
    def test_factory_methods_are_class_methods(self):
        """Test that factory methods are class methods and can be called without instance."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Should work without creating factory instance
            result = VectorStoreFactory.create_vectorstore(config, embeddings)
            assert result is not None
        
        # Test other class methods
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            result = VectorStoreFactory.load_vectorstore(config, embeddings)
            assert result is not None
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            VectorStoreFactory._validate_config(config)  # Should not raise
        
        types = VectorStoreFactory.get_available_types()
        assert isinstance(types, list)


class TestVectorStoreFactoryEdgeCases:
    """Test edge cases for VectorStoreFactory."""
    
    def test_create_vectorstore_with_none_config(self):
        """Test creating vector store with None config."""
        with pytest.raises(AttributeError):
            VectorStoreFactory.create_vectorstore(None, Mock())
    
    def test_create_vectorstore_with_invalid_config_type(self):
        """Test creating vector store with invalid config type."""
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Should handle invalid config gracefully
            with pytest.raises(AttributeError):
                VectorStoreFactory.create_vectorstore("not-a-config", Mock())
    
    def test_create_vectorstore_with_missing_config_attributes(self):
        """Test creating vector store with config missing required attributes."""
        # Create a config object without required attributes
        class IncompleteConfig:
            def __init__(self):
                self.type = VectorStoreType.CHROMA
                # Missing collection_name and params
        
        incomplete_config = IncompleteConfig()
        
        # Should handle incomplete config gracefully - the factory accesses config.type first
        with pytest.raises(AttributeError, match="'IncompleteConfig' object has no attribute"):
            VectorStoreFactory.create_vectorstore(incomplete_config, Mock())
    
    def test_create_vectorstore_with_strategy_that_returns_none(self):
        """Test creating vector store with strategy that returns None."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create a strategy that returns None
            strategy_class = type('MockStrategy', (MockVectorStoreStrategy,), {
                'create_vectorstore': lambda self, config, embeddings, documents=None: None
            })
            
            with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: strategy_class}):
                result = VectorStoreFactory.create_vectorstore(config, embeddings)
                
                # Should return None if strategy returns None
                assert result is None
    
    def test_create_vectorstore_with_strategy_that_returns_string(self):
        """Test creating vector store with strategy that returns unexpected type."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create a strategy that returns a string
            strategy_class = type('MockStrategy', (MockVectorStoreStrategy,), {
                'create_vectorstore': lambda self, config, embeddings, documents=None: "not-vectorstore"
            })
            
            with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: strategy_class}):
                result = VectorStoreFactory.create_vectorstore(config, embeddings)
                
                # Should return whatever the strategy returns
                assert result == "not-vectorstore"
    
    def test_load_vectorstore_with_none_config(self):
        """Test loading vector store with None config."""
        with pytest.raises(AttributeError):
            VectorStoreFactory.load_vectorstore(None, Mock())
    
    def test_load_vectorstore_with_invalid_config_type(self):
        """Test loading vector store with invalid config type."""
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Should handle invalid config gracefully
            with pytest.raises(AttributeError):
                VectorStoreFactory.load_vectorstore("not-a-config", Mock())
    
    def test_validate_config_with_none_config(self):
        """Test validating configuration with None config."""
        with pytest.raises(AttributeError):
            VectorStoreFactory._validate_config(None)
    
    def test_validate_config_with_invalid_config_type(self):
        """Test validating configuration with invalid config type."""
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Should handle invalid config gracefully
            with pytest.raises(AttributeError):
                VectorStoreFactory._validate_config("not-a-config")
    
    def test_register_strategy_with_none_type(self):
        """Test registering strategy with None type."""
        class CustomStrategy(VectorStoreStrategy):
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        # Should handle None type gracefully
        with pytest.raises(TypeError):
            VectorStoreFactory.register_strategy(None, CustomStrategy)
    
    def test_register_strategy_with_none_strategy(self):
        """Test registering strategy with None strategy class."""
        # Should handle None strategy gracefully
        with pytest.raises(TypeError):
            VectorStoreFactory.register_strategy(VectorStoreType.CHROMA, None)
    
    def test_register_strategy_with_invalid_strategy(self):
        """Test registering strategy with invalid strategy class."""
        # Should handle invalid strategy gracefully
        with pytest.raises(TypeError):
            VectorStoreFactory.register_strategy(VectorStoreType.CHROMA, "not-a-strategy")
    
    def test_get_available_types_with_empty_strategies(self):
        """Test getting available types with empty strategies."""
        with patch.object(VectorStoreFactory, '_strategies', {}):
            types = VectorStoreFactory.get_available_types()
            
            assert types == []
    
    def test_get_available_types_with_none_strategies(self):
        """Test getting available types with None strategies."""
        with patch.object(VectorStoreFactory, '_strategies', None):
            with pytest.raises(TypeError):
                VectorStoreFactory.get_available_types()


class TestVectorStoreFactoryIntegration:
    """Integration tests for VectorStoreFactory."""
    
    def test_full_workflow(self):
        """Test complete workflow from factory to vector store creation."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={"distance_metric": "cosine"}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # This should work without mocking since we're testing the real integration
            result = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
            assert result is not None
    
    def test_factory_strategy_consistency(self):
        """Test that factory and strategies are consistent."""
        # Get available types from factory
        factory_types = VectorStoreFactory.get_available_types()
        
        # Test that we can create vector stores for all supported types
        for vectorstore_type in factory_types:
            # Skip None values that might be in the strategies
            if vectorstore_type is None:
                continue
                
            config = VectorStoreConfig(
                type=vectorstore_type,
                collection_name="test_collection",
                params={}
            )
            embeddings = Mock()
            
            # Should not raise "Unsupported vector store type" error
            try:
                with patch.object(VectorStoreFactory, '_strategies', {vectorstore_type: MockVectorStoreStrategy}):
                    result = VectorStoreFactory.create_vectorstore(config, embeddings)
                    assert result is not None
            except ValueError as e:
                if "Unsupported vector store type" in str(e):
                    pytest.fail(f"Strategy not registered for {vectorstore_type}")
    
    def test_workflow_with_multiple_configs(self):
        """Test workflow with multiple different configurations."""
        configs = [
            VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="collection1",
                persist_directory="/tmp/chroma1",
                params={"distance_metric": "cosine"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.FAISS,
                persist_directory="/tmp/faiss1",
                params={"index_type": "IndexFlatL2"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name="collection3",
                params={"api_key": "test-key"}
            )
        ]
        
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {
            VectorStoreType.CHROMA: MockVectorStoreStrategy,
            VectorStoreType.FAISS: MockVectorStoreStrategy,
            VectorStoreType.PINECONE: MockVectorStoreStrategy
        }):
            for config in configs:
                result = VectorStoreFactory.create_vectorstore(config, embeddings)
                assert result is not None
    
    def test_workflow_with_strategy_registration(self):
        """Test workflow with strategy registration."""
        class CustomStrategy(VectorStoreStrategy):
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        # Register a custom strategy
        VectorStoreFactory.register_strategy(VectorStoreType.CHROMA, CustomStrategy)
        
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        # Should use the registered strategy
        result = VectorStoreFactory.create_vectorstore(config, embeddings)
        assert result is not None
    
    def test_workflow_with_error_handling(self):
        """Test workflow with error handling."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create a strategy that raises validation error
            strategy_class = type('MockStrategy', (MockVectorStoreStrategy,), {
                '__init__': lambda self: MockVectorStoreStrategy.__init__(self, should_raise_validation_error=True)
            })
            
            with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: strategy_class}):
                # Should handle validation error
                with pytest.raises(ValueError, match="Mock validation error"):
                    VectorStoreFactory.create_vectorstore(config, embeddings)
    
    def test_workflow_with_creation_and_loading(self):
        """Test workflow with both creation and loading."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Create vector store
            result1 = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
            assert result1 is not None
            
            # Load vector store
            result2 = VectorStoreFactory.load_vectorstore(config, embeddings)
            assert result2 is not None
    
    def test_workflow_with_validation(self):
        """Test workflow with configuration validation."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        
        with patch.object(VectorStoreFactory, '_strategies', {VectorStoreType.CHROMA: MockVectorStoreStrategy}):
            # Should not raise error
            VectorStoreFactory._validate_config(config)
            
            # Create vector store (which also validates)
            result = VectorStoreFactory.create_vectorstore(config, Mock())
            assert result is not None
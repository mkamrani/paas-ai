"""
Unit tests for base vector store strategy interface.

Tests the abstract VectorStoreStrategy base class including:
- Abstract method enforcement
- Interface compliance
- Error handling for abstract methods
- Type checking and validation
"""

import pytest
from abc import ABC
from unittest.mock import Mock, patch
from pathlib import Path

from src.paas_ai.core.rag.vectorstore.base import VectorStoreStrategy
from src.paas_ai.core.config.schemas import VectorStoreConfig, VectorStoreType


class TestVectorStoreStrategy:
    """Test the VectorStoreStrategy abstract base class."""
    
    def test_is_abstract_base_class(self):
        """Test that VectorStoreStrategy is an abstract base class."""
        assert issubclass(VectorStoreStrategy, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that VectorStoreStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VectorStoreStrategy()
    
    def test_has_required_abstract_methods(self):
        """Test that VectorStoreStrategy has the required abstract methods."""
        abstract_methods = VectorStoreStrategy.__abstractmethods__
        
        assert 'create_vectorstore' in abstract_methods
        assert 'load_vectorstore' in abstract_methods
        assert 'validate_config' in abstract_methods
        assert len(abstract_methods) == 3
    
    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of VectorStoreStrategy works."""
        class ConcreteVectorStoreStrategy(VectorStoreStrategy):
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        # Should be able to instantiate
        strategy = ConcreteVectorStoreStrategy()
        assert isinstance(strategy, VectorStoreStrategy)
        
        # Should be able to call methods
        config = VectorStoreConfig(type=VectorStoreType.CHROMA, collection_name="test")
        embeddings = Mock()
        vectorstore = strategy.create_vectorstore(config, embeddings)
        assert vectorstore is not None
        
        loaded_vectorstore = strategy.load_vectorstore(config, embeddings)
        assert loaded_vectorstore is not None
        
        # Should not raise error
        strategy.validate_config(config)
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation fails to instantiate."""
        class IncompleteVectorStoreStrategy(VectorStoreStrategy):
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            # Missing load_vectorstore and validate_config methods
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteVectorStoreStrategy()
    
    def test_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        import inspect
        
        # Check create_vectorstore signature
        create_vectorstore_sig = inspect.signature(VectorStoreStrategy.create_vectorstore)
        assert len(create_vectorstore_sig.parameters) == 4  # self + config + embeddings + documents
        assert 'config' in create_vectorstore_sig.parameters
        assert 'embeddings' in create_vectorstore_sig.parameters
        assert 'documents' in create_vectorstore_sig.parameters
        
        # Check load_vectorstore signature
        load_vectorstore_sig = inspect.signature(VectorStoreStrategy.load_vectorstore)
        assert len(load_vectorstore_sig.parameters) == 3  # self + config + embeddings
        assert 'config' in load_vectorstore_sig.parameters
        assert 'embeddings' in load_vectorstore_sig.parameters
        
        # Check validate_config signature
        validate_config_sig = inspect.signature(VectorStoreStrategy.validate_config)
        assert len(validate_config_sig.parameters) == 2  # self + config
        assert 'config' in validate_config_sig.parameters
    
    def test_docstrings_exist(self):
        """Test that abstract methods have docstrings."""
        assert VectorStoreStrategy.create_vectorstore.__doc__ is not None
        assert VectorStoreStrategy.load_vectorstore.__doc__ is not None
        assert VectorStoreStrategy.validate_config.__doc__ is not None
        
        # Check that docstrings are meaningful
        assert len(VectorStoreStrategy.create_vectorstore.__doc__.strip()) > 10
        assert len(VectorStoreStrategy.load_vectorstore.__doc__.strip()) > 10
        assert len(VectorStoreStrategy.validate_config.__doc__.strip()) > 10
    
    def test_class_docstring(self):
        """Test that the class has a proper docstring."""
        assert VectorStoreStrategy.__doc__ is not None
        assert "Base strategy interface for vector stores" in VectorStoreStrategy.__doc__
    
    def test_inheritance_chain(self):
        """Test the inheritance chain of VectorStoreStrategy."""
        # Should inherit from ABC
        assert ABC in VectorStoreStrategy.__mro__
        
        # Should be in the MRO
        mro = VectorStoreStrategy.__mro__
        assert VectorStoreStrategy in mro
        assert ABC in mro
        assert object in mro
    
    def test_abstract_methods_are_callable(self):
        """Test that abstract methods are callable (though they raise NotImplementedError)."""
        class TestStrategy(VectorStoreStrategy):
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        strategy = TestStrategy()
        config = VectorStoreConfig(type=VectorStoreType.CHROMA, collection_name="test")
        embeddings = Mock()
        
        # Should be callable
        assert callable(strategy.create_vectorstore)
        assert callable(strategy.load_vectorstore)
        assert callable(strategy.validate_config)
        
        # Should work when called
        result = strategy.create_vectorstore(config, embeddings)
        assert result is not None
        
        loaded_result = strategy.load_vectorstore(config, embeddings)
        assert loaded_result is not None
        
        strategy.validate_config(config)  # Should not raise


class TestVectorStoreStrategyInterface:
    """Test the VectorStoreStrategy interface compliance."""
    
    def test_interface_consistency(self):
        """Test that the interface is consistent and well-defined."""
        # All abstract methods should be present
        required_methods = {'create_vectorstore', 'load_vectorstore', 'validate_config'}
        abstract_methods = VectorStoreStrategy.__abstractmethods__
        
        assert required_methods == abstract_methods
    
    def test_method_return_types(self):
        """Test that method return types are properly documented."""
        import inspect
        
        # create_vectorstore should return VectorStore
        create_vectorstore_sig = inspect.signature(VectorStoreStrategy.create_vectorstore)
        return_annotation = create_vectorstore_sig.return_annotation
        
        # Should have return annotation (though it might be string in some cases)
        assert return_annotation is not inspect.Signature.empty
        
        # load_vectorstore should return Optional[VectorStore]
        load_vectorstore_sig = inspect.signature(VectorStoreStrategy.load_vectorstore)
        return_annotation = load_vectorstore_sig.return_annotation
        
        # Should have return annotation
        assert return_annotation is not inspect.Signature.empty
        
        # validate_config should return None
        validate_config_sig = inspect.signature(VectorStoreStrategy.validate_config)
        return_annotation = validate_config_sig.return_annotation
        
        # Should have return annotation
        assert return_annotation is not inspect.Signature.empty
    
    def test_parameter_types(self):
        """Test that method parameters have proper type annotations."""
        import inspect
        
        # create_vectorstore should take VectorStoreConfig, Embeddings, Optional[List[Document]]
        create_vectorstore_sig = inspect.signature(VectorStoreStrategy.create_vectorstore)
        config_param = create_vectorstore_sig.parameters['config']
        embeddings_param = create_vectorstore_sig.parameters['embeddings']
        documents_param = create_vectorstore_sig.parameters['documents']
        
        assert config_param.annotation is not inspect.Signature.empty
        assert embeddings_param.annotation is not inspect.Signature.empty
        assert documents_param.annotation is not inspect.Signature.empty
        
        # load_vectorstore should take VectorStoreConfig, Embeddings
        load_vectorstore_sig = inspect.signature(VectorStoreStrategy.load_vectorstore)
        config_param = load_vectorstore_sig.parameters['config']
        embeddings_param = load_vectorstore_sig.parameters['embeddings']
        
        assert config_param.annotation is not inspect.Signature.empty
        assert embeddings_param.annotation is not inspect.Signature.empty
        
        # validate_config should take VectorStoreConfig
        validate_config_sig = inspect.signature(VectorStoreStrategy.validate_config)
        config_param = validate_config_sig.parameters['config']
        
        assert config_param.annotation is not inspect.Signature.empty


class TestVectorStoreStrategyEdgeCases:
    """Test edge cases for VectorStoreStrategy."""
    
    def test_multiple_inheritance(self):
        """Test VectorStoreStrategy with multiple inheritance."""
        class Mixin:
            def extra_method(self):
                return "mixin"
        
        class MultiInheritanceStrategy(VectorStoreStrategy, Mixin):
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        strategy = MultiInheritanceStrategy()
        assert isinstance(strategy, VectorStoreStrategy)
        assert isinstance(strategy, Mixin)
        assert strategy.extra_method() == "mixin"
    
    def test_strategy_with_properties(self):
        """Test VectorStoreStrategy with properties."""
        class StrategyWithProperties(VectorStoreStrategy):
            def __init__(self):
                self._name = "test"
            
            @property
            def name(self):
                return self._name
            
            @name.setter
            def name(self, value):
                self._name = value
            
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        strategy = StrategyWithProperties()
        assert strategy.name == "test"
        
        strategy.name = "new_name"
        assert strategy.name == "new_name"
    
    def test_strategy_with_class_methods(self):
        """Test VectorStoreStrategy with class methods."""
        class StrategyWithClassMethods(VectorStoreStrategy):
            @classmethod
            def get_supported_types(cls):
                return ["chroma", "faiss", "pinecone"]
            
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        strategy = StrategyWithClassMethods()
        types = StrategyWithClassMethods.get_supported_types()
        assert types == ["chroma", "faiss", "pinecone"]
    
    def test_strategy_with_static_methods(self):
        """Test VectorStoreStrategy with static methods."""
        class StrategyWithStaticMethods(VectorStoreStrategy):
            @staticmethod
            def validate_collection_name(collection_name: str) -> bool:
                return len(collection_name) > 0
            
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        strategy = StrategyWithStaticMethods()
        assert StrategyWithStaticMethods.validate_collection_name("test") is True
        assert StrategyWithStaticMethods.validate_collection_name("") is False
    
    def test_strategy_with_documents_parameter(self):
        """Test VectorStoreStrategy with documents parameter handling."""
        class TestStrategy(VectorStoreStrategy):
            def __init__(self):
                self.documents_received = None
            
            def create_vectorstore(self, config, embeddings, documents=None):
                self.documents_received = documents
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        strategy = TestStrategy()
        config = VectorStoreConfig(type=VectorStoreType.CHROMA, collection_name="test")
        embeddings = Mock()
        documents = [Mock(), Mock()]
        
        # Test with documents
        result = strategy.create_vectorstore(config, embeddings, documents)
        assert result is not None
        assert strategy.documents_received == documents
        
        # Test without documents
        result = strategy.create_vectorstore(config, embeddings)
        assert result is not None
        assert strategy.documents_received is None
        
        # Test with empty documents list
        result = strategy.create_vectorstore(config, embeddings, [])
        assert result is not None
        assert strategy.documents_received == []
    
    def test_strategy_with_config_validation(self):
        """Test VectorStoreStrategy with config validation."""
        class TestStrategy(VectorStoreStrategy):
            def __init__(self):
                self.validation_calls = []
            
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                self.validation_calls.append(config)
                if not config.collection_name:
                    raise ValueError("collection_name is required")
        
        strategy = TestStrategy()
        config = VectorStoreConfig(type=VectorStoreType.CHROMA, collection_name="test")
        embeddings = Mock()
        
        # Should not raise error
        strategy.validate_config(config)
        assert len(strategy.validation_calls) == 1
        assert strategy.validation_calls[0] == config
        
        # Test with invalid config
        invalid_config = VectorStoreConfig(type=VectorStoreType.CHROMA, collection_name="")
        with pytest.raises(ValueError, match="collection_name is required"):
            strategy.validate_config(invalid_config)
        
        assert len(strategy.validation_calls) == 2

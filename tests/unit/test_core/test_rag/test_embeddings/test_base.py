"""
Unit tests for base embedding strategy interface.

Tests the abstract EmbeddingStrategy base class including:
- Abstract method enforcement
- Interface compliance
- Error handling for abstract methods
- Type checking and validation
"""

import pytest
from abc import ABC
from unittest.mock import Mock, patch

from src.paas_ai.core.rag.embeddings.base import EmbeddingStrategy
from src.paas_ai.core.rag.config import EmbeddingConfig, EmbeddingType


class TestEmbeddingStrategy:
    """Test the EmbeddingStrategy abstract base class."""
    
    def test_is_abstract_base_class(self):
        """Test that EmbeddingStrategy is an abstract base class."""
        assert issubclass(EmbeddingStrategy, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that EmbeddingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EmbeddingStrategy()
    
    def test_has_required_abstract_methods(self):
        """Test that EmbeddingStrategy has the required abstract methods."""
        abstract_methods = EmbeddingStrategy.__abstractmethods__
        
        assert 'create_embeddings' in abstract_methods
        assert 'validate_config' in abstract_methods
        assert len(abstract_methods) == 2
    
    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of EmbeddingStrategy works."""
        class ConcreteEmbeddingStrategy(EmbeddingStrategy):
            def create_embeddings(self, config: EmbeddingConfig):
                return Mock()
            
            def validate_config(self, config: EmbeddingConfig):
                pass
        
        # Should be able to instantiate
        strategy = ConcreteEmbeddingStrategy()
        assert isinstance(strategy, EmbeddingStrategy)
        
        # Should be able to call methods
        config = EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="test-model")
        embeddings = strategy.create_embeddings(config)
        assert embeddings is not None
        
        # Should not raise error
        strategy.validate_config(config)
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation fails to instantiate."""
        class IncompleteEmbeddingStrategy(EmbeddingStrategy):
            def create_embeddings(self, config: EmbeddingConfig):
                return Mock()
            # Missing validate_config method
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteEmbeddingStrategy()
    
    def test_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        import inspect
        
        # Check create_embeddings signature
        create_embeddings_sig = inspect.signature(EmbeddingStrategy.create_embeddings)
        assert len(create_embeddings_sig.parameters) == 2  # self + config
        assert 'config' in create_embeddings_sig.parameters
        
        # Check validate_config signature
        validate_config_sig = inspect.signature(EmbeddingStrategy.validate_config)
        assert len(validate_config_sig.parameters) == 2  # self + config
        assert 'config' in validate_config_sig.parameters
    
    def test_docstrings_exist(self):
        """Test that abstract methods have docstrings."""
        assert EmbeddingStrategy.create_embeddings.__doc__ is not None
        assert EmbeddingStrategy.validate_config.__doc__ is not None
        
        # Check that docstrings are meaningful
        assert len(EmbeddingStrategy.create_embeddings.__doc__.strip()) > 10
        assert len(EmbeddingStrategy.validate_config.__doc__.strip()) > 10
    
    def test_class_docstring(self):
        """Test that the class has a proper docstring."""
        assert EmbeddingStrategy.__doc__ is not None
        assert "Base strategy interface for embeddings" in EmbeddingStrategy.__doc__
    
    def test_inheritance_chain(self):
        """Test the inheritance chain of EmbeddingStrategy."""
        # Should inherit from ABC
        assert ABC in EmbeddingStrategy.__mro__
        
        # Should be in the MRO
        mro = EmbeddingStrategy.__mro__
        assert EmbeddingStrategy in mro
        assert ABC in mro
        assert object in mro
    
    def test_abstract_methods_are_callable(self):
        """Test that abstract methods are callable (though they raise NotImplementedError)."""
        class TestStrategy(EmbeddingStrategy):
            def create_embeddings(self, config: EmbeddingConfig):
                return Mock()
            
            def validate_config(self, config: EmbeddingConfig):
                pass
        
        strategy = TestStrategy()
        config = EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="test-model")
        
        # Should be callable
        assert callable(strategy.create_embeddings)
        assert callable(strategy.validate_config)
        
        # Should work when called
        result = strategy.create_embeddings(config)
        assert result is not None
        
        strategy.validate_config(config)  # Should not raise


class TestEmbeddingStrategyInterface:
    """Test the EmbeddingStrategy interface compliance."""
    
    def test_interface_consistency(self):
        """Test that the interface is consistent and well-defined."""
        # All abstract methods should be present
        required_methods = {'create_embeddings', 'validate_config'}
        abstract_methods = EmbeddingStrategy.__abstractmethods__
        
        assert required_methods == abstract_methods
    
    def test_method_return_types(self):
        """Test that method return types are properly documented."""
        import inspect
        
        # create_embeddings should return Embeddings
        create_embeddings_sig = inspect.signature(EmbeddingStrategy.create_embeddings)
        return_annotation = create_embeddings_sig.return_annotation
        
        # Should have return annotation (though it might be string in some cases)
        assert return_annotation is not inspect.Signature.empty
        
        # validate_config should return None
        validate_config_sig = inspect.signature(EmbeddingStrategy.validate_config)
        return_annotation = validate_config_sig.return_annotation
        
        # Should have return annotation
        assert return_annotation is not inspect.Signature.empty
    
    def test_parameter_types(self):
        """Test that method parameters have proper type annotations."""
        import inspect
        
        # create_embeddings should take EmbeddingConfig
        create_embeddings_sig = inspect.signature(EmbeddingStrategy.create_embeddings)
        config_param = create_embeddings_sig.parameters['config']
        
        assert config_param.annotation is not inspect.Signature.empty
        
        # validate_config should take EmbeddingConfig
        validate_config_sig = inspect.signature(EmbeddingStrategy.validate_config)
        config_param = validate_config_sig.parameters['config']
        
        assert config_param.annotation is not inspect.Signature.empty


class TestEmbeddingStrategyEdgeCases:
    """Test edge cases for EmbeddingStrategy."""
    
    def test_multiple_inheritance(self):
        """Test EmbeddingStrategy with multiple inheritance."""
        class Mixin:
            def extra_method(self):
                return "mixin"
        
        class MultiInheritanceStrategy(EmbeddingStrategy, Mixin):
            def create_embeddings(self, config: EmbeddingConfig):
                return Mock()
            
            def validate_config(self, config: EmbeddingConfig):
                pass
        
        strategy = MultiInheritanceStrategy()
        assert isinstance(strategy, EmbeddingStrategy)
        assert isinstance(strategy, Mixin)
        assert strategy.extra_method() == "mixin"
    
    def test_strategy_with_properties(self):
        """Test EmbeddingStrategy with properties."""
        class StrategyWithProperties(EmbeddingStrategy):
            def __init__(self):
                self._name = "test"
            
            @property
            def name(self):
                return self._name
            
            @name.setter
            def name(self, value):
                self._name = value
            
            def create_embeddings(self, config: EmbeddingConfig):
                return Mock()
            
            def validate_config(self, config: EmbeddingConfig):
                pass
        
        strategy = StrategyWithProperties()
        assert strategy.name == "test"
        
        strategy.name = "new_name"
        assert strategy.name == "new_name"
    
    def test_strategy_with_class_methods(self):
        """Test EmbeddingStrategy with class methods."""
        class StrategyWithClassMethods(EmbeddingStrategy):
            @classmethod
            def get_supported_models(cls):
                return ["model1", "model2"]
            
            def create_embeddings(self, config: EmbeddingConfig):
                return Mock()
            
            def validate_config(self, config: EmbeddingConfig):
                pass
        
        strategy = StrategyWithClassMethods()
        models = StrategyWithClassMethods.get_supported_models()
        assert models == ["model1", "model2"]
    
    def test_strategy_with_static_methods(self):
        """Test EmbeddingStrategy with static methods."""
        class StrategyWithStaticMethods(EmbeddingStrategy):
            @staticmethod
            def validate_model_name(model_name: str) -> bool:
                return len(model_name) > 0
            
            def create_embeddings(self, config: EmbeddingConfig):
                return Mock()
            
            def validate_config(self, config: EmbeddingConfig):
                pass
        
        strategy = StrategyWithStaticMethods()
        assert StrategyWithStaticMethods.validate_model_name("test") is True
        assert StrategyWithStaticMethods.validate_model_name("") is False

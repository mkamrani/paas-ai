"""
Unit tests for base retriever strategy interface.

Tests the abstract RetrieverStrategy base class including:
- Abstract method enforcement
- Interface compliance
- Error handling for abstract methods
- Type checking and validation
"""

import pytest
from abc import ABC
from unittest.mock import Mock, patch

from src.paas_ai.core.rag.retrievers.base import RetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


class TestRetrieverStrategy:
    """Test the RetrieverStrategy abstract base class."""
    
    def test_is_abstract_base_class(self):
        """Test that RetrieverStrategy is an abstract base class."""
        assert issubclass(RetrieverStrategy, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that RetrieverStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RetrieverStrategy()
    
    def test_has_required_abstract_methods(self):
        """Test that RetrieverStrategy has the required abstract methods."""
        abstract_methods = RetrieverStrategy.__abstractmethods__
        
        assert 'create_retriever' in abstract_methods
        assert 'validate_config' in abstract_methods
        assert len(abstract_methods) == 2
    
    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of RetrieverStrategy works."""
        class ConcreteRetrieverStrategy(RetrieverStrategy):
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        # Should be able to instantiate
        strategy = ConcreteRetrieverStrategy()
        assert isinstance(strategy, RetrieverStrategy)
        
        # Should be able to call methods
        config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
        vectorstore = Mock()
        retriever = strategy.create_retriever(config, vectorstore)
        assert retriever is not None
        
        # Should not raise error
        strategy.validate_config(config)
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation fails to instantiate."""
        class IncompleteRetrieverStrategy(RetrieverStrategy):
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                return Mock()
            # Missing validate_config method
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteRetrieverStrategy()
    
    def test_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        import inspect
        
        # Check create_retriever signature
        create_retriever_sig = inspect.signature(RetrieverStrategy.create_retriever)
        assert len(create_retriever_sig.parameters) == 4  # self + config + vectorstore + llm
        assert 'config' in create_retriever_sig.parameters
        assert 'vectorstore' in create_retriever_sig.parameters
        assert 'llm' in create_retriever_sig.parameters
        
        # Check validate_config signature
        validate_config_sig = inspect.signature(RetrieverStrategy.validate_config)
        assert len(validate_config_sig.parameters) == 2  # self + config
        assert 'config' in validate_config_sig.parameters
    
    def test_docstrings_exist(self):
        """Test that abstract methods have docstrings."""
        assert RetrieverStrategy.create_retriever.__doc__ is not None
        assert RetrieverStrategy.validate_config.__doc__ is not None
        
        # Check that docstrings are meaningful
        assert len(RetrieverStrategy.create_retriever.__doc__.strip()) > 10
        assert len(RetrieverStrategy.validate_config.__doc__.strip()) > 10
    
    def test_class_docstring(self):
        """Test that the class has a proper docstring."""
        assert RetrieverStrategy.__doc__ is not None
        assert "Base strategy interface for retrievers" in RetrieverStrategy.__doc__
    
    def test_inheritance_chain(self):
        """Test the inheritance chain of RetrieverStrategy."""
        # Should inherit from ABC
        assert ABC in RetrieverStrategy.__mro__
        
        # Should be in the MRO
        mro = RetrieverStrategy.__mro__
        assert RetrieverStrategy in mro
        assert ABC in mro
        assert object in mro
    
    def test_abstract_methods_are_callable(self):
        """Test that abstract methods are callable (though they raise NotImplementedError)."""
        class TestStrategy(RetrieverStrategy):
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = TestStrategy()
        config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
        vectorstore = Mock()
        
        # Should be callable
        assert callable(strategy.create_retriever)
        assert callable(strategy.validate_config)
        
        # Should work when called
        result = strategy.create_retriever(config, vectorstore)
        assert result is not None
        
        strategy.validate_config(config)  # Should not raise


class TestRetrieverStrategyInterface:
    """Test the RetrieverStrategy interface compliance."""
    
    def test_interface_consistency(self):
        """Test that the interface is consistent and well-defined."""
        # All abstract methods should be present
        required_methods = {'create_retriever', 'validate_config'}
        abstract_methods = RetrieverStrategy.__abstractmethods__
        
        assert required_methods == abstract_methods
    
    def test_method_return_types(self):
        """Test that method return types are properly documented."""
        import inspect
        
        # create_retriever should return BaseRetriever
        create_retriever_sig = inspect.signature(RetrieverStrategy.create_retriever)
        return_annotation = create_retriever_sig.return_annotation
        
        # Should have return annotation (though it might be string in some cases)
        assert return_annotation is not inspect.Signature.empty
        
        # validate_config should return None
        validate_config_sig = inspect.signature(RetrieverStrategy.validate_config)
        return_annotation = validate_config_sig.return_annotation
        
        # Should have return annotation
        assert return_annotation is not inspect.Signature.empty
    
    def test_parameter_types(self):
        """Test that method parameters have proper type annotations."""
        import inspect
        
        # create_retriever should take RetrieverConfig, VectorStore, and optional LLM
        create_retriever_sig = inspect.signature(RetrieverStrategy.create_retriever)
        config_param = create_retriever_sig.parameters['config']
        vectorstore_param = create_retriever_sig.parameters['vectorstore']
        llm_param = create_retriever_sig.parameters['llm']
        
        assert config_param.annotation is not inspect.Signature.empty
        assert vectorstore_param.annotation is not inspect.Signature.empty
        # llm can be None, so annotation might be empty or Union
        
        # validate_config should take RetrieverConfig
        validate_config_sig = inspect.signature(RetrieverStrategy.validate_config)
        config_param = validate_config_sig.parameters['config']
        
        assert config_param.annotation is not inspect.Signature.empty


class TestRetrieverStrategyEdgeCases:
    """Test edge cases for RetrieverStrategy."""
    
    def test_multiple_inheritance(self):
        """Test RetrieverStrategy with multiple inheritance."""
        class Mixin:
            def extra_method(self):
                return "mixin"
        
        class MultiInheritanceStrategy(RetrieverStrategy, Mixin):
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = MultiInheritanceStrategy()
        assert isinstance(strategy, RetrieverStrategy)
        assert isinstance(strategy, Mixin)
        assert strategy.extra_method() == "mixin"
    
    def test_strategy_with_properties(self):
        """Test RetrieverStrategy with properties."""
        class StrategyWithProperties(RetrieverStrategy):
            def __init__(self):
                self._name = "test"
            
            @property
            def name(self):
                return self._name
            
            @name.setter
            def name(self, value):
                self._name = value
            
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = StrategyWithProperties()
        assert strategy.name == "test"
        
        strategy.name = "new_name"
        assert strategy.name == "new_name"
    
    def test_strategy_with_class_methods(self):
        """Test RetrieverStrategy with class methods."""
        class StrategyWithClassMethods(RetrieverStrategy):
            @classmethod
            def get_supported_types(cls):
                return ["similarity", "mmr"]
            
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = StrategyWithClassMethods()
        types = StrategyWithClassMethods.get_supported_types()
        assert types == ["similarity", "mmr"]
    
    def test_strategy_with_static_methods(self):
        """Test RetrieverStrategy with static methods."""
        class StrategyWithStaticMethods(RetrieverStrategy):
            @staticmethod
            def validate_k_value(k: int) -> bool:
                return k > 0
            
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = StrategyWithStaticMethods()
        assert StrategyWithStaticMethods.validate_k_value(5) is True
        assert StrategyWithStaticMethods.validate_k_value(0) is False
    
    def test_strategy_with_llm_parameter(self):
        """Test RetrieverStrategy with LLM parameter handling."""
        class LLMAwareStrategy(RetrieverStrategy):
            def __init__(self):
                self.llm_used = None
            
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                self.llm_used = llm
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = LLMAwareStrategy()
        config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
        vectorstore = Mock()
        llm = Mock()
        
        # Test with LLM
        result = strategy.create_retriever(config, vectorstore, llm)
        assert result is not None
        assert strategy.llm_used is llm
        
        # Test without LLM
        result = strategy.create_retriever(config, vectorstore)
        assert result is not None
        assert strategy.llm_used is None
    
    def test_strategy_with_vectorstore_parameter(self):
        """Test RetrieverStrategy with VectorStore parameter handling."""
        class VectorStoreAwareStrategy(RetrieverStrategy):
            def __init__(self):
                self.vectorstore_used = None
            
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                self.vectorstore_used = vectorstore
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = VectorStoreAwareStrategy()
        config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
        vectorstore = Mock()
        
        result = strategy.create_retriever(config, vectorstore)
        assert result is not None
        assert strategy.vectorstore_used is vectorstore
    
    def test_strategy_with_config_parameter(self):
        """Test RetrieverStrategy with RetrieverConfig parameter handling."""
        class ConfigAwareStrategy(RetrieverStrategy):
            def __init__(self):
                self.config_used = None
            
            def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
                self.config_used = config
                return Mock()
            
            def validate_config(self, config: RetrieverConfig):
                pass
        
        strategy = ConfigAwareStrategy()
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY, 
            search_kwargs={"k": 10},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        
        result = strategy.create_retriever(config, vectorstore)
        assert result is not None
        assert strategy.config_used is config
        assert strategy.config_used.type == RetrieverType.SIMILARITY
        assert strategy.config_used.search_kwargs["k"] == 10
        assert strategy.config_used.params["custom_param"] == "value"

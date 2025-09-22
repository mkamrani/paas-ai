"""
Unit tests for base splitter strategy interface.

Tests the abstract SplitterStrategy base class including:
- Abstract method enforcement
- Interface compliance
- Error handling for abstract methods
- Type checking and validation
"""

import pytest
from abc import ABC
from unittest.mock import Mock, patch
from langchain_core.documents import Document

from src.paas_ai.core.rag.splitters.base import SplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType


class TestSplitterStrategy:
    """Test the SplitterStrategy abstract base class."""
    
    def test_is_abstract_base_class(self):
        """Test that SplitterStrategy is an abstract base class."""
        assert issubclass(SplitterStrategy, ABC)
    
    def test_cannot_instantiate_directly(self):
        """Test that SplitterStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SplitterStrategy()
    
    def test_has_required_abstract_methods(self):
        """Test that SplitterStrategy has the required abstract methods."""
        abstract_methods = SplitterStrategy.__abstractmethods__
        
        assert 'create_splitter' in abstract_methods
        assert 'validate_config' in abstract_methods
        assert len(abstract_methods) == 2
    
    def test_concrete_implementation_works(self):
        """Test that a concrete implementation of SplitterStrategy works."""
        class ConcreteSplitterStrategy(SplitterStrategy):
            def create_splitter(self, config: SplitterConfig):
                return Mock()
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        # Should be able to instantiate
        strategy = ConcreteSplitterStrategy()
        assert isinstance(strategy, SplitterStrategy)
        
        # Should be able to call methods
        config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
        splitter = strategy.create_splitter(config)
        assert splitter is not None
        
        # Should not raise error
        strategy.validate_config(config)
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementation fails to instantiate."""
        class IncompleteSplitterStrategy(SplitterStrategy):
            def create_splitter(self, config: SplitterConfig):
                return Mock()
            # Missing validate_config method
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteSplitterStrategy()
    
    def test_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        import inspect
        
        # Check create_splitter signature
        create_splitter_sig = inspect.signature(SplitterStrategy.create_splitter)
        assert len(create_splitter_sig.parameters) == 2  # self + config
        assert 'config' in create_splitter_sig.parameters
        
        # Check validate_config signature
        validate_config_sig = inspect.signature(SplitterStrategy.validate_config)
        assert len(validate_config_sig.parameters) == 2  # self + config
        assert 'config' in validate_config_sig.parameters
    
    def test_docstrings_exist(self):
        """Test that abstract methods have docstrings."""
        assert SplitterStrategy.create_splitter.__doc__ is not None
        assert SplitterStrategy.validate_config.__doc__ is not None
        
        # Check that docstrings are meaningful
        assert len(SplitterStrategy.create_splitter.__doc__.strip()) > 10
        assert len(SplitterStrategy.validate_config.__doc__.strip()) > 10
    
    def test_class_docstring(self):
        """Test that the class has a proper docstring."""
        assert SplitterStrategy.__doc__ is not None
        assert "Base strategy interface for text splitters" in SplitterStrategy.__doc__
    
    def test_inheritance_chain(self):
        """Test the inheritance chain of SplitterStrategy."""
        # Should inherit from ABC
        assert ABC in SplitterStrategy.__mro__
        
        # Should be in the MRO
        mro = SplitterStrategy.__mro__
        assert SplitterStrategy in mro
        assert ABC in mro
        assert object in mro
    
    def test_abstract_methods_are_callable(self):
        """Test that abstract methods are callable (though they raise NotImplementedError)."""
        class TestStrategy(SplitterStrategy):
            def create_splitter(self, config: SplitterConfig):
                return Mock()
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = TestStrategy()
        config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
        
        # Should be callable
        assert callable(strategy.create_splitter)
        assert callable(strategy.validate_config)
        
        # Should work when called
        result = strategy.create_splitter(config)
        assert result is not None
        
        strategy.validate_config(config)  # Should not raise


class TestSplitterStrategyInterface:
    """Test the SplitterStrategy interface compliance."""
    
    def test_interface_consistency(self):
        """Test that the interface is consistent and well-defined."""
        # All abstract methods should be present
        required_methods = {'create_splitter', 'validate_config'}
        abstract_methods = SplitterStrategy.__abstractmethods__
        
        assert required_methods == abstract_methods
    
    def test_method_return_types(self):
        """Test that method return types are properly documented."""
        import inspect
        
        # create_splitter should return a splitter object
        create_splitter_sig = inspect.signature(SplitterStrategy.create_splitter)
        return_annotation = create_splitter_sig.return_annotation
        
        # Should have return annotation (though it might be string in some cases)
        assert return_annotation is not inspect.Signature.empty
        
        # validate_config should return None
        validate_config_sig = inspect.signature(SplitterStrategy.validate_config)
        return_annotation = validate_config_sig.return_annotation
        
        # Should have return annotation
        assert return_annotation is not inspect.Signature.empty
    
    def test_parameter_types(self):
        """Test that method parameters have proper type annotations."""
        import inspect
        
        # create_splitter should take SplitterConfig
        create_splitter_sig = inspect.signature(SplitterStrategy.create_splitter)
        config_param = create_splitter_sig.parameters['config']
        
        assert config_param.annotation is not inspect.Signature.empty
        
        # validate_config should take SplitterConfig
        validate_config_sig = inspect.signature(SplitterStrategy.validate_config)
        config_param = validate_config_sig.parameters['config']
        
        assert config_param.annotation is not inspect.Signature.empty


class TestSplitterStrategyEdgeCases:
    """Test edge cases for SplitterStrategy."""
    
    def test_multiple_inheritance(self):
        """Test SplitterStrategy with multiple inheritance."""
        class Mixin:
            def extra_method(self):
                return "mixin"
        
        class MultiInheritanceStrategy(SplitterStrategy, Mixin):
            def create_splitter(self, config: SplitterConfig):
                return Mock()
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = MultiInheritanceStrategy()
        assert isinstance(strategy, SplitterStrategy)
        assert isinstance(strategy, Mixin)
        assert strategy.extra_method() == "mixin"
    
    def test_strategy_with_properties(self):
        """Test SplitterStrategy with properties."""
        class StrategyWithProperties(SplitterStrategy):
            def __init__(self):
                self._name = "test"
            
            @property
            def name(self):
                return self._name
            
            @name.setter
            def name(self, value):
                self._name = value
            
            def create_splitter(self, config: SplitterConfig):
                return Mock()
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = StrategyWithProperties()
        assert strategy.name == "test"
        
        strategy.name = "new_name"
        assert strategy.name == "new_name"
    
    def test_strategy_with_class_methods(self):
        """Test SplitterStrategy with class methods."""
        class StrategyWithClassMethods(SplitterStrategy):
            @classmethod
            def get_supported_types(cls):
                return ["character", "recursive"]
            
            def create_splitter(self, config: SplitterConfig):
                return Mock()
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = StrategyWithClassMethods()
        types = StrategyWithClassMethods.get_supported_types()
        assert types == ["character", "recursive"]
    
    def test_strategy_with_static_methods(self):
        """Test SplitterStrategy with static methods."""
        class StrategyWithStaticMethods(SplitterStrategy):
            @staticmethod
            def validate_chunk_size(chunk_size: int) -> bool:
                return chunk_size > 0
            
            def create_splitter(self, config: SplitterConfig):
                return Mock()
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = StrategyWithStaticMethods()
        assert StrategyWithStaticMethods.validate_chunk_size(1000) is True
        assert StrategyWithStaticMethods.validate_chunk_size(0) is False


class TestSplitterStrategySplitDocuments:
    """Test the split_documents method of SplitterStrategy."""
    
    def test_split_documents_basic(self):
        """Test basic split_documents functionality."""
        class TestStrategy(SplitterStrategy):
            def create_splitter(self, config: SplitterConfig):
                mock_splitter = Mock()
                mock_splitter.split_documents.return_value = [
                    Document(page_content="chunk1"),
                    Document(page_content="chunk2")
                ]
                return mock_splitter
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = TestStrategy()
        config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content="test document")]
        
        result = strategy.split_documents(config, documents)
        
        assert len(result) == 2
        assert result[0].page_content == "chunk1"
        assert result[1].page_content == "chunk2"
    
    def test_split_documents_with_empty_documents(self):
        """Test split_documents with empty document list."""
        class TestStrategy(SplitterStrategy):
            def create_splitter(self, config: SplitterConfig):
                mock_splitter = Mock()
                mock_splitter.split_documents.return_value = []
                return mock_splitter
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = TestStrategy()
        config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
        documents = []
        
        result = strategy.split_documents(config, documents)
        
        assert result == []
    
    def test_split_documents_with_multiple_documents(self):
        """Test split_documents with multiple input documents."""
        class TestStrategy(SplitterStrategy):
            def create_splitter(self, config: SplitterConfig):
                mock_splitter = Mock()
                mock_splitter.split_documents.return_value = [
                    Document(page_content="chunk1"),
                    Document(page_content="chunk2"),
                    Document(page_content="chunk3")
                ]
                return mock_splitter
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = TestStrategy()
        config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
        documents = [
            Document(page_content="document1"),
            Document(page_content="document2")
        ]
        
        result = strategy.split_documents(config, documents)
        
        assert len(result) == 3
        assert all(isinstance(doc, Document) for doc in result)
    
    def test_split_documents_calls_create_splitter(self):
        """Test that split_documents calls create_splitter."""
        class TestStrategy(SplitterStrategy):
            def __init__(self):
                self.create_splitter_called = False
                self.config_passed = None
            
            def create_splitter(self, config: SplitterConfig):
                self.create_splitter_called = True
                self.config_passed = config
                mock_splitter = Mock()
                mock_splitter.split_documents.return_value = []
                return mock_splitter
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = TestStrategy()
        config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content="test")]
        
        strategy.split_documents(config, documents)
        
        assert strategy.create_splitter_called is True
        assert strategy.config_passed == config
    
    def test_split_documents_passes_documents_to_splitter(self):
        """Test that split_documents passes documents to the splitter."""
        class TestStrategy(SplitterStrategy):
            def create_splitter(self, config: SplitterConfig):
                mock_splitter = Mock()
                mock_splitter.split_documents.return_value = []
                return mock_splitter
            
            def validate_config(self, config: SplitterConfig):
                pass
        
        strategy = TestStrategy()
        config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content="test")]
        
        with patch.object(strategy, 'create_splitter') as mock_create:
            mock_splitter = Mock()
            mock_splitter.split_documents.return_value = []
            mock_create.return_value = mock_splitter
            
            strategy.split_documents(config, documents)
            
            mock_splitter.split_documents.assert_called_once_with(documents)

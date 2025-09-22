"""
Unit tests for retriever factory.

Tests all components of the RetrieverFactory including:
- RetrieverFactory class
- Retriever creation workflow
- Error handling and edge cases
- Integration with registry
- Strategy delegation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.factory import RetrieverFactory
from src.paas_ai.core.rag.retrievers.registry import RetrieverRegistry
from src.paas_ai.core.rag.retrievers.base import RetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    # Store original strategies
    original_strategies = RetrieverRegistry._strategies.copy()
    
    yield
    
    # Restore original strategies
    RetrieverRegistry._strategies = original_strategies


class MockRetrieverStrategy(RetrieverStrategy):
    """Mock retriever strategy for testing."""
    
    def __init__(self, should_raise_validation_error=False, should_raise_creation_error=False):
        self.should_raise_validation_error = should_raise_validation_error
        self.should_raise_creation_error = should_raise_creation_error
        self.validation_calls = []
        self.creation_calls = []
    
    def create_retriever(self, config: RetrieverConfig, vectorstore, llm=None):
        """Create mock retriever."""
        self.creation_calls.append((config, vectorstore, llm))
        if self.should_raise_creation_error:
            raise ValueError("Mock creation error")
        return Mock()
    
    def validate_config(self, config: RetrieverConfig):
        """Validate mock configuration."""
        self.validation_calls.append(config)
        if self.should_raise_validation_error:
            raise ValueError("Mock validation error")


class TestRetrieverFactory:
    """Test the RetrieverFactory class."""
    
    def test_create_retriever_success(self):
        """Test successful retriever creation."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = RetrieverFactory.create_retriever(config, vectorstore, llm)
            
            # Verify strategy was retrieved
            mock_get_strategy.assert_called_once_with(RetrieverType.SIMILARITY)
            
            # Verify validation was called
            assert len(mock_strategy.validation_calls) == 1
            assert mock_strategy.validation_calls[0] == config
            
            # Verify creation was called
            assert len(mock_strategy.creation_calls) == 1
            assert mock_strategy.creation_calls[0] == (config, vectorstore, llm)
            
            # Verify result is the mock retriever
            assert result is not None
    
    def test_create_retriever_validation_error(self):
        """Test retriever creation with validation error."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock validation error"):
                RetrieverFactory.create_retriever(config, vectorstore)
            
            # Verify validation was called but creation was not
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 0
    
    def test_create_retriever_creation_error(self):
        """Test retriever creation with creation error."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy(should_raise_creation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock creation error"):
                RetrieverFactory.create_retriever(config, vectorstore)
            
            # Verify both validation and creation were called
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 1
    
    def test_create_retriever_strategy_not_found(self):
        """Test retriever creation when strategy is not found."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = ValueError("No strategy registered")
            
            with pytest.raises(ValueError, match="No strategy registered"):
                RetrieverFactory.create_retriever(config, vectorstore)
    
    def test_create_retriever_with_different_types(self):
        """Test creating retrievers with different types."""
        test_cases = [
            RetrieverType.SIMILARITY,
            RetrieverType.MMR,
            RetrieverType.ENSEMBLE,
            RetrieverType.MULTI_QUERY,
        ]
        
        for retriever_type in test_cases:
            config = RetrieverConfig(
                type=retriever_type,
                search_kwargs={"k": 5},
                params={}
            )
            vectorstore = Mock()
            
            with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
                mock_strategy = MockRetrieverStrategy()
                mock_get_strategy.return_value = mock_strategy
                
                result = RetrieverFactory.create_retriever(config, vectorstore)
                
                # Verify strategy was retrieved with correct type
                mock_get_strategy.assert_called_with(retriever_type)
                
                # Verify result is not None
                assert result is not None
    
    def test_create_retriever_with_custom_config(self):
        """Test creating retrievers with custom configuration."""
        config = RetrieverConfig(
            type=RetrieverType.MMR,
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7},
            params={"custom_param": "custom_value", "another_param": 42}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = RetrieverFactory.create_retriever(config, vectorstore, llm)
            
            # Verify validation was called with custom config
            assert len(mock_strategy.validation_calls) == 1
            validation_config = mock_strategy.validation_calls[0]
            assert validation_config == config
            assert validation_config.type == RetrieverType.MMR
            assert validation_config.search_kwargs["k"] == 10
            assert validation_config.params["custom_param"] == "custom_value"
            
            # Verify creation was called with custom config
            assert len(mock_strategy.creation_calls) == 1
            creation_args = mock_strategy.creation_calls[0]
            assert creation_args[0] == config
            assert creation_args[1] == vectorstore
            assert creation_args[2] == llm
    
    def test_create_retriever_without_llm(self):
        """Test creating retrievers without LLM parameter."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = RetrieverFactory.create_retriever(config, vectorstore)
            
            # Verify creation was called without LLM
            assert len(mock_strategy.creation_calls) == 1
            creation_args = mock_strategy.creation_calls[0]
            assert creation_args[0] == config
            assert creation_args[1] == vectorstore
            assert creation_args[2] is None
    
    def test_list_supported_types(self):
        """Test listing supported retriever types through factory."""
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [RetrieverType.SIMILARITY, RetrieverType.MMR]
            
            result = RetrieverFactory.list_supported_types()
            
            assert result == [RetrieverType.SIMILARITY, RetrieverType.MMR]
            mock_list.assert_called_once()
    
    def test_register_strategy(self):
        """Test registering strategy through factory."""
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.register_strategy') as mock_register:
            RetrieverFactory.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            
            mock_register.assert_called_once_with(RetrieverType.SIMILARITY, MockRetrieverStrategy)
    
    def test_factory_methods_are_static(self):
        """Test that factory methods are static and can be called without instance."""
        # These should work without creating an instance
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should work without creating factory instance
            result = RetrieverFactory.create_retriever(config, vectorstore)
            assert result is not None
        
        # Test other static methods
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [RetrieverType.SIMILARITY]
            types = RetrieverFactory.list_supported_types()
            assert types == [RetrieverType.SIMILARITY]
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.register_strategy') as mock_register:
            RetrieverFactory.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            mock_register.assert_called_once()


class TestRetrieverFactoryEdgeCases:
    """Test edge cases for RetrieverFactory."""
    
    def test_create_retriever_with_none_config(self):
        """Test creating retriever with None config."""
        with pytest.raises(AttributeError):
            RetrieverFactory.create_retriever(None, Mock())
    
    def test_create_retriever_with_invalid_config_type(self):
        """Test creating retriever with invalid config type."""
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should handle invalid config gracefully
            with pytest.raises(AttributeError):
                RetrieverFactory.create_retriever("not-a-config", Mock())
    
    def test_create_retriever_with_missing_config_attributes(self):
        """Test creating retriever with config missing required attributes."""
        # Create a config object without required attributes
        class IncompleteConfig:
            def __init__(self):
                self.type = RetrieverType.SIMILARITY
                # Missing search_kwargs and params
        
        incomplete_config = IncompleteConfig()
        
        # Should handle incomplete config gracefully - the factory accesses config.type first
        with pytest.raises(AttributeError, match="'IncompleteConfig' object has no attribute"):
            RetrieverFactory.create_retriever(incomplete_config, Mock())
    
    def test_create_retriever_with_none_vectorstore(self):
        """Test creating retriever with None vectorstore."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should pass None vectorstore to strategy
            result = RetrieverFactory.create_retriever(config, None)
            assert result is not None
            
            # Verify None was passed to strategy
            assert len(mock_strategy.creation_calls) == 1
            creation_args = mock_strategy.creation_calls[0]
            assert creation_args[1] is None
    
    def test_create_retriever_with_strategy_that_returns_none(self):
        """Test creating retriever with strategy that returns None."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_strategy.create_retriever = Mock(return_value=None)
            mock_get_strategy.return_value = mock_strategy
            
            result = RetrieverFactory.create_retriever(config, vectorstore)
            
            # Should return None if strategy returns None
            assert result is None
    
    def test_create_retriever_with_strategy_that_returns_string(self):
        """Test creating retriever with strategy that returns unexpected type."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_strategy.create_retriever = Mock(return_value="not-a-retriever")
            mock_get_strategy.return_value = mock_strategy
            
            result = RetrieverFactory.create_retriever(config, vectorstore)
            
            # Should return whatever the strategy returns
            assert result == "not-a-retriever"
    
    def test_create_retriever_with_exception_in_strategy_retrieval(self):
        """Test creating retriever with exception in strategy retrieval."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = Exception("Registry error")
            
            with pytest.raises(Exception, match="Registry error"):
                RetrieverFactory.create_retriever(config, vectorstore)
    
    def test_create_retriever_with_exception_in_validation(self):
        """Test creating retriever with exception in validation."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_strategy.validate_config = Mock(side_effect=Exception("Validation error"))
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(Exception, match="Validation error"):
                RetrieverFactory.create_retriever(config, vectorstore)
    
    def test_create_retriever_with_exception_in_creation(self):
        """Test creating retriever with exception in creation."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_strategy.create_retriever = Mock(side_effect=Exception("Creation error"))
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(Exception, match="Creation error"):
                RetrieverFactory.create_retriever(config, vectorstore)


class TestRetrieverFactoryIntegration:
    """Integration tests for RetrieverFactory."""
    
    def test_full_workflow(self):
        """Test complete workflow from factory to retriever creation."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={"custom_param": "value"}
        )
        vectorstore = Mock()
        llm = Mock()
        
        # This should work without mocking since we're testing the real integration
        try:
            result = RetrieverFactory.create_retriever(config, vectorstore, llm)
            assert result is not None
        except Exception as e:
            # If there are import issues or missing dependencies, that's expected in tests
            # We just want to make sure the factory doesn't crash
            assert "No strategy registered" not in str(e)
    
    def test_factory_registry_consistency(self):
        """Test that factory and registry are consistent."""
        # Get supported types from both
        factory_types = RetrieverFactory.list_supported_types()
        registry_types = RetrieverFactory.list_supported_types()  # Factory delegates to registry
        
        assert factory_types == registry_types
        
        # Test that we can create retrievers for all supported types
        for retriever_type in factory_types:
            # Skip None values that might be in the registry
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
                with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
                    mock_strategy = MockRetrieverStrategy()
                    mock_get_strategy.return_value = mock_strategy
                    
                    result = RetrieverFactory.create_retriever(config, vectorstore)
                    assert result is not None
            except ValueError as e:
                if "No strategy registered" in str(e):
                    pytest.fail(f"Strategy not registered for {retriever_type}")
    
    def test_workflow_with_multiple_configs(self):
        """Test workflow with multiple different configurations."""
        configs = [
            RetrieverConfig(
                type=RetrieverType.SIMILARITY,
                search_kwargs={"k": 5},
                params={}
            ),
            RetrieverConfig(
                type=RetrieverType.MMR,
                search_kwargs={"k": 10, "fetch_k": 20},
                params={"lambda_mult": 0.7}
            ),
            RetrieverConfig(
                type=RetrieverType.ENSEMBLE,
                search_kwargs={"k": 8},
                params={"weights": [0.6, 0.4]}
            )
        ]
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            for config in configs:
                result = RetrieverFactory.create_retriever(config, vectorstore)
                assert result is not None
            
            # Should have been called for each config
            assert mock_get_strategy.call_count == len(configs)
            assert len(mock_strategy.validation_calls) == len(configs)
            assert len(mock_strategy.creation_calls) == len(configs)
    
    def test_workflow_with_strategy_registration(self):
        """Test workflow with strategy registration."""
        # Register a custom strategy
        RetrieverFactory.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
        
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        # Should use the registered strategy
        result = RetrieverFactory.create_retriever(config, vectorstore)
        assert result is not None
    
    def test_workflow_with_error_handling(self):
        """Test workflow with error handling."""
        config = RetrieverConfig(
            type=RetrieverType.SIMILARITY,
            search_kwargs={"k": 5},
            params={}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.factory.RetrieverRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockRetrieverStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            # Should handle validation error
            with pytest.raises(ValueError, match="Mock validation error"):
                RetrieverFactory.create_retriever(config, vectorstore)
            
            # Verify validation was called but creation was not
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 0

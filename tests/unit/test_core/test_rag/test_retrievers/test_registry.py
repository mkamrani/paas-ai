"""
Unit tests for retriever strategy registry.

Tests all components of the RetrieverRegistry including:
- RetrieverRegistry class
- Strategy registration and retrieval
- Error handling and edge cases
- Integration with different retriever types
- Strategy instance management
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

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


class TestRetrieverRegistry:
    """Test the RetrieverRegistry class."""
    
    def test_get_strategy_existing_type(self):
        """Test getting strategy for existing retriever type."""
        strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
        
        assert strategy is not None
        assert hasattr(strategy, 'create_retriever')
        assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_another_existing_type(self):
        """Test getting strategy for another existing retriever type."""
        strategy = RetrieverRegistry.get_strategy(RetrieverType.MMR)
        
        assert strategy is not None
        assert hasattr(strategy, 'create_retriever')
        assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_all_existing_types(self):
        """Test getting strategy for all existing retriever types."""
        existing_types = [
            RetrieverType.SIMILARITY,
            RetrieverType.MMR,
            RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            RetrieverType.ENSEMBLE,
            RetrieverType.MULTI_QUERY,
            RetrieverType.PARENT_DOCUMENT,
        ]
        
        for retriever_type in existing_types:
            strategy = RetrieverRegistry.get_strategy(retriever_type)
            assert strategy is not None
            assert hasattr(strategy, 'create_retriever')
            assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_nonexistent_type(self):
        """Test getting strategy for non-existent retriever type."""
        with pytest.raises(ValueError, match="No strategy registered for retriever type"):
            RetrieverRegistry.get_strategy("nonexistent_type")
    
    def test_get_strategy_with_none_type(self):
        """Test getting strategy with None type."""
        with pytest.raises(ValueError, match="No strategy registered for retriever type"):
            RetrieverRegistry.get_strategy(None)
    
    def test_register_strategy(self):
        """Test registering a custom strategy."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            
            # Get the strategy and verify it's our custom one
            strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy, MockRetrieverStrategy)
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_register_strategy_overwrite(self):
        """Test registering a strategy overwrites existing one."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            
            # Verify it's registered
            strategy1 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy1, MockRetrieverStrategy)
            
            # Register a different strategy
            class AnotherMockStrategy(MockRetrieverStrategy):
                pass
            
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, AnotherMockStrategy)
            
            # Verify it's overwritten
            strategy2 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy2, AnotherMockStrategy)
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_register_strategy_with_none_type(self):
        """Test registering strategy with None type."""
        # Store original strategy for cleanup
        original_strategy = RetrieverRegistry._strategies.get(RetrieverType.SIMILARITY)
        
        try:
            # The registry doesn't validate None types, it just stores them
            # This is the current behavior - no validation
            RetrieverRegistry.register_strategy(None, MockRetrieverStrategy)
            # Should not raise an error
        finally:
            # Clean up by removing the None entry and restoring original
            if None in RetrieverRegistry._strategies:
                del RetrieverRegistry._strategies[None]
            if original_strategy:
                RetrieverRegistry._strategies[RetrieverType.SIMILARITY] = original_strategy
    
    def test_register_strategy_with_none_class(self):
        """Test registering strategy with None class."""
        # Store original strategy for cleanup
        original_strategy = RetrieverRegistry._strategies.get(RetrieverType.SIMILARITY)
        
        try:
            # The registry doesn't validate None classes, it just stores them
            # This is the current behavior - no validation
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, None)
            # Should not raise an error
        finally:
            # Restore original strategy
            if original_strategy:
                RetrieverRegistry._strategies[RetrieverType.SIMILARITY] = original_strategy
    
    def test_list_supported_types(self):
        """Test listing all supported retriever types."""
        supported_types = RetrieverRegistry.list_supported_types()
        
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        
        # Check that expected types are present
        expected_types = {
            RetrieverType.SIMILARITY,
            RetrieverType.MMR,
            RetrieverType.SIMILARITY_SCORE_THRESHOLD,
            RetrieverType.ENSEMBLE,
            RetrieverType.MULTI_QUERY,
            RetrieverType.PARENT_DOCUMENT,
        }
        
        # Should contain at least the expected types
        assert expected_types.issubset(set(supported_types))
    
    def test_get_strategy_returns_new_instance(self):
        """Test that get_strategy returns a new instance each time."""
        # Store original strategy
        original_strategy = RetrieverRegistry._strategies.get(RetrieverType.SIMILARITY)
        
        try:
            strategy1 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            strategy2 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            
            # Should be different instances
            assert strategy1 is not strategy2
            # But should be of the same class
            assert type(strategy1) == type(strategy2)
        finally:
            # Restore original strategy
            if original_strategy:
                RetrieverRegistry._strategies[RetrieverType.SIMILARITY] = original_strategy
    
    def test_get_strategy_different_types_different_instances(self):
        """Test that different types return different strategy instances."""
        # Store original strategies
        original_similarity = RetrieverRegistry._strategies.get(RetrieverType.SIMILARITY)
        original_mmr = RetrieverRegistry._strategies.get(RetrieverType.MMR)
        
        try:
            similarity_strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            mmr_strategy = RetrieverRegistry.get_strategy(RetrieverType.MMR)
            
            # Should be different instances and different types
            assert similarity_strategy is not mmr_strategy
            assert type(similarity_strategy) != type(mmr_strategy)
        finally:
            # Restore original strategies
            if original_similarity:
                RetrieverRegistry._strategies[RetrieverType.SIMILARITY] = original_similarity
            if original_mmr:
                RetrieverRegistry._strategies[RetrieverType.MMR] = original_mmr
    
    def test_registry_persistence(self):
        """Test that registry changes persist across calls."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            
            # Verify it's registered
            strategy1 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy1, MockRetrieverStrategy)
            
            # Get it again - should still be registered
            strategy2 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy2, MockRetrieverStrategy)
            
            # Verify it's in the supported types
            supported_types = RetrieverRegistry.list_supported_types()
            assert RetrieverType.SIMILARITY in supported_types
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies


class TestRetrieverRegistryEdgeCases:
    """Test edge cases for RetrieverRegistry."""
    
    def test_register_strategy_with_invalid_class(self):
        """Test registering strategy with invalid class."""
        class NotAStrategy:
            pass
        
        # The registry doesn't validate class types, it just stores them
        # This is the current behavior - no validation
        RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, NotAStrategy)
        # Should not raise an error
    
    def test_get_strategy_with_invalid_type(self):
        """Test getting strategy with invalid type."""
        with pytest.raises(ValueError, match="No strategy registered for retriever type"):
            RetrieverRegistry.get_strategy("invalid_type")
    
    def test_get_strategy_with_empty_string_type(self):
        """Test getting strategy with empty string type."""
        with pytest.raises(ValueError, match="No strategy registered for retriever type"):
            RetrieverRegistry.get_strategy("")
    
    def test_register_strategy_multiple_times(self):
        """Test registering the same strategy multiple times."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register the same strategy multiple times
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            
            # Should still work
            strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy, MockRetrieverStrategy)
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_register_strategy_with_different_types_same_class(self):
        """Test registering the same class for different types."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register the same class for different types
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            RetrieverRegistry.register_strategy(RetrieverType.MMR, MockRetrieverStrategy)
            
            # Both should work
            similarity_strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            mmr_strategy = RetrieverRegistry.get_strategy(RetrieverType.MMR)
            
            assert isinstance(similarity_strategy, MockRetrieverStrategy)
            assert isinstance(mmr_strategy, MockRetrieverStrategy)
            
            # Should be different instances
            assert similarity_strategy is not mmr_strategy
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_list_supported_types_after_registration(self):
        """Test that list_supported_types reflects new registrations."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Get initial supported types
            initial_types = set(RetrieverRegistry.list_supported_types())
            
            # Register a new strategy for a new type
            class NewRetrieverType:
                def __str__(self):
                    return "new_type"
            
            new_type = NewRetrieverType()
            RetrieverRegistry.register_strategy(new_type, MockRetrieverStrategy)
            
            # Get updated supported types
            updated_types = set(RetrieverRegistry.list_supported_types())
            
            # Should include the new type
            assert new_type in updated_types
            assert len(updated_types) == len(initial_types) + 1
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_strategy_instance_isolation(self):
        """Test that strategy instances are isolated."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register a strategy that tracks calls
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            
            # Get two instances
            strategy1 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            strategy2 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            
            # They should be different instances
            assert strategy1 is not strategy2
            
            # Test that calls to one don't affect the other
            config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
            vectorstore = Mock()
            
            strategy1.validate_config(config)
            strategy1.create_retriever(config, vectorstore)
            
            # strategy2 should not have any calls
            assert len(strategy2.validation_calls) == 0
            assert len(strategy2.creation_calls) == 0
            
            # strategy1 should have the calls
            assert len(strategy1.validation_calls) == 1
            assert len(strategy1.creation_calls) == 1
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies


class TestRetrieverRegistryIntegration:
    """Integration tests for RetrieverRegistry."""
    
    def test_full_workflow_with_custom_strategy(self):
        """Test complete workflow with custom strategy."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register custom strategy
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            
            # Get strategy
            strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy, MockRetrieverStrategy)
            
            # Use strategy
            config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
            vectorstore = Mock()
            strategy.validate_config(config)
            retriever = strategy.create_retriever(config, vectorstore)
            
            assert retriever is not None
            assert len(strategy.validation_calls) == 1
            assert len(strategy.creation_calls) == 1
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_multiple_strategies_workflow(self):
        """Test workflow with multiple different strategies."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register different strategies for different types
            class StrategyA(MockRetrieverStrategy):
                pass
            
            class StrategyB(MockRetrieverStrategy):
                pass
            
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, StrategyA)
            RetrieverRegistry.register_strategy(RetrieverType.MMR, StrategyB)
            
            # Get strategies
            similarity_strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            mmr_strategy = RetrieverRegistry.get_strategy(RetrieverType.MMR)
            
            assert isinstance(similarity_strategy, StrategyA)
            assert isinstance(mmr_strategy, StrategyB)
            
            # Use both strategies
            similarity_config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
            mmr_config = RetrieverConfig(type=RetrieverType.MMR, search_kwargs={"k": 10})
            vectorstore = Mock()
            
            similarity_strategy.validate_config(similarity_config)
            mmr_strategy.validate_config(mmr_config)
            
            assert len(similarity_strategy.validation_calls) == 1
            assert len(mmr_strategy.validation_calls) == 1
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_strategy_replacement_workflow(self):
        """Test workflow with strategy replacement."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register initial strategy
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, MockRetrieverStrategy)
            strategy1 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy1, MockRetrieverStrategy)
            
            # Replace with different strategy
            class ReplacementStrategy(MockRetrieverStrategy):
                pass
            
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, ReplacementStrategy)
            strategy2 = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy2, ReplacementStrategy)
            
            # Should be different instances
            assert strategy1 is not strategy2
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies
    
    def test_error_handling_workflow(self):
        """Test workflow with error handling."""
        original_strategies = RetrieverRegistry._strategies.copy()
        
        try:
            # Register strategy that raises errors
            class ErrorStrategyClass(MockRetrieverStrategy):
                def __init__(self):
                    super().__init__(
                        should_raise_validation_error=True,
                        should_raise_creation_error=True
                    )
            
            RetrieverRegistry.register_strategy(RetrieverType.SIMILARITY, ErrorStrategyClass)
            
            # Get strategy
            strategy = RetrieverRegistry.get_strategy(RetrieverType.SIMILARITY)
            assert isinstance(strategy, ErrorStrategyClass)
            
            # Test validation error
            config = RetrieverConfig(type=RetrieverType.SIMILARITY, search_kwargs={"k": 5})
            with pytest.raises(ValueError, match="Mock validation error"):
                strategy.validate_config(config)
            
            # Test creation error
            vectorstore = Mock()
            with pytest.raises(ValueError, match="Mock creation error"):
                strategy.create_retriever(config, vectorstore)
            
        finally:
            # Restore original strategies
            RetrieverRegistry._strategies = original_strategies

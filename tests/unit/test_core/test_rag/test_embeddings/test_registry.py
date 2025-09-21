"""
Unit tests for embedding strategy registry.

Tests all components of the EmbeddingRegistry including:
- EmbeddingRegistry class
- Strategy registration and retrieval
- Error handling and edge cases
- Integration with different embedding types
- Strategy instance management
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.embeddings.registry import EmbeddingRegistry
from src.paas_ai.core.rag.embeddings.base import EmbeddingStrategy
from src.paas_ai.core.rag.config import EmbeddingConfig, EmbeddingType


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before and after each test."""
    # Reset to defaults before each test
    EmbeddingRegistry.reset_to_defaults()
    
    yield
    
    # Reset to defaults after each test
    EmbeddingRegistry.reset_to_defaults()


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


class TestEmbeddingRegistry:
    """Test the EmbeddingRegistry class."""
    
    def test_get_strategy_existing_type(self):
        """Test getting strategy for existing embedding type."""
        strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
        
        assert strategy is not None
        assert hasattr(strategy, 'create_embeddings')
        assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_another_existing_type(self):
        """Test getting strategy for another existing embedding type."""
        strategy = EmbeddingRegistry.get_strategy(EmbeddingType.SENTENCE_TRANSFORMERS)
        
        assert strategy is not None
        assert hasattr(strategy, 'create_embeddings')
        assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_nonexistent_type(self):
        """Test getting strategy for non-existent embedding type."""
        with pytest.raises(ValueError, match="No strategy registered for embedding type"):
            EmbeddingRegistry.get_strategy("nonexistent_type")
    
    def test_get_strategy_with_none_type(self):
        """Test getting strategy with None type."""
        with pytest.raises(ValueError, match="No strategy registered for embedding type"):
            EmbeddingRegistry.get_strategy(None)
    
    def test_register_strategy(self):
        """Test registering a custom strategy."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            
            # Get the strategy and verify it's our custom one
            strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy, MockEmbeddingStrategy)
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_register_strategy_overwrite(self):
        """Test registering a strategy overwrites existing one."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            
            # Verify it's registered
            strategy1 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy1, MockEmbeddingStrategy)
            
            # Register a different strategy
            class AnotherMockStrategy(MockEmbeddingStrategy):
                pass
            
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, AnotherMockStrategy)
            
            # Verify it's overwritten
            strategy2 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy2, AnotherMockStrategy)
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_register_strategy_with_none_type(self):
        """Test registering strategy with None type."""
        # Store original strategy for cleanup
        original_strategy = EmbeddingRegistry._strategies.get(EmbeddingType.OPENAI)
        
        try:
            # The registry doesn't validate None types, it just stores them
            # This is the current behavior - no validation
            EmbeddingRegistry.register_strategy(None, MockEmbeddingStrategy)
            # Should not raise an error
        finally:
            # Clean up by removing the None entry and restoring original
            if None in EmbeddingRegistry._strategies:
                del EmbeddingRegistry._strategies[None]
            if original_strategy:
                EmbeddingRegistry._strategies[EmbeddingType.OPENAI] = original_strategy
    
    def test_register_strategy_with_none_class(self):
        """Test registering strategy with None class."""
        # Store original strategy for cleanup
        original_strategy = EmbeddingRegistry._strategies.get(EmbeddingType.OPENAI)
        
        try:
            # The registry doesn't validate None classes, it just stores them
            # This is the current behavior - no validation
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, None)
            # Should not raise an error
        finally:
            # Restore original strategy
            if original_strategy:
                EmbeddingRegistry._strategies[EmbeddingType.OPENAI] = original_strategy
    
    def test_list_supported_types(self):
        """Test listing all supported embedding types."""
        supported_types = EmbeddingRegistry.list_supported_types()
        
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        
        # Check that expected types are present
        expected_types = {
            EmbeddingType.OPENAI,
            EmbeddingType.SENTENCE_TRANSFORMERS,
        }
        
        # Should contain at least the expected types
        assert expected_types.issubset(set(supported_types))
    
    def test_get_strategy_returns_new_instance(self):
        """Test that get_strategy returns a new instance each time."""
        # Store original strategy
        original_strategy = EmbeddingRegistry._strategies.get(EmbeddingType.OPENAI)
        
        try:
            strategy1 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            strategy2 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            
            # Should be different instances
            assert strategy1 is not strategy2
            # But should be of the same class
            assert type(strategy1) == type(strategy2)
        finally:
            # Restore original strategy
            if original_strategy:
                EmbeddingRegistry._strategies[EmbeddingType.OPENAI] = original_strategy
    
    def test_get_strategy_different_types_different_instances(self):
        """Test that different types return different strategy instances."""
        # Store original strategies
        original_openai = EmbeddingRegistry._strategies.get(EmbeddingType.OPENAI)
        original_sentence_transformers = EmbeddingRegistry._strategies.get(EmbeddingType.SENTENCE_TRANSFORMERS)
        
        try:
            openai_strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            sentence_transformers_strategy = EmbeddingRegistry.get_strategy(EmbeddingType.SENTENCE_TRANSFORMERS)
            
            # Should be different instances and different types
            assert openai_strategy is not sentence_transformers_strategy
            assert type(openai_strategy) != type(sentence_transformers_strategy)
        finally:
            # Restore original strategies
            if original_openai:
                EmbeddingRegistry._strategies[EmbeddingType.OPENAI] = original_openai
            if original_sentence_transformers:
                EmbeddingRegistry._strategies[EmbeddingType.SENTENCE_TRANSFORMERS] = original_sentence_transformers
    
    def test_registry_persistence(self):
        """Test that registry changes persist across calls."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            
            # Verify it's registered
            strategy1 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy1, MockEmbeddingStrategy)
            
            # Get it again - should still be registered
            strategy2 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy2, MockEmbeddingStrategy)
            
            # Verify it's in the supported types
            supported_types = EmbeddingRegistry.list_supported_types()
            assert EmbeddingType.OPENAI in supported_types
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies


class TestEmbeddingRegistryEdgeCases:
    """Test edge cases for EmbeddingRegistry."""
    
    def test_register_strategy_with_invalid_class(self):
        """Test registering strategy with invalid class."""
        class NotAStrategy:
            pass
        
        # The registry doesn't validate class types, it just stores them
        # This is the current behavior - no validation
        EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, NotAStrategy)
        # Should not raise an error
    
    def test_get_strategy_with_invalid_type(self):
        """Test getting strategy with invalid type."""
        with pytest.raises(ValueError, match="No strategy registered for embedding type"):
            EmbeddingRegistry.get_strategy("invalid_type")
    
    def test_get_strategy_with_empty_string_type(self):
        """Test getting strategy with empty string type."""
        with pytest.raises(ValueError, match="No strategy registered for embedding type"):
            EmbeddingRegistry.get_strategy("")
    
    def test_register_strategy_multiple_times(self):
        """Test registering the same strategy multiple times."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register the same strategy multiple times
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            
            # Should still work
            strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy, MockEmbeddingStrategy)
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_register_strategy_with_different_types_same_class(self):
        """Test registering the same class for different types."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register the same class for different types
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            EmbeddingRegistry.register_strategy(EmbeddingType.SENTENCE_TRANSFORMERS, MockEmbeddingStrategy)
            
            # Both should work
            openai_strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            sentence_transformers_strategy = EmbeddingRegistry.get_strategy(EmbeddingType.SENTENCE_TRANSFORMERS)
            
            assert isinstance(openai_strategy, MockEmbeddingStrategy)
            assert isinstance(sentence_transformers_strategy, MockEmbeddingStrategy)
            
            # Should be different instances
            assert openai_strategy is not sentence_transformers_strategy
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_list_supported_types_after_registration(self):
        """Test that list_supported_types reflects new registrations."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Get initial supported types
            initial_types = set(EmbeddingRegistry.list_supported_types())
            
            # Register a new strategy for a new type
            class NewEmbeddingType:
                def __str__(self):
                    return "new_type"
            
            new_type = NewEmbeddingType()
            EmbeddingRegistry.register_strategy(new_type, MockEmbeddingStrategy)
            
            # Get updated supported types
            updated_types = set(EmbeddingRegistry.list_supported_types())
            
            # Should include the new type
            assert new_type in updated_types
            assert len(updated_types) == len(initial_types) + 1
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_strategy_instance_isolation(self):
        """Test that strategy instances are isolated."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register a strategy that tracks calls
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            
            # Get two instances
            strategy1 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            strategy2 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            
            # They should be different instances
            assert strategy1 is not strategy2
            
            # Test that calls to one don't affect the other
            config = EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="test-model")
            
            strategy1.validate_config(config)
            strategy1.create_embeddings(config)
            
            # strategy2 should not have any calls
            assert len(strategy2.validation_calls) == 0
            assert len(strategy2.creation_calls) == 0
            
            # strategy1 should have the calls
            assert len(strategy1.validation_calls) == 1
            assert len(strategy1.creation_calls) == 1
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies


class TestEmbeddingRegistryIntegration:
    """Integration tests for EmbeddingRegistry."""
    
    def test_full_workflow_with_custom_strategy(self):
        """Test complete workflow with custom strategy."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register custom strategy
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            
            # Get strategy
            strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy, MockEmbeddingStrategy)
            
            # Use strategy
            config = EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="test-model")
            strategy.validate_config(config)
            embeddings = strategy.create_embeddings(config)
            
            assert embeddings is not None
            assert len(strategy.validation_calls) == 1
            assert len(strategy.creation_calls) == 1
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_multiple_strategies_workflow(self):
        """Test workflow with multiple different strategies."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register different strategies for different types
            class StrategyA(MockEmbeddingStrategy):
                pass
            
            class StrategyB(MockEmbeddingStrategy):
                pass
            
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, StrategyA)
            EmbeddingRegistry.register_strategy(EmbeddingType.SENTENCE_TRANSFORMERS, StrategyB)
            
            # Get strategies
            openai_strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            sentence_transformers_strategy = EmbeddingRegistry.get_strategy(EmbeddingType.SENTENCE_TRANSFORMERS)
            
            assert isinstance(openai_strategy, StrategyA)
            assert isinstance(sentence_transformers_strategy, StrategyB)
            
            # Use both strategies
            openai_config = EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="test-model")
            sentence_transformers_config = EmbeddingConfig(type=EmbeddingType.SENTENCE_TRANSFORMERS, model_name="test-model")
            
            openai_strategy.validate_config(openai_config)
            sentence_transformers_strategy.validate_config(sentence_transformers_config)
            
            assert len(openai_strategy.validation_calls) == 1
            assert len(sentence_transformers_strategy.validation_calls) == 1
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_strategy_replacement_workflow(self):
        """Test workflow with strategy replacement."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register initial strategy
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, MockEmbeddingStrategy)
            strategy1 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy1, MockEmbeddingStrategy)
            
            # Replace with different strategy
            class ReplacementStrategy(MockEmbeddingStrategy):
                pass
            
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, ReplacementStrategy)
            strategy2 = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy2, ReplacementStrategy)
            
            # Should be different instances
            assert strategy1 is not strategy2
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies
    
    def test_error_handling_workflow(self):
        """Test workflow with error handling."""
        original_strategies = EmbeddingRegistry._strategies.copy()
        
        try:
            # Register strategy that raises errors
            error_strategy = MockEmbeddingStrategy(
                should_raise_validation_error=True,
                should_raise_creation_error=True
            )
            
            class ErrorStrategyClass(MockEmbeddingStrategy):
                def __init__(self):
                    super().__init__(
                        should_raise_validation_error=True,
                        should_raise_creation_error=True
                    )
            
            EmbeddingRegistry.register_strategy(EmbeddingType.OPENAI, ErrorStrategyClass)
            
            # Get strategy
            strategy = EmbeddingRegistry.get_strategy(EmbeddingType.OPENAI)
            assert isinstance(strategy, ErrorStrategyClass)
            
            # Test validation error
            config = EmbeddingConfig(type=EmbeddingType.OPENAI, model_name="test-model")
            with pytest.raises(ValueError, match="Mock validation error"):
                strategy.validate_config(config)
            
            # Test creation error
            with pytest.raises(ValueError, match="Mock creation error"):
                strategy.create_embeddings(config)
            
        finally:
            # Restore original strategies
            EmbeddingRegistry._strategies = original_strategies

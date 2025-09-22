"""
Unit tests for splitter strategy registry.

Tests all components of the SplitterRegistry including:
- SplitterRegistry class
- Strategy registration and retrieval
- Error handling and edge cases
- Integration with different splitter types
- Strategy instance management
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.splitters.registry import SplitterRegistry
from src.paas_ai.core.rag.splitters.base import SplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType




class MockSplitterStrategy(SplitterStrategy):
    """Mock splitter strategy for testing."""
    
    def __init__(self, should_raise_validation_error=False, should_raise_creation_error=False):
        self.should_raise_validation_error = should_raise_validation_error
        self.should_raise_creation_error = should_raise_creation_error
        self.validation_calls = []
        self.creation_calls = []
        self.split_documents_calls = []
        # Create a mock for split_documents method
        self.split_documents = Mock()
    
    def create_splitter(self, config: SplitterConfig):
        """Create mock splitter."""
        self.creation_calls.append(config)
        if self.should_raise_creation_error:
            raise ValueError("Mock creation error")
        return Mock()
    
    def validate_config(self, config: SplitterConfig):
        """Validate mock configuration."""
        self.validation_calls.append(config)
        if self.should_raise_validation_error:
            raise ValueError("Mock validation error")


class TestSplitterRegistry:
    """Test the SplitterRegistry class."""
    
    def test_get_strategy_existing_type(self):
        """Test getting strategy for existing splitter type."""
        strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
        
        assert strategy is not None
        assert hasattr(strategy, 'create_splitter')
        assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_another_existing_type(self):
        """Test getting strategy for another existing splitter type."""
        strategy = SplitterRegistry.get_strategy(SplitterType.RECURSIVE_CHARACTER)
        
        assert strategy is not None
        assert hasattr(strategy, 'create_splitter')
        assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_nonexistent_type(self):
        """Test getting strategy for non-existent splitter type."""
        with pytest.raises(ValueError, match="No strategy registered for splitter type"):
            SplitterRegistry.get_strategy("nonexistent_type")
    
    def test_get_strategy_with_none_type(self):
        """Test getting strategy with None type."""
        with pytest.raises(ValueError, match="No strategy registered for splitter type"):
            SplitterRegistry.get_strategy(None)
    
    def test_register_strategy(self):
        """Test registering a custom strategy."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            
            # Get the strategy and verify it's our custom one
            strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy, MockSplitterStrategy)
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_register_strategy_overwrite(self):
        """Test registering a strategy overwrites existing one."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            
            # Verify it's registered
            strategy1 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy1, MockSplitterStrategy)
            
            # Register a different strategy
            class AnotherMockStrategy(MockSplitterStrategy):
                pass
            
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, AnotherMockStrategy)
            
            # Verify it's overwritten
            strategy2 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy2, AnotherMockStrategy)
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_register_strategy_with_none_type(self):
        """Test registering strategy with None type."""
        # Store original strategy for cleanup
        original_strategy = SplitterRegistry._strategies.get(SplitterType.CHARACTER)
        
        try:
            # The registry doesn't validate None types, it just stores them
            # This is the current behavior - no validation
            SplitterRegistry.register_strategy(None, MockSplitterStrategy)
            # Should not raise an error
        finally:
            # Clean up by removing the None entry and restoring original
            if None in SplitterRegistry._strategies:
                del SplitterRegistry._strategies[None]
            if original_strategy:
                SplitterRegistry._strategies[SplitterType.CHARACTER] = original_strategy
    
    def test_register_strategy_with_none_class(self):
        """Test registering strategy with None class."""
        # Store original strategy for cleanup
        original_strategy = SplitterRegistry._strategies.get(SplitterType.CHARACTER)
        
        try:
            # The registry doesn't validate None classes, it just stores them
            # This is the current behavior - no validation
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, None)
            # Should not raise an error
        finally:
            # Restore original strategy
            if original_strategy:
                SplitterRegistry._strategies[SplitterType.CHARACTER] = original_strategy
    
    def test_list_supported_types(self):
        """Test listing all supported splitter types."""
        supported_types = SplitterRegistry.list_supported_types()
        
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        
        # Check that expected types are present
        expected_types = {
            SplitterType.CHARACTER,
            SplitterType.RECURSIVE_CHARACTER,
            SplitterType.MARKDOWN,
            SplitterType.HTML,
            SplitterType.JSON,
            SplitterType.CODE,
            SplitterType.TOKEN,
        }
        
        # Should contain at least the expected types
        assert expected_types.issubset(set(supported_types))
    
    def test_get_strategy_returns_new_instance(self):
        """Test that get_strategy returns a new instance each time."""
        # Store original strategy
        original_strategy = SplitterRegistry._strategies.get(SplitterType.CHARACTER)
        
        try:
            strategy1 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            strategy2 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            
            # Should be different instances
            assert strategy1 is not strategy2
            # But should be of the same class
            assert type(strategy1) == type(strategy2)
        finally:
            # Restore original strategy
            if original_strategy:
                SplitterRegistry._strategies[SplitterType.CHARACTER] = original_strategy
    
    def test_get_strategy_different_types_different_instances(self):
        """Test that different types return different strategy instances."""
        # Store original strategies
        original_character = SplitterRegistry._strategies.get(SplitterType.CHARACTER)
        original_recursive = SplitterRegistry._strategies.get(SplitterType.RECURSIVE_CHARACTER)
        
        try:
            character_strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            recursive_strategy = SplitterRegistry.get_strategy(SplitterType.RECURSIVE_CHARACTER)
            
            # Should be different instances and different types
            assert character_strategy is not recursive_strategy
            assert type(character_strategy) != type(recursive_strategy)
        finally:
            # Restore original strategies
            if original_character:
                SplitterRegistry._strategies[SplitterType.CHARACTER] = original_character
            if original_recursive:
                SplitterRegistry._strategies[SplitterType.RECURSIVE_CHARACTER] = original_recursive
    
    def test_registry_persistence(self):
        """Test that registry changes persist across calls."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            
            # Verify it's registered
            strategy1 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy1, MockSplitterStrategy)
            
            # Get it again - should still be registered
            strategy2 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy2, MockSplitterStrategy)
            
            # Verify it's in the supported types
            supported_types = SplitterRegistry.list_supported_types()
            assert SplitterType.CHARACTER in supported_types
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies


class TestSplitterRegistryEdgeCases:
    """Test edge cases for SplitterRegistry."""
    
    def test_register_strategy_with_invalid_class(self):
        """Test registering strategy with invalid class."""
        class NotAStrategy:
            pass
        
        # The registry doesn't validate class types, it just stores them
        # This is the current behavior - no validation
        SplitterRegistry.register_strategy(SplitterType.CHARACTER, NotAStrategy)
        # Should not raise an error
    
    def test_get_strategy_with_invalid_type(self):
        """Test getting strategy with invalid type."""
        with pytest.raises(ValueError, match="No strategy registered for splitter type"):
            SplitterRegistry.get_strategy("invalid_type")
    
    def test_get_strategy_with_empty_string_type(self):
        """Test getting strategy with empty string type."""
        with pytest.raises(ValueError, match="No strategy registered for splitter type"):
            SplitterRegistry.get_strategy("")
    
    def test_register_strategy_multiple_times(self):
        """Test registering the same strategy multiple times."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register the same strategy multiple times
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            
            # Should still work
            strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy, MockSplitterStrategy)
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_register_strategy_with_different_types_same_class(self):
        """Test registering the same class for different types."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register the same class for different types
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            SplitterRegistry.register_strategy(SplitterType.RECURSIVE_CHARACTER, MockSplitterStrategy)
            
            # Both should work
            character_strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            recursive_strategy = SplitterRegistry.get_strategy(SplitterType.RECURSIVE_CHARACTER)
            
            assert isinstance(character_strategy, MockSplitterStrategy)
            assert isinstance(recursive_strategy, MockSplitterStrategy)
            
            # Should be different instances
            assert character_strategy is not recursive_strategy
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_list_supported_types_after_registration(self):
        """Test that list_supported_types reflects new registrations."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Get initial supported types
            initial_types = set(SplitterRegistry.list_supported_types())
            
            # Register a new strategy for a new type
            class NewSplitterType:
                def __str__(self):
                    return "new_type"
            
            new_type = NewSplitterType()
            SplitterRegistry.register_strategy(new_type, MockSplitterStrategy)
            
            # Get updated supported types
            updated_types = set(SplitterRegistry.list_supported_types())
            
            # Should include the new type
            assert new_type in updated_types
            assert len(updated_types) == len(initial_types) + 1
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_strategy_instance_isolation(self):
        """Test that strategy instances are isolated."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register a strategy that tracks calls
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            
            # Get two instances
            strategy1 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            strategy2 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            
            # They should be different instances
            assert strategy1 is not strategy2
            
            # Test that calls to one don't affect the other
            config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
            
            strategy1.validate_config(config)
            strategy1.create_splitter(config)
            
            # strategy2 should not have any calls
            assert len(strategy2.validation_calls) == 0
            assert len(strategy2.creation_calls) == 0
            
            # strategy1 should have the calls
            assert len(strategy1.validation_calls) == 1
            assert len(strategy1.creation_calls) == 1
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies


class TestSplitterRegistryIntegration:
    """Integration tests for SplitterRegistry."""
    
    def test_full_workflow_with_custom_strategy(self):
        """Test complete workflow with custom strategy."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register custom strategy
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            
            # Get strategy
            strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy, MockSplitterStrategy)
            
            # Use strategy
            config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
            strategy.validate_config(config)
            splitter = strategy.create_splitter(config)
            
            assert splitter is not None
            assert len(strategy.validation_calls) == 1
            assert len(strategy.creation_calls) == 1
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_multiple_strategies_workflow(self):
        """Test workflow with multiple different strategies."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register different strategies for different types
            class StrategyA(MockSplitterStrategy):
                pass
            
            class StrategyB(MockSplitterStrategy):
                pass
            
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, StrategyA)
            SplitterRegistry.register_strategy(SplitterType.RECURSIVE_CHARACTER, StrategyB)
            
            # Get strategies
            character_strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            recursive_strategy = SplitterRegistry.get_strategy(SplitterType.RECURSIVE_CHARACTER)
            
            assert isinstance(character_strategy, StrategyA)
            assert isinstance(recursive_strategy, StrategyB)
            
            # Use both strategies
            character_config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
            recursive_config = SplitterConfig(type=SplitterType.RECURSIVE_CHARACTER, chunk_size=1000, chunk_overlap=200)
            
            character_strategy.validate_config(character_config)
            recursive_strategy.validate_config(recursive_config)
            
            assert len(character_strategy.validation_calls) == 1
            assert len(recursive_strategy.validation_calls) == 1
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_strategy_replacement_workflow(self):
        """Test workflow with strategy replacement."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register initial strategy
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            strategy1 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy1, MockSplitterStrategy)
            
            # Replace with different strategy
            class ReplacementStrategy(MockSplitterStrategy):
                pass
            
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, ReplacementStrategy)
            strategy2 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy2, ReplacementStrategy)
            
            # Should be different instances
            assert strategy1 is not strategy2
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies
    
    def test_error_handling_workflow(self):
        """Test workflow with error handling."""
        original_strategies = SplitterRegistry._strategies.copy()
        
        try:
            # Register strategy that raises errors
            error_strategy = MockSplitterStrategy(
                should_raise_validation_error=True,
                should_raise_creation_error=True
            )
            
            class ErrorStrategyClass(MockSplitterStrategy):
                def __init__(self):
                    super().__init__(
                        should_raise_validation_error=True,
                        should_raise_creation_error=True
                    )
            
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, ErrorStrategyClass)
            
            # Get strategy
            strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy, ErrorStrategyClass)
            
            # Test validation error
            config = SplitterConfig(type=SplitterType.CHARACTER, chunk_size=1000, chunk_overlap=200)
            with pytest.raises(ValueError, match="Mock validation error"):
                strategy.validate_config(config)
            
            # Test creation error
            with pytest.raises(ValueError, match="Mock creation error"):
                strategy.create_splitter(config)
            
        finally:
            # Restore original strategies
            SplitterRegistry._strategies = original_strategies

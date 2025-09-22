"""
Unit tests for citation strategy registry.

Tests all components of the CitationStrategyRegistry including:
- Strategy registration and retrieval
- Error handling and edge cases
- Integration with different strategy types
- Strategy instance management
- Registry persistence and consistency
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.citations.strategies import (
    CitationStrategyRegistry,
    CitationStrategy,
    DefaultCitationStrategy,
    TechnicalCitationStrategy,
    WebCitationStrategy,
    PolicyCitationStrategy
)
from src.paas_ai.core.rag.citations.models import SourceReference, CitationMetadata
from src.paas_ai.core.rag.processing.base import ProcessingContext
from src.paas_ai.core.config.schemas import ResourceType, CitationVerbosity
from langchain_core.documents import Document


class MockCitationStrategy(CitationStrategy):
    """Mock citation strategy for testing."""
    
    def __init__(self, name="MockStrategy", should_raise_error=False):
        self.name = name
        self.should_raise_error = should_raise_error
        self.extract_calls = []
        self.format_calls = []
        self.link_calls = []
    
    def extract_citation_metadata(self, document: Document, context: ProcessingContext) -> CitationMetadata:
        """Extract mock citation metadata."""
        self.extract_calls.append((document, context))
        if self.should_raise_error:
            raise Exception("Mock extraction error")
        return CitationMetadata()
    
    def format_citation(self, source_ref: SourceReference, verbosity: CitationVerbosity) -> str:
        """Format mock citation."""
        self.format_calls.append((source_ref, verbosity))
        if self.should_raise_error:
            raise Exception("Mock formatting error")
        return f"[{self.name} Citation]"
    
    def generate_citation_link(self, source_ref: SourceReference) -> str:
        """Generate mock citation link."""
        self.link_calls.append(source_ref)
        if self.should_raise_error:
            raise Exception("Mock link generation error")
        return f"https://{self.name.lower()}.com"


class TestCitationStrategyRegistry:
    """Test the CitationStrategyRegistry class."""
    
    def test_init(self):
        """Test registry initialization."""
        registry = CitationStrategyRegistry()
        assert registry is not None
        assert isinstance(registry, CitationStrategyRegistry)
        assert hasattr(registry, '_strategies')
        assert isinstance(registry._strategies, dict)
    
    def test_get_strategy_existing_default(self):
        """Test getting existing default strategy."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("default_citation")
        assert strategy is not None
        assert isinstance(strategy, DefaultCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_get_strategy_existing_technical(self):
        """Test getting existing technical strategy."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("technical_citation")
        assert strategy is not None
        assert isinstance(strategy, TechnicalCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_get_strategy_existing_web(self):
        """Test getting existing web strategy."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("web_citation")
        assert strategy is not None
        assert isinstance(strategy, WebCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_get_strategy_existing_policy(self):
        """Test getting existing policy strategy."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("policy_citation")
        assert strategy is not None
        assert isinstance(strategy, PolicyCitationStrategy)
        assert isinstance(strategy, CitationStrategy)
    
    def test_get_strategy_existing_rule(self):
        """Test getting existing rule strategy."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("rule_citation")
        assert strategy is not None
        assert isinstance(strategy, PolicyCitationStrategy)  # Rule strategy reuses policy strategy
        assert isinstance(strategy, CitationStrategy)
    
    def test_get_strategy_nonexistent_returns_default(self):
        """Test getting non-existent strategy returns default."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("nonexistent_strategy")
        assert strategy is not None
        assert isinstance(strategy, DefaultCitationStrategy)
    
    def test_get_strategy_with_none_name(self):
        """Test getting strategy with None name returns default."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy(None)
        assert strategy is not None
        assert isinstance(strategy, DefaultCitationStrategy)
    
    def test_get_strategy_with_empty_name(self):
        """Test getting strategy with empty name returns default."""
        registry = CitationStrategyRegistry()
        
        strategy = registry.get_strategy("")
        assert strategy is not None
        assert isinstance(strategy, DefaultCitationStrategy)
    
    def test_register_strategy(self):
        """Test registering a custom strategy."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register custom strategy
            custom_strategy = MockCitationStrategy()
            registry.register_strategy("default_citation", custom_strategy)
            
            # Get the strategy and verify it's our custom one
            strategy = registry.get_strategy("default_citation")
            assert isinstance(strategy, MockCitationStrategy)
            assert strategy.name == "MockStrategy"
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy
    
    def test_register_strategy_overwrite(self):
        """Test registering a strategy overwrites existing one."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register first custom strategy
            strategy1_instance = MockCitationStrategy()
            registry.register_strategy("default_citation", strategy1_instance)
            strategy1 = registry.get_strategy("default_citation")
            assert isinstance(strategy1, MockCitationStrategy)
            
            # Register different strategy
            class AnotherMockStrategy(MockCitationStrategy):
                def __init__(self):
                    super().__init__(name="AnotherMockStrategy")
            
            strategy2_instance = AnotherMockStrategy()
            registry.register_strategy("default_citation", strategy2_instance)
            strategy2 = registry.get_strategy("default_citation")
            assert isinstance(strategy2, AnotherMockStrategy)
            assert strategy2.name == "AnotherMockStrategy"
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy
    
    def test_register_strategy_with_none_name(self):
        """Test registering strategy with None name."""
        registry = CitationStrategyRegistry()
        
        # Should not raise error
        mock_strategy = MockCitationStrategy()
        registry.register_strategy(None, mock_strategy)
        
        # Should be able to retrieve it
        strategy = registry.get_strategy(None)
        assert isinstance(strategy, MockCitationStrategy)
    
    def test_register_strategy_with_none_class(self):
        """Test registering strategy with None class."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Should not raise error
            registry.register_strategy("default_citation", None)
            
            # Should return None when getting None class
            strategy = registry.get_strategy("default_citation")
            assert strategy is None
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        registry = CitationStrategyRegistry()
        
        strategies = registry.list_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        # Check that expected strategies are present
        expected_strategies = {
            "default_citation",
            "technical_citation",
            "web_citation",
            "policy_citation",
            "rule_citation"
        }
        
        assert expected_strategies.issubset(set(strategies))
    
    def test_get_strategy_returns_new_instance(self):
        """Test that get_strategy returns a new instance each time."""
        registry = CitationStrategyRegistry()
        
        strategy1 = registry.get_strategy("default_citation")
        strategy2 = registry.get_strategy("default_citation")
        
        # Should be the same instance (registry stores instances, not classes)
        assert strategy1 is strategy2
        # But should be of the same class
        assert type(strategy1) == type(strategy2)
    
    def test_get_strategy_different_types_different_instances(self):
        """Test that different types return different strategy instances."""
        registry = CitationStrategyRegistry()
        
        default_strategy = registry.get_strategy("default_citation")
        technical_strategy = registry.get_strategy("technical_citation")
        
        # Should be different instances and different types
        assert default_strategy is not technical_strategy
        assert type(default_strategy) != type(technical_strategy)
    
    def test_registry_persistence(self):
        """Test that registry changes persist across calls."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register a custom strategy
            mock_strategy_instance = MockCitationStrategy()
            registry.register_strategy("default_citation", mock_strategy_instance)
            
            # Verify it's registered
            strategy1 = registry.get_strategy("default_citation")
            assert isinstance(strategy1, MockCitationStrategy)
            
            # Get it again - should still be registered
            strategy2 = registry.get_strategy("default_citation")
            assert isinstance(strategy2, MockCitationStrategy)
            
            # Verify it's in the list
            strategies = registry.list_strategies()
            assert "default_citation" in strategies
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy


class TestCitationStrategyRegistryEdgeCases:
    """Test edge cases for CitationStrategyRegistry."""
    
    def test_register_strategy_with_invalid_class(self):
        """Test registering strategy with invalid class."""
        registry = CitationStrategyRegistry()
        
        class NotAStrategy:
            pass
        
        # Should not raise error
        not_a_strategy_instance = NotAStrategy()
        registry.register_strategy("test_strategy", not_a_strategy_instance)
        
        # Should return the registered instance
        strategy = registry.get_strategy("test_strategy")
        assert isinstance(strategy, NotAStrategy)
    
    def test_register_strategy_multiple_times(self):
        """Test registering the same strategy multiple times."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register the same strategy multiple times
            mock_strategy_instance = MockCitationStrategy()
            registry.register_strategy("default_citation", mock_strategy_instance)
            registry.register_strategy("default_citation", mock_strategy_instance)
            registry.register_strategy("default_citation", mock_strategy_instance)
            
            # Should still work
            strategy = registry.get_strategy("default_citation")
            assert isinstance(strategy, MockCitationStrategy)
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy
    
    def test_register_strategy_with_different_names_same_class(self):
        """Test registering the same class for different names."""
        registry = CitationStrategyRegistry()
        
        # Store original strategies for cleanup
        original_strategies = {}
        for name in ["test_strategy1", "test_strategy2"]:
            original_strategies[name] = registry._strategies.get(name)
        
        try:
            # Register the same class for different names
            mock_strategy_instance1 = MockCitationStrategy()
            mock_strategy_instance2 = MockCitationStrategy()
            registry.register_strategy("test_strategy1", mock_strategy_instance1)
            registry.register_strategy("test_strategy2", mock_strategy_instance2)
            
            # Both should work
            strategy1 = registry.get_strategy("test_strategy1")
            strategy2 = registry.get_strategy("test_strategy2")
            
            assert isinstance(strategy1, MockCitationStrategy)
            assert isinstance(strategy2, MockCitationStrategy)
            
            # Should be different instances
            assert strategy1 is not strategy2
        finally:
            # Clean up
            for name, original_strategy in original_strategies.items():
                if original_strategy:
                    registry._strategies[name] = original_strategy
                elif name in registry._strategies:
                    del registry._strategies[name]
    
    def test_list_strategies_after_registration(self):
        """Test that list_strategies reflects new registrations."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("new_test_strategy")
        
        try:
            # Get initial strategies
            initial_strategies = set(registry.list_strategies())
            
            # Register a new strategy
            registry.register_strategy("new_test_strategy", MockCitationStrategy)
            
            # Get updated strategies
            updated_strategies = set(registry.list_strategies())
            
            # Should include the new strategy
            assert "new_test_strategy" in updated_strategies
            assert len(updated_strategies) == len(initial_strategies) + 1
        finally:
            # Clean up
            if original_strategy:
                registry._strategies["new_test_strategy"] = original_strategy
            elif "new_test_strategy" in registry._strategies:
                del registry._strategies["new_test_strategy"]
    
    def test_strategy_instance_isolation(self):
        """Test that strategy instances are isolated."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register a strategy that tracks calls
            mock_strategy_instance = MockCitationStrategy()
            registry.register_strategy("default_citation", mock_strategy_instance)
            
            # Get two instances
            strategy1 = registry.get_strategy("default_citation")
            strategy2 = registry.get_strategy("default_citation")
            
            # They should be the same instance (registry stores instances)
            assert strategy1 is strategy2
            
            # Test that calls to one affect the other (since they're the same instance)
            document = Document(page_content="test", metadata={})
            context = Mock(spec=ProcessingContext)
            source_ref = SourceReference(
                source_url="https://example.com",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            )
            
            strategy1.extract_citation_metadata(document, context)
            strategy1.format_citation(source_ref, CitationVerbosity.STANDARD)
            strategy1.generate_citation_link(source_ref)
            
            # strategy2 should have the same calls (since they're the same instance)
            assert len(strategy2.extract_calls) == 1
            assert len(strategy2.format_calls) == 1
            assert len(strategy2.link_calls) == 1
            
            # strategy1 should have the calls
            assert len(strategy1.extract_calls) == 1
            assert len(strategy1.format_calls) == 1
            assert len(strategy1.link_calls) == 1
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy


class TestCitationStrategyRegistryIntegration:
    """Integration tests for CitationStrategyRegistry."""
    
    def test_full_workflow_with_custom_strategy(self):
        """Test complete workflow with custom strategy."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register custom strategy
            mock_strategy_instance = MockCitationStrategy()
            registry.register_strategy("default_citation", mock_strategy_instance)
            
            # Get strategy
            strategy = registry.get_strategy("default_citation")
            assert isinstance(strategy, MockCitationStrategy)
            
            # Use strategy
            document = Document(page_content="test", metadata={})
            context = Mock(spec=ProcessingContext)
            source_ref = SourceReference(
                source_url="https://example.com",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            )
            
            metadata = strategy.extract_citation_metadata(document, context)
            citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
            link = strategy.generate_citation_link(source_ref)
            
            assert isinstance(metadata, CitationMetadata)
            assert citation == "[MockStrategy Citation]"
            assert link == "https://mockstrategy.com"
            
            assert len(strategy.extract_calls) == 1
            assert len(strategy.format_calls) == 1
            assert len(strategy.link_calls) == 1
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy
    
    def test_multiple_strategies_workflow(self):
        """Test workflow with multiple different strategies."""
        registry = CitationStrategyRegistry()
        
        # Store original strategies for cleanup
        original_strategies = {}
        for name in ["test_strategy_a", "test_strategy_b"]:
            original_strategies[name] = registry._strategies.get(name)
        
        try:
            # Register different strategies
            class StrategyA(MockCitationStrategy):
                def __init__(self):
                    super().__init__(name="StrategyA")
            
            class StrategyB(MockCitationStrategy):
                def __init__(self):
                    super().__init__(name="StrategyB")
            
            strategy_a_instance = StrategyA()
            strategy_b_instance = StrategyB()
            registry.register_strategy("test_strategy_a", strategy_a_instance)
            registry.register_strategy("test_strategy_b", strategy_b_instance)
            
            # Get strategies
            strategy_a = registry.get_strategy("test_strategy_a")
            strategy_b = registry.get_strategy("test_strategy_b")
            
            assert isinstance(strategy_a, StrategyA)
            assert isinstance(strategy_b, StrategyB)
            
            # Use both strategies
            document = Document(page_content="test", metadata={})
            context = Mock(spec=ProcessingContext)
            source_ref = SourceReference(
                source_url="https://example.com",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            )
            
            strategy_a.extract_citation_metadata(document, context)
            strategy_b.extract_citation_metadata(document, context)
            
            assert len(strategy_a.extract_calls) == 1
            assert len(strategy_b.extract_calls) == 1
        finally:
            # Clean up
            for name, original_strategy in original_strategies.items():
                if original_strategy:
                    registry._strategies[name] = original_strategy
                elif name in registry._strategies:
                    del registry._strategies[name]
    
    def test_strategy_replacement_workflow(self):
        """Test workflow with strategy replacement."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register initial strategy
            mock_strategy_instance1 = MockCitationStrategy()
            registry.register_strategy("default_citation", mock_strategy_instance1)
            strategy1 = registry.get_strategy("default_citation")
            assert isinstance(strategy1, MockCitationStrategy)
            
            # Replace with different strategy
            class ReplacementStrategy(MockCitationStrategy):
                def __init__(self):
                    super().__init__(name="ReplacementStrategy")
            
            replacement_strategy_instance = ReplacementStrategy()
            registry.register_strategy("default_citation", replacement_strategy_instance)
            strategy2 = registry.get_strategy("default_citation")
            assert isinstance(strategy2, ReplacementStrategy)
            
            # Should be different instances
            assert strategy1 is not strategy2
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy
    
    def test_error_handling_workflow(self):
        """Test workflow with error handling."""
        registry = CitationStrategyRegistry()
        
        # Store original strategy for cleanup
        original_strategy = registry._strategies.get("default_citation")
        
        try:
            # Register strategy that raises errors
            error_strategy = MockCitationStrategy(should_raise_error=True)
            
            class ErrorStrategyClass(MockCitationStrategy):
                def __init__(self):
                    super().__init__(should_raise_error=True)
            
            error_strategy_instance = ErrorStrategyClass()
            registry.register_strategy("default_citation", error_strategy_instance)
            
            # Get strategy
            strategy = registry.get_strategy("default_citation")
            assert isinstance(strategy, ErrorStrategyClass)
            
            # Test extraction error
            document = Document(page_content="test", metadata={})
            context = Mock(spec=ProcessingContext)
            with pytest.raises(Exception, match="Mock extraction error"):
                strategy.extract_citation_metadata(document, context)
            
            # Test formatting error
            source_ref = SourceReference(
                source_url="https://example.com",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            )
            with pytest.raises(Exception, match="Mock formatting error"):
                strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
            
            # Test link generation error
            with pytest.raises(Exception, match="Mock link generation error"):
                strategy.generate_citation_link(source_ref)
        finally:
            # Restore original strategy
            if original_strategy:
                registry._strategies["default_citation"] = original_strategy
    
    def test_registry_with_real_strategies(self):
        """Test registry with real strategy implementations."""
        registry = CitationStrategyRegistry()
        
        # Test all built-in strategies
        strategy_names = [
            "default_citation",
            "technical_citation",
            "web_citation",
            "policy_citation",
            "rule_citation"
        ]
        
        for strategy_name in strategy_names:
            strategy = registry.get_strategy(strategy_name)
            assert strategy is not None
            assert isinstance(strategy, CitationStrategy)
            
            # Test that strategy can be used
            document = Document(page_content="test", metadata={})
            context = Mock(spec=ProcessingContext)
            source_ref = SourceReference(
                source_url="https://example.com",
                resource_type=ResourceType.DSL,
                document_id="doc123",
                chunk_id="chunk456"
            )
            
            # Should not raise exceptions
            metadata = strategy.extract_citation_metadata(document, context)
            citation = strategy.format_citation(source_ref, CitationVerbosity.STANDARD)
            link = strategy.generate_citation_link(source_ref)
            
            assert isinstance(metadata, CitationMetadata)
            assert isinstance(citation, str)
            assert isinstance(link, str)
    
    def test_registry_consistency(self):
        """Test that registry maintains consistency."""
        registry = CitationStrategyRegistry()
        
        # Test that list_strategies and get_strategy are consistent
        strategies = registry.list_strategies()
        
        for strategy_name in strategies:
            # Should be able to get each listed strategy
            strategy = registry.get_strategy(strategy_name)
            assert strategy is not None
        
        # Test that registering a new strategy updates the list
        original_strategy = registry._strategies.get("consistency_test_strategy")
        
        try:
            initial_count = len(registry.list_strategies())
            
            mock_strategy_instance = MockCitationStrategy()
            registry.register_strategy("consistency_test_strategy", mock_strategy_instance)
            
            updated_count = len(registry.list_strategies())
            assert updated_count == initial_count + 1
            
            assert "consistency_test_strategy" in registry.list_strategies()
            
            strategy = registry.get_strategy("consistency_test_strategy")
            assert isinstance(strategy, MockCitationStrategy)
        finally:
            # Clean up
            if original_strategy:
                registry._strategies["consistency_test_strategy"] = original_strategy
            elif "consistency_test_strategy" in registry._strategies:
                del registry._strategies["consistency_test_strategy"]

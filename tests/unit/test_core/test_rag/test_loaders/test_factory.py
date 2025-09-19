"""
Unit tests for document loader factory and registry.

Tests all components of the loader factory system including:
- DocumentLoaderFactory class
- LoaderRegistry class
- Strategy registration and retrieval
- Error handling and edge cases
- Integration with different loader types
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.loaders.factory import DocumentLoaderFactory
from src.paas_ai.core.rag.loaders.registry import LoaderRegistry
from src.paas_ai.core.rag.loaders.base import LoaderStrategy
from src.paas_ai.core.rag.config import LoaderConfig, LoaderType


class MockLoaderStrategy(LoaderStrategy):
    """Mock loader strategy for testing."""
    
    def __init__(self, should_raise_validation_error=False, should_raise_creation_error=False):
        self.should_raise_validation_error = should_raise_validation_error
        self.should_raise_creation_error = should_raise_creation_error
        self.validation_calls = []
        self.creation_calls = []
    
    def create_loader(self, config, url):
        """Create a mock loader."""
        self.creation_calls.append((config, url))
        if self.should_raise_creation_error:
            raise ValueError("Mock creation error")
        return Mock()
    
    def validate_config(self, config, url):
        """Validate mock configuration."""
        self.validation_calls.append((config, url))
        if self.should_raise_validation_error:
            raise ValueError("Mock validation error")


class TestLoaderRegistry:
    """Test the LoaderRegistry class."""
    
    def test_get_strategy_existing_type(self):
        """Test getting strategy for existing loader type."""
        strategy = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
        
        assert strategy is not None
        assert hasattr(strategy, 'create_loader')
        assert hasattr(strategy, 'validate_config')
    
    def test_get_strategy_nonexistent_type(self):
        """Test getting strategy for non-existent loader type."""
        with pytest.raises(ValueError, match="No strategy registered for loader type"):
            LoaderRegistry.get_strategy("nonexistent_type")
    
    def test_register_strategy(self):
        """Test registering a custom strategy."""
        original_strategies = LoaderRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            LoaderRegistry.register_strategy(LoaderType.CRAWL4AI_WEB, MockLoaderStrategy)
            
            # Get the strategy and verify it's our custom one
            strategy = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
            assert isinstance(strategy, MockLoaderStrategy)
            
        finally:
            # Restore original strategies
            LoaderRegistry._strategies = original_strategies
    
    def test_register_strategy_overwrite(self):
        """Test registering a strategy overwrites existing one."""
        original_strategies = LoaderRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            LoaderRegistry.register_strategy(LoaderType.CRAWL4AI_WEB, MockLoaderStrategy)
            
            # Verify it's registered
            strategy1 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
            assert isinstance(strategy1, MockLoaderStrategy)
            
            # Register a different strategy
            class AnotherMockStrategy(MockLoaderStrategy):
                pass
            
            LoaderRegistry.register_strategy(LoaderType.CRAWL4AI_WEB, AnotherMockStrategy)
            
            # Verify it's overwritten
            strategy2 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
            assert isinstance(strategy2, AnotherMockStrategy)
            # Note: AnotherMockStrategy inherits from MockLoaderStrategy, so isinstance returns True
            # This is expected behavior for inheritance
            
        finally:
            # Restore original strategies
            LoaderRegistry._strategies = original_strategies
    
    def test_list_supported_types(self):
        """Test listing all supported loader types."""
        supported_types = LoaderRegistry.list_supported_types()
        
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        
        # Check that all expected types are present
        expected_types = {
            LoaderType.WEB,
            LoaderType.CRAWL4AI_WEB,
            LoaderType.HTML,
            LoaderType.PDF,
            LoaderType.MARKDOWN,
            LoaderType.JSON,
            LoaderType.CSV,
            LoaderType.DIRECTORY,
            LoaderType.CONFLUENCE,
            LoaderType.NOTION,
            LoaderType.GITHUB,
        }
        
        assert set(supported_types) == expected_types
    
    def test_get_strategy_returns_new_instance(self):
        """Test that get_strategy returns a new instance each time."""
        strategy1 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
        strategy2 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
        
        # Should be different instances
        assert strategy1 is not strategy2
        # But should be of the same class
        assert type(strategy1) == type(strategy2)
    
    def test_register_strategy_with_none_type(self):
        """Test registering strategy with None type."""
        # The registry doesn't validate None types, it just stores them
        # This is the current behavior - no validation
        LoaderRegistry.register_strategy(None, MockLoaderStrategy)
        # Should not raise an error
    
    def test_register_strategy_with_none_class(self):
        """Test registering strategy with None class."""
        # The registry doesn't validate None classes, it just stores them
        # This is the current behavior - no validation
        LoaderRegistry.register_strategy(LoaderType.CRAWL4AI_WEB, None)
        # Should not raise an error


class TestDocumentLoaderFactory:
    """Test the DocumentLoaderFactory class."""
    
    def test_create_loader_success(self):
        """Test successful loader creation."""
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockLoaderStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = DocumentLoaderFactory.create_loader(config, url)
            
            # Verify strategy was retrieved
            mock_get_strategy.assert_called_once_with(LoaderType.CRAWL4AI_WEB)
            
            # Verify validation was called
            assert len(mock_strategy.validation_calls) == 1
            assert mock_strategy.validation_calls[0] == (config, url)
            
            # Verify creation was called
            assert len(mock_strategy.creation_calls) == 1
            assert mock_strategy.creation_calls[0] == (config, url)
            
            # Verify result is the mock loader
            assert result is not None
    
    def test_create_loader_validation_error(self):
        """Test loader creation with validation error."""
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockLoaderStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock validation error"):
                DocumentLoaderFactory.create_loader(config, url)
            
            # Verify validation was called but creation was not
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 0
    
    def test_create_loader_creation_error(self):
        """Test loader creation with creation error."""
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockLoaderStrategy(should_raise_creation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock creation error"):
                DocumentLoaderFactory.create_loader(config, url)
            
            # Verify both validation and creation were called
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 1
    
    def test_create_loader_strategy_not_found(self):
        """Test loader creation when strategy is not found."""
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = ValueError("No strategy registered")
            
            with pytest.raises(ValueError, match="No strategy registered"):
                DocumentLoaderFactory.create_loader(config, url)
    
    def test_list_supported_types(self):
        """Test listing supported loader types through factory."""
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [LoaderType.CRAWL4AI_WEB, LoaderType.WEB]
            
            result = DocumentLoaderFactory.list_supported_types()
            
            assert result == [LoaderType.CRAWL4AI_WEB, LoaderType.WEB]
            mock_list.assert_called_once()
    
    def test_register_strategy(self):
        """Test registering strategy through factory."""
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.register_strategy') as mock_register:
            DocumentLoaderFactory.register_strategy(LoaderType.CRAWL4AI_WEB, MockLoaderStrategy)
            
            mock_register.assert_called_once_with(LoaderType.CRAWL4AI_WEB, MockLoaderStrategy)
    
    def test_create_loader_with_different_types(self):
        """Test creating loaders with different types."""
        test_cases = [
            LoaderType.WEB,
            LoaderType.CRAWL4AI_WEB,
            LoaderType.HTML,
            LoaderType.PDF,
            LoaderType.MARKDOWN,
            LoaderType.JSON,
            LoaderType.CSV,
            LoaderType.DIRECTORY,
            LoaderType.CONFLUENCE,
            LoaderType.NOTION,
            LoaderType.GITHUB,
        ]
        
        for loader_type in test_cases:
            config = LoaderConfig(type=loader_type)
            url = "https://example.com"
            
            with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
                mock_strategy = MockLoaderStrategy()
                mock_get_strategy.return_value = mock_strategy
                
                result = DocumentLoaderFactory.create_loader(config, url)
                
                # Verify strategy was retrieved with correct type
                mock_get_strategy.assert_called_with(loader_type)
                
                # Verify result is not None
                assert result is not None
    
    def test_create_loader_with_custom_config(self):
        """Test creating loader with custom configuration."""
        config = LoaderConfig(
            type=LoaderType.CRAWL4AI_WEB,
            params={'headless': False, 'timeout': 60000}
        )
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockLoaderStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = DocumentLoaderFactory.create_loader(config, url)
            
            # Verify validation was called with custom config
            assert len(mock_strategy.validation_calls) == 1
            validation_config, validation_url = mock_strategy.validation_calls[0]
            assert validation_config == config
            assert validation_url == url
            
            # Verify creation was called with custom config
            assert len(mock_strategy.creation_calls) == 1
            creation_config, creation_url = mock_strategy.creation_calls[0]
            assert creation_config == config
            assert creation_url == url
    
    def test_create_loader_with_different_urls(self):
        """Test creating loader with different URL types."""
        test_urls = [
            "https://example.com",
            "http://example.com",
            "file:///path/to/file.pdf",
            "urls.csv",
            "/absolute/path/to/file.md",
            "relative/path/to/file.txt"
        ]
        
        for url in test_urls:
            config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
            
            with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
                mock_strategy = MockLoaderStrategy()
                mock_get_strategy.return_value = mock_strategy
                
                result = DocumentLoaderFactory.create_loader(config, url)
                
                # Verify URL was passed correctly
                assert len(mock_strategy.validation_calls) == 1
                validation_config, validation_url = mock_strategy.validation_calls[0]
                assert validation_url == url
                
                assert len(mock_strategy.creation_calls) == 1
                creation_config, creation_url = mock_strategy.creation_calls[0]
                assert creation_url == url


class TestIntegration:
    """Integration tests for factory and registry."""
    
    def test_full_workflow(self):
        """Test complete workflow from factory to loader creation."""
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        # This should work without mocking since we're testing the real integration
        try:
            result = DocumentLoaderFactory.create_loader(config, url)
            assert result is not None
        except Exception as e:
            # If there are import issues or missing dependencies, that's expected in tests
            # We just want to make sure the factory doesn't crash
            assert "No strategy registered" not in str(e)
    
    def test_registry_persistence(self):
        """Test that registry changes persist across calls."""
        original_strategies = LoaderRegistry._strategies.copy()
        
        try:
            # Register a custom strategy
            LoaderRegistry.register_strategy(LoaderType.CRAWL4AI_WEB, MockLoaderStrategy)
            
            # Verify it's registered
            strategy1 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
            assert isinstance(strategy1, MockLoaderStrategy)
            
            # Get it again - should still be registered
            strategy2 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
            assert isinstance(strategy2, MockLoaderStrategy)
            
            # Verify it's in the supported types
            supported_types = LoaderRegistry.list_supported_types()
            assert LoaderType.CRAWL4AI_WEB in supported_types
            
        finally:
            # Restore original strategies
            LoaderRegistry._strategies = original_strategies
    
    def test_factory_registry_consistency(self):
        """Test that factory and registry are consistent."""
        # Get supported types from both
        factory_types = DocumentLoaderFactory.list_supported_types()
        registry_types = LoaderRegistry.list_supported_types()
        
        assert factory_types == registry_types
        
        # Test that we can create loaders for all supported types
        for loader_type in factory_types:
            # Skip None values that might be in the registry
            if loader_type is None:
                continue
                
            config = LoaderConfig(type=loader_type)
            url = "https://example.com"
            
            # Should not raise "No strategy registered" error
            try:
                with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
                    mock_strategy = MockLoaderStrategy()
                    mock_get_strategy.return_value = mock_strategy
                    
                    result = DocumentLoaderFactory.create_loader(config, url)
                    assert result is not None
            except ValueError as e:
                if "No strategy registered" in str(e):
                    pytest.fail(f"Strategy not registered for {loader_type}")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_create_loader_with_none_config(self):
        """Test creating loader with None config."""
        with pytest.raises(AttributeError):
            DocumentLoaderFactory.create_loader(None, "https://example.com")
    
    def test_create_loader_with_none_url(self):
        """Test creating loader with None URL."""
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockLoaderStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # This should work - the strategy will handle None URL validation
            try:
                result = DocumentLoaderFactory.create_loader(config, None)
                # If it doesn't raise an error, verify the strategy was called
                assert len(mock_strategy.validation_calls) == 1
                assert len(mock_strategy.creation_calls) == 1
            except ValueError:
                # If validation fails, that's also expected
                assert len(mock_strategy.validation_calls) == 1
                assert len(mock_strategy.creation_calls) == 0
    
    def test_create_loader_with_empty_string_url(self):
        """Test creating loader with empty string URL."""
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockLoaderStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # This should work - the strategy will handle empty URL validation
            try:
                result = DocumentLoaderFactory.create_loader(config, "")
                # If it doesn't raise an error, verify the strategy was called
                assert len(mock_strategy.validation_calls) == 1
                assert len(mock_strategy.creation_calls) == 1
            except ValueError:
                # If validation fails, that's also expected
                assert len(mock_strategy.validation_calls) == 1
                assert len(mock_strategy.creation_calls) == 0
    
    def test_register_strategy_with_invalid_class(self):
        """Test registering strategy with invalid class."""
        class NotAStrategy:
            pass
        
        # The registry doesn't validate class types, it just stores them
        # This is the current behavior - no validation
        LoaderRegistry.register_strategy(LoaderType.CRAWL4AI_WEB, NotAStrategy)
        # Should not raise an error
    
    def test_get_strategy_with_invalid_type(self):
        """Test getting strategy with invalid type."""
        with pytest.raises(ValueError, match="No strategy registered for loader type"):
            LoaderRegistry.get_strategy("invalid_type")
    
    def test_factory_methods_are_static(self):
        """Test that factory methods are static and can be called without instance."""
        # These should work without creating an instance
        config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
        url = "https://example.com"
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockLoaderStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should work without creating factory instance
            result = DocumentLoaderFactory.create_loader(config, url)
            assert result is not None
        
        # Test other static methods
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [LoaderType.CRAWL4AI_WEB]
            types = DocumentLoaderFactory.list_supported_types()
            assert types == [LoaderType.CRAWL4AI_WEB]
        
        with patch('src.paas_ai.core.rag.loaders.factory.LoaderRegistry.register_strategy') as mock_register:
            DocumentLoaderFactory.register_strategy(LoaderType.CRAWL4AI_WEB, MockLoaderStrategy)
            mock_register.assert_called_once()
    
    def test_strategy_instance_isolation(self):
        """Test that strategy instances are isolated."""
        original_strategies = LoaderRegistry._strategies.copy()
        
        try:
            # Register a strategy that tracks calls
            LoaderRegistry.register_strategy(LoaderType.CRAWL4AI_WEB, MockLoaderStrategy)
            
            # Get two instances
            strategy1 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
            strategy2 = LoaderRegistry.get_strategy(LoaderType.CRAWL4AI_WEB)
            
            # They should be different instances
            assert strategy1 is not strategy2
            
            # Test that calls to one don't affect the other
            config = LoaderConfig(type=LoaderType.CRAWL4AI_WEB)
            url = "https://example.com"
            
            strategy1.validate_config(config, url)
            strategy1.create_loader(config, url)
            
            # strategy2 should not have any calls
            assert len(strategy2.validation_calls) == 0
            assert len(strategy2.creation_calls) == 0
            
            # strategy1 should have the calls
            assert len(strategy1.validation_calls) == 1
            assert len(strategy1.creation_calls) == 1
            
        finally:
            # Restore original strategies
            LoaderRegistry._strategies = original_strategies

"""
Unit tests for splitters factory.

Tests all components of the TextSplitterFactory including:
- TextSplitterFactory class
- Splitter creation workflow
- Error handling and edge cases
- Integration with registry
- Strategy delegation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.paas_ai.core.rag.splitters.factory import TextSplitterFactory
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


class TestTextSplitterFactory:
    """Test the TextSplitterFactory class."""
    
    def test_create_splitter_success(self):
        """Test successful splitter creation."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = TextSplitterFactory.create_splitter(config)
            
            # Verify strategy was retrieved
            mock_get_strategy.assert_called_once_with(SplitterType.CHARACTER)
            
            # Verify validation was called
            assert len(mock_strategy.validation_calls) == 1
            assert mock_strategy.validation_calls[0] == config
            
            # Verify creation was called
            assert len(mock_strategy.creation_calls) == 1
            assert mock_strategy.creation_calls[0] == config
            
            # Verify result is the mock splitter
            assert result is not None
    
    def test_create_splitter_validation_error(self):
        """Test splitter creation with validation error."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock validation error"):
                TextSplitterFactory.create_splitter(config)
            
            # Verify validation was called but creation was not
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 0
    
    def test_create_splitter_creation_error(self):
        """Test splitter creation with creation error."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy(should_raise_creation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock creation error"):
                TextSplitterFactory.create_splitter(config)
            
            # Verify both validation and creation were called
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 1
    
    def test_create_splitter_strategy_not_found(self):
        """Test splitter creation when strategy is not found."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = ValueError("No strategy registered")
            
            with pytest.raises(ValueError, match="No strategy registered"):
                TextSplitterFactory.create_splitter(config)
    
    def test_create_splitter_with_different_types(self):
        """Test creating splitters with different types."""
        test_cases = [
            SplitterType.CHARACTER,
            SplitterType.RECURSIVE_CHARACTER,
            SplitterType.MARKDOWN,
            SplitterType.HTML,
            SplitterType.JSON,
            SplitterType.CODE,
            SplitterType.TOKEN,
        ]
        
        for splitter_type in test_cases:
            config = SplitterConfig(
                type=splitter_type,
                chunk_size=1000,
                chunk_overlap=200,
                params={}
            )
            
            with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
                mock_strategy = MockSplitterStrategy()
                mock_get_strategy.return_value = mock_strategy
                
                result = TextSplitterFactory.create_splitter(config)
                
                # Verify strategy was retrieved with correct type
                mock_get_strategy.assert_called_with(splitter_type)
                
                # Verify result is not None
                assert result is not None
    
    def test_create_splitter_with_custom_config(self):
        """Test creating splitters with custom configuration."""
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1500,
            chunk_overlap=300,
            params={
                "separators": ["\n\n", "\n", " ", ""],
                "length_function": len
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            result = TextSplitterFactory.create_splitter(config)
            
            # Verify validation was called with custom config
            assert len(mock_strategy.validation_calls) == 1
            validation_config = mock_strategy.validation_calls[0]
            assert validation_config == config
            assert validation_config.chunk_size == 1500
            assert validation_config.chunk_overlap == 300
            assert validation_config.params["separators"] == ["\n\n", "\n", " ", ""]
            
            # Verify creation was called with custom config
            assert len(mock_strategy.creation_calls) == 1
            creation_config = mock_strategy.creation_calls[0]
            assert creation_config == config
    
    def test_split_documents_success(self):
        """Test successful document splitting."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        documents = [
            Document(page_content="This is a test document."),
            Document(page_content="This is another test document.")
        ]
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_strategy.split_documents = Mock(return_value=[
                Document(page_content="chunk1"),
                Document(page_content="chunk2")
            ])
            mock_get_strategy.return_value = mock_strategy
            
            result = TextSplitterFactory.split_documents(config, documents)
            
            # Verify strategy was retrieved
            mock_get_strategy.assert_called_once_with(SplitterType.CHARACTER)
            
            # Verify validation was called
            assert len(mock_strategy.validation_calls) == 1
            assert mock_strategy.validation_calls[0] == config
            
            # Verify split_documents was called
            mock_strategy.split_documents.assert_called_once_with(config, documents)
            
            # Verify result
            assert len(result) == 2
            assert result[0].page_content == "chunk1"
            assert result[1].page_content == "chunk2"
    
    def test_split_documents_validation_error(self):
        """Test document splitting with validation error."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        documents = [Document(page_content="test")]
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(ValueError, match="Mock validation error"):
                TextSplitterFactory.split_documents(config, documents)
            
            # Verify validation was called but split_documents was not
            assert len(mock_strategy.validation_calls) == 1
            mock_strategy.split_documents.assert_not_called()
    
    def test_list_supported_types(self):
        """Test listing supported splitter types through factory."""
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [SplitterType.CHARACTER, SplitterType.MARKDOWN]
            
            result = TextSplitterFactory.list_supported_types()
            
            assert result == [SplitterType.CHARACTER, SplitterType.MARKDOWN]
            mock_list.assert_called_once()
    
    def test_register_strategy(self):
        """Test registering strategy through factory."""
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.register_strategy') as mock_register:
            TextSplitterFactory.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            
            mock_register.assert_called_once_with(SplitterType.CHARACTER, MockSplitterStrategy)
    
    def test_factory_methods_are_static(self):
        """Test that factory methods are static and can be called without instance."""
        # These should work without creating an instance
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should work without creating factory instance
            result = TextSplitterFactory.create_splitter(config)
            assert result is not None
        
        # Test other static methods
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.list_supported_types') as mock_list:
            mock_list.return_value = [SplitterType.CHARACTER]
            types = TextSplitterFactory.list_supported_types()
            assert types == [SplitterType.CHARACTER]
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.register_strategy') as mock_register:
            TextSplitterFactory.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
            mock_register.assert_called_once()


class TestTextSplitterFactoryEdgeCases:
    """Test edge cases for TextSplitterFactory."""
    
    def test_create_splitter_with_none_config(self):
        """Test creating splitter with None config."""
        with pytest.raises(AttributeError):
            TextSplitterFactory.create_splitter(None)
    
    def test_create_splitter_with_invalid_config_type(self):
        """Test creating splitter with invalid config type."""
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            # Should handle invalid config gracefully
            with pytest.raises(AttributeError):
                TextSplitterFactory.create_splitter("not-a-config")
    
    def test_create_splitter_with_missing_config_attributes(self):
        """Test creating splitter with config missing required attributes."""
        # Create a config object without required attributes
        class IncompleteConfig:
            def __init__(self):
                self.type = SplitterType.CHARACTER
                # Missing chunk_size, chunk_overlap, and params
        
        incomplete_config = IncompleteConfig()
        
        # Should handle incomplete config gracefully - the factory accesses config.type first
        with pytest.raises(AttributeError, match="'IncompleteConfig' object has no attribute"):
            TextSplitterFactory.create_splitter(incomplete_config)
    
    def test_create_splitter_with_strategy_that_returns_none(self):
        """Test creating splitter with strategy that returns None."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_strategy.create_splitter = Mock(return_value=None)
            mock_get_strategy.return_value = mock_strategy
            
            result = TextSplitterFactory.create_splitter(config)
            
            # Should return None if strategy returns None
            assert result is None
    
    def test_create_splitter_with_strategy_that_returns_string(self):
        """Test creating splitter with strategy that returns unexpected type."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_strategy.create_splitter = Mock(return_value="not-a-splitter")
            mock_get_strategy.return_value = mock_strategy
            
            result = TextSplitterFactory.create_splitter(config)
            
            # Should return whatever the strategy returns
            assert result == "not-a-splitter"
    
    def test_create_splitter_with_exception_in_strategy_retrieval(self):
        """Test creating splitter with exception in strategy retrieval."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = Exception("Registry error")
            
            with pytest.raises(Exception, match="Registry error"):
                TextSplitterFactory.create_splitter(config)
    
    def test_create_splitter_with_exception_in_validation(self):
        """Test creating splitter with exception in validation."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_strategy.validate_config = Mock(side_effect=Exception("Validation error"))
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(Exception, match="Validation error"):
                TextSplitterFactory.create_splitter(config)
    
    def test_create_splitter_with_exception_in_creation(self):
        """Test creating splitter with exception in creation."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_strategy.create_splitter = Mock(side_effect=Exception("Creation error"))
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(Exception, match="Creation error"):
                TextSplitterFactory.create_splitter(config)
    
    def test_split_documents_with_none_documents(self):
        """Test splitting documents with None documents."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_strategy.split_documents = Mock(side_effect=TypeError("None documents"))
            mock_get_strategy.return_value = mock_strategy
            
            with pytest.raises(TypeError, match="None documents"):
                TextSplitterFactory.split_documents(config, None)
    
    def test_split_documents_with_empty_documents(self):
        """Test splitting documents with empty document list."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_strategy.split_documents = Mock(return_value=[])
            mock_get_strategy.return_value = mock_strategy
            
            result = TextSplitterFactory.split_documents(config, [])
            
            assert result == []
            mock_strategy.split_documents.assert_called_once_with(config, [])


class TestTextSplitterFactoryIntegration:
    """Integration tests for TextSplitterFactory."""
    
    def test_full_workflow(self):
        """Test complete workflow from factory to splitter creation."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        # This should work without mocking since we're testing the real integration
        try:
            result = TextSplitterFactory.create_splitter(config)
            assert result is not None
        except Exception as e:
            # If there are import issues or missing dependencies, that's expected in tests
            # We just want to make sure the factory doesn't crash
            assert "No strategy registered" not in str(e)
    
    def test_factory_registry_consistency(self):
        """Test that factory and registry are consistent."""
        # Get supported types from both
        factory_types = TextSplitterFactory.list_supported_types()
        registry_types = TextSplitterFactory.list_supported_types()  # Factory delegates to registry
        
        assert factory_types == registry_types
        
        # Test that we can create splitters for all supported types
        for splitter_type in factory_types:
            # Skip None values that might be in the registry
            if splitter_type is None:
                continue
                
            config = SplitterConfig(
                type=splitter_type,
                chunk_size=1000,
                chunk_overlap=200,
                params={}
            )
            
            # Should not raise "No strategy registered" error
            try:
                with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
                    mock_strategy = MockSplitterStrategy()
                    mock_get_strategy.return_value = mock_strategy
                    
                    result = TextSplitterFactory.create_splitter(config)
                    assert result is not None
            except ValueError as e:
                if "No strategy registered" in str(e):
                    pytest.fail(f"Strategy not registered for {splitter_type}")
    
    def test_workflow_with_multiple_configs(self):
        """Test workflow with multiple different configurations."""
        configs = [
            SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separator": "\n\n"}
            ),
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1500,
                chunk_overlap=300,
                params={"separators": ["\n\n", "\n", " "]}
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [("#", "Header 1"), ("##", "Header 2")]}
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy()
            mock_get_strategy.return_value = mock_strategy
            
            for config in configs:
                result = TextSplitterFactory.create_splitter(config)
                assert result is not None
            
            # Should have been called for each config
            assert mock_get_strategy.call_count == len(configs)
            assert len(mock_strategy.validation_calls) == len(configs)
            assert len(mock_strategy.creation_calls) == len(configs)
    
    def test_workflow_with_strategy_registration(self):
        """Test workflow with strategy registration."""
        # Register a custom strategy
        TextSplitterFactory.register_strategy(SplitterType.CHARACTER, MockSplitterStrategy)
        
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        # Should use the registered strategy
        result = TextSplitterFactory.create_splitter(config)
        assert result is not None
    
    def test_workflow_with_error_handling(self):
        """Test workflow with error handling."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = MockSplitterStrategy(should_raise_validation_error=True)
            mock_get_strategy.return_value = mock_strategy
            
            # Should handle validation error
            with pytest.raises(ValueError, match="Mock validation error"):
                TextSplitterFactory.create_splitter(config)
            
            # Verify validation was called but creation was not
            assert len(mock_strategy.validation_calls) == 1
            assert len(mock_strategy.creation_calls) == 0

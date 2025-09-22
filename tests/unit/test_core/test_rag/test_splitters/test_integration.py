"""
Integration tests for the splitters system.

Tests the complete splitters system integration including:
- End-to-end workflows
- Cross-component interactions
- Real splitter creation (with mocking)
- Error propagation
- Configuration validation across components
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.paas_ai.core.rag.splitters.factory import TextSplitterFactory
from src.paas_ai.core.rag.splitters.registry import SplitterRegistry
from src.paas_ai.core.rag.splitters.character import CharacterSplitterStrategy
from src.paas_ai.core.rag.splitters.recursive_character import RecursiveCharacterSplitterStrategy
from src.paas_ai.core.rag.splitters.markdown import MarkdownSplitterStrategy
from src.paas_ai.core.rag.splitters.html import HTMLSplitterStrategy
from src.paas_ai.core.rag.splitters.json import JSONSplitterStrategy
from src.paas_ai.core.rag.splitters.code import CodeSplitterStrategy
from src.paas_ai.core.rag.splitters.token import TokenSplitterStrategy
from src.paas_ai.core.rag.config import SplitterConfig, SplitterType




class TestSplittersSystemIntegration:
    """Integration tests for the complete splitters system."""
    
    def test_character_workflow_integration(self):
        """Test complete character splitter workflow integration."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Test through factory
            result = TextSplitterFactory.create_splitter(config)
            
            assert result is not None
            # The mock should be called with the correct parameters
            mock_splitter_class.assert_called_once()
            call_args = mock_splitter_class.call_args
            assert call_args[1]['chunk_size'] == 1000
            assert call_args[1]['chunk_overlap'] == 200
            assert call_args[1]['separator'] == "\n\n"
    
    def test_recursive_character_workflow_integration(self):
        """Test complete recursive character splitter workflow integration."""
        config = SplitterConfig(
            type=SplitterType.RECURSIVE_CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separators": ["\n\n", "\n", " ", ""]}
        )
        
        with patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Test through factory
            result = TextSplitterFactory.create_splitter(config)
            
            assert result is not None
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
    
    def test_markdown_workflow_integration(self):
        """Test complete markdown splitter workflow integration."""
        config = SplitterConfig(
            type=SplitterType.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Test through factory
            result = TextSplitterFactory.create_splitter(config)
            
            assert result is not None
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ]
            )
    
    def test_html_workflow_integration(self):
        """Test complete HTML splitter workflow integration."""
        config = SplitterConfig(
            type=SplitterType.HTML,
            chunk_size=1000,
            chunk_overlap=200,
            params={
                "headers_to_split_on": [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3")
                ]
            }
        )
        
        with patch('src.paas_ai.core.rag.splitters.html.HTMLHeaderTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Test through factory
            result = TextSplitterFactory.create_splitter(config)
            
            assert result is not None
            mock_splitter_class.assert_called_once_with(
                headers_to_split_on=[
                    ("h1", "Header 1"),
                    ("h2", "Header 2"),
                    ("h3", "Header 3")
                ]
            )
    
    def test_json_workflow_integration(self):
        """Test complete JSON splitter workflow integration."""
        config = SplitterConfig(
            type=SplitterType.JSON,
            chunk_size=1000,
            chunk_overlap=200,
            params={"max_depth": 3}
        )
        
        with patch('src.paas_ai.core.rag.splitters.json.RecursiveJsonSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Test through factory
            result = TextSplitterFactory.create_splitter(config)
            
            assert result is not None
            mock_splitter_class.assert_called_once_with(
                max_chunk_size=1000,
                max_depth=3
            )
    
    def test_code_workflow_integration(self):
        """Test complete code splitter workflow integration."""
        config = SplitterConfig(
            type=SplitterType.CODE,
            chunk_size=1000,
            chunk_overlap=200,
            params={"language": "python"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.code.CodeSplitter') as mock_splitter_class:
            
            mock_splitter = Mock()
            mock_splitter_class.from_language.return_value = mock_splitter
            
            # Test through factory
            result = TextSplitterFactory.create_splitter(config)
            
            assert result is not None
            mock_splitter_class.from_language.assert_called_once()
    
    def test_token_workflow_integration(self):
        """Test complete token splitter workflow integration."""
        config = SplitterConfig(
            type=SplitterType.TOKEN,
            chunk_size=1000,
            chunk_overlap=200,
            params={"encoding_name": "cl100k_base", "model_name": "gpt-3.5-turbo"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.token.TokenTextSplitter') as mock_splitter_class, \
             patch('src.paas_ai.core.rag.splitters.token.tiktoken') as mock_tiktoken:
            
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            mock_tiktoken.get_encoding.return_value = Mock()
            
            # Test through factory
            result = TextSplitterFactory.create_splitter(config)
            
            assert result is not None
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000,
                chunk_overlap=200,
                encoding_name="cl100k_base",
                model_name="gpt-3.5-turbo"
            )
    
    def test_registry_factory_integration(self):
        """Test integration between registry and factory."""
        # Test that factory uses registry correctly
        supported_types = TextSplitterFactory.list_supported_types()
        registry_types = SplitterRegistry.list_supported_types()
        
        assert supported_types == registry_types
        
        # Test that factory can create splitters for all registry types
        for splitter_type in registry_types:
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
                    mock_strategy = Mock()
                    mock_strategy.validate_config = Mock()
                    mock_strategy.create_splitter = Mock(return_value=Mock())
                    mock_get_strategy.return_value = mock_strategy
                    
                    result = TextSplitterFactory.create_splitter(config)
                    assert result is not None
            except ValueError as e:
                if "No strategy registered" in str(e):
                    pytest.fail(f"Strategy not registered for {splitter_type}")
    
    def test_strategy_registration_workflow(self):
        """Test complete strategy registration workflow."""
        # Store original strategy
        original_strategy = SplitterRegistry._strategies.get(SplitterType.CHARACTER)
        
        try:
            # Track calls at class level
            validation_calls = []
            creation_calls = []
            
            # Register a custom strategy
            class CustomSplitterStrategy:
                def validate_config(self, config):
                    validation_calls.append(config)
                
                def create_splitter(self, config):
                    creation_calls.append(config)
                    return Mock()
            
            # Register through factory
            TextSplitterFactory.register_strategy(SplitterType.CHARACTER, CustomSplitterStrategy)
            
            # Verify it's registered
            strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert isinstance(strategy, CustomSplitterStrategy)
            
            # Test workflow
            config = SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={}
            )
            
            result = TextSplitterFactory.create_splitter(config)
            assert result is not None
            assert len(validation_calls) == 1
            assert len(creation_calls) == 1
        finally:
            # Restore original strategy
            if original_strategy:
                SplitterRegistry._strategies[SplitterType.CHARACTER] = original_strategy
    
    def test_error_propagation_integration(self):
        """Test error propagation through the system."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=-100,  # Invalid chunk size
            chunk_overlap=200,
            params={}
        )
        
        # Should propagate validation error from strategy through factory
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitterFactory.create_splitter(config)
    
    def test_configuration_validation_integration(self):
        """Test configuration validation across components."""
        # Test valid configuration
        valid_config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = TextSplitterFactory.create_splitter(valid_config)
            assert result is not None
        
        # Test invalid configuration
        invalid_config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=-100,  # Invalid
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitterFactory.create_splitter(invalid_config)
    
    def test_multiple_splitter_types_integration(self):
        """Test integration with multiple splitter types."""
        configs = [
            SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separator": "\n\n"}
            ),
            SplitterConfig(
                type=SplitterType.RECURSIVE_CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separators": ["\n\n", "\n", " "]}
            ),
            SplitterConfig(
                type=SplitterType.MARKDOWN,
                chunk_size=1000,
                chunk_overlap=200,
                params={"headers_to_split_on": [("#", "Header 1"), ("##", "Header 2")]}
            )
        ]
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_character_class, \
             patch('src.paas_ai.core.rag.splitters.recursive_character.RecursiveCharacterTextSplitter') as mock_recursive_class, \
             patch('src.paas_ai.core.rag.splitters.markdown.MarkdownHeaderTextSplitter') as mock_markdown_class:
            
            mock_character_splitter = Mock()
            mock_recursive_splitter = Mock()
            mock_markdown_splitter = Mock()
            mock_character_class.return_value = mock_character_splitter
            mock_recursive_class.return_value = mock_recursive_splitter
            mock_markdown_class.return_value = mock_markdown_splitter
            
            # Test character config
            result1 = TextSplitterFactory.create_splitter(configs[0])
            assert result1 is not None
            
            # Test recursive character config
            result2 = TextSplitterFactory.create_splitter(configs[1])
            assert result2 is not None
            
            # Test markdown config
            result3 = TextSplitterFactory.create_splitter(configs[2])
            assert result3 is not None
            
            # Verify all were called
            mock_character_class.assert_called_once()
            mock_recursive_class.assert_called_once()
            mock_markdown_class.assert_called_once()
    
    def test_document_splitting_integration(self):
        """Test integration with document splitting."""
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
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter.split_documents.return_value = [
                Document(page_content="chunk1"),
                Document(page_content="chunk2")
            ]
            mock_splitter_class.return_value = mock_splitter
            
            result = TextSplitterFactory.split_documents(config, documents)
            
            assert len(result) == 2
            assert result[0].page_content == "chunk1"
            assert result[1].page_content == "chunk2"
            mock_splitter.split_documents.assert_called_once_with(documents)


class TestSplittersSystemErrorHandling:
    """Test error handling across the splitters system."""
    
    def test_missing_strategy_error_handling(self):
        """Test error handling for missing strategies."""
        # Try to get a strategy that doesn't exist
        with pytest.raises(ValueError, match="No strategy registered for splitter type"):
            SplitterRegistry.get_strategy("nonexistent_type")
    
    def test_invalid_config_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with negative chunk size
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=-100,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitterFactory.create_splitter(config)
    
    def test_strategy_creation_error_handling(self):
        """Test error handling for strategy creation errors."""
        # Mock a strategy that raises an error during creation
        class ErrorStrategy:
            def validate_config(self, config):
                pass
            
            def create_splitter(self, config):
                raise RuntimeError("Strategy creation error")
        
        # Register the error strategy
        SplitterRegistry.register_strategy(SplitterType.CHARACTER, ErrorStrategy)
        
        try:
            config = SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={}
            )
            
            with pytest.raises(RuntimeError, match="Strategy creation error"):
                TextSplitterFactory.create_splitter(config)
        finally:
            # Restore original strategy
            SplitterRegistry.register_strategy(SplitterType.CHARACTER, CharacterSplitterStrategy)
    
    def test_validation_error_handling(self):
        """Test error handling for validation errors."""
        # Test with invalid chunk size
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=-100,
            chunk_overlap=200,
            params={}
        )
        
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextSplitterFactory.create_splitter(config)
    
    def test_import_error_handling(self):
        """Test error handling for import errors."""
        # Mock import error
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter', side_effect=ImportError("Module not found")):
            config = SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={"separator": "\n\n"}
            )
            
            with pytest.raises(ImportError, match="Module not found"):
                TextSplitterFactory.create_splitter(config)


class TestSplittersSystemPerformance:
    """Test performance characteristics of the splitters system."""
    
    def test_strategy_instance_reuse(self):
        """Test that strategy instances are created fresh each time."""
        # Get the same strategy multiple times
        strategy1 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
        strategy2 = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
        
        # Should be different instances
        assert strategy1 is not strategy2
        assert type(strategy1) == type(strategy2)
    
    def test_factory_method_performance(self):
        """Test that factory methods are efficient."""
        import time
        
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            # Time multiple calls
            start_time = time.time()
            for _ in range(100):
                result = TextSplitterFactory.create_splitter(config)
                assert result == mock_splitter
            end_time = time.time()
            
            # Should be reasonably fast (less than 1 second for 100 calls)
            assert (end_time - start_time) < 1.0
    
    def test_registry_lookup_performance(self):
        """Test that registry lookups are efficient."""
        import time
        
        # Time multiple lookups
        start_time = time.time()
        for _ in range(1000):
            strategy = SplitterRegistry.get_strategy(SplitterType.CHARACTER)
            assert strategy is not None
        end_time = time.time()
        
        # Should be very fast (less than 0.1 seconds for 1000 lookups)
        assert (end_time - start_time) < 0.1


class TestSplittersSystemCompatibility:
    """Test compatibility and interoperability of the splitters system."""
    
    def test_config_compatibility(self):
        """Test that configurations work across different components."""
        config = SplitterConfig(
            type=SplitterType.CHARACTER,
            chunk_size=1000,
            chunk_overlap=200,
            params={"separator": "\n\n"}
        )
        
        # Test that config works with registry
        strategy = SplitterRegistry.get_strategy(config.type)
        assert strategy is not None
        
        # Test that config works with factory
        with patch('src.paas_ai.core.rag.splitters.character.CharacterTextSplitter') as mock_splitter_class:
            mock_splitter = Mock()
            mock_splitter_class.return_value = mock_splitter
            
            result = TextSplitterFactory.create_splitter(config)
            assert result == mock_splitter
    
    def test_strategy_interface_compatibility(self):
        """Test that all strategies implement the required interface."""
        for splitter_type in SplitterRegistry.list_supported_types():
            if splitter_type is None:
                continue
                
            strategy = SplitterRegistry.get_strategy(splitter_type)
            
            # Check required methods
            assert hasattr(strategy, 'create_splitter')
            assert hasattr(strategy, 'validate_config')
            assert callable(strategy.create_splitter)
            assert callable(strategy.validate_config)
    
    def test_factory_registry_compatibility(self):
        """Test that factory and registry are compatible."""
        # Test that factory delegates to registry correctly
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_strategy = Mock()
            mock_strategy.validate_config = Mock()
            mock_strategy.create_splitter = Mock(return_value=Mock())
            mock_get_strategy.return_value = mock_strategy
            
            config = SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={}
            )
            
            result = TextSplitterFactory.create_splitter(config)
            assert result is not None
            mock_get_strategy.assert_called_once_with(SplitterType.CHARACTER)
    
    def test_cross_component_error_handling(self):
        """Test error handling across components."""
        # Test that errors from registry propagate to factory
        with patch('src.paas_ai.core.rag.splitters.factory.SplitterRegistry.get_strategy') as mock_get_strategy:
            mock_get_strategy.side_effect = ValueError("Registry error")
            
            config = SplitterConfig(
                type=SplitterType.CHARACTER,
                chunk_size=1000,
                chunk_overlap=200,
                params={}
            )
            
            with pytest.raises(ValueError, match="Registry error"):
                TextSplitterFactory.create_splitter(config)

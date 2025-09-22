"""
Unit tests for Pinecone vector store strategy.

Tests all components of the PineconeVectorStoreStrategy including:
- PineconeVectorStoreStrategy class
- Configuration validation
- Vector store creation
- Vector store loading
- Error handling and edge cases
- Import error handling
- Collection name validation
- Pinecone-specific constraints
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.paas_ai.core.rag.vectorstore.pinecone import PineconeVectorStoreStrategy
from src.paas_ai.core.config.schemas import VectorStoreConfig, VectorStoreType


class TestPineconeVectorStoreStrategy:
    """Test the PineconeVectorStoreStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = PineconeVectorStoreStrategy()
        assert strategy is not None
        assert isinstance(strategy, PineconeVectorStoreStrategy)
    
    def test_create_vectorstore_with_documents(self):
        """Test creating vector store with documents."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={"api_key": "test-key", "environment": "us-west1-gcp"}
        )
        embeddings = Mock()
        documents = [
            Document(page_content="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"})
        ]
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.from_documents.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings, documents)
            
            # Verify PineconeVectorStore.from_documents was called correctly
            mock_pinecone_class.from_documents.assert_called_once_with(
                documents=documents,
                embedding=embeddings,
                index_name="test-collection",
                api_key="test-key",
                environment="us-west1-gcp"
            )
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_without_documents(self):
        """Test creating vector store without documents."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={"api_key": "test-key", "environment": "us-west1-gcp"}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings)
            
            # Verify PineconeVectorStore constructor was called correctly
            mock_pinecone_class.assert_called_once_with(
                embedding=embeddings,
                index_name="test-collection",
                api_key="test-key",
                environment="us-west1-gcp"
            )
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_with_empty_params(self):
        """Test creating vector store with empty params."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings)
            
            # Verify PineconeVectorStore constructor was called with only required params
            mock_pinecone_class.assert_called_once_with(
                embedding=embeddings,
                index_name="test-collection"
            )
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_import_error(self):
        """Test creating vector store when PineconeVectorStore is not available."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore', None):
            with pytest.raises(ImportError, match="Pinecone integration requires pinecone-client package"):
                strategy.create_vectorstore(config, embeddings)
    
    def test_load_vectorstore_success(self):
        """Test loading existing vector store successfully."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={"api_key": "test-key", "environment": "us-west1-gcp"}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = strategy.load_vectorstore(config, embeddings)
            
            # Verify PineconeVectorStore constructor was called correctly
            mock_pinecone_class.assert_called_once_with(
                embedding=embeddings,
                index_name="test-collection",
                api_key="test-key",
                environment="us-west1-gcp"
            )
            
            assert result == mock_vectorstore
    
    def test_load_vectorstore_import_error(self):
        """Test loading vector store when PineconeVectorStore is not available."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore', None):
            result = strategy.load_vectorstore(config, embeddings)
            
            # Should return None when import fails
            assert result is None
    
    def test_load_vectorstore_exception_handling(self):
        """Test loading vector store with exception handling."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            # Make PineconeVectorStore constructor raise an exception
            mock_pinecone_class.side_effect = Exception("Pinecone error")
            
            result = strategy.load_vectorstore(config, embeddings)
            
            # Should return None when exception occurs
            assert result is None
    
    def test_validate_config_valid_collection_name(self):
        """Test configuration validation with valid collection name."""
        strategy = PineconeVectorStoreStrategy()
        valid_names = [
            "test-collection",
            "test123",
            "test-collection-123",
            "a",
            "collection-with-hyphens",
            "collection123",
            "test-collection-name"
        ]
        
        for collection_name in valid_names:
            config = VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name=collection_name,
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_empty_collection_name(self):
        """Test configuration validation with empty collection name."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="",
            params={}
        )
        
        with pytest.raises(ValueError, match="collection_name is required for Pinecone"):
            strategy.validate_config(config)
    
    def test_validate_config_none_collection_name(self):
        """Test configuration validation with None collection name."""
        strategy = PineconeVectorStoreStrategy()
        # Pydantic doesn't allow None for collection_name, so we test with empty string instead
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="",
            params={}
        )
        
        with pytest.raises(ValueError, match="collection_name is required for Pinecone"):
            strategy.validate_config(config)
    
    def test_validate_config_uppercase_collection_name(self):
        """Test configuration validation with uppercase collection name."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="TEST-COLLECTION",
            params={}
        )
        
        with pytest.raises(ValueError, match="Pinecone collection_name must be lowercase"):
            strategy.validate_config(config)
    
    def test_validate_config_mixed_case_collection_name(self):
        """Test configuration validation with mixed case collection name."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="Test-Collection",
            params={}
        )
        
        with pytest.raises(ValueError, match="Pinecone collection_name must be lowercase"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_characters_collection_name(self):
        """Test configuration validation with invalid characters in collection name."""
        strategy = PineconeVectorStoreStrategy()
        invalid_names = [
            "test@collection",  # @ symbol
            "test#collection",  # # symbol
            "test$collection",  # $ symbol
            "test%collection",  # % symbol
            "test collection",  # space
            "test.collection",  # dot
            "test/collection",  # slash
            "test\\collection",  # backslash
            "test+collection",  # plus
            "test=collection",  # equals
            "test_collection",  # underscore
        ]
        
        for collection_name in invalid_names:
            config = VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name=collection_name,
                params={}
            )
            
            with pytest.raises(ValueError, match="Pinecone collection_name must contain only letters, numbers, and hyphens"):
                strategy.validate_config(config)
    
    def test_validate_config_unicode_collection_name(self):
        """Test configuration validation with unicode characters in collection name."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="测试-collection",
            params={}
        )
        
        with pytest.raises(ValueError, match="Pinecone collection_name must contain only letters, numbers, and hyphens"):
            strategy.validate_config(config)
    
    def test_validate_config_whitespace_collection_name(self):
        """Test configuration validation with whitespace in collection name."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="  test-collection  ",
            params={}
        )
        
        with pytest.raises(ValueError, match="Pinecone collection_name must contain only letters, numbers, and hyphens"):
            strategy.validate_config(config)
    
    def test_validate_config_too_long_collection_name(self):
        """Test configuration validation with collection name that's too long."""
        strategy = PineconeVectorStoreStrategy()
        long_name = "a" * 46  # 46 characters (over the 45 character limit)
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name=long_name,
            params={}
        )
        
        with pytest.raises(ValueError, match="Pinecone collection_name must be 45 characters or less"):
            strategy.validate_config(config)
    
    def test_validate_config_exactly_45_characters_collection_name(self):
        """Test configuration validation with collection name that's exactly 45 characters."""
        strategy = PineconeVectorStoreStrategy()
        exact_name = "a" * 45  # Exactly 45 characters
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name=exact_name,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)


class TestPineconeVectorStoreStrategyEdgeCases:
    """Test edge cases for PineconeVectorStoreStrategy."""
    
    def test_create_vectorstore_with_complex_params(self):
        """Test creating vector store with complex parameters."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={
                "api_key": "test-key",
                "environment": "us-west1-gcp",
                "namespace": "test-namespace",
                "text_key": "text",
                "pinecone_api_key": "pinecone-key"
            }
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings)
            
            # Verify all params were passed
            mock_pinecone_class.assert_called_once_with(
                embedding=embeddings,
                index_name="test-collection",
                api_key="test-key",
                environment="us-west1-gcp",
                namespace="test-namespace",
                text_key="text",
                pinecone_api_key="pinecone-key"
            )
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = PineconeVectorStoreStrategy()
        original_params = {"api_key": "test-key", "environment": "us-west1-gcp"}
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params=original_params
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_vectorstore_with_none_documents(self):
        """Test creating vector store with None documents."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings, None)
            
            # Should use PineconeVectorStore constructor, not from_documents
            mock_pinecone_class.assert_called_once_with(
                embedding=embeddings,
                index_name="test-collection"
            )
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_with_empty_documents_list(self):
        """Test creating vector store with empty documents list."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        documents = []
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings, documents)
            
            # Empty list evaluates to False, so it uses constructor instead of from_documents
            mock_pinecone_class.assert_called_once_with(
                embedding=embeddings,
                index_name="test-collection"
            )
            
            assert result == mock_vectorstore
    
    def test_validate_config_with_missing_collection_name_attribute(self):
        """Test configuration validation with config that has no collection_name attribute."""
        strategy = PineconeVectorStoreStrategy()
        
        # Create a config object without collection_name attribute
        class ConfigWithoutCollectionName:
            def __init__(self):
                self.type = VectorStoreType.PINECONE
                # No collection_name attribute
        
        config = ConfigWithoutCollectionName()
        
        with pytest.raises(AttributeError):
            strategy.validate_config(config)
    
    def test_validate_config_with_non_string_collection_name(self):
        """Test configuration validation with non-string collection name."""
        strategy = PineconeVectorStoreStrategy()
        
        # Create a config object with non-string collection_name
        class ConfigWithNonStringCollectionName:
            def __init__(self):
                self.type = VectorStoreType.PINECONE
                self.collection_name = 123  # Not a string
        
        config = ConfigWithNonStringCollectionName()
        
        # Should handle non-string collection name gracefully
        with pytest.raises(TypeError, match="collection_name must be a string"):
            strategy.validate_config(config)
    
    def test_validate_config_edge_case_names(self):
        """Test configuration validation with edge case collection names."""
        strategy = PineconeVectorStoreStrategy()
        
        # Test various edge cases
        test_cases = [
            ("a", True),  # Single character
            ("1", True),  # Single digit
            ("a1", True),  # Letter and digit
            ("1a", True),  # Digit and letter
            ("a-1", True),  # Letter, hyphen, digit
            ("1-a", True),  # Digit, hyphen, letter
            ("a1b2c3", True),  # Mixed alphanumeric
            ("a-b-c", True),  # Multiple hyphens
            ("a1-b2-c3", True),  # Mixed with hyphens
        ]
        
        for collection_name, should_pass in test_cases:
            config = VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name=collection_name,
                params={}
            )
            
            if should_pass:
                # Should not raise any exception
                strategy.validate_config(config)
            else:
                # Should raise an exception
                with pytest.raises(ValueError):
                    strategy.validate_config(config)
    
    def test_validate_config_length_boundary_cases(self):
        """Test configuration validation with length boundary cases."""
        strategy = PineconeVectorStoreStrategy()
        
        # Test length boundary cases
        test_cases = [
            ("a" * 1, True),  # 1 character
            ("a" * 10, True),  # 10 characters
            ("a" * 44, True),  # 44 characters
            ("a" * 45, True),  # Exactly 45 characters
            ("a" * 46, False),  # 46 characters (over limit)
            ("a" * 100, False),  # 100 characters (way over limit)
        ]
        
        for collection_name, should_pass in test_cases:
            config = VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name=collection_name,
                params={}
            )
            
            if should_pass:
                # Should not raise any exception
                strategy.validate_config(config)
            else:
                # Should raise an exception
                with pytest.raises(ValueError, match="Pinecone collection_name must be 45 characters or less"):
                    strategy.validate_config(config)


class TestPineconeVectorStoreStrategyIntegration:
    """Integration tests for PineconeVectorStoreStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={"api_key": "test-key", "environment": "us-west1-gcp"}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.from_documents.return_value = mock_vectorstore
            
            # Validate config
            strategy.validate_config(config)
            
            # Create vector store
            result = strategy.create_vectorstore(config, embeddings, documents)
            
            assert result == mock_vectorstore
            mock_pinecone_class.from_documents.assert_called_once()
    
    def test_workflow_with_loading(self):
        """Test workflow with vector store loading."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            # Validate config
            strategy.validate_config(config)
            
            # Load vector store
            result = strategy.load_vectorstore(config, embeddings)
            
            assert result == mock_vectorstore
            mock_pinecone_class.assert_called_once()
    
    def test_workflow_with_multiple_configs(self):
        """Test workflow with multiple different configurations."""
        strategy = PineconeVectorStoreStrategy()
        
        configs = [
            VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name="collection1",
                params={"api_key": "key1", "environment": "us-west1-gcp"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name="collection2",
                params={"api_key": "key2", "environment": "us-east1-gcp"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name="collection3",
                params={}
            )
        ]
        
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            for config in configs:
                # Validate config
                strategy.validate_config(config)
                
                # Create vector store
                result = strategy.create_vectorstore(config, embeddings)
                
                assert result == mock_vectorstore
            
            # Should have been called 3 times
            assert mock_pinecone_class.call_count == 3
    
    def test_workflow_with_error_handling(self):
        """Test workflow with error handling."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="",  # Invalid collection name
            params={}
        )
        embeddings = Mock()
        
        # Should handle validation error
        with pytest.raises(ValueError, match="collection_name is required for Pinecone"):
            strategy.validate_config(config)
    
    def test_workflow_with_import_error(self):
        """Test workflow with import error."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore', None):
            # Should handle import error during creation
            with pytest.raises(ImportError, match="Pinecone integration requires pinecone-client package"):
                strategy.create_vectorstore(config, embeddings)
            
            # Should handle import error during loading
            result = strategy.load_vectorstore(config, embeddings)
            assert result is None
    
    def test_workflow_with_documents_and_loading(self):
        """Test workflow with both document creation and loading."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.from_documents.return_value = mock_vectorstore
            mock_pinecone_class.return_value = mock_vectorstore
            
            # Validate config
            strategy.validate_config(config)
            
            # Create vector store with documents
            result1 = strategy.create_vectorstore(config, embeddings, documents)
            assert result1 == mock_vectorstore
            
            # Load vector store
            result2 = strategy.load_vectorstore(config, embeddings)
            assert result2 == mock_vectorstore
            
            # Both methods should have been called
            mock_pinecone_class.from_documents.assert_called_once()
            mock_pinecone_class.assert_called_once()
    
    def test_workflow_with_exception_handling(self):
        """Test workflow with exception handling during loading."""
        strategy = PineconeVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            # Make PineconeVectorStore constructor raise an exception
            mock_pinecone_class.side_effect = Exception("Pinecone connection error")
            
            # Validate config
            strategy.validate_config(config)
            
            # Load vector store should handle exception gracefully
            result = strategy.load_vectorstore(config, embeddings)
            assert result is None

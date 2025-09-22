"""
Unit tests for Chroma vector store strategy.

Tests all components of the ChromaVectorStoreStrategy including:
- ChromaVectorStoreStrategy class
- Configuration validation
- Vector store creation
- Vector store loading
- Error handling and edge cases
- Directory handling
- Collection name validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

from paas_ai.core.rag.vectorstore.chroma import ChromaVectorStoreStrategy
from paas_ai.core.config.schemas import VectorStoreConfig, VectorStoreType


class TestChromaVectorStoreStrategy:
    """Test the ChromaVectorStoreStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = ChromaVectorStoreStrategy()
        assert strategy is not None
        assert isinstance(strategy, ChromaVectorStoreStrategy)
    
    def test_create_vectorstore_with_documents(self):
        """Test creating vector store with documents."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={"distance_metric": "cosine"}
        )
        embeddings = Mock()
        documents = [
            Document(page_content="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"})
        ]
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_chroma_class.from_documents.return_value = mock_vectorstore
                
                result = strategy.create_vectorstore(config, embeddings, documents)
                
                # Verify directory creation
                mock_path.assert_called_once_with("/tmp/chroma_test")
                mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
                
                # Verify Chroma.from_documents was called correctly
                mock_chroma_class.from_documents.assert_called_once_with(
                    documents=documents,
                    embedding=embeddings,
                    collection_name="test_collection",
                    persist_directory="/tmp/chroma_test",
                    distance_metric="cosine"
                )
                
                assert result == mock_vectorstore
    
    def test_create_vectorstore_without_documents(self):
        """Test creating vector store without documents."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={"distance_metric": "cosine"}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                result = strategy.create_vectorstore(config, embeddings)
                
                # Verify directory creation
                mock_path.assert_called_once_with("/tmp/chroma_test")
                mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
                
                # Verify Chroma constructor was called correctly
                mock_chroma_class.assert_called_once_with(
                    embedding_function=embeddings,
                    collection_name="test_collection",
                    persist_directory="/tmp/chroma_test",
                    distance_metric="cosine"
                )
                
                assert result == mock_vectorstore
    
    def test_create_vectorstore_without_persist_directory(self):
        """Test creating vector store without persist directory."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={"distance_metric": "cosine"}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            mock_vectorstore = Mock()
            mock_chroma_class.from_documents.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings, documents)
            
            # Verify Chroma.from_documents was called with persist_directory=None
            mock_chroma_class.from_documents.assert_called_once_with(
                documents=documents,
                embedding=embeddings,
                collection_name="test_collection",
                persist_directory=None,
                distance_metric="cosine"
            )
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_with_empty_params(self):
        """Test creating vector store with empty params."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            mock_vectorstore = Mock()
            mock_chroma_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings)
            
            # Verify Chroma constructor was called with only required params
            mock_chroma_class.assert_called_once_with(
                embedding_function=embeddings,
                collection_name="test_collection",
                persist_directory=None
            )
            
            assert result == mock_vectorstore
    
    def test_load_vectorstore_success(self):
        """Test loading existing vector store successfully."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={"distance_metric": "cosine"}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                result = strategy.load_vectorstore(config, embeddings)
                
                # Verify path existence check
                mock_path.assert_called_once_with("/tmp/chroma_test")
                mock_path_instance.exists.assert_called_once()
                
                # Verify Chroma constructor was called correctly
                # Note: The implementation calls str(persist_dir) where persist_dir is a mock Path object
                mock_chroma_class.assert_called_once_with(
                    embedding_function=embeddings,
                    collection_name="test_collection",
                    persist_directory=str(mock_path_instance),
                    distance_metric="cosine"
                )
                
                assert result == mock_vectorstore
    
    def test_load_vectorstore_no_persist_directory(self):
        """Test loading vector store without persist directory."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        result = strategy.load_vectorstore(config, embeddings)
        
        # Should return None when no persist directory
        assert result is None
    
    def test_load_vectorstore_directory_not_exists(self):
        """Test loading vector store when directory doesn't exist."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/nonexistent",
            params={}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.exists.return_value = False
            
            result = strategy.load_vectorstore(config, embeddings)
            
            # Should return None when directory doesn't exist
            assert result is None
    
    def test_load_vectorstore_exception_handling(self):
        """Test loading vector store with exception handling."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                # Make Chroma constructor raise an exception
                mock_chroma_class.side_effect = Exception("Chroma error")
                
                result = strategy.load_vectorstore(config, embeddings)
                
                # Should return None when exception occurs
                assert result is None
    
    def test_validate_config_valid_collection_name(self):
        """Test configuration validation with valid collection name."""
        strategy = ChromaVectorStoreStrategy()
        valid_names = [
            "test_collection",
            "test-collection",
            "test123",
            "test_collection_123",
            "test-collection-123",
            "a",
            "collection_with_underscores",
            "collection-with-hyphens"
        ]
        
        for collection_name in valid_names:
            config = VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name=collection_name,
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_empty_collection_name(self):
        """Test configuration validation with empty collection name."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="",
            params={}
        )
        
        with pytest.raises(ValueError, match="collection_name is required for Chroma"):
            strategy.validate_config(config)
    
    def test_validate_config_none_collection_name(self):
        """Test configuration validation with None collection name."""
        strategy = ChromaVectorStoreStrategy()
        # Pydantic doesn't allow None for collection_name, so we test with empty string instead
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="",
            params={}
        )
        
        with pytest.raises(ValueError, match="collection_name is required for Chroma"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_collection_name_characters(self):
        """Test configuration validation with invalid characters in collection name."""
        strategy = ChromaVectorStoreStrategy()
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
        ]
        
        for collection_name in invalid_names:
            config = VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name=collection_name,
                params={}
            )
            
            with pytest.raises(ValueError, match="collection_name must contain only alphanumeric characters, hyphens, and underscores"):
                strategy.validate_config(config)
    
    def test_validate_config_unicode_collection_name(self):
        """Test configuration validation with unicode characters in collection name."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="测试_collection",
            params={}
        )
        
        # Unicode characters are actually allowed by the current validation logic
        # The validation only checks if the cleaned name (without hyphens/underscores) is alphanumeric
        # and unicode characters like "测试" are considered alphanumeric by Python
        strategy.validate_config(config)  # Should not raise
    
    def test_validate_config_whitespace_collection_name(self):
        """Test configuration validation with whitespace in collection name."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="  test_collection  ",
            params={}
        )
        
        with pytest.raises(ValueError, match="collection_name must contain only alphanumeric characters, hyphens, and underscores"):
            strategy.validate_config(config)


class TestChromaVectorStoreStrategyEdgeCases:
    """Test edge cases for ChromaVectorStoreStrategy."""
    
    def test_create_vectorstore_with_path_object(self):
        """Test creating vector store with Path object as persist_directory."""
        strategy = ChromaVectorStoreStrategy()
        persist_path = Path("/tmp/chroma_test")
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory=str(persist_path),
            params={}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                result = strategy.create_vectorstore(config, embeddings)
                
                # Should convert Path to string
                mock_path.assert_called_once_with(str(persist_path))
                assert result == mock_vectorstore
    
    def test_create_vectorstore_with_complex_params(self):
        """Test creating vector store with complex parameters."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={
                "distance_metric": "cosine",
                "collection_metadata": {"description": "test collection"},
                "client_settings": {"host": "localhost", "port": 8000}
            }
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                result = strategy.create_vectorstore(config, embeddings)
                
                # Verify all params were passed
                mock_chroma_class.assert_called_once_with(
                    embedding_function=embeddings,
                    collection_name="test_collection",
                    persist_directory="/tmp/chroma_test",
                    distance_metric="cosine",
                    collection_metadata={"description": "test collection"},
                    client_settings={"host": "localhost", "port": 8000}
                )
                
                assert result == mock_vectorstore
    
    def test_create_vectorstore_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = ChromaVectorStoreStrategy()
        original_params = {"distance_metric": "cosine", "test_param": "value"}
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params=original_params
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            mock_vectorstore = Mock()
            mock_chroma_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_load_vectorstore_with_path_object(self):
        """Test loading vector store with Path object as persist_directory."""
        strategy = ChromaVectorStoreStrategy()
        persist_path = Path("/tmp/chroma_test")
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory=str(persist_path),
            params={}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                result = strategy.load_vectorstore(config, embeddings)
                
                # Should convert Path to string
                mock_path.assert_called_once_with(str(persist_path))
                assert result == mock_vectorstore
    
    def test_create_vectorstore_with_none_documents(self):
        """Test creating vector store with None documents."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            mock_vectorstore = Mock()
            mock_chroma_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings, None)
            
            # Should use Chroma constructor, not from_documents
            mock_chroma_class.assert_called_once_with(
                embedding_function=embeddings,
                collection_name="test_collection",
                persist_directory=None
            )
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_with_empty_documents_list(self):
        """Test creating vector store with empty documents list."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        documents = []
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            mock_vectorstore = Mock()
            mock_chroma_class.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings, documents)
            
            # Empty list evaluates to False, so it uses constructor instead of from_documents
            mock_chroma_class.assert_called_once_with(
                embedding_function=embeddings,
                collection_name="test_collection",
                persist_directory=None
            )
            
            assert result == mock_vectorstore
    
    def test_validate_config_with_missing_collection_name_attribute(self):
        """Test configuration validation with config that has no collection_name attribute."""
        strategy = ChromaVectorStoreStrategy()
        
        # Create a config object without collection_name attribute
        class ConfigWithoutCollectionName:
            def __init__(self):
                self.type = VectorStoreType.CHROMA
                # No collection_name attribute
        
        config = ConfigWithoutCollectionName()
        
        with pytest.raises(AttributeError):
            strategy.validate_config(config)
    
    def test_validate_config_with_non_string_collection_name(self):
        """Test configuration validation with non-string collection name."""
        strategy = ChromaVectorStoreStrategy()
        
        # Create a config object with non-string collection_name
        class ConfigWithNonStringCollectionName:
            def __init__(self):
                self.type = VectorStoreType.CHROMA
                self.collection_name = 123  # Not a string
        
        config = ConfigWithNonStringCollectionName()
        
        # Should handle non-string collection name gracefully
        with pytest.raises(AttributeError):
            strategy.validate_config(config)


class TestChromaVectorStoreStrategyIntegration:
    """Integration tests for ChromaVectorStoreStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={"distance_metric": "cosine"}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_chroma_class.from_documents.return_value = mock_vectorstore
                
                # Validate config
                strategy.validate_config(config)
                
                # Create vector store
                result = strategy.create_vectorstore(config, embeddings, documents)
                
                assert result == mock_vectorstore
                mock_chroma_class.from_documents.assert_called_once()
    
    def test_workflow_with_loading(self):
        """Test workflow with vector store loading."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                # Validate config
                strategy.validate_config(config)
                
                # Load vector store
                result = strategy.load_vectorstore(config, embeddings)
                
                assert result == mock_vectorstore
                mock_chroma_class.assert_called_once()
    
    def test_workflow_with_multiple_configs(self):
        """Test workflow with multiple different configurations."""
        strategy = ChromaVectorStoreStrategy()
        
        configs = [
            VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="collection1",
                persist_directory="/tmp/chroma1",
                params={"distance_metric": "cosine"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="collection2",
                persist_directory="/tmp/chroma2",
                params={"distance_metric": "euclidean"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="collection3",
                params={}
            )
        ]
        
        embeddings = Mock()
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                for config in configs:
                    # Validate config
                    strategy.validate_config(config)
                    
                    # Create vector store
                    result = strategy.create_vectorstore(config, embeddings)
                    
                    assert result == mock_vectorstore
                
                # Should have been called 3 times
                assert mock_chroma_class.call_count == 3
    
    def test_workflow_with_error_handling(self):
        """Test workflow with error handling."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="",  # Invalid collection name
            params={}
        )
        embeddings = Mock()
        
        # Should handle validation error
        with pytest.raises(ValueError, match="collection_name is required for Chroma"):
            strategy.validate_config(config)
    
    def test_workflow_with_documents_and_loading(self):
        """Test workflow with both document creation and loading."""
        strategy = ChromaVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_chroma_class.from_documents.return_value = mock_vectorstore
                mock_chroma_class.return_value = mock_vectorstore
                
                # Validate config
                strategy.validate_config(config)
                
                # Create vector store with documents
                result1 = strategy.create_vectorstore(config, embeddings, documents)
                assert result1 == mock_vectorstore
                
                # Load vector store
                result2 = strategy.load_vectorstore(config, embeddings)
                assert result2 == mock_vectorstore
                
                # Both methods should have been called
                mock_chroma_class.from_documents.assert_called_once()
                mock_chroma_class.assert_called_once()
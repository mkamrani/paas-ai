"""
Unit tests for FAISS vector store strategy.

Tests all components of the FAISSVectorStoreStrategy including:
- FAISSVectorStoreStrategy class
- Configuration validation
- Vector store creation
- Vector store loading
- Error handling and edge cases
- Directory handling
- Empty index creation
- Embedding dimension calculation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

from src.paas_ai.core.rag.vectorstore.faiss import FAISSVectorStoreStrategy
from src.paas_ai.core.config.schemas import VectorStoreConfig, VectorStoreType


class TestFAISSVectorStoreStrategy:
    """Test the FAISSVectorStoreStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = FAISSVectorStoreStrategy()
        assert strategy is not None
        assert isinstance(strategy, FAISSVectorStoreStrategy)
    
    def test_create_vectorstore_with_documents(self):
        """Test creating vector store with documents."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={"index_type": "IndexFlatL2"}
        )
        embeddings = Mock()
        documents = [
            Document(page_content="Test content 1", metadata={"source": "test1"}),
            Document(page_content="Test content 2", metadata={"source": "test2"})
        ]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_faiss_class.from_documents.return_value = mock_vectorstore
                mock_vectorstore.save_local = Mock()
                
                result = strategy.create_vectorstore(config, embeddings, documents)
                
                # Verify directory creation
                mock_path.assert_called_once_with("/tmp/faiss_test")
                mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
                
                # Verify FAISS.from_documents was called correctly
                mock_faiss_class.from_documents.assert_called_once_with(
                    documents=documents,
                    embedding=embeddings,
                    index_type="IndexFlatL2"
                )
                
                # Verify save_local was called
                mock_vectorstore.save_local.assert_called_once_with("/tmp/faiss_test")
                
                assert result == mock_vectorstore
    
    def test_create_vectorstore_without_documents(self):
        """Test creating vector store without documents (empty index)."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={"index_type": "IndexFlatL2"}
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]  # 4-dimensional embedding
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
          with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
              with patch('src.paas_ai.core.rag.vectorstore.faiss.np') as mock_np:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_index = Mock()
                mock_faiss.IndexFlatL2.return_value = mock_index
                
                mock_vectorstore = Mock()
                mock_faiss_class.return_value = mock_vectorstore
                mock_vectorstore.save_local = Mock()
                
                result = strategy.create_vectorstore(config, embeddings)
                
                # Verify embedding dimension calculation
                embeddings.embed_query.assert_called_once_with("sample text for dimension calculation")
                
                # Verify FAISS index creation
                mock_faiss.IndexFlatL2.assert_called_once_with(4)
                
                # Verify FAISS constructor was called correctly
                mock_faiss_class.assert_called_once_with(
                    embedding_function=embeddings,
                    index=mock_index,
                    docstore={},
                    index_to_docstore_id={}
                )
                
                # Verify save_local was called
                mock_vectorstore.save_local.assert_called_once_with("/tmp/faiss_test")
                
                assert result == mock_vectorstore
    
    def test_create_vectorstore_without_persist_directory(self):
        """Test creating vector store without persist directory."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={"index_type": "IndexFlatL2"}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            mock_vectorstore = Mock()
            mock_faiss_class.from_documents.return_value = mock_vectorstore
            
            result = strategy.create_vectorstore(config, embeddings, documents)
            
            # Verify FAISS.from_documents was called correctly
            mock_faiss_class.from_documents.assert_called_once_with(
                documents=documents,
                embedding=embeddings,
                index_type="IndexFlatL2"
            )
            
            # Should not call save_local when no persist directory
            mock_vectorstore.save_local.assert_not_called()
            
            assert result == mock_vectorstore
    
    def test_create_vectorstore_with_empty_params(self):
        """Test creating vector store with empty params."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]  # 3-dimensional embedding
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                mock_index = Mock()
                mock_faiss.IndexFlatL2.return_value = mock_index
                
                mock_vectorstore = Mock()
                mock_faiss_class.return_value = mock_vectorstore
                
                result = strategy.create_vectorstore(config, embeddings)
                
                # Verify FAISS constructor was called with only required params
                mock_faiss_class.assert_called_once_with(
                    embedding_function=embeddings,
                    index=mock_index,
                    docstore={},
                    index_to_docstore_id={}
                )
                
                assert result == mock_vectorstore
    
    def test_load_vectorstore_success(self):
        """Test loading existing vector store successfully."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={"index_type": "IndexFlatL2"}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_faiss_class.load_local.return_value = mock_vectorstore
                
                result = strategy.load_vectorstore(config, embeddings)
                
                # Verify path existence check
                mock_path.assert_called_once_with("/tmp/faiss_test")
                mock_path_instance.exists.assert_called_once()
                
                # Verify FAISS.load_local was called correctly
                # Note: The implementation calls str(persist_dir) where persist_dir is a mock Path object
                mock_faiss_class.load_local.assert_called_once_with(
                    folder_path=str(mock_path_instance),
                    embeddings=embeddings,
                    index_type="IndexFlatL2"
                )
                
                assert result == mock_vectorstore
    
    def test_load_vectorstore_no_persist_directory(self):
        """Test loading vector store without persist directory."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        embeddings = Mock()
        
        result = strategy.load_vectorstore(config, embeddings)
        
        # Should return None when no persist directory
        assert result is None
    
    def test_load_vectorstore_directory_not_exists(self):
        """Test loading vector store when directory doesn't exist."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/nonexistent",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.exists.return_value = False
            
            result = strategy.load_vectorstore(config, embeddings)
            
            # Should return None when directory doesn't exist
            assert result is None
    
    def test_load_vectorstore_exception_handling(self):
        """Test loading vector store with exception handling."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                # Make FAISS.load_local raise an exception
                mock_faiss_class.load_local.side_effect = Exception("FAISS error")
                
                result = strategy.load_vectorstore(config, embeddings)
                
                # Should return None when exception occurs
                assert result is None
    
    def test_validate_config_valid_persist_directory(self):
        """Test configuration validation with valid persist directory."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_no_persist_directory(self):
        """Test configuration validation without persist directory."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_invalid_persist_directory_parent(self):
        """Test configuration validation with invalid persist directory parent."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.parent = Mock()
            mock_path_instance.parent.exists.return_value = True
            mock_path_instance.parent.is_dir.return_value = False  # Parent exists but is not a directory
            
            with pytest.raises(ValueError, match="persist_directory parent must be a directory"):
                strategy.validate_config(config)
    
    def test_validate_config_persist_directory_parent_not_exists(self):
        """Test configuration validation when persist directory parent doesn't exist."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.parent = Mock()
            mock_path_instance.parent.exists.return_value = False  # Parent doesn't exist
            
            # Should not raise error when parent doesn't exist
            strategy.validate_config(config)


class TestFAISSVectorStoreStrategyEdgeCases:
    """Test edge cases for FAISSVectorStoreStrategy."""
    
    def test_create_vectorstore_with_path_object(self):
        """Test creating vector store with Path object as persist_directory."""
        strategy = FAISSVectorStoreStrategy()
        persist_path = Path("/tmp/faiss_test")
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory=str(persist_path),
            params={}
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                    mock_path_instance = Mock()
                    mock_path.return_value = mock_path_instance
                    mock_path_instance.mkdir = Mock()
                    
                    mock_index = Mock()
                    mock_faiss.IndexFlatL2.return_value = mock_index
                    
                    mock_vectorstore = Mock()
                    mock_faiss_class.return_value = mock_vectorstore
                    mock_vectorstore.save_local = Mock()
                    
                    result = strategy.create_vectorstore(config, embeddings)
                    
                    # Should convert Path to string
                    mock_path.assert_called_once_with(str(persist_path))
                    assert result == mock_vectorstore
    
    def test_create_vectorstore_with_complex_params(self):
        """Test creating vector store with complex parameters."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={
                "index_type": "IndexFlatIP",
                "metric": "cosine",
                "normalize_embeddings": True
            }
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                    mock_path_instance = Mock()
                    mock_path.return_value = mock_path_instance
                    mock_path_instance.mkdir = Mock()
                    
                    mock_index = Mock()
                    mock_faiss.IndexFlatL2.return_value = mock_index
                    
                    mock_vectorstore = Mock()
                    mock_faiss_class.return_value = mock_vectorstore
                    mock_vectorstore.save_local = Mock()
                    
                    result = strategy.create_vectorstore(config, embeddings)
                    
                    # Verify FAISS constructor was called correctly
                    # Note: FAISS constructor doesn't accept index_type, metric, normalize_embeddings
                    mock_faiss_class.assert_called_once_with(
                        embedding_function=embeddings,
                        index=mock_index,
                        docstore={},
                        index_to_docstore_id={}
                    )
                    
                    assert result == mock_vectorstore
    
    def test_create_vectorstore_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = FAISSVectorStoreStrategy()
        original_params = {"index_type": "IndexFlatL2", "test_param": "value"}
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params=original_params
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                mock_index = Mock()
                mock_faiss.IndexFlatL2.return_value = mock_index
                
                mock_vectorstore = Mock()
                mock_faiss_class.return_value = mock_vectorstore
                
                result = strategy.create_vectorstore(config, embeddings)
                
                # Original params should not be modified
                assert config.params == original_params
                assert config.params is not original_params  # Should be a copy
    
    def test_create_vectorstore_with_different_embedding_dimensions(self):
        """Test creating vector store with different embedding dimensions."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        embeddings = Mock()
        
        test_cases = [
            ([0.1], 1),  # 1-dimensional
            ([0.1, 0.2], 2),  # 2-dimensional
            ([0.1, 0.2, 0.3, 0.4, 0.5], 5),  # 5-dimensional
            ([0.1] * 100, 100),  # 100-dimensional
        ]
        
        for embedding_vector, expected_dimension in test_cases:
            embeddings.embed_query.return_value = embedding_vector
            
            with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
                with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                    mock_index = Mock()
                    mock_faiss.IndexFlatL2.return_value = mock_index
                    
                    mock_vectorstore = Mock()
                    mock_faiss_class.return_value = mock_vectorstore
                    
                    result = strategy.create_vectorstore(config, embeddings)
                    
                    # Verify correct dimension was used
                    mock_faiss.IndexFlatL2.assert_called_with(expected_dimension)
                    assert result == mock_vectorstore
    
    def test_create_vectorstore_with_none_documents(self):
        """Test creating vector store with None documents."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                mock_index = Mock()
                mock_faiss.IndexFlatL2.return_value = mock_index
                
                mock_vectorstore = Mock()
                mock_faiss_class.return_value = mock_vectorstore
                
                result = strategy.create_vectorstore(config, embeddings, None)
                
                # Should use FAISS constructor, not from_documents
                mock_faiss_class.assert_called_once_with(
                    embedding_function=embeddings,
                    index=mock_index,
                    docstore={},
                    index_to_docstore_id={}
                )
                
                assert result == mock_vectorstore
    
    def test_create_vectorstore_with_empty_documents_list(self):
        """Test creating vector store with empty documents list."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]  # Return a list for len() to work
        documents = []
    
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                mock_index = Mock()
                mock_faiss.IndexFlatL2.return_value = mock_index
    
                mock_vectorstore = Mock()
                mock_faiss_class.return_value = mock_vectorstore
                mock_vectorstore.save_local = Mock()
    
                result = strategy.create_vectorstore(config, embeddings, documents)
                
                # Empty list evaluates to False, so it uses constructor instead of from_documents
                mock_faiss_class.assert_called_once_with(
                    embedding_function=embeddings,
                    index=mock_index,
                    docstore={},
                    index_to_docstore_id={}
                )
                
                assert result == mock_vectorstore
    
    def test_load_vectorstore_with_path_object(self):
        """Test loading vector store with Path object as persist_directory."""
        strategy = FAISSVectorStoreStrategy()
        persist_path = Path("/tmp/faiss_test")
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory=str(persist_path),
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_faiss_class.load_local.return_value = mock_vectorstore
                
                result = strategy.load_vectorstore(config, embeddings)
                
                # Should convert Path to string
                mock_path.assert_called_once_with(str(persist_path))
                assert result == mock_vectorstore
    
    def test_validate_config_with_missing_persist_directory_attribute(self):
        """Test configuration validation with config that has no persist_directory attribute."""
        strategy = FAISSVectorStoreStrategy()
        
        # Create a config object without persist_directory attribute
        class ConfigWithoutPersistDirectory:
            def __init__(self):
                self.type = VectorStoreType.FAISS
                # No persist_directory attribute
        
        config = ConfigWithoutPersistDirectory()
        
        # Should not raise error - persist_directory is optional
        strategy.validate_config(config)
    
    def test_validate_config_with_non_string_persist_directory(self):
        """Test configuration validation with non-string persist_directory."""
        strategy = FAISSVectorStoreStrategy()
        
        # Create a config object with non-string persist_directory
        class ConfigWithNonStringPersistDirectory:
            def __init__(self):
                self.type = VectorStoreType.FAISS
                self.persist_directory = 123  # Not a string
        
        config = ConfigWithNonStringPersistDirectory()
        
        # Should handle non-string persist_directory gracefully
        with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.parent = Mock()
            mock_path_instance.parent.exists.return_value = False
            
            # Should not raise error
            strategy.validate_config(config)


class TestFAISSVectorStoreStrategyIntegration:
    """Integration tests for FAISSVectorStoreStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={"index_type": "IndexFlatL2"}
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_faiss_class.from_documents.return_value = mock_vectorstore
                mock_vectorstore.save_local = Mock()
                
                # Validate config
                strategy.validate_config(config)
                
                # Create vector store
                result = strategy.create_vectorstore(config, embeddings, documents)
                
                assert result == mock_vectorstore
                mock_faiss_class.from_documents.assert_called_once()
                mock_vectorstore.save_local.assert_called_once()
    
    def test_workflow_with_loading(self):
        """Test workflow with vector store loading."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        embeddings = Mock()
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_faiss_class.load_local.return_value = mock_vectorstore
                
                # Validate config
                strategy.validate_config(config)
                
                # Load vector store
                result = strategy.load_vectorstore(config, embeddings)
                
                assert result == mock_vectorstore
                mock_faiss_class.load_local.assert_called_once()
    
    def test_workflow_with_multiple_configs(self):
        """Test workflow with multiple different configurations."""
        strategy = FAISSVectorStoreStrategy()
        
        configs = [
            VectorStoreConfig(
                type=VectorStoreType.FAISS,
                persist_directory="/tmp/faiss1",
                params={"index_type": "IndexFlatL2"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.FAISS,
                persist_directory="/tmp/faiss2",
                params={"index_type": "IndexFlatIP"}
            ),
            VectorStoreConfig(
                type=VectorStoreType.FAISS,
                params={}
            )
        ]
        
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                    mock_path_instance = Mock()
                    mock_path.return_value = mock_path_instance
                    mock_path_instance.mkdir = Mock()
                    
                    mock_index = Mock()
                    mock_faiss.IndexFlatL2.return_value = mock_index
                    
                    mock_vectorstore = Mock()
                    mock_faiss_class.return_value = mock_vectorstore
                    mock_vectorstore.save_local = Mock()
                    
                    for config in configs:
                        # Validate config
                        strategy.validate_config(config)
                        
                        # Create vector store
                        result = strategy.create_vectorstore(config, embeddings)
                        
                        assert result == mock_vectorstore
                    
                    # Should have been called 3 times
                    assert mock_faiss_class.call_count == 3
    
    def test_workflow_with_error_handling(self):
        """Test workflow with error handling."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
            mock_path_instance = Mock()
            mock_path.return_value = mock_path_instance
            mock_path_instance.parent = Mock()
            mock_path_instance.parent.exists.return_value = True
            mock_path_instance.parent.is_dir.return_value = False  # Invalid parent
            
            # Should handle validation error
            with pytest.raises(ValueError, match="persist_directory parent must be a directory"):
                strategy.validate_config(config)
    
    def test_workflow_with_documents_and_loading(self):
        """Test workflow with both document creation and loading."""
        strategy = FAISSVectorStoreStrategy()
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                with patch('src.paas_ai.core.rag.vectorstore.faiss.faiss') as mock_faiss:
                    mock_path_instance = Mock()
                    mock_path.return_value = mock_path_instance
                    mock_path_instance.mkdir = Mock()
                    mock_path_instance.exists.return_value = True
                    
                    mock_index = Mock()
                    mock_faiss.IndexFlatL2.return_value = mock_index
                    
                    mock_vectorstore = Mock()
                    mock_faiss_class.from_documents.return_value = mock_vectorstore
                    mock_faiss_class.load_local.return_value = mock_vectorstore
                    mock_vectorstore.save_local = Mock()
                    
                    # Validate config
                    strategy.validate_config(config)
                    
                    # Create vector store with documents
                    result1 = strategy.create_vectorstore(config, embeddings, documents)
                    assert result1 == mock_vectorstore
                    
                    # Load vector store
                    result2 = strategy.load_vectorstore(config, embeddings)
                    assert result2 == mock_vectorstore
                    
                    # Both methods should have been called
                    mock_faiss_class.from_documents.assert_called_once()
                    mock_faiss_class.load_local.assert_called_once()

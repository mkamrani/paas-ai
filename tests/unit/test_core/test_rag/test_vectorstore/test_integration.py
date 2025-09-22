"""
Integration tests for vector store module.

Tests the integration between different components of the vector store module including:
- End-to-end workflows
- Cross-component interactions
- Real-world usage scenarios
- Error propagation
- Configuration flow
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

from src.paas_ai.core.rag.vectorstore.factory import VectorStoreFactory
from src.paas_ai.core.rag.vectorstore.chroma import ChromaVectorStoreStrategy
from src.paas_ai.core.rag.vectorstore.faiss import FAISSVectorStoreStrategy
from src.paas_ai.core.rag.vectorstore.pinecone import PineconeVectorStoreStrategy
from src.paas_ai.core.config.schemas import VectorStoreConfig, VectorStoreType


@pytest.fixture
def mock_vectorstore():
    """Fixture providing a consistent mock vectorstore."""
    return Mock()


@pytest.fixture
def mock_embeddings():
    """Fixture providing a consistent mock embeddings object."""
    mock_emb = Mock()
    mock_emb.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock_emb


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        Document(page_content="Test content", metadata={"source": "test"})
    ]


@pytest.fixture(autouse=True)
def reset_factory_registry():
    """Fixture to reset the factory registry after each test to ensure isolation."""
    # Store original strategies
    original_strategies = VectorStoreFactory._strategies.copy()
    
    yield
    
    # Restore original strategies after test
    VectorStoreFactory._strategies = original_strategies


class TestVectorStoreIntegration:
    """Integration tests for vector store components."""
    
    def test_factory_with_chroma_strategy(self):
        """Test factory integration with Chroma strategy."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        embeddings = Mock()
        documents = [Document(page_content="Test content", metadata={"source": "test"})]
        
        with patch('src.paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('src.paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.mkdir = Mock()
                
                mock_vectorstore = Mock()
                mock_chroma_class.from_documents.return_value = mock_vectorstore
                
                # Test factory creation
                result = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
                
                # Verify Chroma was called correctly
                mock_chroma_class.from_documents.assert_called_once_with(
                    documents=documents,
                    embedding=embeddings,
                    collection_name="test_collection",
                    persist_directory="/tmp/chroma_test",
                )
                
                assert result == mock_vectorstore
    
    def test_factory_with_faiss_strategy(self):
        """Test factory integration with FAISS strategy."""
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
                
                # Test factory creation
                result = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
                
                # Verify FAISS was called correctly
                mock_faiss_class.from_documents.assert_called_once_with(
                    documents=documents,
                    embedding=embeddings,
                    index_type="IndexFlatL2"
                )
                
                assert result == mock_vectorstore
    
    def test_factory_with_pinecone_strategy(self):
        """Test factory integration with Pinecone strategy."""
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
            
            # Test factory creation
            result = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
            
            # Verify Pinecone was called correctly
            mock_pinecone_class.from_documents.assert_called_once_with(
                documents=documents,
                embedding=embeddings,
                index_name="test-collection",
                api_key="test-key",
                environment="us-west1-gcp"
            )
            
            assert result == mock_vectorstore
    
    def test_factory_validation_flow(self):
        """Test factory validation flow with different strategies."""
        test_cases = [
            # Chroma - valid
            (VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="test_collection",
                params={}
            ), True),
            # Chroma - invalid (empty collection name)
            (VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="",
                params={}
            ), False),
            # FAISS - valid
            (VectorStoreConfig(
                type=VectorStoreType.FAISS,
                persist_directory="/tmp/faiss_test",
                params={}
            ), True),
            # Pinecone - valid
            (VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name="test-collection",
                params={}
            ), True),
            # Pinecone - invalid (uppercase collection name)
            (VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name="TEST-COLLECTION",
                params={}
            ), False),
        ]
        
        for config, should_pass in test_cases:
            if should_pass:
                # Should not raise any exception
                VectorStoreFactory._validate_config(config)
            else:
                # Should raise an exception
                with pytest.raises(ValueError):
                    VectorStoreFactory._validate_config(config)
    
    def test_factory_loading_flow(self):
        """Test factory loading flow with different strategies."""
        embeddings = Mock()
        
        # Test Chroma loading
        chroma_config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch('src.paas_ai.core.rag.vectorstore.chroma.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_chroma_class.return_value = mock_vectorstore
                
                result = VectorStoreFactory.load_vectorstore(chroma_config, embeddings)
                assert result == mock_vectorstore
        
        # Test FAISS loading
        faiss_config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path.return_value = mock_path_instance
                mock_path_instance.exists.return_value = True
                
                mock_vectorstore = Mock()
                mock_faiss_class.load_local.return_value = mock_vectorstore
                
                result = VectorStoreFactory.load_vectorstore(faiss_config, embeddings)
                assert result == mock_vectorstore
        
        # Test Pinecone loading
        pinecone_config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.return_value = mock_vectorstore
            
            result = VectorStoreFactory.load_vectorstore(pinecone_config, embeddings)
            assert result == mock_vectorstore
    
    def test_strategy_registration_workflow(self):
        """Test strategy registration workflow."""
        class CustomStrategy:
            def create_vectorstore(self, config, embeddings, documents=None):
                return Mock()
            
            def load_vectorstore(self, config, embeddings):
                return Mock()
            
            def validate_config(self, config):
                pass
        
        # Register custom strategy
        VectorStoreFactory.register_strategy(VectorStoreType.CHROMA, CustomStrategy)
        
        # Verify it was registered
        assert VectorStoreFactory._strategies[VectorStoreType.CHROMA] == CustomStrategy
        
        # Test using the registered strategy
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        embeddings = Mock()
        
        result = VectorStoreFactory.create_vectorstore(config, embeddings)
        assert result is not None
    
    def test_error_propagation_workflow(self, mock_embeddings):
        """Test error propagation through the factory."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        
        # Test validation error propagation using patch
        with patch.object(VectorStoreFactory, '_validate_config') as mock_validate:
            mock_validate.side_effect = ValueError("Custom validation error")
            
            with pytest.raises(ValueError, match="Custom validation error"):
                VectorStoreFactory.create_vectorstore(config, mock_embeddings)
        
        # Test creation error propagation
        with patch('src.paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            mock_chroma_class.side_effect = Exception("Chroma creation error")
            
            with pytest.raises(Exception, match="Chroma creation error"):
                VectorStoreFactory.create_vectorstore(config, mock_embeddings)
    
    def test_configuration_flow_workflow(self):
        """Test configuration flow through different components."""
        # Test with different configuration scenarios
        configs = [
            # Minimal config
            VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="minimal_collection",
                params={}
            ),
            # Full config
            VectorStoreConfig(
                type=VectorStoreType.CHROMA,
                collection_name="full_collection",
                persist_directory="/tmp/full_test",
                params={
                    "collection_metadata": {"description": "full test collection"}
                }
            ),
            # FAISS config
            VectorStoreConfig(
                type=VectorStoreType.FAISS,
                persist_directory="/tmp/faiss_test",
                params={"index_type": "IndexFlatL2"}
            ),
            # Pinecone config
            VectorStoreConfig(
                type=VectorStoreType.PINECONE,
                collection_name="pinecone-collection",
                params={"api_key": "test-key", "environment": "us-west1-gcp"}
            )
        ]
        
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        for config in configs:
            # Test validation
            VectorStoreFactory._validate_config(config)
            
            # Test creation based on config type
            if config.type == VectorStoreType.CHROMA:
                with patch('src.paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
                    with patch.object(Path, 'mkdir') as mock_mkdir:
                        mock_vectorstore = Mock()
                        mock_chroma_class.return_value = mock_vectorstore
                        
                        result = VectorStoreFactory.create_vectorstore(config, embeddings)
                        # Verify the mock was called and returned the expected result
                        mock_chroma_class.assert_called_once()
                        assert result is not None
                        
            elif config.type == VectorStoreType.FAISS:
                with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
                    with patch.object(Path, 'mkdir') as mock_mkdir:
                        mock_vectorstore = Mock()
                        mock_faiss_class.return_value = mock_vectorstore
                        mock_vectorstore.save_local = Mock()
                        
                        result = VectorStoreFactory.create_vectorstore(config, embeddings)
                        # Verify the mock was called and returned the expected result
                        mock_faiss_class.assert_called_once()
                        assert result is not None
                        
            elif config.type == VectorStoreType.PINECONE:
                with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
                    mock_vectorstore = Mock()
                    mock_pinecone_class.return_value = mock_vectorstore
                    
                    result = VectorStoreFactory.create_vectorstore(config, embeddings)
                    # Verify the mock was called and returned the expected result
                    mock_pinecone_class.assert_called_once()
                    assert result is not None
    
    def test_document_handling_workflow(self):
        """Test document handling workflow across different strategies."""
        documents = [
            Document(page_content="Document 1", metadata={"source": "doc1", "page": 1}),
            Document(page_content="Document 2", metadata={"source": "doc2", "page": 2}),
            Document(page_content="Document 3", metadata={"source": "doc3", "page": 3})
        ]
        
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Test with Chroma
        chroma_config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch.object(Path, 'mkdir') as mock_mkdir:
                mock_vectorstore = Mock()
                mock_chroma_class.from_documents.return_value = mock_vectorstore
                
                result = VectorStoreFactory.create_vectorstore(chroma_config, embeddings, documents)
                
                # Verify documents were passed correctly
                mock_chroma_class.from_documents.assert_called_once()
                call_args = mock_chroma_class.from_documents.call_args
                assert call_args[1]['documents'] == documents
                
                assert result == mock_vectorstore
        
        # Test with FAISS
        faiss_config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch.object(Path, 'mkdir') as mock_mkdir:
                mock_vectorstore = Mock()
                mock_faiss_class.from_documents.return_value = mock_vectorstore
                mock_vectorstore.save_local = Mock()
                
                result = VectorStoreFactory.create_vectorstore(faiss_config, embeddings, documents)
                
                # Verify documents were passed correctly
                mock_faiss_class.from_documents.assert_called_once()
                call_args = mock_faiss_class.from_documents.call_args
                assert call_args[1]['documents'] == documents
                
                assert result == mock_vectorstore
        
        # Test with Pinecone
        pinecone_config = VectorStoreConfig(
            type=VectorStoreType.PINECONE,
            collection_name="test-collection",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.pinecone.PineconeVectorStore') as mock_pinecone_class:
            mock_vectorstore = Mock()
            mock_pinecone_class.from_documents.return_value = mock_vectorstore
            
            result = VectorStoreFactory.create_vectorstore(pinecone_config, embeddings, documents)
            
            # Verify documents were passed correctly
            mock_pinecone_class.from_documents.assert_called_once()
            call_args = mock_pinecone_class.from_documents.call_args
            assert call_args[1]['documents'] == documents
            
            assert result == mock_vectorstore
    
    def test_embeddings_integration_workflow(self):
        """Test embeddings integration workflow."""
        # Test with different embedding configurations
        embeddings_configs = [
            Mock(embed_query=Mock(return_value=[0.1, 0.2, 0.3])),  # 3D embeddings
            Mock(embed_query=Mock(return_value=[0.1, 0.2, 0.3, 0.4])),  # 4D embeddings
            Mock(embed_query=Mock(return_value=[0.1] * 100)),  # 100D embeddings
        ]
        
        config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            params={}
        )
        
        for embeddings in embeddings_configs:
            with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
                mock_vectorstore = Mock()
                mock_faiss_class.return_value = mock_vectorstore
                
                result = VectorStoreFactory.create_vectorstore(config, embeddings)
                
                # Verify embeddings were passed correctly
                mock_faiss_class.assert_called_once()
                call_args = mock_faiss_class.call_args
                assert call_args[1]['embedding_function'] == embeddings
                
                assert result == mock_vectorstore
    
    def test_persistence_workflow(self):
        """Test persistence workflow across different strategies."""
        embeddings = Mock()
        embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # Test Chroma persistence
        chroma_config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="test_collection",
            persist_directory="/tmp/chroma_test",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch.object(Path, 'mkdir') as mock_mkdir:
                with patch.object(Path, 'exists', return_value=True) as mock_exists:
                    # Test creation with persistence (no documents, uses constructor)
                    mock_vectorstore_created = Mock()
                    mock_chroma_class.return_value = mock_vectorstore_created
                    
                    result1 = VectorStoreFactory.create_vectorstore(chroma_config, embeddings)
                    assert result1 == mock_vectorstore_created
                    
                    # Reset mock for loading test
                    mock_chroma_class.reset_mock()
                    
                    # Test loading from persistence (uses constructor)
                    mock_vectorstore_loaded = Mock()
                    mock_chroma_class.return_value = mock_vectorstore_loaded
                    
                    result2 = VectorStoreFactory.load_vectorstore(chroma_config, embeddings)
                    assert result2 == mock_vectorstore_loaded
                    
                    # Verify directory operations
                    mock_mkdir.assert_called_once()
                    mock_exists.assert_called_once()
        
        # Test FAISS persistence
        faiss_config = VectorStoreConfig(
            type=VectorStoreType.FAISS,
            persist_directory="/tmp/faiss_test",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.vectorstore.faiss.FAISS') as mock_faiss_class:
            with patch.object(Path, 'mkdir') as mock_mkdir:
                with patch.object(Path, 'exists', return_value=True) as mock_exists:
                    mock_vectorstore = Mock()
                    mock_faiss_class.return_value = mock_vectorstore
                    mock_faiss_class.load_local.return_value = mock_vectorstore
                    mock_vectorstore.save_local = Mock()
                    
                    # Test creation with persistence
                    result1 = VectorStoreFactory.create_vectorstore(faiss_config, embeddings)
                    assert result1 == mock_vectorstore
                    
                    # Test loading from persistence
                    result2 = VectorStoreFactory.load_vectorstore(faiss_config, embeddings)
                    assert result2 == mock_vectorstore
                    
                    # Verify save operation was called
                    mock_vectorstore.save_local.assert_called_once_with("/tmp/faiss_test")
                    mock_faiss_class.load_local.assert_called_once()
    
    def test_comprehensive_workflow(self):
        """Test comprehensive end-to-end workflow."""
        config = VectorStoreConfig(
            type=VectorStoreType.CHROMA,
            collection_name="comprehensive_test",
            persist_directory="/tmp/comprehensive_test",
            params={
                "collection_metadata": {"description": "comprehensive test"}
            }
        )
        embeddings = Mock()
        documents = [
            Document(page_content="Comprehensive test document", metadata={"source": "test"})
        ]

        with patch('src.paas_ai.core.rag.vectorstore.chroma.Chroma') as mock_chroma_class:
            with patch.object(Path, 'mkdir') as mock_mkdir:
                with patch.object(Path, 'exists', return_value=True) as mock_exists:
                    # Step 1: Validate configuration
                    VectorStoreFactory._validate_config(config)

                    # Step 2: Create vector store with documents (uses from_documents)
                    mock_vectorstore_with_docs = Mock()
                    mock_chroma_class.from_documents.return_value = mock_vectorstore_with_docs

                    result1 = VectorStoreFactory.create_vectorstore(config, embeddings, documents)
                    assert result1 == mock_vectorstore_with_docs

                    # Step 3: Load vector store (uses constructor)
                    mock_vectorstore_loaded = Mock()
                    mock_chroma_class.return_value = mock_vectorstore_loaded

                    result2 = VectorStoreFactory.load_vectorstore(config, embeddings)
                    assert result2 == mock_vectorstore_loaded

                    # Step 4: Create vector store without documents (uses constructor)
                    mock_vectorstore_no_docs = Mock()
                    mock_chroma_class.return_value = mock_vectorstore_no_docs

                    result3 = VectorStoreFactory.create_vectorstore(config, embeddings)
                    assert result3 == mock_vectorstore_no_docs

                    # Verify all operations were called
                    mock_chroma_class.from_documents.assert_called_once()
                    mock_chroma_class.assert_called()
                    mock_mkdir.assert_called()
                    mock_exists.assert_called()

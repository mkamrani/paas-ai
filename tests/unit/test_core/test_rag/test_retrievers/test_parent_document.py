"""
Unit tests for parent document retriever strategy.

Tests all components of the ParentDocumentRetrieverStrategy including:
- ParentDocumentRetrieverStrategy class
- Configuration validation
- Retriever creation
- Error handling and edge cases
- Parent document specific parameter validation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.retrievers.parent_document import ParentDocumentRetrieverStrategy
from src.paas_ai.core.rag.config import RetrieverConfig, RetrieverType


class TestParentDocumentRetrieverStrategy:
    """Test the ParentDocumentRetrieverStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = ParentDocumentRetrieverStrategy()
        assert strategy is not None
        assert isinstance(strategy, ParentDocumentRetrieverStrategy)
    
    def test_create_retriever_basic(self):
        """Test creating retriever with basic configuration."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_parent_doc_class.assert_called_once_with(
                vectorstore=vectorstore,
                docstore=None,
                child_splitter=mock_child_splitter
            )
            assert result == mock_parent_doc
    
    def test_create_retriever_with_docstore(self):
        """Test creating retriever with docstore."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter and docstore
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        mock_docstore = Mock()
        mock_docstore.add = Mock()
        mock_docstore.search = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "docstore": mock_docstore
            }
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_parent_doc_class.assert_called_once_with(
                vectorstore=vectorstore,
                docstore=mock_docstore,
                child_splitter=mock_child_splitter
            )
            assert result == mock_parent_doc
    
    def test_create_retriever_with_parent_splitter(self):
        """Test creating retriever with parent splitter."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child and parent splitters
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        mock_parent_splitter = Mock()
        mock_parent_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "parent_splitter": mock_parent_splitter
            }
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_parent_doc_class.assert_called_once_with(
                vectorstore=vectorstore,
                docstore=None,
                child_splitter=mock_child_splitter,
                parent_splitter=mock_parent_splitter
            )
            assert result == mock_parent_doc
    
    def test_create_retriever_with_additional_params(self):
        """Test creating retriever with additional parameters."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "custom_param": "value",
                "another_param": 42
            }
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_parent_doc_class.assert_called_once_with(
                vectorstore=vectorstore,
                docstore=None,
                child_splitter=mock_child_splitter,
                custom_param="value",
                another_param=42
            )
            assert result == mock_parent_doc
    
    def test_create_retriever_with_llm(self):
        """Test creating retriever with LLM parameter (should be ignored)."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter}
        )
        vectorstore = Mock()
        llm = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore, llm)
            
            # LLM should be ignored for parent document retriever
            assert result == mock_parent_doc
    
    def test_create_retriever_missing_child_splitter(self):
        """Test creating retriever without child_splitter (should raise error)."""
        strategy = ParentDocumentRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={}  # No child_splitter
        )
        vectorstore = Mock()
        
        with pytest.raises(ValueError, match="child_splitter is required for ParentDocumentRetriever"):
            strategy.create_retriever(config, vectorstore)
    
    def test_validate_config_with_valid_parameters(self):
        """Test configuration validation with valid parameters."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_valid_docstore(self):
        """Test configuration validation with valid docstore."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter and docstore
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        mock_docstore = Mock()
        mock_docstore.add = Mock()
        mock_docstore.search = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "docstore": mock_docstore
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_valid_parent_splitter(self):
        """Test configuration validation with valid parent splitter."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child and parent splitters
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        mock_parent_splitter = Mock()
        mock_parent_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "parent_splitter": mock_parent_splitter
            }
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_missing_child_splitter(self):
        """Test configuration validation without child_splitter."""
        strategy = ParentDocumentRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={}  # No child_splitter
        )
        
        with pytest.raises(ValueError, match="params\\['child_splitter'\\] is required for ParentDocumentRetriever"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_child_splitter_no_split_documents(self):
        """Test configuration validation with child_splitter missing split_documents method."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter without split_documents method
        mock_child_splitter = Mock()
        del mock_child_splitter.split_documents  # Remove the method
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter}
        )
        
        with pytest.raises(ValueError, match="params\\['child_splitter'\\] must have a split_documents method"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_docstore_no_add_method(self):
        """Test configuration validation with docstore missing add method."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter and invalid docstore
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        mock_docstore = Mock()
        mock_docstore.search = Mock()
        del mock_docstore.add  # Remove the add method
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "docstore": mock_docstore
            }
        )
        
        with pytest.raises(ValueError, match="params\\['docstore'\\] must have add and search methods"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_docstore_no_search_method(self):
        """Test configuration validation with docstore missing search method."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter and invalid docstore
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        mock_docstore = Mock()
        mock_docstore.add = Mock()
        del mock_docstore.search  # Remove the search method
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "docstore": mock_docstore
            }
        )
        
        with pytest.raises(ValueError, match="params\\['docstore'\\] must have add and search methods"):
            strategy.validate_config(config)
    
    def test_validate_config_invalid_parent_splitter_no_split_documents(self):
        """Test configuration validation with parent_splitter missing split_documents method."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter and invalid parent splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        mock_parent_splitter = Mock()
        del mock_parent_splitter.split_documents  # Remove the method
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "parent_splitter": mock_parent_splitter
            }
        )
        
        with pytest.raises(ValueError, match="params\\['parent_splitter'\\] must have a split_documents method"):
            strategy.validate_config(config)


class TestParentDocumentRetrieverStrategyEdgeCases:
    """Test edge cases for ParentDocumentRetrieverStrategy."""
    
    def test_create_retriever_with_special_characters_in_params(self):
        """Test creating retriever with special characters in parameters."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "custom_param": "value with spaces",
                "unicode_param": "测试参数",
                "special_chars": "!@#$%^&*()",
                "nested_dict": {"key": "value"},
                "list_param": [1, 2, 3]
            }
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore)
            
            mock_parent_doc_class.assert_called_once_with(
                vectorstore=vectorstore,
                docstore=None,
                child_splitter=mock_child_splitter,
                custom_param="value with spaces",
                unicode_param="测试参数",
                special_chars="!@#$%^&*()",
                nested_dict={"key": "value"},
                list_param=[1, 2, 3]
            )
            assert result == mock_parent_doc
    
    def test_create_retriever_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        original_params = {"child_splitter": mock_child_splitter, "custom_param": "value"}
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params=original_params
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_create_retriever_search_kwargs_copy_behavior(self):
        """Test that search_kwargs are copied and not modified."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        original_search_kwargs = {"k": 5}
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs=original_search_kwargs,
            params={"child_splitter": mock_child_splitter}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            result = strategy.create_retriever(config, vectorstore)
            
            # Original search_kwargs should not be modified
            assert config.search_kwargs == original_search_kwargs
            assert config.search_kwargs is not original_search_kwargs  # Should be a copy
    
    def test_validate_config_with_none_child_splitter(self):
        """Test configuration validation with None child_splitter."""
        strategy = ParentDocumentRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": None}
        )
        
        with pytest.raises(ValueError, match="params\\['child_splitter'\\] must have a split_documents method"):
            strategy.validate_config(config)
    
    def test_validate_config_with_string_child_splitter(self):
        """Test configuration validation with string child_splitter."""
        strategy = ParentDocumentRetrieverStrategy()
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": "not-a-splitter"}
        )
        
        with pytest.raises(ValueError, match="params\\['child_splitter'\\] must have a split_documents method"):
            strategy.validate_config(config)
    
    def test_validate_config_with_none_docstore(self):
        """Test configuration validation with None docstore."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "docstore": None
            }
        )
        
        with pytest.raises(ValueError, match="params\\['docstore'\\] must have add and search methods"):
            strategy.validate_config(config)
    
    def test_validate_config_with_none_parent_splitter(self):
        """Test configuration validation with None parent_splitter."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": mock_child_splitter,
                "parent_splitter": None
            }
        )
        
        with pytest.raises(ValueError, match="params\\['parent_splitter'\\] must have a split_documents method"):
            strategy.validate_config(config)
    
    def test_validate_config_with_missing_params_attribute(self):
        """Test configuration validation with config that has no params attribute."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create a config object without params attribute
        class ConfigWithoutParams:
            def __init__(self):
                self.type = RetrieverType.PARENT_DOCUMENT
                self.search_kwargs = {"k": 5}
                # No params attribute
        
        config = ConfigWithoutParams()
        
        # Should raise error when trying to access params
        with pytest.raises(AttributeError):
            strategy.validate_config(config)
    
    def test_validate_config_with_non_dict_params(self):
        """Test configuration validation with non-dict params."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Pydantic doesn't allow non-dict params, so we test with a custom config object
        class ConfigWithNonDictParams:
            def __init__(self):
                self.type = RetrieverType.PARENT_DOCUMENT
                self.search_kwargs = {"k": 5}
                self.params = "not-a-dict"
        
        config = ConfigWithNonDictParams()
        
        # Should raise error when trying to access params as dict
        with pytest.raises(TypeError):
            strategy.validate_config(config)


class TestParentDocumentRetrieverStrategyIntegration:
    """Integration tests for ParentDocumentRetrieverStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter, "custom_param": "value"}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            # Validate config
            strategy.validate_config(config)
            
            # Create retriever
            result = strategy.create_retriever(config, vectorstore)
            
            assert result == mock_parent_doc
            mock_parent_doc_class.assert_called_once()
    
    def test_workflow_with_different_splitter_types(self):
        """Test workflow with different splitter types."""
        strategy = ParentDocumentRetrieverStrategy()
        vectorstore = Mock()
        
        # Test with different splitter types
        splitter_types = [
            Mock(split_documents=Mock()),
            Mock(split_documents=Mock()),
            Mock(split_documents=Mock())
        ]
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            for child_splitter in splitter_types:
                config = RetrieverConfig(
                    type=RetrieverType.PARENT_DOCUMENT,
                    search_kwargs={"k": 5},
                    params={"child_splitter": child_splitter}
                )
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_parent_doc
            
            # Should have been called for each splitter
            assert mock_parent_doc_class.call_count == len(splitter_types)
    
    def test_workflow_with_different_docstore_types(self):
        """Test workflow with different docstore types."""
        strategy = ParentDocumentRetrieverStrategy()
        vectorstore = Mock()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        # Test with different docstore types
        docstore_types = [
            Mock(add=Mock(), search=Mock()),
            Mock(add=Mock(), search=Mock()),
            Mock(add=Mock(), search=Mock())
        ]
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            for docstore in docstore_types:
                config = RetrieverConfig(
                    type=RetrieverType.PARENT_DOCUMENT,
                    search_kwargs={"k": 5},
                    params={
                        "child_splitter": mock_child_splitter,
                        "docstore": docstore
                    }
                )
                
                # Validate config
                strategy.validate_config(config)
                
                # Create retriever
                result = strategy.create_retriever(config, vectorstore)
                
                assert result == mock_parent_doc
            
            # Should have been called for each docstore
            assert mock_parent_doc_class.call_count == len(docstore_types)
    
    def test_workflow_with_mixed_valid_and_invalid_configs(self):
        """Test workflow with mix of valid and invalid configurations."""
        strategy = ParentDocumentRetrieverStrategy()
        vectorstore = Mock()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        # Valid config
        valid_config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter}
        )
        
        # Invalid config (no child_splitter)
        invalid_config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={}
        )
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_retriever(valid_config, vectorstore)
            assert result == mock_parent_doc
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="params\\['child_splitter'\\] is required for ParentDocumentRetriever"):
                strategy.validate_config(invalid_config)
    
    def test_workflow_with_parent_document_retriever_errors(self):
        """Test workflow with ParentDocumentRetriever errors."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter}
        )
        vectorstore = Mock()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever', side_effect=Exception("ParentDocumentRetriever error")):
            with pytest.raises(Exception, match="ParentDocumentRetriever error"):
                strategy.create_retriever(config, vectorstore)
    
    def test_workflow_with_none_vectorstore(self):
        """Test workflow with None vectorstore."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter}
        )
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            # Should handle None vectorstore gracefully
            result = strategy.create_retriever(config, None)
            assert result == mock_parent_doc
    
    def test_workflow_with_custom_vectorstore_implementation(self):
        """Test workflow with custom vectorstore implementation."""
        strategy = ParentDocumentRetrieverStrategy()
        
        # Create mock child splitter
        mock_child_splitter = Mock()
        mock_child_splitter.split_documents = Mock()
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={"child_splitter": mock_child_splitter, "custom_param": "value"}
        )
        
        # Custom vectorstore
        class CustomVectorStore:
            def __init__(self):
                self.name = "custom_vectorstore"
        
        vectorstore = CustomVectorStore()
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            # Validate config
            strategy.validate_config(config)
            
            # Create retriever
            result = strategy.create_retriever(config, vectorstore)
            
            assert result == mock_parent_doc
            mock_parent_doc_class.assert_called_once()
            call_kwargs = mock_parent_doc_class.call_args[1]
            assert call_kwargs["vectorstore"] == vectorstore
            assert call_kwargs["child_splitter"] == mock_child_splitter
            assert call_kwargs["custom_param"] == "value"
    
    def test_workflow_with_complex_splitter_implementations(self):
        """Test workflow with complex splitter implementations."""
        strategy = ParentDocumentRetrieverStrategy()
        vectorstore = Mock()
        
        # Create complex splitter implementations
        class ComplexChildSplitter:
            def __init__(self, chunk_size=1000):
                self.chunk_size = chunk_size
            
            def split_documents(self, documents):
                return [f"chunk_{i}" for i in range(len(documents) * 2)]
        
        class ComplexParentSplitter:
            def __init__(self, chunk_size=5000):
                self.chunk_size = chunk_size
            
            def split_documents(self, documents):
                return [f"parent_chunk_{i}" for i in range(len(documents))]
        
        child_splitter = ComplexChildSplitter(chunk_size=1000)
        parent_splitter = ComplexParentSplitter(chunk_size=5000)
        
        config = RetrieverConfig(
            type=RetrieverType.PARENT_DOCUMENT,
            search_kwargs={"k": 5},
            params={
                "child_splitter": child_splitter,
                "parent_splitter": parent_splitter
            }
        )
        
        with patch('src.paas_ai.core.rag.retrievers.parent_document.ParentDocumentRetriever') as mock_parent_doc_class:
            mock_parent_doc = Mock()
            mock_parent_doc_class.return_value = mock_parent_doc
            
            # Validate config
            strategy.validate_config(config)
            
            # Create retriever
            result = strategy.create_retriever(config, vectorstore)
            
            assert result == mock_parent_doc
            mock_parent_doc_class.assert_called_once()
            call_kwargs = mock_parent_doc_class.call_args[1]
            assert call_kwargs["child_splitter"] == child_splitter
            assert call_kwargs["parent_splitter"] == parent_splitter

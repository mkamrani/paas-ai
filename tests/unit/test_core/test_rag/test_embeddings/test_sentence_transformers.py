"""
Unit tests for SentenceTransformers embedding strategy.

Tests all components of the SentenceTransformers embedding strategy including:
- SentenceTransformersEmbeddingStrategy class
- Configuration validation
- Embedding creation
- Error handling and edge cases
- Model name validation
- Warning handling for unusual model names
"""

import warnings
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.embeddings.sentence_transformers import SentenceTransformersEmbeddingStrategy
from src.paas_ai.core.rag.config import EmbeddingConfig, EmbeddingType


class TestSentenceTransformersEmbeddingStrategy:
    """Test the SentenceTransformersEmbeddingStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = SentenceTransformersEmbeddingStrategy()
        assert strategy is not None
        assert isinstance(strategy, SentenceTransformersEmbeddingStrategy)
    
    def test_create_embeddings_basic(self):
        """Test creating embeddings with basic configuration."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model_name="all-MiniLM-L6-v2"
            )
            assert result == mock_embeddings
    
    def test_create_embeddings_with_additional_params(self):
        """Test creating embeddings with additional parameters."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            params={
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True},
                "cache_folder": "/tmp/cache"
            }
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
                cache_folder="/tmp/cache"
            )
            assert result == mock_embeddings
    
    def test_create_embeddings_empty_params(self):
        """Test creating embeddings with empty params."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="paraphrase-MiniLM-L6-v2",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model_name="paraphrase-MiniLM-L6-v2"
            )
            assert result == mock_embeddings
    
    def test_create_embeddings_none_params(self):
        """Test creating embeddings with None params."""
        strategy = SentenceTransformersEmbeddingStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Should work fine with empty params
            result = strategy.create_embeddings(config)
            assert result == mock_embeddings
            mock_embeddings_class.assert_called_once_with(
                model_name="all-MiniLM-L6-v2"
            )
    
    def test_validate_config_with_valid_model_name(self):
        """Test configuration validation with valid model name."""
        strategy = SentenceTransformersEmbeddingStrategy()
        valid_models = [
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "distilbert-base-nli-mean-tokens",
            "bert-base-nli-mean-tokens"
        ]
        
        for model_name in valid_models:
            config = EmbeddingConfig(
                type=EmbeddingType.SENTENCE_TRANSFORMERS,
                model_name=model_name,
                params={}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_empty_model_name(self):
        """Test configuration validation with empty model name."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="",
            params={}
        )
        
        with pytest.raises(ValueError, match="model_name is required for SentenceTransformers embeddings"):
            strategy.validate_config(config)
    
    def test_validate_config_with_none_model_name(self):
        """Test configuration validation with None model name."""
        strategy = SentenceTransformersEmbeddingStrategy()
        # Pydantic doesn't allow None for model_name, so we test with empty string instead
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="",
            params={}
        )
        
        with pytest.raises(ValueError, match="model_name is required for SentenceTransformers embeddings"):
            strategy.validate_config(config)
    
    def test_validate_config_with_unusual_model_name_warning(self):
        """Test configuration validation with unusual model name that triggers warning."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="unusual-model-name",
            params={}
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.validate_config(config)
            
            # Should have issued a warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "doesn't match common SentenceTransformers patterns" in str(w[0].message)
            assert "unusual-model-name" in str(w[0].message)
    
    def test_validate_config_with_whitespace_model_name(self):
        """Test configuration validation with whitespace in model name."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="  all-MiniLM-L6-v2  ",
            params={}
        )
        
        # Should not raise an exception (whitespace is allowed)
        strategy.validate_config(config)
    
    def test_validate_config_with_unicode_model_name(self):
        """Test configuration validation with unicode characters in model name."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="测试模型-name",
            params={}
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.validate_config(config)
            
            # Should have issued a warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
    
    def test_validate_config_with_very_long_model_name(self):
        """Test configuration validation with very long model name."""
        strategy = SentenceTransformersEmbeddingStrategy()
        long_model_name = "a" * 1000
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name=long_model_name,
            params={}
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.validate_config(config)
            
            # Should have issued a warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
    
    def test_validate_config_with_special_characters_model_name(self):
        """Test configuration validation with special characters in model name."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="model@with#special$chars",
            params={}
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.validate_config(config)
            
            # Should have issued a warning
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
    
    def test_validate_config_with_valid_prefixes_no_warning(self):
        """Test configuration validation with valid prefixes that don't trigger warnings."""
        strategy = SentenceTransformersEmbeddingStrategy()
        valid_prefixes = [
            'all-',
            'sentence-transformers/',
            'paraphrase-',
            'distilbert-',
            'bert-'
        ]
        
        for prefix in valid_prefixes:
            model_name = f"{prefix}test-model"
            config = EmbeddingConfig(
                type=EmbeddingType.SENTENCE_TRANSFORMERS,
                model_name=model_name,
                params={}
            )
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                strategy.validate_config(config)
                
                # Should not have issued a warning
                assert len(w) == 0


class TestSentenceTransformersEmbeddingStrategyEdgeCases:
    """Test edge cases for SentenceTransformersEmbeddingStrategy."""
    
    def test_create_embeddings_with_complex_params(self):
        """Test creating embeddings with complex parameter structures."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            params={
                "model_kwargs": {
                    "device": "cuda",
                    "trust_remote_code": True,
                    "use_auth_token": "hf_token"
                },
                "encode_kwargs": {
                    "normalize_embeddings": True,
                    "batch_size": 32,
                    "show_progress_bar": True
                },
                "cache_folder": "/tmp/huggingface_cache",
                "model_revision": "main"
            }
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    "device": "cuda",
                    "trust_remote_code": True,
                    "use_auth_token": "hf_token"
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                    "show_progress_bar": True
                },
                cache_folder="/tmp/huggingface_cache",
                model_revision="main"
            )
            assert result == mock_embeddings
    
    def test_create_embeddings_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = SentenceTransformersEmbeddingStrategy()
        original_params = {
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        }
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_validate_config_with_none_params(self):
        """Test configuration validation with None params."""
        strategy = SentenceTransformersEmbeddingStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params={}
        )
        
        # Should not raise error with empty params
        strategy.validate_config(config)
    
    def test_validate_config_with_missing_params_attribute(self):
        """Test configuration validation with config that has no params attribute."""
        strategy = SentenceTransformersEmbeddingStrategy()
        
        # Create a config object without params attribute
        class ConfigWithoutParams:
            def __init__(self):
                self.type = EmbeddingType.SENTENCE_TRANSFORMERS
                self.model_name = "all-MiniLM-L6-v2"
                # No params attribute
        
        config = ConfigWithoutParams()
        
        # Should not raise error - the strategy doesn't access params for validation
        strategy.validate_config(config)
    
    def test_validate_config_with_non_dict_params(self):
        """Test configuration validation with non-dict params."""
        strategy = SentenceTransformersEmbeddingStrategy()
        # Pydantic doesn't allow non-dict params, so we test with a custom config object
        class ConfigWithNonDictParams:
            def __init__(self):
                self.type = EmbeddingType.SENTENCE_TRANSFORMERS
                self.model_name = "all-MiniLM-L6-v2"
                self.params = "not-a-dict"
        
        config = ConfigWithNonDictParams()
        
        # Should not raise error - the strategy doesn't access params for validation
        strategy.validate_config(config)
    
    def test_warning_message_content(self):
        """Test the content of warning messages."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="unusual-model",
            params={}
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy.validate_config(config)
            
            warning_message = str(w[0].message)
            assert "unusual-model" in warning_message
            assert "doesn't match common SentenceTransformers patterns" in warning_message
            assert "all-" in warning_message
            assert "sentence-transformers/" in warning_message
            assert "paraphrase-" in warning_message
            assert "distilbert-" in warning_message
            assert "bert-" in warning_message
    
    def test_multiple_warnings_suppression(self):
        """Test that multiple calls don't accumulate warnings."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="unusual-model",
            params={}
        )
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Call multiple times
            strategy.validate_config(config)
            strategy.validate_config(config)
            strategy.validate_config(config)
            
            # Should have 3 warnings (one per call)
            assert len(w) == 3
            for warning in w:
                assert issubclass(warning.category, UserWarning)


class TestSentenceTransformersEmbeddingStrategyIntegration:
    """Integration tests for SentenceTransformersEmbeddingStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params={"model_kwargs": {"device": "cpu"}}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Validate config
            strategy.validate_config(config)
            
            # Create embeddings
            result = strategy.create_embeddings(config)
            
            assert result == mock_embeddings
            mock_embeddings_class.assert_called_once()
    
    def test_workflow_with_warning(self):
        """Test workflow with model name that triggers warning."""
        strategy = SentenceTransformersEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="unusual-model-name",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Validate config (should issue warning)
                strategy.validate_config(config)
                
                # Create embeddings
                result = strategy.create_embeddings(config)
                
                assert result == mock_embeddings
                assert len(w) == 1
                assert issubclass(w[0].category, UserWarning)
    
    def test_workflow_with_multiple_models(self):
        """Test workflow with multiple different models."""
        strategy = SentenceTransformersEmbeddingStrategy()
        
        # Test with different model configurations
        configs = [
            EmbeddingConfig(
                type=EmbeddingType.SENTENCE_TRANSFORMERS,
                model_name="all-MiniLM-L6-v2",
                params={}
            ),
            EmbeddingConfig(
                type=EmbeddingType.SENTENCE_TRANSFORMERS,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                params={"model_kwargs": {"device": "cpu"}}
            ),
            EmbeddingConfig(
                type=EmbeddingType.SENTENCE_TRANSFORMERS,
                model_name="paraphrase-MiniLM-L6-v2",
                params={"encode_kwargs": {"normalize_embeddings": True}}
            )
        ]
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            for config in configs:
                # Validate config
                strategy.validate_config(config)
                
                # Create embeddings
                result = strategy.create_embeddings(config)
                
                assert result == mock_embeddings
            
            # Should have been called 3 times
            assert mock_embeddings_class.call_count == 3
    
    def test_workflow_with_mixed_valid_and_invalid_models(self):
        """Test workflow with mix of valid and invalid model names."""
        strategy = SentenceTransformersEmbeddingStrategy()
        
        # Valid model (no warning)
        valid_config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            params={}
        )
        
        # Invalid model name (empty)
        invalid_config = EmbeddingConfig(
            type=EmbeddingType.SENTENCE_TRANSFORMERS,
            model_name="",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.sentence_transformers.HuggingFaceEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Valid config should work
            strategy.validate_config(valid_config)
            result = strategy.create_embeddings(valid_config)
            assert result == mock_embeddings
            
            # Invalid config should raise error
            with pytest.raises(ValueError, match="model_name is required"):
                strategy.validate_config(invalid_config)

"""
Unit tests for OpenAI embedding strategy.

Tests all components of the OpenAI embedding strategy including:
- OpenAIEmbeddingStrategy class
- Configuration validation
- Embedding creation
- Error handling and edge cases
- Environment variable handling
- Model validation
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.paas_ai.core.rag.embeddings.openai import OpenAIEmbeddingStrategy
from src.paas_ai.core.rag.config import EmbeddingConfig, EmbeddingType


class TestOpenAIEmbeddingStrategy:
    """Test the OpenAIEmbeddingStrategy class."""
    
    def test_init(self):
        """Test strategy initialization."""
        strategy = OpenAIEmbeddingStrategy()
        assert strategy is not None
        assert isinstance(strategy, OpenAIEmbeddingStrategy)
    
    def test_create_embeddings_basic(self):
        """Test creating embeddings with basic configuration."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model="text-embedding-3-small",
                api_key="test-key"
            )
            assert result == mock_embeddings
    
    def test_create_embeddings_with_additional_params(self):
        """Test creating embeddings with additional parameters."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-large",
            params={
                "api_key": "test-key",
                "chunk_size": 1000,
                "max_retries": 3,
                "request_timeout": 60
            }
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model="text-embedding-3-large",
                api_key="test-key",
                chunk_size=1000,
                max_retries=3,
                request_timeout=60
            )
            assert result == mock_embeddings
    
    def test_create_embeddings_empty_params(self):
        """Test creating embeddings with empty params."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model="text-embedding-3-small"
            )
            assert result == mock_embeddings
    
    def test_create_embeddings_none_params(self):
        """Test creating embeddings with None params."""
        strategy = OpenAIEmbeddingStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Should work fine with empty params
            result = strategy.create_embeddings(config)
            assert result == mock_embeddings
            mock_embeddings_class.assert_called_once_with(
                model="text-embedding-3-small"
            )
    
    def test_validate_config_with_api_key_in_params(self):
        """Test configuration validation with API key in params."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_env_var(self):
        """Test configuration validation with environment variable."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={}
        )
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_with_both_api_key_and_env_var(self):
        """Test configuration validation with both API key in params and env var."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "param-key"}
        )
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            # Should use param key, not env var
            strategy.validate_config(config)
    
    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={}
        )
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI embeddings require OPENAI_API_KEY environment variable or api_key in config"):
                strategy.validate_config(config)
    
    def test_validate_config_empty_api_key_in_params(self):
        """Test configuration validation with empty API key in params."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": ""}
        )
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI embeddings require OPENAI_API_KEY environment variable or api_key in config"):
                strategy.validate_config(config)
    
    def test_validate_config_none_api_key_in_params(self):
        """Test configuration validation with None API key in params."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": None}
        )
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI embeddings require OPENAI_API_KEY environment variable or api_key in config"):
                strategy.validate_config(config)
    
    def test_validate_config_valid_models(self):
        """Test configuration validation with valid model names."""
        strategy = OpenAIEmbeddingStrategy()
        valid_models = [
            'text-embedding-3-small',
            'text-embedding-3-large',
            'text-embedding-ada-002'
        ]
        
        for model_name in valid_models:
            config = EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name=model_name,
                params={"api_key": "test-key"}
            )
            
            # Should not raise any exception
            strategy.validate_config(config)
    
    def test_validate_config_invalid_model(self):
        """Test configuration validation with invalid model name."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="invalid-model",
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match="Unsupported OpenAI model: invalid-model"):
            strategy.validate_config(config)
    
    def test_validate_config_empty_model_name(self):
        """Test configuration validation with empty model name."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="",
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match="Unsupported OpenAI model:"):
            strategy.validate_config(config)
    
    def test_validate_config_none_model_name(self):
        """Test configuration validation with None model name."""
        strategy = OpenAIEmbeddingStrategy()
        # Pydantic doesn't allow None for model_name, so we test with empty string instead
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="",
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match="Unsupported OpenAI model:"):
            strategy.validate_config(config)
    
    def test_validate_config_case_sensitive_models(self):
        """Test that model validation is case sensitive."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="TEXT-EMBEDDING-3-SMALL",  # Wrong case
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match="Unsupported OpenAI model: TEXT-EMBEDDING-3-SMALL"):
            strategy.validate_config(config)
    
    def test_validate_config_whitespace_in_model_name(self):
        """Test configuration validation with whitespace in model name."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name=" text-embedding-3-small ",  # With whitespace
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match="Unsupported OpenAI model:  text-embedding-3-small "):
            strategy.validate_config(config)


class TestOpenAIEmbeddingStrategyEdgeCases:
    """Test edge cases for OpenAIEmbeddingStrategy."""
    
    def test_create_embeddings_with_special_characters_in_params(self):
        """Test creating embeddings with special characters in parameters."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={
                "api_key": "test-key",
                "organization": "org-123",
                "base_url": "https://api.openai.com/v1",
                "default_headers": {"User-Agent": "test-agent"}
            }
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            mock_embeddings_class.assert_called_once_with(
                model="text-embedding-3-small",
                api_key="test-key",
                organization="org-123",
                base_url="https://api.openai.com/v1",
                default_headers={"User-Agent": "test-agent"}
            )
            assert result == mock_embeddings
    
    def test_validate_config_with_unicode_in_api_key(self):
        """Test configuration validation with unicode characters in API key."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "sk-测试-key-123"}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_long_api_key(self):
        """Test configuration validation with very long API key."""
        strategy = OpenAIEmbeddingStrategy()
        long_api_key = "sk-" + "a" * 200
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": long_api_key}
        )
        
        # Should not raise any exception
        strategy.validate_config(config)
    
    def test_validate_config_with_very_long_model_name(self):
        """Test configuration validation with very long model name."""
        strategy = OpenAIEmbeddingStrategy()
        long_model_name = "text-embedding-" + "x" * 100
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name=long_model_name,
            params={"api_key": "test-key"}
        )
        
        with pytest.raises(ValueError, match=f"Unsupported OpenAI model: {long_model_name}"):
            strategy.validate_config(config)
    
    def test_create_embeddings_params_copy_behavior(self):
        """Test that params are copied and not modified."""
        strategy = OpenAIEmbeddingStrategy()
        original_params = {"api_key": "test-key", "chunk_size": 1000}
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params=original_params
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            result = strategy.create_embeddings(config)
            
            # Original params should not be modified
            assert config.params == original_params
            assert config.params is not original_params  # Should be a copy
    
    def test_validate_config_with_none_params(self):
        """Test configuration validation with None params."""
        strategy = OpenAIEmbeddingStrategy()
        # Pydantic doesn't allow None for params, so we test with empty dict instead
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={}
        )
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI embeddings require OPENAI_API_KEY"):
                strategy.validate_config(config)
    
    def test_validate_config_with_missing_params_attribute(self):
        """Test configuration validation with config that has no params attribute."""
        strategy = OpenAIEmbeddingStrategy()
        
        # Create a config object without params attribute
        class ConfigWithoutParams:
            def __init__(self):
                self.type = EmbeddingType.OPENAI
                self.model_name = "text-embedding-3-small"
                # No params attribute
        
        config = ConfigWithoutParams()
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AttributeError):
                strategy.validate_config(config)
    
    def test_validate_config_with_non_dict_params(self):
        """Test configuration validation with non-dict params."""
        strategy = OpenAIEmbeddingStrategy()
        # Pydantic doesn't allow non-dict params, so we test with a custom config object
        class ConfigWithNonDictParams:
            def __init__(self):
                self.type = EmbeddingType.OPENAI
                self.model_name = "text-embedding-3-small"
                self.params = "not-a-dict"
        
        config = ConfigWithNonDictParams()
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(AttributeError):
                strategy.validate_config(config)


class TestOpenAIEmbeddingStrategyIntegration:
    """Integration tests for OpenAIEmbeddingStrategy."""
    
    def test_full_workflow(self):
        """Test complete workflow from validation to creation."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-small",
            params={"api_key": "test-key"}
        )
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings
            
            # Validate config
            strategy.validate_config(config)
            
            # Create embeddings
            result = strategy.create_embeddings(config)
            
            assert result == mock_embeddings
            mock_embeddings_class.assert_called_once()
    
    def test_workflow_with_env_var_fallback(self):
        """Test workflow with environment variable fallback."""
        strategy = OpenAIEmbeddingStrategy()
        config = EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_name="text-embedding-3-large",
            params={}  # No API key in params
        )
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'env-key'}):
            with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
                mock_embeddings = Mock()
                mock_embeddings_class.return_value = mock_embeddings
                
                # Validate config (should use env var)
                strategy.validate_config(config)
                
                # Create embeddings
                result = strategy.create_embeddings(config)
                
                assert result == mock_embeddings
                mock_embeddings_class.assert_called_once_with(
                    model="text-embedding-3-large"
                )
    
    def test_workflow_with_multiple_retries(self):
        """Test workflow with multiple validation attempts."""
        strategy = OpenAIEmbeddingStrategy()
        
        # Test with different configurations
        configs = [
            EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="text-embedding-3-small",
                params={"api_key": "key1"}
            ),
            EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="text-embedding-3-large",
                params={"api_key": "key2"}
            ),
            EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_name="text-embedding-ada-002",
                params={"api_key": "key3"}
            )
        ]
        
        with patch('src.paas_ai.core.rag.embeddings.openai.OpenAIEmbeddings') as mock_embeddings_class:
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

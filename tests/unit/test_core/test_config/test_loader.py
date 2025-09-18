"""
Unit tests for core/config/loader.py

Tests all configuration loading functionality including:
- Environment variable loading from .env files
- YAML configuration file loading and parsing
- Environment variable substitution
- Profile resolution
- Configuration file management
- Error handling and edge cases
"""

import os
import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from contextlib import contextmanager
from pydantic import ValidationError

from src.paas_ai.core.config.loader import (
    # Exception classes
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigFileEmptyError,
    InvalidYAMLError,
    InvalidConfigurationError,
    MissingEnvironmentVariableError,
    
    # Functions
    _load_dotenv_files,
    _substitute_env_vars,
    _load_yaml_config,
    _resolve_profile,
    load_config,
    load_config_file,
    _convert_enums_to_strings,
    save_config_file,
    create_sample_config,
    list_available_profiles,
    get_current_profile,
)

from src.paas_ai.core.config.schemas import (
    Config,
    ConfigFile,
    EmbeddingConfig,
    VectorStoreConfig,
    RetrieverConfig,
    EmbeddingType,
    VectorStoreType,
    RetrieverType,
    DEFAULT_CONFIG_PROFILES,
)


@contextmanager
def temp_env_file(content: str):
    """Context manager for creating temporary .env files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(content)
        f.flush()
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@contextmanager
def temp_yaml_file(content: str):
    """Context manager for creating temporary YAML files."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(content)
        f.flush()
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


@contextmanager
def temp_directory():
    """Context manager for creating temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestExceptionClasses:
    """Test custom exception classes."""
    
    def test_configuration_error(self):
        """Test ConfigurationError base exception."""
        error = ConfigurationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError exception."""
        error = ConfigFileNotFoundError("File not found")
        assert str(error) == "File not found"
        assert isinstance(error, ConfigurationError)
    
    def test_config_file_empty_error(self):
        """Test ConfigFileEmptyError exception."""
        error = ConfigFileEmptyError("File is empty")
        assert str(error) == "File is empty"
        assert isinstance(error, ConfigurationError)
    
    def test_invalid_yaml_error(self):
        """Test InvalidYAMLError exception."""
        error = InvalidYAMLError("Invalid YAML")
        assert str(error) == "Invalid YAML"
        assert isinstance(error, ConfigurationError)
    
    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError exception."""
        error = InvalidConfigurationError("Invalid config")
        assert str(error) == "Invalid config"
        assert isinstance(error, ConfigurationError)
    
    def test_missing_environment_variable_error(self):
        """Test MissingEnvironmentVariableError exception."""
        error = MissingEnvironmentVariableError("Missing VAR")
        assert str(error) == "Missing VAR"
        assert isinstance(error, ConfigurationError)


class TestLoadDotenvFiles:
    """Test _load_dotenv_files function."""
    
    def test_load_dotenv_files_no_dotenv_available(self):
        """Test behavior when dotenv is not available."""
        with patch('src.paas_ai.core.config.loader.DOTENV_AVAILABLE', False):
            # Should not raise any errors
            _load_dotenv_files()
    
    def test_load_dotenv_files_dotenv_available(self):
        """Test loading .env files when dotenv is available."""
        with patch('src.paas_ai.core.config.loader.DOTENV_AVAILABLE', True):
            with patch('src.paas_ai.core.config.loader.load_dotenv') as mock_load_dotenv:
                with temp_directory() as tmpdir:
                    # Create test .env files
                    env_file = tmpdir / '.env'
                    env_file.write_text('TEST_VAR=test_value')
                    
                    with patch('pathlib.Path.cwd', return_value=tmpdir):
                        _load_dotenv_files()
                    
                    # Should call load_dotenv for the .env file
                    mock_load_dotenv.assert_called()
    
    def test_load_dotenv_files_priority_order(self):
        """Test that .env files are loaded in correct priority order."""
        with patch('src.paas_ai.core.config.loader.DOTENV_AVAILABLE', True):
            with patch('src.paas_ai.core.config.loader.load_dotenv') as mock_load_dotenv:
                with temp_directory() as tmpdir:
                    # Create multiple .env files
                    (tmpdir / '.env.local').write_text('LOCAL_VAR=local')
                    (tmpdir / '.env').write_text('MAIN_VAR=main')
                    (tmpdir / '.env.default').write_text('DEFAULT_VAR=default')
                    
                    with patch('pathlib.Path.cwd', return_value=tmpdir):
                        _load_dotenv_files()
                    
                    # Should call load_dotenv multiple times
                    assert mock_load_dotenv.call_count >= 1
    
    def test_load_dotenv_files_no_override(self):
        """Test that existing environment variables are not overridden."""
        with patch('src.paas_ai.core.config.loader.DOTENV_AVAILABLE', True):
            with patch('src.paas_ai.core.config.loader.load_dotenv') as mock_load_dotenv:
                # Set existing environment variable
                os.environ['EXISTING_VAR'] = 'existing_value'
                
                try:
                    with temp_directory() as tmpdir:
                        env_file = tmpdir / '.env'
                        env_file.write_text('EXISTING_VAR=new_value')
                        
                        with patch('pathlib.Path.cwd', return_value=tmpdir):
                            _load_dotenv_files()
                        
                        # Should call load_dotenv with override=False
                        mock_load_dotenv.assert_called_with(env_file, override=False)
                finally:
                    # Clean up
                    if 'EXISTING_VAR' in os.environ:
                        del os.environ['EXISTING_VAR']


class TestSubstituteEnvVars:
    """Test _substitute_env_vars function."""
    
    def test_substitute_env_vars_string_with_variable(self):
        """Test substituting environment variables in strings."""
        os.environ['TEST_VAR'] = 'test_value'
        try:
            config_dict = "Hello ${TEST_VAR}!"
            result = _substitute_env_vars(config_dict)
            assert result == "Hello test_value!"
        finally:
            del os.environ['TEST_VAR']
    
    def test_substitute_env_vars_string_without_variable(self):
        """Test string without environment variables."""
        config_dict = "Hello world!"
        result = _substitute_env_vars(config_dict)
        assert result == "Hello world!"
    
    def test_substitute_env_vars_string_with_multiple_variables(self):
        """Test string with multiple environment variables."""
        os.environ['VAR1'] = 'value1'
        os.environ['VAR2'] = 'value2'
        try:
            config_dict = "${VAR1} and ${VAR2}"
            result = _substitute_env_vars(config_dict)
            assert result == "value1 and value2"
        finally:
            del os.environ['VAR1']
            del os.environ['VAR2']
    
    def test_substitute_env_vars_missing_variable(self):
        """Test error when environment variable is missing."""
        config_dict = "Hello ${MISSING_VAR}!"
        with pytest.raises(MissingEnvironmentVariableError, match="Environment variable 'MISSING_VAR' not found"):
            _substitute_env_vars(config_dict)
    
    def test_substitute_env_vars_dict(self):
        """Test substituting environment variables in dictionaries."""
        os.environ['API_KEY'] = 'secret_key'
        try:
            config_dict = {
                "api_key": "${API_KEY}",
                "nested": {
                    "value": "${API_KEY}"
                }
            }
            result = _substitute_env_vars(config_dict)
            expected = {
                "api_key": "secret_key",
                "nested": {
                    "value": "secret_key"
                }
            }
            assert result == expected
        finally:
            del os.environ['API_KEY']
    
    def test_substitute_env_vars_list(self):
        """Test substituting environment variables in lists."""
        os.environ['ITEM1'] = 'value1'
        os.environ['ITEM2'] = 'value2'
        try:
            config_dict = ["${ITEM1}", "static", "${ITEM2}"]
            result = _substitute_env_vars(config_dict)
            assert result == ["value1", "static", "value2"]
        finally:
            del os.environ['ITEM1']
            del os.environ['ITEM2']
    
    def test_substitute_env_vars_nested_structures(self):
        """Test substituting environment variables in nested structures."""
        os.environ['BASE_URL'] = 'https://api.example.com'
        os.environ['VERSION'] = 'v1'
        try:
            config_dict = {
                "api": {
                    "url": "${BASE_URL}",
                    "version": "${VERSION}",
                    "endpoints": ["${BASE_URL}/${VERSION}/users", "${BASE_URL}/${VERSION}/posts"]
                }
            }
            result = _substitute_env_vars(config_dict)
            expected = {
                "api": {
                    "url": "https://api.example.com",
                    "version": "v1",
                    "endpoints": ["https://api.example.com/v1/users", "https://api.example.com/v1/posts"]
                }
            }
            assert result == expected
        finally:
            del os.environ['BASE_URL']
            del os.environ['VERSION']
    
    def test_substitute_env_vars_non_string_types(self):
        """Test that non-string types are returned unchanged."""
        config_dict = 42
        result = _substitute_env_vars(config_dict)
        assert result == 42
        
        config_dict = True
        result = _substitute_env_vars(config_dict)
        assert result is True
        
        config_dict = None
        result = _substitute_env_vars(config_dict)
        assert result is None


class TestLoadYamlConfig:
    """Test _load_yaml_config function."""
    
    def test_load_yaml_config_success(self):
        """Test successful YAML config loading."""
        yaml_content = """
embedding:
  type: openai
  model_name: text-embedding-3-small
vectorstore:
  type: chroma
  collection_name: test_collection
retriever:
  type: similarity
  search_kwargs:
    k: 5
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            result = _load_yaml_config(yaml_path)
            
            assert isinstance(result, dict)
            assert result["embedding"]["type"] == "openai"
            assert result["vectorstore"]["type"] == "chroma"
            assert result["retriever"]["search_kwargs"]["k"] == 5
    
    def test_load_yaml_config_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        non_existent_path = Path("/non/existent/file.yaml")
        
        with pytest.raises(ConfigFileNotFoundError, match="Config file not found"):
            _load_yaml_config(non_existent_path)
    
    def test_load_yaml_config_empty_file(self):
        """Test error when YAML file is empty."""
        with temp_yaml_file("") as yaml_path:
            with pytest.raises(ConfigFileEmptyError, match="Config file is empty"):
                _load_yaml_config(yaml_path)
    
    def test_load_yaml_config_invalid_yaml(self):
        """Test error when YAML file contains invalid YAML."""
        invalid_yaml = """
embedding:
  type: openai
  invalid_yaml: [unclosed list
"""
        with temp_yaml_file(invalid_yaml) as yaml_path:
            with pytest.raises(InvalidYAMLError, match="Invalid YAML in config file"):
                _load_yaml_config(yaml_path)
    
    def test_load_yaml_config_with_env_vars(self):
        """Test YAML config loading with environment variable substitution."""
        os.environ['MODEL_NAME'] = 'text-embedding-3-large'
        os.environ['COLLECTION_NAME'] = 'my_collection'
        try:
            yaml_content = """
embedding:
  type: openai
  model_name: ${MODEL_NAME}
vectorstore:
  type: chroma
  collection_name: ${COLLECTION_NAME}
retriever:
  type: similarity
  search_kwargs:
    k: 5
"""
            with temp_yaml_file(yaml_content) as yaml_path:
                result = _load_yaml_config(yaml_path)
                
                assert result["embedding"]["model_name"] == "text-embedding-3-large"
                assert result["vectorstore"]["collection_name"] == "my_collection"
        finally:
            del os.environ['MODEL_NAME']
            del os.environ['COLLECTION_NAME']
    
    def test_load_yaml_config_missing_env_var(self):
        """Test error when YAML references missing environment variable."""
        yaml_content = """
embedding:
  type: openai
  model_name: ${MISSING_VAR}
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with pytest.raises(MissingEnvironmentVariableError, match="Environment variable 'MISSING_VAR' not found"):
                _load_yaml_config(yaml_path)
    
    def test_load_yaml_config_general_exception(self):
        """Test handling of general exceptions during file loading."""
        with patch('pathlib.Path.exists', return_value=True), patch('builtins.open', side_effect=PermissionError("Permission denied")):
            yaml_path = Path("/test/config.yaml")
            with pytest.raises(ConfigurationError, match="Failed to load config file"):
                _load_yaml_config(yaml_path)


class TestResolveProfile:
    """Test _resolve_profile function."""
    
    def test_resolve_profile_predefined_default(self):
        """Test resolving predefined default profile."""
        config_file = ConfigFile(current="default")
        config, profile_name = _resolve_profile(config_file)
        
        assert profile_name == "default"
        assert config.embedding.type == EmbeddingType.OPENAI
    
    def test_resolve_profile_predefined_local(self):
        """Test resolving predefined local profile."""
        config_file = ConfigFile(current="local")
        config, profile_name = _resolve_profile(config_file)
        
        assert profile_name == "local"
        assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
    
    def test_resolve_profile_custom_profile(self):
        """Test resolving custom profile from config file."""
        custom_config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        config_file = ConfigFile(
            current="custom",
            profiles={"custom": custom_config}
        )
        
        config, profile_name = _resolve_profile(config_file)
        
        assert profile_name == "custom"
        assert config.embedding.type == EmbeddingType.OPENAI
    
    def test_resolve_profile_with_override(self):
        """Test resolving profile with override."""
        config_file = ConfigFile(current="default")
        config, profile_name = _resolve_profile(config_file, profile_override="local")
        
        assert profile_name == "local"
        assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
    
    def test_resolve_profile_unknown_profile(self):
        """Test error when profile is unknown."""
        config_file = ConfigFile(current="unknown")
        
        with pytest.raises(InvalidConfigurationError, match="Unknown profile 'unknown'"):
            _resolve_profile(config_file)
    
    def test_resolve_profile_override_unknown(self):
        """Test error when profile override is unknown."""
        config_file = ConfigFile(current="default")
        
        with pytest.raises(InvalidConfigurationError, match="Unknown profile 'unknown'"):
            _resolve_profile(config_file, profile_override="unknown")


class TestLoadConfig:
    """Test load_config function."""
    
    def test_load_config_custom_config_path(self):
        """Test loading config from custom path via environment variable."""
        yaml_content = """
current: custom
profiles:
  custom:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: custom_collection
    retriever:
      type: similarity
      search_kwargs:
        k: 5
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch.dict(os.environ, {'PAAS_AI_CONFIG': str(yaml_path)}):
                config = load_config()
                assert isinstance(config, Config)
                assert config.embedding.type == EmbeddingType.OPENAI
                assert config.vectorstore.collection_name == "custom_collection"
    
    def test_load_config_custom_config_not_found(self):
        """Test fallback when custom config file is not found."""
        with patch.dict(os.environ, {'PAAS_AI_CONFIG': '/non/existent/config.yaml'}):
            config = load_config()
            # Should fall back to local profile
            assert isinstance(config, Config)
            assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
    
    def test_load_config_default_location(self):
        """Test loading config from default location."""
        yaml_content = """
current: default
profiles:
  default:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: paas_ai_rag
    retriever:
      type: similarity
      search_kwargs:
        k: 4
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch('pathlib.Path.home', return_value=yaml_path.parent):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('src.paas_ai.core.config.loader._load_yaml_config') as mock_load:
                        mock_load.return_value = yaml.safe_load(yaml_content)
                        
                        config = load_config()
                        assert isinstance(config, Config)
                        assert config.embedding.type == EmbeddingType.OPENAI
                        assert config.vectorstore.collection_name == "paas_ai_rag"
    
    def test_load_config_default_location_not_found(self):
        """Test fallback when default config file is not found."""
        with patch('pathlib.Path.home', return_value=Path('/home/user')):
            with patch('pathlib.Path.exists', return_value=False):
                config = load_config()
                # Should fall back to local profile
                assert isinstance(config, Config)
                assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
    
    def test_load_config_profile_override(self):
        """Test loading config with profile override."""
        with patch.dict(os.environ, {'PAAS_AI_PROFILE': 'production'}):
            config = load_config()
            # Should use production profile
            assert isinstance(config, Config)
            assert config.embedding.type == EmbeddingType.OPENAI
    
    def test_load_config_fallback_to_local(self):
        """Test fallback to local profile when no config files exist."""
        with patch('pathlib.Path.exists', return_value=False):
            config = load_config()
            # Should fall back to local profile
            assert isinstance(config, Config)
            assert config.embedding.type == EmbeddingType.SENTENCE_TRANSFORMERS
    
    def test_load_config_custom_config_error(self):
        """Test error handling for custom config file errors."""
        yaml_content = """
current: invalid
profiles:
  invalid:
    embedding:
      type: invalid_type  # This should cause validation error
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch.dict(os.environ, {'PAAS_AI_CONFIG': str(yaml_path)}):
                with pytest.raises((ConfigurationError, ValidationError)):
                    load_config()
    
    def test_load_config_default_config_error(self):
        """Test error handling for default config file errors."""
        yaml_content = """
current: invalid
profiles:
  invalid:
    embedding:
      type: invalid_type  # This should cause validation error
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch('pathlib.Path.home', return_value=yaml_path.parent):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('src.paas_ai.core.config.loader._load_yaml_config') as mock_load:
                        mock_load.return_value = yaml.safe_load(yaml_content)
                        
                        with pytest.raises((ConfigurationError, ValidationError)):
                            load_config()


class TestLoadConfigFile:
    """Test load_config_file function."""
    
    def test_load_config_file_with_path(self):
        """Test loading config file with specific path."""
        yaml_content = """
current: test
profiles:
  test:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: test_collection
    retriever:
      type: similarity
      search_kwargs:
        k: 5
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            config_file, path = load_config_file(yaml_path)
            
            assert isinstance(config_file, ConfigFile)
            assert config_file.current == "test"
            assert "test" in config_file.profiles
            assert path == yaml_path.resolve()
    
    def test_load_config_file_with_custom_env_var(self):
        """Test loading config file with custom environment variable."""
        yaml_content = """
current: env_test
profiles:
  env_test:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: env_collection
    retriever:
      type: similarity
      search_kwargs:
        k: 5
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch.dict(os.environ, {'PAAS_AI_CONFIG': str(yaml_path)}):
                config_file, path = load_config_file()
                
                assert isinstance(config_file, ConfigFile)
                assert config_file.current == "env_test"
                assert path == yaml_path.resolve()
    
    def test_load_config_file_default_location(self):
        """Test loading config file from default location."""
        yaml_content = """
current: default_test
profiles:
  default_test:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: paas_ai_rag
    retriever:
      type: similarity
      search_kwargs:
        k: 5
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch('pathlib.Path.home', return_value=yaml_path.parent):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('src.paas_ai.core.config.loader._load_yaml_config') as mock_load:
                        mock_load.return_value = yaml.safe_load(yaml_content)
                        
                        config_file, path = load_config_file()
                        
                        assert isinstance(config_file, ConfigFile)
                        assert config_file.current == "default_test"
    
    def test_load_config_file_not_found(self):
        """Test error when config file is not found."""
        non_existent_path = Path("/non/existent/config.yaml")
        
        with pytest.raises(ConfigFileNotFoundError):
            load_config_file(non_existent_path)


class TestConvertEnumsToStrings:
    """Test _convert_enums_to_strings function."""
    
    def test_convert_enums_to_strings_dict(self):
        """Test converting enums to strings in dictionary."""
        from src.paas_ai.core.config.schemas import EmbeddingType, VectorStoreType
        
        obj = {
            "embedding_type": EmbeddingType.OPENAI,
            "vectorstore_type": VectorStoreType.CHROMA,
            "nested": {
                "type": EmbeddingType.SENTENCE_TRANSFORMERS
            }
        }
        
        result = _convert_enums_to_strings(obj)
        
        assert result["embedding_type"] == "openai"
        assert result["vectorstore_type"] == "chroma"
        assert result["nested"]["type"] == "sentence_transformers"
    
    def test_convert_enums_to_strings_list(self):
        """Test converting enums to strings in list."""
        from src.paas_ai.core.config.schemas import EmbeddingType, VectorStoreType
        
        obj = [EmbeddingType.OPENAI, "string", VectorStoreType.CHROMA]
        
        result = _convert_enums_to_strings(obj)
        
        assert result == ["openai", "string", "chroma"]
    
    def test_convert_enums_to_strings_enum(self):
        """Test converting single enum to string."""
        from src.paas_ai.core.config.schemas import EmbeddingType
        
        result = _convert_enums_to_strings(EmbeddingType.OPENAI)
        assert result == "openai"
    
    def test_convert_enums_to_strings_non_enum(self):
        """Test that non-enum values are returned unchanged."""
        result = _convert_enums_to_strings("string")
        assert result == "string"
        
        result = _convert_enums_to_strings(42)
        assert result == 42
        
        result = _convert_enums_to_strings(True)
        assert result is True
        
        result = _convert_enums_to_strings(None)
        assert result is None


class TestSaveConfigFile:
    """Test save_config_file function."""
    
    def test_save_config_file_with_path(self):
        """Test saving config file to specific path."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        config_file = ConfigFile(current="test", profiles={"test": config})
        
        with temp_directory() as tmpdir:
            config_path = tmpdir / "test_config.yaml"
            result_path = save_config_file(config_file, config_path)
            
            assert result_path.resolve() == config_path.resolve()
            assert config_path.exists()
            
            # Verify content
            with open(config_path, 'r') as f:
                content = f.read()
                assert "current: test" in content
                assert "type: openai" in content
    
    def test_save_config_file_with_env_var(self):
        """Test saving config file using environment variable path."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        config_file = ConfigFile(current="env_test", profiles={"env_test": config})
        
        with temp_directory() as tmpdir:
            config_path = tmpdir / "env_config.yaml"
            with patch.dict(os.environ, {'PAAS_AI_CONFIG': str(config_path)}):
                result_path = save_config_file(config_file)
                
                assert result_path.resolve() == config_path.resolve()
                assert config_path.exists()
    
    def test_save_config_file_default_location(self):
        """Test saving config file to default location."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        config_file = ConfigFile(current="default_test", profiles={"default_test": config})
        
        with temp_directory() as tmpdir: 
            with patch('pathlib.Path.home', return_value=tmpdir):
                result_path = save_config_file(config_file)
                
                expected_path = tmpdir / '.paas-ai' / 'config.yaml'
                assert result_path == expected_path
                assert expected_path.exists()
                assert expected_path.parent.exists()  # Directory should be created
    
    def test_save_config_file_creates_directory(self):
        """Test that save_config_file creates parent directories."""
        config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        config_file = ConfigFile(current="test", profiles={"test": config})
        
        with temp_directory() as tmpdir:
            config_path = tmpdir / "nested" / "deep" / "config.yaml"
            result_path = save_config_file(config_file, config_path)
            
            assert result_path.resolve() == config_path.resolve()
            assert config_path.exists()
            assert config_path.parent.exists()
            assert config_path.parent.parent.exists()


class TestCreateSampleConfig:
    """Test create_sample_config function."""
    
    def test_create_sample_config_default_location(self):
        """Test creating sample config at default location."""
        with temp_directory() as tmpdir:
            with patch('pathlib.Path.home', return_value=tmpdir):
                result_path = create_sample_config()
                
                expected_path = tmpdir / '.paas-ai' / 'config.yaml'
                assert result_path == expected_path
                assert expected_path.exists()
                assert expected_path.parent.exists()
                
                # Verify content
                with open(expected_path, 'r') as f:
                    content = f.read()
                    assert "current: my-custom-profile" in content
                    assert "type: sentence_transformers" in content
                    assert "type: chroma" in content
    
    def test_create_sample_config_custom_location(self):
        """Test creating sample config at custom location."""
        with temp_directory() as tmpdir:
            config_path = tmpdir / "custom_config.yaml"
            result_path = create_sample_config(config_path)
            
            assert result_path.resolve() == config_path.resolve()
            assert config_path.exists()
            
            # Verify content
            with open(config_path, 'r') as f:
                content = f.read()
                assert "current: my-custom-profile" in content
                assert "type: sentence_transformers" in content
    
    def test_create_sample_config_creates_directory(self):
        """Test that create_sample_config creates parent directories."""
        with temp_directory() as tmpdir:
            config_path = tmpdir / "nested" / "deep" / "sample_config.yaml"
            result_path = create_sample_config(config_path)
            
            assert result_path.resolve() == config_path.resolve()
            assert config_path.exists()
            assert config_path.parent.exists()
            assert config_path.parent.parent.exists()


class TestListAvailableProfiles:
    """Test list_available_profiles function."""
    
    def test_list_available_profiles_builtin_only(self):
        """Test listing only built-in profiles."""
        profiles = list_available_profiles(include_custom=False)
        
        expected_profiles = {"default", "local", "production", "verbose"}
        assert set(profiles.keys()) == expected_profiles
        
        # Check descriptions
        assert "OpenAI embeddings + Chroma vector store" in profiles["default"]
        assert "SentenceTransformers + Chroma" in profiles["local"]
        assert "OpenAI large embeddings + Pinecone" in profiles["production"]
        assert "Verbose mode with token tracking" in profiles["verbose"]
    
    def test_list_available_profiles_with_custom(self):
        """Test listing profiles including custom ones."""
        yaml_content = """
current: custom
profiles:
  custom:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: custom_collection
    retriever:
      type: similarity
      search_kwargs:
        k: 5
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch('pathlib.Path.home', return_value=yaml_path.parent):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('src.paas_ai.core.config.loader._load_yaml_config') as mock_load:
                        mock_load.return_value = yaml.safe_load(yaml_content)
                        
                        profiles = list_available_profiles(include_custom=True)
                        
                        # Should include both built-in and custom profiles
                        assert "default" in profiles
                        assert "local" in profiles
                        assert "custom" in profiles
                        assert "Custom profile: custom" in profiles["custom"]
    
    def test_list_available_profiles_no_custom_config(self):
        """Test listing profiles when no custom config exists."""
        with patch('pathlib.Path.exists', return_value=False):
            profiles = list_available_profiles(include_custom=True)
            
            # Should only include built-in profiles
            expected_profiles = {"default", "local", "production", "verbose"}
            assert set(profiles.keys()) == expected_profiles


class TestGetCurrentProfile:
    """Test get_current_profile function."""
    
    def test_get_current_profile_env_override(self):
        """Test getting current profile from environment override."""
        with patch.dict(os.environ, {'PAAS_AI_PROFILE': 'production'}):
            profile = get_current_profile()
            assert profile == "production"
    
    def test_get_current_profile_from_config_file(self):
        """Test getting current profile from config file."""
        yaml_content = """
current: custom_profile
profiles:
  custom_profile:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: custom_collection
    retriever:
      type: similarity
      search_kwargs:
        k: 5
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch('pathlib.Path.home', return_value=yaml_path.parent):
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('src.paas_ai.core.config.loader._load_yaml_config') as mock_load:
                        mock_load.return_value = yaml.safe_load(yaml_content)
                        
                        profile = get_current_profile()
                        assert profile == "custom_profile"
    
    def test_get_current_profile_fallback(self):
        """Test fallback when no config file exists."""
        with patch('pathlib.Path.exists', return_value=False):
            profile = get_current_profile()
            assert profile == "default"


class TestIntegration:
    """Integration tests for the configuration loading system."""
    
    def test_full_config_loading_workflow(self):
        """Test complete configuration loading workflow."""
        yaml_content = """
current: integration_test
profiles:
  integration_test:
    embedding:
      type: openai
      model_name: text-embedding-3-small
    vectorstore:
      type: chroma
      collection_name: integration_collection
    retriever:
      type: similarity
      search_kwargs:
        k: 5
    llm:
      provider: openai
      model_name: gpt-3.5-turbo
      temperature: 0.1
    citation:
      enabled: true
      verbosity: standard
      format: inline
    batch_size: 32
    validate_urls: true
    max_parallel: 5
    timeout: 30
    log_level: INFO
    multi_agent:
      enabled: true
      mode: supervisor
      default_agent: designer
      track_tokens: false
      verbose: false
    agents:
      designer:
        model: gpt-4o-mini
        temperature: 0.1
      paas_manifest_generator:
        model: gpt-4o-mini
        temperature: 0.0
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch.dict(os.environ, {'PAAS_AI_CONFIG': str(yaml_path)}):
                config = load_config()
                assert isinstance(config, Config)
                assert config.embedding.type == EmbeddingType.OPENAI
                assert config.vectorstore.collection_name == "integration_collection"
    
    def test_config_roundtrip(self):
        """Test saving and loading configuration in a roundtrip."""
        # Create original config
        original_config = Config(
            embedding=EmbeddingConfig(type=EmbeddingType.OPENAI),
            vectorstore=VectorStoreConfig(type=VectorStoreType.CHROMA),
            retriever=RetrieverConfig(type=RetrieverType.SIMILARITY)
        )
        
        original_config_file = ConfigFile(
            current="roundtrip_test",
            profiles={"roundtrip_test": original_config}
        )
        
        with temp_directory() as tmpdir:
            config_path = tmpdir / "roundtrip_config.yaml"
            
            # Save config
            save_config_file(original_config_file, config_path)
            
            # Load config back
            loaded_config_file, _ = load_config_file(config_path)
            
            # Verify they match
            assert loaded_config_file.current == "roundtrip_test"
            assert "roundtrip_test" in loaded_config_file.profiles


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_yaml_file(self):
        """Test handling of completely empty YAML file."""
        with temp_yaml_file("") as yaml_path:
            with pytest.raises(ConfigFileEmptyError):
                _load_yaml_config(yaml_path)
    
    def test_yaml_file_with_only_comments(self):
        """Test handling of YAML file with only comments."""
        yaml_content = """
# This is a comment
# Another comment
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with pytest.raises(ConfigFileEmptyError):
                _load_yaml_config(yaml_path)
    
    def test_yaml_file_with_null_content(self):
        """Test handling of YAML file with null content."""
        yaml_content = "null"
        with temp_yaml_file(yaml_content) as yaml_path:
            with pytest.raises(ConfigFileEmptyError):
                _load_yaml_config(yaml_path)
    
    def test_environment_variable_with_special_characters(self):
        """Test environment variable substitution with special characters."""
        os.environ['SPECIAL_VAR'] = 'value with spaces and "quotes"'
        try:
            config_dict = "Test: ${SPECIAL_VAR}"
            result = _substitute_env_vars(config_dict)
            assert result == 'Test: value with spaces and "quotes"'
        finally:
            del os.environ['SPECIAL_VAR']
    
    def test_nested_environment_variable_substitution(self):
        """Test deeply nested environment variable substitution."""
        os.environ['BASE'] = 'api'
        os.environ['VERSION'] = 'v1'
        os.environ['ENDPOINT'] = 'users'
        try:
            config_dict = {
                "deeply": {
                    "nested": {
                        "structure": {
                            "url": "${BASE}/${VERSION}/${ENDPOINT}"
                        }
                    }
                }
            }
            result = _substitute_env_vars(config_dict)
            assert result["deeply"]["nested"]["structure"]["url"] == "api/v1/users"
        finally:
            del os.environ['BASE']
            del os.environ['VERSION']
            del os.environ['ENDPOINT']
    
    def test_config_file_with_invalid_profile_structure(self):
        """Test handling of config file with invalid profile structure."""
        yaml_content = """
current: invalid_profile
profiles:
  invalid_profile:
    # Missing required fields
    embedding:
      type: openai
    # Missing vectorstore and retriever
"""
        with temp_yaml_file(yaml_content) as yaml_path:
            with patch.dict(os.environ, {'PAAS_AI_CONFIG': str(yaml_path)}):
                with pytest.raises((ConfigurationError, ValidationError)):
                    load_config()
    
    def test_profile_override_with_invalid_profile(self):
        """Test profile override with invalid profile name."""
        with patch.dict(os.environ, {'PAAS_AI_PROFILE': 'invalid_profile'}):
            with pytest.raises(InvalidConfigurationError, match="Unknown profile 'invalid_profile'"):
                load_config()

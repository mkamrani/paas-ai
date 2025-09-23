"""
Unit tests for RAG status CLI command.

Tests all components of the status CLI module including:
- Status command functionality
- Configuration profile handling
- Output formatting (text and JSON)
- Detailed vs basic status display
- Error handling and edge cases
- Configuration validation
- System health indicators
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.paas_ai.cli.commands.rag.status import status
from src.paas_ai.core.config import DEFAULT_CONFIG_PROFILES, ConfigurationError


class TestStatusCommand:
    """Test the status command."""

    def test_status_basic_text_output(self):
        """Test basic status with text output."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.batch_size = 10
            mock_config.max_retries = 3
            mock_config.timeout = 30
            mock_config.validate_urls = True
            mock_config.skip_invalid_docs = False
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.collection_name = "rag_collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = "similarity"
            mock_config.retriever.search_kwargs = {"k": 5}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 25,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command
            result = runner.invoke(status, [])

            # Verify
            assert result.exit_code == 0
            assert "RAG System Status" in result.output
            assert "📚 Knowledge Base Status: ready" in result.output
            assert "📊 Total Documents: 25" in result.output
            assert "🔧 Vector Store: chroma" in result.output
            assert "🤖 Embedding Model: text-embedding-3-small" in result.output
            assert "🔍 Retriever Type: similarity" in result.output
            assert "✅ RAG system operational" in result.output

    def test_status_detailed_text_output(self):
        """Test detailed status with text output."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.batch_size = 20
            mock_config.max_retries = 5
            mock_config.timeout = 60
            mock_config.validate_urls = False
            mock_config.skip_invalid_docs = True
            mock_config.embedding.type = "sentence_transformers"
            mock_config.embedding.model_name = "all-MiniLM-L6-v2"
            mock_config.vectorstore.type = "faiss"
            mock_config.vectorstore.collection_name = "test_collection"
            mock_config.vectorstore.persist_directory = Mock()
            mock_config.vectorstore.persist_directory.exists.return_value = True
            mock_config.retriever.type = "mmr"
            mock_config.retriever.search_kwargs = {"k": 10, "fetch_k": 20}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 50,
                "vectorstore_type": "faiss",
                "embedding_model": "all-MiniLM-L6-v2",
                "retriever_type": "mmr",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with detailed flag
            result = runner.invoke(status, ["--detailed"])

            # Verify basic status
            assert result.exit_code == 0
            assert "RAG System Status" in result.output
            assert "📚 Knowledge Base Status: ready" in result.output
            assert "📊 Total Documents: 50" in result.output

            # Verify detailed configuration
            assert "🔍 Detailed Configuration:" in result.output
            assert "Profile: None" in result.output
            assert "Batch Size: 20" in result.output
            assert "Max Retries: 5" in result.output
            assert "Timeout: 60s" in result.output
            assert "URL Validation: False" in result.output
            assert "Skip Invalid Docs: True" in result.output
            assert "Directory Exists: True" in result.output

            # Verify embedding configuration
            assert "⚙️ Embedding Configuration:" in result.output
            assert "Type: sentence_transformers" in result.output
            assert "Model: all-MiniLM-L6-v2" in result.output

            # Verify vector store configuration
            assert "🗄️ Vector Store Configuration:" in result.output
            assert "Type: faiss" in result.output
            assert "Collection: test_collection" in result.output

            # Verify retriever configuration
            assert "🔍 Retriever Configuration:" in result.output
            assert "Type: mmr" in result.output
            assert "Search Kwargs: {'k': 10, 'fetch_k': 20}" in result.output

    def test_status_json_output(self):
        """Test status with JSON output format."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.batch_size = 15
            mock_config.max_retries = 3
            mock_config.timeout = 45
            mock_config.validate_urls = True
            mock_config.skip_invalid_docs = False
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-large"
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.collection_name = "json_collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = "similarity"
            mock_config.retriever.search_kwargs = {"k": 7}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 30,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-large",
                "retriever_type": "similarity",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with JSON format
            result = runner.invoke(status, ["--format", "json"])

            # Verify
            assert result.exit_code == 0

            # Parse JSON output - extract JSON from the logged output
            # The JSON is logged as multi-line output with ANSI color codes
            import re

            # Find the JSON part after the log prefix - handle multi-line JSON
            json_match = re.search(r"\{.*\}", result.output, re.DOTALL)
            assert json_match is not None, f"Could not find JSON in output: {result.output}"
            json_output = json_match.group(0)
            parsed_output = json.loads(json_output)

            assert parsed_output["status"] == "ready"
            assert parsed_output["total_documents"] == 30
            assert parsed_output["vectorstore_type"] == "chroma"
            assert parsed_output["embedding_model"] == "text-embedding-3-large"
            assert parsed_output["retriever_type"] == "similarity"
            assert parsed_output["config_profile"] is None

    def test_status_detailed_json_output(self):
        """Test detailed status with JSON output format."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.batch_size = 25
            mock_config.max_retries = 4
            mock_config.timeout = 90
            mock_config.validate_urls = False
            mock_config.skip_invalid_docs = True
            mock_config.embedding.type = "sentence_transformers"
            mock_config.embedding.model_name = "paraphrase-MiniLM-L6-v2"
            mock_config.vectorstore.type = "faiss"
            mock_config.vectorstore.collection_name = "detailed_collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = "mmr"
            mock_config.retriever.search_kwargs = {"k": 8, "fetch_k": 16}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 75,
                "vectorstore_type": "faiss",
                "embedding_model": "paraphrase-MiniLM-L6-v2",
                "retriever_type": "mmr",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with detailed and JSON flags
            result = runner.invoke(status, ["--detailed", "--format", "json"])

            # Verify
            assert result.exit_code == 0

            # Parse JSON output - extract JSON from the logged output
            import re

            # Find the JSON part after the log prefix - handle multi-line JSON
            json_match = re.search(r"\{.*\}", result.output, re.DOTALL)
            assert json_match is not None, f"Could not find JSON in output: {result.output}"
            json_output = json_match.group(0)
            parsed_output = json.loads(json_output)

            # Verify basic stats
            assert parsed_output["status"] == "ready"
            assert parsed_output["total_documents"] == 75
            assert parsed_output["vectorstore_type"] == "faiss"
            assert parsed_output["embedding_model"] == "paraphrase-MiniLM-L6-v2"
            assert parsed_output["retriever_type"] == "mmr"
            assert parsed_output["config_profile"] is None

            # Verify detailed configuration
            assert "configuration" in parsed_output
            config = parsed_output["configuration"]
            assert config["batch_size"] == 25
            assert config["max_retries"] == 4
            assert config["timeout"] == 90
            assert config["validate_urls"] == False
            assert config["skip_invalid_docs"] == True

            # Verify embedding configuration
            assert config["embedding"]["type"] == "sentence_transformers"
            assert config["embedding"]["model_name"] == "paraphrase-MiniLM-L6-v2"

            # Verify vector store configuration
            assert config["vectorstore"]["type"] == "faiss"
            assert config["vectorstore"]["collection_name"] == "detailed_collection"
            assert config["vectorstore"]["persist_directory"] is None

            # Verify retriever configuration
            assert config["retriever"]["type"] == "mmr"
            assert config["retriever"]["search_kwargs"] == {"k": 8, "fetch_k": 16}

    def test_status_empty_knowledge_base(self):
        """Test status when knowledge base is empty."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats for empty knowledge base
            mock_stats = {
                "status": "empty",
                "total_documents": 0,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command
            result = runner.invoke(status, [])

            # Verify
            assert result.exit_code == 0
            assert "📚 Knowledge Base Status: empty" in result.output
            assert "📊 Total Documents: 0" in result.output
            assert "⚠️ Knowledge base is empty" in result.output
            assert "💡 Add resources using: paas-ai rag resources add" in result.output
            # Should not show retriever type for empty knowledge base
            assert "🔍 Retriever Type:" not in result.output

    def test_status_with_config_profile(self):
        """Test status with specific config profile."""
        runner = CliRunner()

        with patch(
            "src.paas_ai.cli.commands.rag.status.DEFAULT_CONFIG_PROFILES"
        ) as mock_profiles, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mock profile
            mock_profile = Mock()
            mock_profile.batch_size = 30
            mock_profile.max_retries = 2
            mock_profile.timeout = 120
            mock_profile.validate_urls = True
            mock_profile.skip_invalid_docs = False
            mock_profile.embedding.type = "openai"
            mock_profile.embedding.model_name = "text-embedding-ada-002"
            mock_profile.vectorstore.type = "chroma"
            mock_profile.vectorstore.collection_name = "profile_collection"
            mock_profile.vectorstore.persist_directory = None
            mock_profile.retriever.type = "similarity"
            mock_profile.retriever.search_kwargs = {"k": 3}
            mock_profiles.__getitem__.return_value = mock_profile

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 40,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-ada-002",
                "retriever_type": "similarity",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with config profile
            result = runner.invoke(status, ["--config-profile", "production"])

            # Verify
            assert result.exit_code == 0
            assert "📚 Knowledge Base Status: ready" in result.output
            assert "📊 Total Documents: 40" in result.output
            mock_profiles.__getitem__.assert_called_once_with("production")

    def test_status_configuration_error(self):
        """Test status with configuration error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config:
            mock_load_config.side_effect = ConfigurationError("Config not found")

            result = runner.invoke(status, [])

            assert result.exit_code == 1
            # Verify the error is handled properly - focus on exit code and that error is in output
            assert "Config not found" in result.output

    def test_status_processing_error(self):
        """Test status with processing error."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock error in get_stats
            mock_processor.get_stats.side_effect = Exception("Stats failed")

            result = runner.invoke(status, [])

            assert result.exit_code == 1
            assert "Status check failed" in result.output


class TestStatusEdgeCases:
    """Test edge cases for status command."""

    def test_status_with_none_persist_directory(self):
        """Test status with None persist directory."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.batch_size = 10
            mock_config.max_retries = 3
            mock_config.timeout = 30
            mock_config.validate_urls = True
            mock_config.skip_invalid_docs = False
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.collection_name = "test_collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = "similarity"
            mock_config.retriever.search_kwargs = {"k": 5}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 15,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with detailed flag
            result = runner.invoke(status, ["--detailed"])

            # Verify
            assert result.exit_code == 0
            # When persist_directory is None, it's not shown in detailed output
            # The code only shows persist directory when it's not None (line 81 in status.py)
            assert "Persist Directory:" not in result.output
            assert "Directory Exists:" not in result.output

    def test_status_with_path_persist_directory(self):
        """Test status with Path object persist directory."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.__str__ = Mock(return_value="/tmp/rag_data")

            mock_config = Mock()
            mock_config.batch_size = 10
            mock_config.max_retries = 3
            mock_config.timeout = 30
            mock_config.validate_urls = True
            mock_config.skip_invalid_docs = False
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.collection_name = "test_collection"
            mock_config.vectorstore.persist_directory = mock_path
            mock_config.retriever.type = "similarity"
            mock_config.retriever.search_kwargs = {"k": 5}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 20,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with detailed flag
            result = runner.invoke(status, ["--detailed"])

            # Verify
            assert result.exit_code == 0
            assert "Persist Directory: /tmp/rag_data" in result.output
            assert "Directory Exists: True" in result.output

    def test_status_missing_retriever_type(self):
        """Test status when retriever type is missing from stats."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats without retriever_type
            mock_stats = {
                "status": "ready",
                "total_documents": 10,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small"
                # Missing retriever_type
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command
            result = runner.invoke(status, [])

            # Verify
            assert result.exit_code == 0
            assert "🔍 Retriever Type: N/A" in result.output

    def test_status_with_complex_search_kwargs(self):
        """Test status with complex search kwargs."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            complex_search_kwargs = {
                "k": 10,
                "fetch_k": 20,
                "lambda_mult": 0.5,
                "filter": {"type": "dsl"},
                "score_threshold": 0.7,
            }

            mock_config = Mock()
            mock_config.batch_size = 10
            mock_config.max_retries = 3
            mock_config.timeout = 30
            mock_config.validate_urls = True
            mock_config.skip_invalid_docs = False
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.collection_name = "test_collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = "mmr"
            mock_config.retriever.search_kwargs = complex_search_kwargs
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 25,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "mmr",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with detailed flag
            result = runner.invoke(status, ["--detailed"])

            # Verify
            assert result.exit_code == 0
            assert (
                "Search Kwargs: {'k': 10, 'fetch_k': 20, 'lambda_mult': 0.5, 'filter': {'type': 'dsl'}, 'score_threshold': 0.7}"
                in result.output
            )

    def test_status_json_with_none_persist_directory(self):
        """Test JSON status with None persist directory."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.batch_size = 10
            mock_config.max_retries = 3
            mock_config.timeout = 30
            mock_config.validate_urls = True
            mock_config.skip_invalid_docs = False
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.collection_name = "test_collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = "similarity"
            mock_config.retriever.search_kwargs = {"k": 5}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 15,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with detailed and JSON flags
            result = runner.invoke(status, ["--detailed", "--format", "json"])

            # Verify
            assert result.exit_code == 0

            # Parse JSON output - extract JSON from the logged output
            import re

            # Find the JSON part after the log prefix - handle multi-line JSON
            json_match = re.search(r"\{.*\}", result.output, re.DOTALL)
            assert json_match is not None, f"Could not find JSON in output: {result.output}"
            json_output = json_match.group(0)
            parsed_output = json.loads(json_output)

            # Verify persist directory is None in JSON
            assert parsed_output["configuration"]["vectorstore"]["persist_directory"] is None

    def test_status_with_invalid_config_profile(self):
        """Test status with invalid config profile."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.DEFAULT_CONFIG_PROFILES") as mock_profiles:
            # Mock KeyError for invalid profile
            mock_profiles.__getitem__.side_effect = KeyError("Invalid profile")

            result = runner.invoke(status, ["--config-profile", "invalid"])

            # Should handle KeyError gracefully - Click returns exit code 2 for usage errors
            assert result.exit_code == 2


class TestStatusIntegration:
    """Integration tests for status command."""

    def test_full_workflow_basic_status(self):
        """Test complete workflow for basic status."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.embedding.type = "openai"
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock comprehensive stats
            mock_stats = {
                "status": "ready",
                "total_documents": 100,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-large",
                "retriever_type": "mmr",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command
            result = runner.invoke(status, [])

            # Verify complete workflow
            assert result.exit_code == 0
            assert "RAG System Status" in result.output
            assert "📚 Knowledge Base Status: ready" in result.output
            assert "📊 Total Documents: 100" in result.output
            assert "🔧 Vector Store: chroma" in result.output
            assert "🤖 Embedding Model: text-embedding-3-large" in result.output
            assert "🔍 Retriever Type: mmr" in result.output
            assert "✅ RAG system operational" in result.output

            # Verify all components were called
            mock_load_config.assert_called_once()
            mock_processor_class.assert_called_once_with(mock_config)
            mock_processor.get_stats.assert_called_once()

    def test_full_workflow_detailed_status(self):
        """Test complete workflow for detailed status."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup comprehensive mocks
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.__str__ = Mock(return_value="/data/rag_persist")

            mock_config = Mock()
            mock_config.batch_size = 50
            mock_config.max_retries = 5
            mock_config.timeout = 120
            mock_config.validate_urls = True
            mock_config.skip_invalid_docs = False
            mock_config.embedding.type = "sentence_transformers"
            mock_config.embedding.model_name = "all-mpnet-base-v2"
            mock_config.vectorstore.type = "faiss"
            mock_config.vectorstore.collection_name = "production_collection"
            mock_config.vectorstore.persist_directory = mock_path
            mock_config.retriever.type = "mmr"
            mock_config.retriever.search_kwargs = {"k": 15, "fetch_k": 30, "lambda_mult": 0.7}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock comprehensive stats
            mock_stats = {
                "status": "ready",
                "total_documents": 500,
                "vectorstore_type": "faiss",
                "embedding_model": "all-mpnet-base-v2",
                "retriever_type": "mmr",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with detailed flag
            result = runner.invoke(status, ["--detailed"])

            # Verify complete workflow
            assert result.exit_code == 0

            # Verify basic status
            assert "RAG System Status" in result.output
            assert "📚 Knowledge Base Status: ready" in result.output
            assert "📊 Total Documents: 500" in result.output
            assert "🔧 Vector Store: faiss" in result.output
            assert "🤖 Embedding Model: all-mpnet-base-v2" in result.output
            assert "🔍 Retriever Type: mmr" in result.output

            # Verify detailed configuration
            assert "🔍 Detailed Configuration:" in result.output
            assert "Profile: None" in result.output
            assert "Batch Size: 50" in result.output
            assert "Max Retries: 5" in result.output
            assert "Timeout: 120s" in result.output
            assert "URL Validation: True" in result.output
            assert "Skip Invalid Docs: False" in result.output
            assert "Persist Directory: /data/rag_persist" in result.output
            assert "Directory Exists: True" in result.output

            # Verify embedding configuration
            assert "⚙️ Embedding Configuration:" in result.output
            assert "Type: sentence_transformers" in result.output
            assert "Model: all-mpnet-base-v2" in result.output

            # Verify vector store configuration
            assert "🗄️ Vector Store Configuration:" in result.output
            assert "Type: faiss" in result.output
            assert "Collection: production_collection" in result.output

            # Verify retriever configuration
            assert "🔍 Retriever Configuration:" in result.output
            assert "Type: mmr" in result.output
            assert "Search Kwargs: {'k': 15, 'fetch_k': 30, 'lambda_mult': 0.7}" in result.output

            # Verify system health
            assert "✅ RAG system operational" in result.output

            # Verify all components were called
            mock_load_config.assert_called_once()
            mock_processor_class.assert_called_once_with(mock_config)
            mock_processor.get_stats.assert_called_once()

    def test_full_workflow_json_status(self):
        """Test complete workflow for JSON status."""
        runner = CliRunner()

        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_config.batch_size = 20
            mock_config.max_retries = 3
            mock_config.timeout = 60
            mock_config.validate_urls = False
            mock_config.skip_invalid_docs = True
            mock_config.embedding.type = "openai"
            mock_config.embedding.model_name = "text-embedding-3-small"
            mock_config.vectorstore.type = "chroma"
            mock_config.vectorstore.collection_name = "json_collection"
            mock_config.vectorstore.persist_directory = None
            mock_config.retriever.type = "similarity"
            mock_config.retriever.search_kwargs = {"k": 8}
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock stats
            mock_stats = {
                "status": "ready",
                "total_documents": 75,
                "vectorstore_type": "chroma",
                "embedding_model": "text-embedding-3-small",
                "retriever_type": "similarity",
            }
            mock_processor.get_stats.return_value = mock_stats

            # Run command with JSON format
            result = runner.invoke(status, ["--format", "json"])

            # Verify complete workflow
            assert result.exit_code == 0

            # Parse JSON output - extract JSON from the logged output
            import re

            # Find the JSON part after the log prefix - handle multi-line JSON
            json_match = re.search(r"\{.*\}", result.output, re.DOTALL)
            assert json_match is not None, f"Could not find JSON in output: {result.output}"
            json_output = json_match.group(0)
            parsed_output = json.loads(json_output)

            # Verify basic stats
            assert parsed_output["status"] == "ready"
            assert parsed_output["total_documents"] == 75
            assert parsed_output["vectorstore_type"] == "chroma"
            assert parsed_output["embedding_model"] == "text-embedding-3-small"
            assert parsed_output["retriever_type"] == "similarity"
            assert parsed_output["config_profile"] is None

            # Verify all components were called
            mock_load_config.assert_called_once()
            mock_processor_class.assert_called_once_with(mock_config)
            mock_processor.get_stats.assert_called_once()

    def test_error_handling_workflow(self):
        """Test error handling across the workflow."""
        runner = CliRunner()

        # Test configuration error
        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config:
            mock_load_config.side_effect = ConfigurationError("Config file not found")

            result = runner.invoke(status, [])

            assert result.exit_code == 1
            # Verify the error is handled properly - focus on exit code and that error is in output
            assert "Config file not found" in result.output

        # Test processing error
        with patch("src.paas_ai.cli.commands.rag.status.load_config") as mock_load_config, patch(
            "src.paas_ai.cli.commands.rag.status.RAGProcessor"
        ) as mock_processor_class:
            # Setup mocks
            mock_config = Mock()
            mock_load_config.return_value = mock_config

            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            # Mock error in get_stats
            mock_processor.get_stats.side_effect = Exception("Database connection failed")

            result = runner.invoke(status, [])

            assert result.exit_code == 1
            assert "Status check failed" in result.output

    def test_config_profile_workflow(self):
        """Test workflow with different config profiles."""
        runner = CliRunner()

        profiles = ["default", "local", "production"]

        for profile in profiles:
            with patch(
                "src.paas_ai.cli.commands.rag.status.DEFAULT_CONFIG_PROFILES"
            ) as mock_profiles, patch(
                "src.paas_ai.cli.commands.rag.status.RAGProcessor"
            ) as mock_processor_class:
                # Setup mock profile
                mock_profile = Mock()
                mock_profile.embedding.type = "openai"
                mock_profiles.__getitem__.return_value = mock_profile

                mock_processor = Mock()
                mock_processor_class.return_value = mock_processor

                # Mock stats
                mock_stats = {
                    "status": "ready",
                    "total_documents": 25,
                    "vectorstore_type": "chroma",
                    "embedding_model": "text-embedding-3-small",
                    "retriever_type": "similarity",
                }
                mock_processor.get_stats.return_value = mock_stats

                # Run command with specific profile
                result = runner.invoke(status, ["--config-profile", profile])

                # Verify
                assert result.exit_code == 0
                assert "📚 Knowledge Base Status: ready" in result.output
                mock_profiles.__getitem__.assert_called_once_with(profile)

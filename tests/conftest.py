"""Common pytest fixtures"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest
from typer.testing import CliRunner

# Import our test utilities
from tests.utils import TempDirectoryManager, CLITestHelper


@pytest.fixture
def sample_data_path():
    """Fixture providing path to the sample data directory."""
    base_dir = Path(__file__).parent
    return str(base_dir / "data")


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
        f.write("This is sample text content for testing Synthetic Data Kit.")
        file_path = f.name

    yield file_path

    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


@pytest.fixture
def sample_qa_pairs():
    """Return sample QA pairs for testing."""
    return [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data that mimics real data.",
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns.",
        },
    ]


@pytest.fixture
def sample_qa_pairs_file():
    """Create a temporary file with sample QA pairs for testing."""
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data that mimics real data.",
        },
        {
            "question": "Why use synthetic data for fine-tuning?",
            "answer": "Synthetic data can help overcome data scarcity and privacy concerns.",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(qa_pairs, f)
        file_path = f.name

    yield file_path

    # Cleanup
    if os.path.exists(file_path):
        os.unlink(file_path)


# Mock factories for reusable test components


class MockLLMClientFactory:
    """Factory for creating mock LLM clients with different configurations."""

    @staticmethod
    def create_qa_client(qa_pairs=None):
        """Create a mock client for QA generation."""
        if qa_pairs is None:
            qa_pairs = [
                {
                    "question": "What is synthetic data?",
                    "answer": "Synthetic data is artificially generated data that mimics real data.",
                },
                {
                    "question": "Why use synthetic data for fine-tuning?",
                    "answer": "Synthetic data can help overcome data scarcity and privacy concerns.",
                },
            ]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps(qa_pairs)
        mock_client.batch_completion.return_value = [json.dumps([pair]) for pair in qa_pairs]
        return mock_client

    @staticmethod
    def create_cot_client(cot_examples=None):
        """Create a mock client for Chain of Thought generation."""
        if cot_examples is None:
            cot_examples = [
                {
                    "reasoning": "Let me think step by step...",
                    "answer": "Based on my analysis, the answer is...",
                }
            ]

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps(cot_examples)
        mock_client.batch_completion.return_value = [
            json.dumps([example]) for example in cot_examples
        ]
        return mock_client

    @staticmethod
    def create_summary_client(summary_text="This is a test summary."):
        """Create a mock client for summary generation."""
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = summary_text
        return mock_client

    @staticmethod
    def create_rating_client(ratings=None):
        """Create a mock client for content rating."""
        if ratings is None:
            ratings = [8, 7, 9]  # Default ratings

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = json.dumps(ratings)
        mock_client.batch_completion.return_value = [json.dumps([rating]) for rating in ratings]
        return mock_client


@pytest.fixture
def llm_client_factory():
    """Factory fixture for creating various mock LLM clients."""
    return MockLLMClientFactory


@pytest.fixture
def mock_llm_client(llm_client_factory):
    """Default mock LLM client for backward compatibility."""
    return llm_client_factory.create_qa_client()


class MockConfigFactory:
    """Factory for creating various mock configurations."""

    @staticmethod
    def create_api_config(provider="api-endpoint", api_key="mock-key", model="mock-model"):
        """Create a mock API endpoint configuration."""
        return {
            "llm": {"provider": provider},
            "api-endpoint": {
                "api_base": "https://api.together.xyz/v1",
                "api_key": api_key,
                "model": model,
                "azure_api_version": "2024-06-01",
                "max_retries": 3,
                "retry_delay": 1,
            },
            "generation": {
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 0.95,
                "batch_size": 32,
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "output",
            },
        }

    @staticmethod
    def create_vllm_config(model="mock-vllm-model"):
        """Create a mock vLLM configuration."""
        return {
            "llm": {"provider": "vllm"},
            "vllm": {
                "api_base": "http://localhost:8000",
                "model": model,
                "max_retries": 3,
                "retry_delay": 1,
            },
            "generation": {
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 0.95,
                "batch_size": 16,
            },
            "paths": {
                "data_dir": "data",
                "output_dir": "output",
            },
        }


@pytest.fixture
def config_factory():
    """Factory fixture for creating various mock configurations."""
    return MockConfigFactory


@pytest.fixture
def mock_config(config_factory):
    """Default mock configuration for backward compatibility."""
    return config_factory.create_api_config()


@pytest.fixture
def test_env():
    """Set test environment variables."""
    original_env = os.environ.copy()
    os.environ["PROJECT_TEST_ENV"] = "1"
    # Only use API_ENDPOINT_KEY for consistency with the code
    os.environ["API_ENDPOINT_KEY"] = "mock-api-key-for-testing"
    os.environ["SDK_VERBOSE"] = "false"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def patch_config(config_factory):
    """Patch the config loader to return a mock configuration."""
    mock_config = config_factory.create_api_config()
    
    with patch("synthetic_data_kit.utils.config.load_config") as mock_load_config:
        mock_load_config.return_value = mock_config
        yield mock_load_config

@pytest.fixture
def patch_llm_client_config(config_factory):
    """Patch the config loader to return a mock configuration."""
    mock_config = config_factory.create_api_config()
    
    with patch("synthetic_data_kit.models.llm_client.load_config") as mock_load_config:
        mock_load_config.return_value = mock_config
        yield mock_load_config

@pytest.fixture
def patch_cli_config(config_factory):
    """Patch the config loader to return a mock configuration."""
    mock_config = config_factory.create_api_config()
        
    with patch("synthetic_data_kit.cli.load_config") as mock_load_config:
        mock_load_config.return_value = mock_config
        yield mock_load_config


@pytest.fixture
def patch_vllm_config(config_factory):
    """Patch the config loader to return a vLLM configuration."""
    with patch("synthetic_data_kit.utils.config.load_config") as mock_load_config:
        mock_load_config.return_value = config_factory.create_vllm_config()
        yield mock_load_config


# Additional utility fixtures for common test patterns


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_file_operations():
    """Mock common file operations for testing."""
    with patch("builtins.open"), patch("os.makedirs"), patch(
        "os.path.exists", return_value=True
    ), patch("pathlib.Path.exists", return_value=True):
        yield


@pytest.fixture
def sample_cot_data():
    """Sample Chain of Thought data for testing."""
    return [
        {
            "query": "What is 2 + 2?",
            "reasoning": "Let me solve this step by step. First, I need to add 2 and 2. This is a basic arithmetic operation.",
            "answer": "2 + 2 = 4",
        },
        {
            "query": "Explain photosynthesis",
            "reasoning": "To explain photosynthesis, I need to break it down into its key components and process.",
            "answer": "Photosynthesis is the process by which plants convert sunlight into energy.",
        },
    ]


@pytest.fixture
def sample_conversations():
    """Sample conversation data for testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "How do I bake a cake?"},
                {
                    "role": "assistant",
                    "content": "To bake a cake, you need flour, eggs, sugar, and butter.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to current weather data."},
            ]
        },
    ]
#New fixtures
@pytest.fixture
def cli_runner():
    """Fixture providing CLI runner for testing CLI commands.
    
    This replaces the repetitive CliRunner() setup in multiple test files.
    """
    return CliRunner()


@pytest.fixture
def cli_helper():
    """Fixture providing CLI test helper with common utilities.
    
    This replaces manual CLI testing setup in functional tests.
    """
    from synthetic_data_kit.cli import app
    return CLITestHelper(app)


@pytest.fixture
def temp_env():
    """Fixture providing temporary directory environment with cleanup.
    
    This replaces manual tempfile.mkdtemp() and cleanup logic in multiple tests.
    """
    with TempDirectoryManager() as temp_mgr:
        yield temp_mgr


@pytest.fixture
def standard_test_files():
    """Fixture providing standard test files in a temporary directory.
    
    This replaces repeated file creation patterns across tests.
    """
    with TempDirectoryManager() as temp_mgr:
        # Create standard test files that many tests need
        files = temp_mgr.create_files({
            "test.txt": "This is test content for processing.",
            "sample.txt": "Sample document content for testing.",
            "document.txt": "Document with multiple sentences. It has detailed content for analysis."
        })
        yield temp_mgr, files


@pytest.fixture
def standard_qa_files():
    """Fixture providing standard QA JSON files in a temporary directory.
    
    This replaces repeated QA file creation in integration tests.
    """
    with TempDirectoryManager() as temp_mgr:
        qa_data = {
            "qa1.json": {
                "qa_pairs": [
                    {"question": "What is AI?", "answer": "Artificial Intelligence"},
                    {"question": "What is ML?", "answer": "Machine Learning"}
                ]
            },
            "qa2.json": {
                "qa_pairs": [
                    {"question": "What is data science?", "answer": "Analyzing data for insights"},
                    {"question": "What is Python?", "answer": "A programming language"}
                ]
            }
        }
        files = temp_mgr.create_json_files(qa_data)
        yield temp_mgr, files


class MockPatchManager:
    """Context manager for common patch patterns used across tests.
    
    This reduces repetitive patching patterns found in multiple test files.
    """
    
    @contextmanager
    def patch_process_file(self, return_value=None):
        """Standard process_file patching for ingest tests."""
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
            mock_process.return_value = return_value or "/tmp/mock_output.txt"
            yield mock_process
    
    @contextmanager
    def patch_create_process_file(self, return_value=None):
        """Standard process_file patching for create tests."""
        with patch("synthetic_data_kit.core.create.process_file") as mock_process:
            mock_process.return_value = return_value or "/tmp/mock_qa.json"
            yield mock_process
    
    @contextmanager
    def patch_curate_qa_pairs(self, return_value=None):
        """Standard curate_qa_pairs patching for curate tests."""
        with patch("synthetic_data_kit.core.curate.curate_qa_pairs") as mock_curate:
            mock_curate.return_value = return_value or "/tmp/mock_curated.json"
            yield mock_curate
    
    @contextmanager
    def patch_convert_format(self, return_value=None):
        """Standard convert_format patching for save-as tests."""
        with patch("synthetic_data_kit.core.save_as.convert_format") as mock_convert:
            mock_convert.return_value = return_value or "/tmp/mock_converted.jsonl"
            yield mock_convert
    
    @contextmanager
    def patch_directory_processor(self, success_count=1, failed_count=0):
        """Standard directory processor patching with configurable results."""
        result = {"total_files": success_count + failed_count, "successful": success_count, "failed": failed_count}
        
        patches = [
            patch("synthetic_data_kit.utils.directory_processor.process_directory_ingest", return_value=result),
            patch("synthetic_data_kit.utils.directory_processor.process_directory_create", return_value=result),
            patch("synthetic_data_kit.utils.directory_processor.process_directory_curate", return_value=result),
            patch("synthetic_data_kit.utils.directory_processor.process_directory_save_as", return_value=result),
        ]
        
        with patch.multiple(
            "synthetic_data_kit.utils.directory_processor",
            process_directory_ingest=patches[0],
            process_directory_create=patches[1], 
            process_directory_curate=patches[2],
            process_directory_save_as=patches[3]
        ):
            yield
    
    @contextmanager
    def patch_llm_client(self, response="Mock LLM response"):
        """Standard LLM client patching for create/curate tests."""
        with patch("synthetic_data_kit.models.llm_client.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.chat_completion.return_value = response
            mock_client_class.return_value = mock_client
            yield mock_client


@pytest.fixture
def mock_patches():
    """Fixture providing mock patch manager for common patching patterns.
    
    This reduces repetitive mock setup across test files.
    """
    return MockPatchManager()


class StandardTestData:
    """Centralized test data to reduce duplication across test files."""
    
    # Standard file content patterns
    FILE_CONTENTS = {
        'short': "Brief test content.",
        'medium': "Medium length test content with more details and information.",
        'long': "Very long test content with extensive details. " * 20,
        'html': "<html><body><h1>Test HTML</h1><p>HTML content for testing.</p></body></html>",
        'json_qa': '{"qa_pairs": [{"question": "Test?", "answer": "Yes."}]}'
    }
    
    # Standard QA pairs for testing
    QA_PAIRS = [
        {"question": "What is synthetic data?", "answer": "Artificially generated data"},
        {"question": "Why use synthetic data?", "answer": "Privacy and diversity benefits"},
        {"question": "How is it created?", "answer": "Using machine learning models"}
    ]
    
    # Standard directory stats for mocking
    DIRECTORY_STATS = {
        'empty': {'total_files': 0, 'supported_files': 0, 'unsupported_files': 0, 'by_extension': {}, 'file_list': []},
        'single_txt': {'total_files': 1, 'supported_files': 1, 'unsupported_files': 0, 'by_extension': {'.txt': 1}, 'file_list': ['test.txt']},
        'mixed': {'total_files': 3, 'supported_files': 2, 'unsupported_files': 1, 'by_extension': {'.txt': 2, '.xyz': 1}, 'file_list': ['test1.txt', 'test2.txt', 'unsupported.xyz']}
    }
    
    # Standard process results
    PROCESS_RESULTS = {
        'success': {"total_files": 1, "successful": 1, "failed": 0},
        'partial_failure': {"total_files": 3, "successful": 2, "failed": 1},
        'complete_failure': {"total_files": 2, "successful": 0, "failed": 2}
    }


@pytest.fixture 
def test_data():
    """Fixture providing centralized test data constants.
    
    This reduces test data duplication across multiple test files.
    """
    return StandardTestData


@pytest.fixture
def mixed_file_directory():
    """Fixture providing a directory with mixed supported/unsupported files.
    
    This replaces the repeated mixed file setup in edge case tests.
    """
    with TempDirectoryManager() as temp_mgr:
        # Create supported files
        supported_files = temp_mgr.create_files({
            "doc1.txt": "Supported text file content",
            "doc2.txt": "Another supported file"
        })
        
        # Create unsupported files
        unsupported_files = temp_mgr.create_files({
            "file.xyz": "Unsupported file content",
            "data.unknown": "Another unsupported file"
        })
        
        yield temp_mgr, supported_files, unsupported_files

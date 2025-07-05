import os
import tempfile
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner


class TestFileFactory:
    """Factory for creating test files with various content types."""
    
    @staticmethod
    def create_test_files(directory: str, file_specs: Dict[str, str]) -> List[str]:
        """Create test files with specified content.
        
        Args:
            directory: Directory to create files in
            file_specs: Dict mapping filename to content
            
        Returns:
            List of created file paths
        """
        created_files = []
        for filename, content in file_specs.items():
            file_path = os.path.join(directory, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            created_files.append(file_path)
        return created_files
    
    @staticmethod
    def create_json_files(directory: str, file_specs: Dict[str, Dict[str, Any]]) -> List[str]:
        """Create JSON test files with specified content.
        
        Args:
            directory: Directory to create files in
            file_specs: Dict mapping filename to JSON content
            
        Returns:
            List of created file paths
        """
        created_files = []
        for filename, content in file_specs.items():
            file_path = os.path.join(directory, filename)
            with open(file_path, 'w') as f:
                json.dump(content, f)
            created_files.append(file_path)
        return created_files

    @staticmethod
    def create_mixed_directory(directory: str, 
                             supported_files: int = 2, 
                             unsupported_files: int = 1) -> Tuple[List[str], List[str]]:
        """Create a directory with mixed supported and unsupported files.
        
        Args:
            directory: Directory to create files in
            supported_files: Number of supported files to create
            unsupported_files: Number of unsupported files to create
            
        Returns:
            Tuple of (supported_file_paths, unsupported_file_paths)
        """
        supported_paths = []
        unsupported_paths = []
        
        # Create supported files
        for i in range(supported_files):
            file_path = os.path.join(directory, f"supported_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Test content {i}")
            supported_paths.append(file_path)
        
        # Create unsupported files
        for i in range(unsupported_files):
            file_path = os.path.join(directory, f"unsupported_{i}.xyz")
            with open(file_path, 'w') as f:
                f.write(f"Unsupported content {i}")
            unsupported_paths.append(file_path)
        
        return supported_paths, unsupported_paths


class CLITestHelper:
    """Helper class for CLI testing with common patterns."""
    
    def __init__(self, app):
        self.app = app
        self.runner = CliRunner()
    
    def run_command(self, cmd_args: List[str], expect_success: bool = True) -> Any:
        """Run a CLI command with common setup.
        
        Args:
            cmd_args: Command arguments
            expect_success: Whether to expect success (exit code 0)
            
        Returns:
            CliRunner result
        """
        result = self.runner.invoke(self.app, cmd_args)
        
        if expect_success:
            assert result.exit_code == 0, f"Command failed: {result.stdout}"
        
        return result
    
    def assert_cli_success(self, result: Any, expected_patterns: List[str]):
        """Assert CLI command succeeded with expected output patterns.
        
        Args:
            result: CliRunner result
            expected_patterns: List of strings that should appear in stdout
        """
        assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}"
        
        for pattern in expected_patterns:
            assert pattern in result.stdout, f"Expected pattern '{pattern}' not found in output: {result.stdout}"
    
    def assert_cli_failure(self, result: Any, expected_patterns: Optional[List[str]] = None):
        """Assert CLI command failed with expected error patterns.
        
        Args:
            result: CliRunner result
            expected_patterns: List of error strings that should appear in output
        """
        assert result.exit_code != 0, f"Command unexpectedly succeeded: {result.stdout}"
        
        if expected_patterns:
            output = result.stdout + result.stderr if hasattr(result, 'stderr') else result.stdout
            for pattern in expected_patterns:
                assert pattern in output, f"Expected error pattern '{pattern}' not found in output: {output}"


class DirectoryStatsHelper:
    """Helper for asserting directory statistics."""
    
    @staticmethod
    def assert_directory_stats(stats: Dict[str, Any], 
                             expected_total: int, 
                             expected_supported: int):
        """Assert directory statistics match expectations.
        
        Args:
            stats: Directory statistics dict
            expected_total: Expected total file count
            expected_supported: Expected supported file count
        """
        assert stats.get('total_files') == expected_total, \
            f"Expected {expected_total} total files, got {stats.get('total_files')}"
        
        assert stats.get('supported_files') == expected_supported, \
            f"Expected {expected_supported} supported files, got {stats.get('supported_files')}"
        
        expected_unsupported = expected_total - expected_supported
        assert stats.get('unsupported_files') == expected_unsupported, \
            f"Expected {expected_unsupported} unsupported files, got {stats.get('unsupported_files')}"


class MockConfigHelper:
    """Helper for creating mock configurations."""
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """Create a default mock configuration."""
        return {
            'paths': {
                'input': 'data/input',  # String format like actual config
                'output': {
                    'parsed': 'data/parsed',
                    'generated': 'data/generated', 
                    'curated': 'data/curated',
                    'final': 'data/final'
                }
            },
            'llm': {
                'provider': 'vllm'
            },
            'vllm': {
                'api_base': 'http://localhost:8000/v1',
                'model': 'test-model'
            }
        }
    
    @staticmethod
    def create_api_endpoint_config() -> Dict[str, Any]:
        """Create a mock configuration for API endpoint provider."""
        config = MockConfigHelper.create_default_config()
        config['llm']['provider'] = 'api-endpoint'
        config['api-endpoint'] = {
            'api_base': 'https://api.example.com/v1',
            'model': 'gpt-4',
            'api_key': 'test-key'
        }
        return config


class TempDirectoryManager:
    """Context manager for handling temporary directories with cleanup."""
    
    def __init__(self):
        self.temp_dir = None
        self.created_files = []
    
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            # Clean up all created files
            for file_path in self.created_files:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except OSError:
                    pass  # Ignore cleanup errors
            
            # Remove directory
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except OSError:
                pass  # Ignore cleanup errors
    
    def create_files(self, file_specs: Dict[str, str]) -> List[str]:
        """Create files in the temporary directory."""
        file_paths = TestFileFactory.create_test_files(self.temp_dir, file_specs)
        self.created_files.extend(file_paths)
        return file_paths
    
    def create_json_files(self, file_specs: Dict[str, Dict[str, Any]]) -> List[str]:
        """Create JSON files in the temporary directory."""
        file_paths = TestFileFactory.create_json_files(self.temp_dir, file_specs)
        self.created_files.extend(file_paths)
        return file_paths
    
    @property
    def path(self) -> str:
        """Get the temporary directory path."""
        return self.temp_dir


# Common test data constants
SAMPLE_QA_PAIRS = {
    "qa_pairs": [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"}
    ]
}

SAMPLE_TEXT_CONTENT = "This is sample text content for testing."

SAMPLE_FILE_SPECS = {
    "test1.txt": "First test file content",
    "test2.txt": "Second test file content",
    "test3.pdf": "PDF-like content",
    "unsupported.xyz": "Unsupported file content"
}

SAMPLE_JSON_SPECS = {
    "qa1.json": SAMPLE_QA_PAIRS,
    "qa2.json": {
        "qa_pairs": [
            {"question": "What is Python?", "answer": "A programming language"},
            {"question": "What is testing?", "answer": "Quality assurance process"}
        ]
    }
}
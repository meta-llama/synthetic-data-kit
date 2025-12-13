"""Integration tests for the create workflow."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.core import create


@pytest.mark.integration
def test_process_file(patch_config, test_env):
    """Test processing a file to generate QA pairs."""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("This is sample text content for testing QA pair generation.")
        input_path = f.name

    output_dir = tempfile.mkdtemp()
    output_path = None

    try:
        # Mock LLMClient
        with patch("synthetic_data_kit.core.create.LLMClient") as mock_llm_client_class:
            # Setup mock LLM client
            mock_llm_client = MagicMock()
            mock_llm_client_class.return_value = mock_llm_client

            # Mock QAGenerator with simplified behavior
            with patch("synthetic_data_kit.core.create.QAGenerator") as mock_qa_gen_class:
                # Create a mock generator that returns a predefined document
                mock_generator = MagicMock()
                mock_generator.process_documents.return_value = {
                    "qa_pairs": [
                        {"question": "What is this?", "answer": "This is sample text."},
                        {"question": "What is it for?", "answer": "For testing QA generation."},
                    ],
                }
                mock_qa_gen_class.return_value = mock_generator

                # Mock file operations
                with patch("builtins.open", create=True), patch("json.dump") as mock_json_dump:
                    # Mock os.path.exists to return True for our output file
                    with patch("os.path.exists", return_value=True), patch(
                        "os.path.join",
                        return_value=os.path.join(output_dir, "output_qa_pairs.json"),
                    ):
                        # Run the process_file function with minimal arguments
                        output_path = create.process_file(
                            file_path=input_path,
                            output_dir=output_dir,
                            config_path=None,
                            api_base=None,
                            model=None,
                            content_type="qa",
                            num_pairs=2,
                            verbose=False,
                            provider="api-endpoint",
                        )

                        # Verify function doesn't raise an exception
                        assert output_path is not None

                        # Verify the LLM client was created
                        mock_llm_client_class.assert_called_once()

                        # Verify QA generator was created and used
                        mock_qa_gen_class.assert_called_once()
                        mock_generator.process_documents.assert_called_once()

                        # Verify data was written to a file
                        mock_json_dump.assert_called()

    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.unlink(input_path)
        try:
            os.rmdir(output_dir)
        except:
            pass


@pytest.mark.integration
def test_process_directory(patch_config, test_env):
    """Test processing a directory to generate QA pairs."""
    # Create a temporary directory with test files
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    try:
        # Create a few test files
        file_paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".txt", dir=temp_dir, delete=False
            ) as f:
                f.write(f"This is sample text content {i} for testing QA pair generation.")
                file_paths.append(f.name)

        # Mock LLMClient
        with patch("synthetic_data_kit.core.create.LLMClient") as mock_llm_client_class:
            # Setup mock LLM client
            mock_llm_client = MagicMock()
            mock_llm_client_class.return_value = mock_llm_client

            # Mock QAGenerator
            with patch("synthetic_data_kit.core.create.QAGenerator") as mock_qa_gen_class:
                # Create a mock generator that returns a predefined document
                mock_generator = MagicMock()
                mock_generator.process_document.return_value = {
                    "qa_pairs": [
                        {"question": "What is this?", "answer": "This is sample text."},
                        {"question": "What is it for?", "answer": "For testing QA generation."},
                    ],
                }
                mock_qa_gen_class.return_value = mock_generator

                # Mock glob to return our test files
                with patch("glob.glob", return_value=file_paths):
                    # Mock file operations
                    with patch("builtins.open", create=True), patch("json.dump"):
                        # Mock the process_file function to return a predictable output path
                        with patch(
                            "synthetic_data_kit.core.create.process_file"
                        ) as mock_process_file:
                            # Have process_file return output paths for each input file
                            output_files = [
                                os.path.join(output_dir, f"output_{i}.json")
                                for i in range(len(file_paths))
                            ]
                            mock_process_file.side_effect = output_files

                            # Import and run the process_directory function from directory_processor
                            from synthetic_data_kit.utils.directory_processor import process_directory_create
                            results = process_directory_create(
                                directory=temp_dir,
                                output_dir=output_dir,
                                config_path=None,
                                api_base=None,
                                model=None,
                                content_type="qa",
                                num_pairs=2,
                                verbose=False,
                                provider="api-endpoint",
                            )

                            # Verify process_file was called the right number of times
                            assert mock_process_file.call_count == len(file_paths)

                            # Verify function returns expected results structure
                            assert isinstance(results, dict)
                            assert "total_files" in results
                            assert "successful" in results
                            assert results["total_files"] == len(file_paths)

    finally:
        # Clean up temporary files and directories
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass

        try:
            os.rmdir(temp_dir)
        except:
            pass

        try:
            os.rmdir(output_dir)
        except:
            pass

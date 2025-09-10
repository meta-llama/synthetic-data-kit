"""Functional tests for the CLI interface."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app


@pytest.mark.functional
def test_system_check_command_vllm(patch_config):
    """Test the system-check command with vLLM provider."""
    runner = CliRunner()

    # Mock the requests.get to simulate a vLLM server response
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["Llama-3-70B-Instruct"]
        mock_get.return_value = mock_response

        result = runner.invoke(app, ["system-check", "--provider", "vllm"])

        assert result.exit_code == 0
        # Check for general success rather than specific message
        assert "vLLM server is running" in result.stdout
        mock_get.assert_called_once()


@pytest.mark.functional
def test_system_check_command_api_endpoint(patch_config, test_env):
    """Test the system-check command with API endpoint provider."""
    runner = CliRunner()

    # Mock OpenAI API client
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_client.models.list.return_value = ["mock-model"]
        mock_openai.return_value = mock_client

        result = runner.invoke(app, ["system-check", "--provider", "api-endpoint"])

        # Just check exit code, not specific message since it varies
        assert result.exit_code == 0
        mock_openai.assert_called_once()


@pytest.mark.functional
def test_ingest_command(patch_config):
    """Test the ingest command with a text file."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
        f.write("Sample text content for testing.")
        input_path = f.name

    try:
        # Create a mock for process_file
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
            # Set up the mock to return a valid output path
            output_path = os.path.join(os.path.dirname(input_path), "output_test.txt")
            mock_process.return_value = output_path

            # Run the ingest command
            result = runner.invoke(app, ["ingest", input_path])

            # Verify the command executed successfully
            assert result.exit_code == 0
            assert "Text successfully extracted" in result.stdout

            # Verify the process_file function was called with correct arguments
            mock_process.assert_called_once()
            # Check that the first argument (file_path) matches
            assert mock_process.call_args[0][0] == input_path

    finally:
        # Clean up the temporary file
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_create_command(patch_config, test_env):
    """Test the create command with a text file."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
        f.write("Sample text content for testing.")
        input_path = f.name

    try:
        # Create a mock for process_file
        with patch("synthetic_data_kit.core.create.process_file") as mock_process:
            # Set up the mock to return a valid output path
            output_path = os.path.join(os.path.dirname(input_path), "output_qa_pairs.json")
            mock_process.return_value = output_path

            # Run the create command
            result = runner.invoke(app, ["create", input_path, "--type", "qa"])

            # Verify the command executed successfully
            assert result.exit_code == 0
            assert "Content saved to" in result.stdout

            # Verify the process_file function was called with correct arguments
            mock_process.assert_called_once()
            # Check that the first argument (file_path) matches
            assert mock_process.call_args[0][0] == input_path

    finally:
        # Clean up the temporary file
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_create_help_shows_difficulty(patch_config):
    """Ensure --difficulty option is visible in help for create."""
    runner = CliRunner()

    result = runner.invoke(app, ["create", "--help"])

    assert result.exit_code == 0
    assert "--difficulty" in result.stdout
    assert "easy|medium|advanced" in result.stdout


@pytest.mark.functional
def test_create_command_with_difficulty_passed(patch_config, test_env):
    """Test the create command passes --difficulty through to process_file."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
        f.write("Sample text content for testing.")
        input_path = f.name

    try:
        with patch("synthetic_data_kit.core.create.process_file") as mock_process:
            output_path = os.path.join(os.path.dirname(input_path), "output_qa_pairs.json")
            mock_process.return_value = output_path

            result = runner.invoke(
                app,
                [
                    "create",
                    input_path,
                    "--type",
                    "qa",
                    "--difficulty",
                    "advanced",
                    "--provider",
                    "api-endpoint",
                ],
            )

            assert result.exit_code == 0
            mock_process.assert_called_once()
            # verify difficulty kwarg is forwarded
            assert mock_process.call_args.kwargs.get("difficulty") == "advanced"
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_curate_command(patch_config, test_env):
    """Test the curate command with a JSON file."""
    runner = CliRunner()

    # Create a temporary QA pairs file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as f:
        json.dump(
            [
                {"question": "What is synthetic data?", "answer": "Sample answer."},
                {"question": "Why use synthetic data?", "answer": "Another sample answer."},
            ],
            f,
        )
        input_path = f.name

    try:
        # Create a mock for curate_qa_pairs
        with patch("synthetic_data_kit.core.curate.curate_qa_pairs") as mock_curate:
            # Set up the mock to return a valid output path
            output_path = os.path.join(os.path.dirname(input_path), "output_cleaned.json")
            mock_curate.return_value = output_path

            # Run the curate command
            result = runner.invoke(app, ["curate", input_path, "--threshold", "7.0"])

            # Verify the command executed successfully
            assert result.exit_code == 0
            assert "Cleaned content saved to" in result.stdout

            # Verify the curate_qa_pairs function was called with correct arguments
            mock_curate.assert_called_once()
            # Check that the first argument (file_path) matches
            assert mock_curate.call_args[0][0] == input_path

    finally:
        # Clean up the temporary file
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_save_as_command(patch_config):
    """Test the save-as command with a JSON file."""
    runner = CliRunner()

    # Create a temporary QA pairs file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as f:
        json.dump(
            [
                {"question": "What is synthetic data?", "answer": "Sample answer."},
                {"question": "Why use synthetic data?", "answer": "Another sample answer."},
            ],
            f,
        )
        input_path = f.name

    try:
        # Create a mock for convert_format
        with patch("synthetic_data_kit.core.save_as.convert_format") as mock_convert:
            # Set up the mock to return a valid output path
            output_path = os.path.join(os.path.dirname(input_path), "output.jsonl")
            mock_convert.return_value = output_path

            # Run the save-as command
            result = runner.invoke(app, ["save-as", input_path, "--format", "jsonl"])

            # Verify the command executed successfully
            assert result.exit_code == 0
            assert "Converted to jsonl format" in result.stdout

            # Verify the convert_format function was called with correct arguments
            mock_convert.assert_called_once()
            # Check that the first argument (file_path) matches
            assert mock_convert.call_args[0][0] == input_path

    finally:
        # Clean up the temporary file
        if os.path.exists(input_path):
            os.unlink(input_path)

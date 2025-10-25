"""Unit tests for error handling."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from synthetic_data_kit.core import create, curate, save_as
from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.llm_processing import parse_qa_pairs


@pytest.mark.unit
def test_parse_qa_pairs_invalid_json():
    """Test handling of invalid JSON in parse_qa_pairs."""
    # Invalid JSON that doesn't parse
    invalid_json = "This is not JSON at all"
    result = parse_qa_pairs(invalid_json)

    # Should return an empty list or a list with partial results rather than crashing
    assert isinstance(result, list)

    # Partial JSON that looks like JSON but is malformed
    partial_json = """
    Here are some results:
    [
        {"question": "What is synthetic data?", "answer": "It's artificial data."},
        {"question": "Why use synthetic data?",
    """
    result = parse_qa_pairs(partial_json)

    # Should return at least something rather than crashing
    assert isinstance(result, list)
    # It may use regex fallback to extract the one valid pair
    if result:
        assert "question" in result[0]


@pytest.mark.unit
def test_llm_client_error_handling(patch_config, test_env):
    """Test error handling in LLM client."""
    with patch("synthetic_data_kit.models.llm_client.OpenAI") as mock_openai:
        # Setup mock to raise an exception
        mock_openai.side_effect = Exception("API Error")

        # Should handle the exception gracefully
        with pytest.raises(Exception) as excinfo:
            LLMClient(provider="api-endpoint")

        # Check that the error message is helpful
        assert "API Error" in str(excinfo.value)


@pytest.mark.unit
def test_save_as_unknown_format():
    """Test error handling for unknown format in save_as."""
    # Create sample QA pairs
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
    ]

    # Create a temporary file with QA pairs
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump({"qa_pairs": qa_pairs}, f)
        input_path = f.name

    # Create temporary output path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        # Try to convert to an unknown format
        with pytest.raises(ValueError) as excinfo:
            save_as.convert_format(
                input_path=input_path, output_path=output_path, format_type="unknown-format"
            )

        # Check that the error message is helpful
        assert "Unknown format type" in str(excinfo.value)
    finally:
        # Clean up
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_save_as_unrecognized_data_format():
    """Test error handling for unrecognized data format in save_as."""
    # Create a file with unrecognized structure
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump({"something_unexpected": "data"}, f)
        input_path = f.name

    # Create temporary output path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        # Try to convert a file with unrecognized structure
        with pytest.raises(ValueError) as excinfo:
            save_as.convert_format(
                input_path=input_path, output_path=output_path, format_type="jsonl"
            )

        # Check that the error message is helpful
        assert "Unrecognized data format" in str(excinfo.value)
    finally:
        # Clean up
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_create_invalid_content_type(patch_config, test_env):
    """Test error handling for invalid content type in create."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
        f.write("Sample text content")
        file_path = f.name

    # Create temporary output directory
    output_dir = tempfile.mkdtemp()

    try:
        # Mock the LLM client
        with patch("synthetic_data_kit.core.create.LLMClient"):
            # Try to create with an invalid content type
            with pytest.raises(ValueError) as excinfo:
                create.process_file(
                    file_path=file_path, output_dir=output_dir, content_type="invalid-type"
                )

            # Check that the error message mentions the content type
            # The actual message is "Unknown content type: invalid-type"
            assert "content type" in str(excinfo.value).lower()
            assert "invalid-type" in str(excinfo.value)
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.unlink(file_path)
        os.rmdir(output_dir)


@pytest.mark.unit
def test_curate_input_validation(patch_config, test_env):
    """Test input validation for curate function."""
    # Create a temporary file with QA pairs
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        }
    ]

    cot_examples = [
        {
            "question": "What is synthetic data?",
            "reasoning": "Synthetic data is artificially generated data.",
            "answer": "Synthetic data is artificially generated data.",
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump({"qa_pairs": qa_pairs, "cot_examples": cot_examples}, f)
        file_path = f.name

    # Create temporary output directory
    output_dir = tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.json")

    try:
        # Create empty file to test error handling
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump({}, f)
            empty_file_path = f.name

        # Mock the LLM client
        with patch("synthetic_data_kit.core.curate.LLMClient"):
            # Try to curate an empty file
            with pytest.raises(ValueError) as excinfo:
                curate.curate_qa_pairs(input_path=empty_file_path, output_path=output_path)

            # Check that the error message is helpful
            assert "No QA pairs or CoT examples found" in str(excinfo.value)
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.unlink(file_path)
        if os.path.exists(empty_file_path):
            os.unlink(empty_file_path)
        if os.path.exists(output_dir):
            try:
                os.rmdir(output_dir)
            except OSError:
                pass

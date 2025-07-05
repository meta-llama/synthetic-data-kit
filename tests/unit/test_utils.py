"""Unit tests for utility functions."""

from pathlib import Path

import pytest

from synthetic_data_kit.utils import config, text


@pytest.mark.unit
def test_split_into_chunks():
    """Test splitting text into chunks."""
    # Create multi-paragraph text
    paragraphs = ["Paragraph one." * 5, "Paragraph two." * 5, "Paragraph three." * 5]
    text_content = "\n\n".join(paragraphs)

    # Using a small chunk size to ensure splitting
    chunks = text.split_into_chunks(text_content, chunk_size=50, overlap=10)

    # Check that chunks were created
    assert len(chunks) > 0

    # For this specific test case, we should have at least 2 chunks
    assert len(chunks) >= 2

    # Due to chunking logic, some content might be trimmed at chunk boundaries.
    # The real test is that we have multiple chunks and they aren't empty.
    assert all(len(chunk) > 0 for chunk in chunks)

    # Test edge case: empty text
    empty_chunks = text.split_into_chunks("", chunk_size=50, overlap=10)

    # Empty text should produce an empty list, not a list with an empty string
    assert empty_chunks == []


@pytest.mark.unit
def test_extract_json_from_text():
    """Test extracting JSON from text."""
    # Test valid JSON in code block
    json_text = """
    Some random text before the JSON
    ```json
    {
        "question": "What is synthetic data?",
        "answer": "Synthetic data is artificially generated data."
    }
    ```
    Some random text after the JSON
    """

    result = text.extract_json_from_text(json_text)

    assert isinstance(result, dict)
    assert "question" in result
    assert result["question"] == "What is synthetic data?"
    assert result["answer"] == "Synthetic data is artificially generated data."

    # Test invalid JSON - should raise ValueError
    invalid_json = """
    This is not JSON at all, but the function should try the various extraction methods
    and ultimately raise an error when no valid JSON is found.
    """

    with pytest.raises(ValueError):
        text.extract_json_from_text(invalid_json)


@pytest.mark.unit
def test_extract_json_list_from_text():
    """Test extracting JSON list from text."""
    json_text = """
    Some random text before the JSON
    ```json
    [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data."
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples."
        }
    ]
    ```
    Some random text after the JSON
    """

    result = text.extract_json_from_text(json_text)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["question"] == "What is synthetic data?"
    assert result[1]["question"] == "Why use synthetic data?"


@pytest.mark.unit
def test_load_config(tmpdir):
    """Test loading config from file."""
    # Create a temporary config file

    config_path = Path(tmpdir) / "test_config.yaml"
    with open(config_path, "w") as f:
        f.write(
            "llm:\n  provider: test-provider\ntest-provider:\n  api_base: http://test-api.com\n  model: test-model"
        )

    # Load the config
    loaded_config = config.load_config(config_path)

    # Check that the config was loaded correctly
    assert loaded_config["llm"]["provider"] == "test-provider"
    assert loaded_config["test-provider"]["api_base"] == "http://test-api.com"
    assert loaded_config["test-provider"]["model"] == "test-model"


@pytest.mark.unit
def test_get_llm_provider(mock_config):
    """Test getting the LLM provider from config."""
    provider = config.get_llm_provider(mock_config)
    assert provider == "api-endpoint"

    # Test with empty config
    empty_config = {}
    default_provider = config.get_llm_provider(empty_config)
    assert default_provider == "vllm"  # Should return the default provider


@pytest.mark.unit
def test_get_path_config():
    """Test getting path configuration."""
    # Create a test config with proper structure
    test_config = {
        "paths": {
            "output": {"default": "data/output", "generated": "data/generated"},
            "input": {"default": "data/input", "pdf": "data/pdf"},
        }
    }

    # Test with valid path type and file type
    output_path = config.get_path_config(test_config, "output", "default")
    assert output_path == "data/output"

    output_path_specific = config.get_path_config(test_config, "output", "generated")
    assert output_path_specific == "data/generated"

    # Test input path type
    input_path = config.get_path_config(test_config, "input", "default")
    assert input_path == "data/input"

    input_path_specific = config.get_path_config(test_config, "input", "pdf")
    assert input_path_specific == "data/pdf"

    # Test with unknown path type - should raise ValueError
    with pytest.raises(ValueError):
        config.get_path_config(test_config, "nonexistent", "default")

    # Test with empty config
    empty_config = {}
    default_path = config.get_path_config(empty_config, "output", "default")
    assert default_path == "data/output"

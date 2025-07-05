"""Unit tests for format converters."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.utils.format_converter import (
    to_alpaca,
    to_chatml,
    to_fine_tuning,
    to_hf_dataset,
    to_jsonl,
)


@pytest.mark.unit
def test_to_jsonl():
    """Test conversion to JSONL format."""
    # Create sample QA pairs
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
        },
    ]

    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_file:
        output_path = temp_file.name

    try:
        # Convert to JSONL
        result_path = to_jsonl(qa_pairs, output_path)

        # Check that the result path is correct
        assert result_path == output_path

        # Check that the file was created
        assert os.path.exists(output_path)

        # Read the file and check content
        with open(output_path) as f:
            lines = f.readlines()

        # Should have two lines (one for each QA pair)
        assert len(lines) == 2

        # Each line should be valid JSON
        line1_data = json.loads(lines[0])
        line2_data = json.loads(lines[1])

        # Check content
        assert line1_data["question"] == "What is synthetic data?"
        assert line2_data["question"] == "Why use synthetic data?"
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_to_alpaca():
    """Test conversion to Alpaca format."""
    # Create sample QA pairs
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
        },
    ]

    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        output_path = temp_file.name

    try:
        # Convert to Alpaca format
        result_path = to_alpaca(qa_pairs, output_path)

        # Check that the result path is correct
        assert result_path == output_path

        # Check that the file was created
        assert os.path.exists(output_path)

        # Read the file and check content
        with open(output_path) as f:
            data = json.load(f)

        # Should have two items in the list
        assert len(data) == 2

        # Check format structure
        assert "instruction" in data[0]
        assert "input" in data[0]
        assert "output" in data[0]

        # Check content
        assert data[0]["instruction"] == "What is synthetic data?"
        assert data[0]["input"] == ""
        assert data[0]["output"] == "Synthetic data is artificially generated data."
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_to_fine_tuning():
    """Test conversion to fine-tuning format."""
    # Create sample QA pairs
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
        },
    ]

    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        output_path = temp_file.name

    try:
        # Convert to fine-tuning format
        result_path = to_fine_tuning(qa_pairs, output_path)

        # Check that the result path is correct
        assert result_path == output_path

        # Check that the file was created
        assert os.path.exists(output_path)

        # Read the file and check content
        with open(output_path) as f:
            data = json.load(f)

        # Should have two items in the list
        assert len(data) == 2

        # Check format structure
        assert "messages" in data[0]
        assert len(data[0]["messages"]) == 3

        # Check message roles and content
        assert data[0]["messages"][0]["role"] == "system"
        assert data[0]["messages"][1]["role"] == "user"
        assert data[0]["messages"][1]["content"] == "What is synthetic data?"
        assert data[0]["messages"][2]["role"] == "assistant"
        assert data[0]["messages"][2]["content"] == "Synthetic data is artificially generated data."
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_to_chatml():
    """Test conversion to ChatML format."""
    # Create sample QA pairs
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
        },
    ]

    # Create a temporary output path
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_file:
        output_path = temp_file.name

    try:
        # Convert to ChatML format
        result_path = to_chatml(qa_pairs, output_path)

        # Check that the result path is correct
        assert result_path == output_path

        # Check that the file was created
        assert os.path.exists(output_path)

        # Read the file and check content
        with open(output_path) as f:
            lines = f.readlines()

        # Should have two lines (one for each QA pair)
        assert len(lines) == 2

        # Each line should be valid JSON
        line1_data = json.loads(lines[0])

        # Check format structure
        assert "messages" in line1_data
        assert len(line1_data["messages"]) == 3

        # Check message roles and content
        assert line1_data["messages"][0]["role"] == "system"
        assert line1_data["messages"][1]["role"] == "user"
        assert line1_data["messages"][1]["content"] == "What is synthetic data?"
        assert line1_data["messages"][2]["role"] == "assistant"
        assert (
            line1_data["messages"][2]["content"] == "Synthetic data is artificially generated data."
        )
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_to_hf_dataset():
    """Test conversion to Hugging Face dataset."""
    # Create sample QA pairs
    qa_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
        },
    ]

    # Create a temporary directory for output
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "test_dataset")

    try:
        # Skip this test if datasets module is not available
        pytest.importorskip("datasets")
        # Mock the datasets module
        with patch("datasets.Dataset") as mock_dataset:
            # Setup mock dataset
            mock_dataset_instance = MagicMock()
            mock_dataset.from_dict.return_value = mock_dataset_instance

            # Convert to HF dataset
            to_hf_dataset(qa_pairs, output_path)

            # Check that Dataset.from_dict was called with the right structure
            mock_dataset.from_dict.assert_called_once()
            call_args = mock_dataset.from_dict.call_args[0][0]
            assert "question" in call_args
            assert "answer" in call_args
            assert call_args["question"] == ["What is synthetic data?", "Why use synthetic data?"]

            # Check that save_to_disk was called
            mock_dataset_instance.save_to_disk.assert_called_once_with(output_path)
    finally:
        # No need to clean up files as we mocked the saving
        pass

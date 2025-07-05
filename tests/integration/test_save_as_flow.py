"""Integration tests for the save-as workflow."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from synthetic_data_kit.core import save_as


@pytest.mark.integration
def test_convert_format():
    """Test converting QA pairs to different formats."""
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

    # Create a temporary file with QA pairs
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump({"qa_pairs": qa_pairs}, f)
        input_path = f.name

    # Create temporary output directory
    output_dir = tempfile.mkdtemp()

    try:
        # Test converting to JSONL format
        jsonl_output = os.path.join(output_dir, "output.jsonl")
        result_path = save_as.convert_format(
            input_path=input_path, output_path=jsonl_output, format_type="jsonl"
        )

        # Check that the file was created
        assert os.path.exists(result_path)

        # Read the file and check content
        with open(result_path) as f:
            lines = f.readlines()

        # Should have two lines (one for each QA pair)
        assert len(lines) == 2

        # Each line should be valid JSON
        line1_data = json.loads(lines[0])
        line2_data = json.loads(lines[1])

        # Check content
        assert line1_data["question"] == "What is synthetic data?"
        assert line2_data["question"] == "Why use synthetic data?"

        # Test converting to Alpaca format
        alpaca_output = os.path.join(output_dir, "output_alpaca.json")
        result_path = save_as.convert_format(
            input_path=input_path, output_path=alpaca_output, format_type="alpaca"
        )

        # Check that the file was created
        assert os.path.exists(result_path)

        # Read the file and check content
        with open(result_path) as f:
            data = json.load(f)

        # Should have two items in the list
        assert len(data) == 2

        # Check format structure
        assert "instruction" in data[0]
        assert "input" in data[0]
        assert "output" in data[0]

        # Check content
        assert data[0]["instruction"] == "What is synthetic data?"
        assert data[0]["output"] == "Synthetic data is artificially generated data."

    finally:
        # Clean up
        if os.path.exists(input_path):
            os.unlink(input_path)

        # Clean up output files
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_convert_format_with_filtered_pairs():
    """Test converting filtered_pairs to different formats."""
    # Create sample QA pairs with ratings
    filtered_pairs = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
            "rating": 8.5,
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
            "rating": 9.0,
        },
    ]

    # Create a temporary file with filtered pairs
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump({"filtered_pairs": filtered_pairs}, f)
        input_path = f.name

    # Create temporary output directory
    output_dir = tempfile.mkdtemp()

    try:
        # Test converting to fine-tuning format
        ft_output = os.path.join(output_dir, "output_ft.json")
        result_path = save_as.convert_format(
            input_path=input_path, output_path=ft_output, format_type="ft"
        )

        # Check that the file was created
        assert os.path.exists(result_path)

        # Read the file and check content
        with open(result_path) as f:
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
        if os.path.exists(input_path):
            os.unlink(input_path)

        # Clean up output files
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_convert_format_with_conversations():
    """Test converting conversations to different formats."""
    # Create sample conversations
    conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"},
            {"role": "assistant", "content": "Synthetic data is artificially generated data."},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Why use synthetic data?"},
            {
                "role": "assistant",
                "content": "To protect privacy and create diverse training examples.",
            },
        ],
    ]

    # Create a temporary file with conversations
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump({"conversations": conversations}, f)
        input_path = f.name

    # Create temporary output directory
    output_dir = tempfile.mkdtemp()

    try:
        # Test converting to ChatML format
        chatml_output = os.path.join(output_dir, "output_chatml.jsonl")
        result_path = save_as.convert_format(
            input_path=input_path, output_path=chatml_output, format_type="chatml"
        )

        # Check that the file was created
        assert os.path.exists(result_path)

        # Read the file and check content
        with open(result_path) as f:
            lines = f.readlines()

        # Should have two lines (one for each conversation)
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
        if os.path.exists(input_path):
            os.unlink(input_path)

        # Clean up output files
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_process_multiple_files():
    """Test processing multiple files."""
    # Create sample QA pairs
    qa_pairs1 = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
    ]

    qa_pairs2 = [
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
        },
    ]

    # Create temporary input files
    input_dir = tempfile.mkdtemp()
    input_files = []

    # Create first input file
    file1_path = os.path.join(input_dir, "file1_cleaned.json")
    with open(file1_path, "w") as f:
        json.dump({"qa_pairs": qa_pairs1}, f)
    input_files.append(file1_path)

    # Create second input file
    file2_path = os.path.join(input_dir, "file2_cleaned.json")
    with open(file2_path, "w") as f:
        json.dump({"qa_pairs": qa_pairs2}, f)
    input_files.append(file2_path)

    # Create temporary output directory
    output_dir = tempfile.mkdtemp()

    try:
        # Process multiple files using directory processor
        from synthetic_data_kit.utils.directory_processor import process_directory_save_as
        results = process_directory_save_as(
            directory=input_dir,
            output_dir=output_dir,
            format="jsonl",
            storage_format="json",
            config=None,
            verbose=False,
        )

        # Check that files were processed successfully
        assert isinstance(results, dict)
        assert results["successful"] == 2
        assert results["failed"] == 0
        
        # Check that output files were created
        output_files = [result["output_file"] for result in results["results"]]
        assert len(output_files) == 2
        for output_file in output_files:
            assert os.path.exists(output_file)

        # Check the content of the first output file
        with open(output_files[0]) as f:
            lines = f.readlines()

        # Should have one line for the first file
        assert len(lines) == 1

        # Check content
        line_data = json.loads(lines[0])
        assert line_data["question"] == "What is synthetic data?"

        # Check the content of the second output file
        with open(output_files[1]) as f:
            lines = f.readlines()

        # Should have one line for the second file
        assert len(lines) == 1

        # Check content
        line_data = json.loads(lines[0])
        assert line_data["question"] == "Why use synthetic data?"

    finally:
        # Clean up input files
        for file_path in input_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        os.rmdir(input_dir)

        # Clean up output files
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_process_directory():
    """Test processing a directory."""
    # Create sample QA pairs
    qa_pairs1 = [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data.",
        },
    ]

    qa_pairs2 = [
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples.",
        },
    ]

    # Create temporary input directory
    input_dir = tempfile.mkdtemp()

    # Create input files
    file1_path = os.path.join(input_dir, "file1_cleaned.json")
    with open(file1_path, "w") as f:
        json.dump({"qa_pairs": qa_pairs1}, f)

    file2_path = os.path.join(input_dir, "file2_cleaned.json")
    with open(file2_path, "w") as f:
        json.dump({"qa_pairs": qa_pairs2}, f)

    # Create temporary output directory
    output_dir = tempfile.mkdtemp()

    try:
        # Process directory using directory processor
        from synthetic_data_kit.utils.directory_processor import process_directory_save_as
        results = process_directory_save_as(
            directory=input_dir,
            output_dir=output_dir,
            format="jsonl",
            storage_format="json",
            config=None,
            verbose=False,
        )

        # Check that files were processed successfully
        assert isinstance(results, dict)
        assert results["successful"] == 2
        assert results["failed"] == 0

        # Check the output files were created
        output_files = [result["output_file"] for result in results["results"]]
        assert len(output_files) == 2
        for output_file in output_files:
            assert os.path.exists(output_file)

    finally:
        # Clean up input files
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(input_dir)

        # Clean up output files
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(output_dir)

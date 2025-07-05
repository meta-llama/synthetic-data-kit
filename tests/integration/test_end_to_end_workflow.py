"""End-to-end integration tests for the complete workflow."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.core import create, ingest, save_as
from synthetic_data_kit.parsers.txt_parser import TXTParser


@pytest.mark.integration
def test_complete_workflow(patch_config, test_env):
    """Test the complete workflow from ingest to save-as."""
    # Create a temporary source document
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
        f.write(
            "This is a sample document about synthetic data. It contains information about generation techniques."
        )
        source_path = f.name

    # Create temporary directories for intermediate outputs
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "output")
    generated_dir = os.path.join(temp_dir, "generated")
    cleaned_dir = os.path.join(temp_dir, "cleaned")
    final_dir = os.path.join(temp_dir, "final")

    # Create the directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Define paths for intermediate files
    parsed_path = os.path.join(output_dir, "sample.txt")
    qa_pairs_path = os.path.join(generated_dir, "sample_qa_pairs.json")
    curated_path = os.path.join(cleaned_dir, "sample_cleaned.json")
    final_path = os.path.join(final_dir, "sample.jsonl")

    try:
        # 1. Ingest step - mock the determine_parser function
        with patch("synthetic_data_kit.core.ingest.determine_parser") as mock_determine_parser:
            parser = TXTParser()
            mock_determine_parser.return_value = parser

            # Parse the document
            output_path = ingest.process_file(
                file_path=source_path,
                output_dir=output_dir,
                output_name="sample.txt",  # Specify output name to match expected path
            )

            # Check that parsing was successful
            assert output_path == parsed_path

            # Copy the source content to the parsed path for the next step
            with open(source_path) as src, open(parsed_path, "w") as dst:
                dst.write(src.read())

        # 2. Create step - mock the LLM client
        with patch("synthetic_data_kit.core.create.LLMClient") as mock_llm_client_class:
            # Setup mock LLM client with config
            mock_llm_client = MagicMock()
            mock_llm_client.config = {
                "prompts": {
                    "qa_generation": "Generate question-answer pairs based on this text: {text}",
                }
            }
            mock_llm_client_class.return_value = mock_llm_client

            # Mock QA Generator
            with patch("synthetic_data_kit.core.create.QAGenerator") as mock_qa_gen_class:
                # Create a mock generator that returns predefined QA pairs
                mock_generator = MagicMock()
                mock_generator.process_document.return_value = {
                    "summary": "A sample document about synthetic data generation.",
                    "qa_pairs": [
                        {
                            "question": "What is the document about?",
                            "answer": "Synthetic data generation techniques.",
                        },
                        {
                            "question": "Why is synthetic data useful?",
                            "answer": "It helps in training machine learning models without real data.",
                        },
                    ],
                }
                mock_qa_gen_class.return_value = mock_generator

                # Generate QA pairs
                output_path = create.process_file(
                    file_path=parsed_path, output_dir=generated_dir, content_type="qa", num_pairs=2
                )

                # Check that QA pairs were generated
                assert output_path == qa_pairs_path

                # Write mock QA pairs to the output path
                with open(qa_pairs_path, "w") as f:
                    json.dump(
                        {
                            "summary": "A sample document about synthetic data generation.",
                            "qa_pairs": [
                                {
                                    "question": "What is the document about?",
                                    "answer": "Synthetic data generation techniques.",
                                },
                                {
                                    "question": "Why is synthetic data useful?",
                                    "answer": "It helps in training machine learning models without real data.",
                                },
                            ],
                        },
                        f,
                    )

        # 3. Curate step - skip actual curation for simplicity
        # Just create the expected output file manually
        with open(curated_path, "w") as f:
            json.dump(
                {
                    "summary": "A sample document about synthetic data generation.",
                    "filtered_pairs": [
                        {
                            "question": "What is the document about?",
                            "answer": "Synthetic data generation techniques.",
                            "rating": 9.0,
                        }
                    ],
                    "conversations": [
                        [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What is the document about?"},
                            {
                                "role": "assistant",
                                "content": "Synthetic data generation techniques.",
                            },
                        ]
                    ],
                    "metrics": {"total": 2, "filtered": 1, "retention_rate": 0.5},
                },
                f,
            )

        # 4. Save-as step
        output_path = save_as.convert_format(
            input_path=curated_path, output_path=final_path, format_type="jsonl"
        )

        # Check that the final file was created
        assert output_path == final_path
        assert os.path.exists(final_path)

        # Read the file and check content
        with open(final_path) as f:
            lines = f.readlines()

        # Should have one line (one QA pair passed the curation)
        assert len(lines) == 1

        # Check content
        line_data = json.loads(lines[0])
        assert line_data["question"] == "What is the document about?"
        assert line_data["answer"] == "Synthetic data generation techniques."

    finally:
        # Clean up
        if os.path.exists(source_path):
            os.unlink(source_path)

        # Clean up temporary directories
        for path in [parsed_path, qa_pairs_path, curated_path, final_path]:
            if os.path.exists(path):
                os.unlink(path)

        # Remove directories
        for dir_path in [output_dir, generated_dir, cleaned_dir, final_dir, temp_dir]:
            try:
                os.rmdir(dir_path)
            except:
                pass

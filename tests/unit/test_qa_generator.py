"""Unit tests for QA generator."""

import json
from unittest.mock import MagicMock

import pytest

from synthetic_data_kit.generators.qa_generator import QAGenerator


@pytest.mark.unit
def test_qa_generator_initialization(patch_config):
    """Test QA generator initialization."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Check that the generator was initialized correctly
    assert generator.client == mock_client
    assert generator.config is not None
    assert generator.generation_config is not None
    assert generator.curate_config is not None


@pytest.mark.unit
def test_generate_summary(patch_config):
    """Test generating summary."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = "This is a summary of the document."

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Generate summary
    summary = generator.generate_summary("This is a document to summarize.")

    # Check that the summary was generated correctly
    assert summary == "This is a summary of the document."
    # Check that client was called
    assert mock_client.chat_completion.called


@pytest.mark.unit
def test_generate_qa_pairs(patch_config):
    """Test generating QA pairs."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.batch_completion.return_value = [
        json.dumps(
            [
                {
                    "question": "What is synthetic data?",
                    "answer": "Synthetic data is artificially generated data.",
                }
            ]
        ),
        json.dumps(
            [
                {
                    "question": "Why use synthetic data?",
                    "answer": "To protect privacy and create diverse training examples.",
                }
            ]
        ),
    ]

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Generate QA pairs
    qa_pairs = generator.generate_qa_pairs(
        document_text="This is a document to generate QA pairs from.",
        summary="This is a summary of the document.",
        num_pairs=2,
    )

    # Check that the QA pairs were generated correctly
    assert len(qa_pairs) == 2
    assert qa_pairs[0]["question"] == "What is synthetic data?"
    assert qa_pairs[1]["question"] == "Why use synthetic data?"
    # Check that client was called
    assert mock_client.batch_completion.called


@pytest.mark.unit
def test_generate_qa_pairs_exact_count_and_dedup(patch_config):
    """Test that generator enforces exact count and deduplicates questions."""
    mock_client = MagicMock()
    # First batch returns duplicates and more than needed
    first_resp = json.dumps([
        {"question": "What is synthetic data?", "answer": "A."},
        {"question": "What is synthetic data?", "answer": "A."},
        {"question": "Why use synthetic data?", "answer": "B."},
        {"question": "How is synthetic data generated?", "answer": "C."},
    ])
    # Second batch provides an extra unique to fill if needed
    second_resp = json.dumps([
        {"question": "Where is synthetic data used?", "answer": "D."}
    ])
    mock_client.batch_completion.side_effect = [[first_resp], [second_resp]]

    generator = QAGenerator(client=mock_client)
    qa_pairs = generator.generate_qa_pairs(
        document_text="Doc text.", summary=None, num_pairs=3
    )
    # Should return exactly 3 unique questions
    assert len(qa_pairs) == 3
    qs = {p["question"] for p in qa_pairs}
    assert len(qs) == 3


@pytest.mark.unit
def test_rate_qa_pairs(patch_config):
    """Test rating QA pairs."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = json.dumps(
        [
            {
                "question": "What is synthetic data?",
                "answer": "Synthetic data is artificially generated data.",
                "rating": 8,
            },
            {
                "question": "Why use synthetic data?",
                "answer": "To protect privacy and create diverse training examples.",
                "rating": 6,
            },
        ]
    )

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Sample QA pairs to rate
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

    # Rate QA pairs with threshold 7
    rated_pairs, metrics = generator.rate_qa_pairs(
        qa_pairs=qa_pairs, summary="This is a summary of the document.", threshold=7.0
    )

    # Check that only pairs with rating >= threshold were kept
    assert len(rated_pairs) == 1
    assert rated_pairs[0]["question"] == "What is synthetic data?"
    assert rated_pairs[0]["rating"] == 8

    # Check metrics
    assert metrics["total"] == 2
    assert metrics["filtered"] == 1
    assert metrics["retention_rate"] == 0.5

    # Check that client was called
    assert mock_client.chat_completion.called


@pytest.mark.unit
def test_process_document(patch_config):
    """Test processing a document end-to-end."""
    # Create mock LLM client
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = "This is a summary of the document."
    mock_client.batch_completion.return_value = [
        json.dumps(
            [
                {
                    "question": "What is synthetic data?",
                    "answer": "Synthetic data is artificially generated data.",
                }
            ]
        ),
        json.dumps(
            [
                {
                    "question": "Why use synthetic data?",
                    "answer": "To protect privacy and create diverse training examples.",
                }
            ]
        ),
    ]

    # Initialize generator
    generator = QAGenerator(client=mock_client)

    # Process document
    result = generator.process_documents(
        documents=[{"text": "This is a document to process."}], num_pairs=2, verbose=False
    )

    # Check that the result contains QA pairs only (no summary)
    assert "qa_pairs" in result
    assert len(result["qa_pairs"]) == 2

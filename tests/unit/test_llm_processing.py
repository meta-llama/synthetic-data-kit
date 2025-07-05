"""Unit tests for LLM processing utilities."""

import pytest

from synthetic_data_kit.utils import llm_processing


@pytest.mark.unit
def test_parse_qa_pairs():
    """Test parsing QA pairs from LLM output."""
    # Test with a JSON array containing multiple QA pairs
    json_text = """
    Here is the result:
    [
        {
            "question": "What is synthetic data?",
            "answer": "Synthetic data is artificially generated data."
        },
        {
            "question": "Why use synthetic data?",
            "answer": "To protect privacy and create diverse training examples."
        },
        {
            "question": "How can synthetic data help with ML?",
            "answer": "It can provide more training examples, especially for rare cases."
        }
    ]
    """

    result = llm_processing.parse_qa_pairs(json_text)

    # Check that all 3 QA pairs were extracted
    assert len(result) == 3
    assert result[0]["question"] == "What is synthetic data?"
    assert result[1]["question"] == "Why use synthetic data?"
    assert result[2]["question"] == "How can synthetic data help with ML?"


@pytest.mark.unit
def test_parse_qa_pairs_with_regex():
    """Test parsing QA pairs using regex fallback."""
    # Test with a non-JSON format that requires regex pattern matching
    # Format the text exactly as the regex pattern expects
    text = """
    Here are the QA pairs:
    {"question": "What is synthetic data?", "answer": "Synthetic data is artificially generated data."}
    {"question": "Why use synthetic data?", "answer": "To protect privacy and create diverse training examples."}
    """

    result = llm_processing.parse_qa_pairs(text)

    # Check that both QA pairs were extracted using regex
    assert len(result) == 2
    assert result[0]["question"] == "What is synthetic data?"
    assert result[1]["question"] == "Why use synthetic data?"


@pytest.mark.unit
def test_convert_to_conversation_format():
    """Test converting QA pairs to conversation format."""
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

    conversations = llm_processing.convert_to_conversation_format(qa_pairs)

    # Check that conversations were created correctly
    assert len(conversations) == 2

    # Check first conversation
    assert len(conversations[0]) == 3  # system, user, assistant
    assert conversations[0][0]["role"] == "system"
    assert conversations[0][1]["role"] == "user"
    assert conversations[0][1]["content"] == "What is synthetic data?"
    assert conversations[0][2]["role"] == "assistant"
    assert conversations[0][2]["content"] == "Synthetic data is artificially generated data."

    # Check second conversation
    assert conversations[1][1]["content"] == "Why use synthetic data?"

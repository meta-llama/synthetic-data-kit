"""Unit tests for COT Generator."""

import json
from unittest.mock import MagicMock

import pytest

from synthetic_data_kit.generators.cot_generator import COTGenerator


@pytest.mark.unit
def test_cot_generator_initialization(patch_config):
    """Test COT generator initialization."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = COTGenerator(client=mock_client)

    # Check that the generator was initialized correctly
    assert generator.client == mock_client
    assert generator.config is not None
    assert generator.generation_config is not None


@pytest.mark.unit
def test_parse_json_output():
    """Test parsing JSON output from LLM."""
    # Create mock LLM client
    mock_client = MagicMock()

    # Initialize generator
    generator = COTGenerator(client=mock_client)

    # Test with valid JSON array
    valid_json_text = """
    Here is the result:
    [
        {
            "question": "What is synthetic data?",
            "reasoning": "Synthetic data is data that is artificially created rather than collected from real-world events. It's generated using algorithms and often mirrors statistical properties of real data.",
            "answer": "Synthetic data is artificially generated data."
        },
        {
            "question": "Why use synthetic data?",
            "reasoning": "There are several reasons to use synthetic data. First, it helps protect privacy by not using real personal data. Second, it can be used to balance datasets for machine learning. Third, it can be generated in large quantities without collecting real data.",
            "answer": "To protect privacy and create diverse training examples."
        }
    ]
    """

    result = generator.parse_json_output(valid_json_text)

    # Check that parsing was successful
    assert result is not None
    assert len(result) == 2
    assert result[0]["question"] == "What is synthetic data?"
    assert "reasoning" in result[0]
    assert result[0]["answer"] == "Synthetic data is artificially generated data."

    # Test with invalid JSON
    invalid_json_text = """
    Here is the result:
    This is not JSON at all.
    """

    result = generator.parse_json_output(invalid_json_text)

    # Check that parsing failed gracefully
    assert result is None


@pytest.mark.unit
def test_generate_cot_examples(patch_config):
    """Test generating chain-of-thought examples."""
    # Create mock LLM client with config
    mock_client = MagicMock()
    mock_client.config = {
        "prompts": {
            "cot_generation": "Generate {num_examples} Chain of Thought reasoning examples from the following text:\n\nText:\n{text}",
            "cot_enhancement": "Enhance the following conversations with Chain of Thought reasoning. Include_simple_steps: {include_simple_steps}\n\nConversations:\n{conversations}",
        }
    }
    mock_client.chat_completion.return_value = json.dumps(
        [
            {
                "question": "What is synthetic data?",
                "reasoning": "Synthetic data is data that is artificially created rather than collected from real-world events. It's generated using algorithms and often mirrors statistical properties of real data.",
                "answer": "Synthetic data is artificially generated data.",
            },
            {
                "question": "Why use synthetic data?",
                "reasoning": "There are several reasons to use synthetic data. First, it helps protect privacy by not using real personal data. Second, it can be used to balance datasets for machine learning. Third, it can be generated in large quantities without collecting real data.",
                "answer": "To protect privacy and create diverse training examples.",
            },
            {
                "question": "How is synthetic data generated?",
                "reasoning": "Synthetic data can be generated using various techniques. These include statistical models, machine learning algorithms, and rule-based systems. The choice of method depends on the type of data and the intended use case.",
                "answer": "Using statistical models, ML algorithms, and rule-based systems.",
            },
        ]
    )

    # Initialize generator
    generator = COTGenerator(client=mock_client)

    # Generate examples - explicitly request 3 examples
    examples = generator.generate_cot_examples(
        document_text="This is a document about synthetic data.", num_examples=3
    )

    # Check that we get 3 examples as requested
    assert len(examples) == 3, f"Expected 3 examples, got {len(examples)}"

    # Check that examples were generated correctly
    assert examples[0]["question"] == "What is synthetic data?"
    assert "reasoning" in examples[0]
    assert examples[0]["answer"] == "Synthetic data is artificially generated data."

    # Check if the document text was used properly in the prompt
    # We may call the API multiple times if needed to get enough examples
    assert mock_client.chat_completion.call_count > 0, "Chat completion was never called"

    # Get the first call's arguments
    call_args = mock_client.chat_completion.call_args_list[0][0][0]
    prompt_content = call_args[0]["content"]

    # Print for debugging
    print(f"DEBUG - Prompt content: {prompt_content}")

    # Bug #2 check: Was the actual document text included?
    assert "This is a document about synthetic data" in prompt_content, (
        f"Document text not included in prompt. Actual prompt: {prompt_content}"
    )


@pytest.mark.unit
def test_enhance_with_cot(patch_config):
    """Test enhancing existing conversations with COT reasoning."""
    # Create mock LLM client with config
    mock_client = MagicMock()
    mock_client.config = {
        "prompts": {
            "cot_generation": "Generate {num_examples} Chain of Thought reasoning examples from the following text:\n\nText:\n{text}",
            "cot_enhancement": "Enhance the following conversations with Chain of Thought reasoning. Include_simple_steps: {include_simple_steps}\n\nConversations:\n{conversations}",
        },
        "generation": {"batch_size": 2},  # Set batch size to match our test case
    }
    # Mock the response to include two enhanced conversations
    mock_client.chat_completion.return_value = json.dumps(
        [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides detailed explanations.",
                },
                {"role": "user", "content": "What is synthetic data?"},
                {
                    "role": "assistant",
                    "content": "Let me think through this step by step:\n\nSynthetic data is data that is artificially created rather than collected from real-world events. It's generated using algorithms and often mirrors statistical properties of real data.\n\nSo the answer is: Synthetic data is artificially generated data.",
                },
            ],
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides detailed explanations.",
                },
                {"role": "user", "content": "Why use synthetic data?"},
                {
                    "role": "assistant",
                    "content": "Let me think through this step by step:\n\nThere are several reasons to use synthetic data. First, it helps protect privacy by not using real personal data. Second, it can be used to balance datasets for machine learning. Third, it can be generated in large quantities without collecting real data.\n\nSo the answer is: To protect privacy and create diverse training examples.",
                },
            ],
        ]
    )

    # Initialize generator
    generator = COTGenerator(client=mock_client)

    # Sample conversations to enhance - create multiple
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

    # Enhance conversations
    enhanced = generator.enhance_with_cot(conversations)

    # Check that ALL conversations were enhanced (not just the first one)
    assert len(enhanced) == 2, f"Expected 2 enhanced conversations, got {len(enhanced)}"

    # Check that enhancement was successful
    assert enhanced[0][0]["role"] == "system"
    assert enhanced[0][1]["role"] == "user"
    assert enhanced[0][2]["role"] == "assistant"

    # Check that reasoning was added
    assert "Let me think through this step by step" in enhanced[0][2]["content"]

    # Check if include_simple_steps parameter was respected and matches what was requested
    assert mock_client.chat_completion.call_count > 0, "Chat completion was never called"
    call_args = mock_client.chat_completion.call_args_list[0][0][0]
    call_args[0]["content"]

    # Reset mock to check second call with include_simple_steps=True
    mock_client.reset_mock()
    mock_client.chat_completion.return_value = json.dumps(
        [
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides detailed explanations.",
                },
                {"role": "user", "content": "What is synthetic data?"},
                {
                    "role": "assistant",
                    "content": "Let me think through this step by step:\n\nSynthetic data is data that is artificially created rather than collected from real-world events.\n\nSo the answer is: Synthetic data is artificially generated data.",
                },
            ],
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides detailed explanations.",
                },
                {"role": "user", "content": "Why use synthetic data?"},
                {
                    "role": "assistant",
                    "content": "Let me think through this step by step:\n\nThere are several reasons to use synthetic data.\n\nSo the answer is: To protect privacy and create diverse training examples.",
                },
            ],
        ]
    )

    # Try with include_simple_steps=True
    generator.enhance_with_cot(conversations, include_simple_steps=True)

    # Should have been called at least once
    assert mock_client.chat_completion.call_count > 0, (
        "Chat completion was never called with include_simple_steps=True"
    )

    call_args_true = mock_client.chat_completion.call_args_list[0][0][0]
    prompt_content_true = call_args_true[0]["content"]

    # The parameter value should be respected, not hardcoded
    assert "include_simple_steps: true" in prompt_content_true.lower(), (
        f"include_simple_steps=True not respected. Actual prompt: {prompt_content_true}"
    )


@pytest.mark.unit
def test_process_document(patch_config):
    """Test processing a document to generate COT examples."""
    # Create mock LLM client with config
    mock_client = MagicMock()
    mock_client.config = {
        "prompts": {
            "cot_generation": "Generate {num_examples} Chain of Thought reasoning examples from the following text:\n\nText:\n{text}",
            "cot_enhancement": "Enhance the following conversations with Chain of Thought reasoning. Include_simple_steps: {include_simple_steps}\n\nConversations:\n{conversations}",
        }
    }

    # Mock the summary generation
    mock_client.chat_completion.side_effect = [
        "This is a summary about synthetic data.",  # First call for summary
        json.dumps(
            [  # Second call for CoT examples
                {
                    "question": "What is synthetic data?",
                    "reasoning": "Synthetic data is data that is artificially created rather than collected from real-world events.",
                    "answer": "Synthetic data is artificially generated data.",
                },
                {
                    "question": "Why use synthetic data?",
                    "reasoning": "There are several reasons to use synthetic data including privacy protection.",
                    "answer": "To protect privacy and create diverse training examples.",
                },
            ]
        ),
    ]

    # Initialize generator
    generator = COTGenerator(client=mock_client)

    # Process document
    result = generator.process_document(
        document_text="This is a document about synthetic data.", num_examples=2
    )

    # Check that the result contains expected fields
    assert "cot_examples" in result
    assert "conversations" in result

    # Check CoT examples
    assert len(result["cot_examples"]) == 2
    assert result["cot_examples"][0]["question"] == "What is synthetic data?"

    # Check conversations
    assert len(result["conversations"]) == 2
    assert len(result["conversations"][0]) == 3  # system, user, assistant
    assert "reasoning" in result["cot_examples"][0]

    # Check that client was called once for the CoT examples (no summary call)
    assert mock_client.chat_completion.call_count == 1

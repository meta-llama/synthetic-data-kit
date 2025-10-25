"""Unit tests for LLM client."""

from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.models.llm_client import LLMClient


@pytest.mark.unit
def test_llm_client_initialization(patch_config, test_env):
    """Test LLM client initialization with API endpoint provider."""
    with patch("synthetic_data_kit.models.llm_client.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize client
        client = LLMClient(provider="api-endpoint")

        # Check that the client was initialized correctly
        assert client.provider == "api-endpoint"
        assert client.api_base is not None
        assert client.model is not None
        # Check that OpenAI client was initialized
        assert mock_openai.called


@pytest.mark.unit
def test_llm_client_vllm_initialization(patch_config, test_env):
    """Test LLM client initialization with vLLM provider."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["mock-model"]
        mock_get.return_value = mock_response

        # Initialize client
        client = LLMClient(provider="vllm")

        # Check that the client was initialized correctly
        assert client.provider == "vllm"
        assert client.api_base is not None
        assert client.model is not None
        # Check that vLLM server was checked
        assert mock_get.called


@pytest.mark.unit
def test_llm_client_chat_completion(patch_config, test_env):
    """Test LLM client chat completion with API endpoint provider."""
    with patch("synthetic_data_kit.models.llm_client.OpenAI") as mock_openai:
        # Create a proper mock chain for OpenAI client
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completions = MagicMock()
        mock_create = MagicMock()

        # Setup the nested mock structure
        mock_openai.return_value = mock_client
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create

        # Setup mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "This is a test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Connect the create function to return our mock response
        mock_create.return_value = mock_response

        # Initialize client
        client = LLMClient(provider="api-endpoint")

        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"},
        ]

        response = client.chat_completion(messages, temperature=0.7)

        # Check that the response is correct
        assert response == "This is a test response"
        # Check that OpenAI client was called
        assert mock_create.called


@pytest.mark.unit
def test_llm_client_vllm_chat_completion(patch_config, test_env):
    """Test LLM client chat completion with vLLM provider."""
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        # Mock vLLM server check
        mock_check_response = MagicMock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = ["mock-model"]
        mock_get.return_value = mock_check_response

        # Mock vLLM API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a test response"}}]
        }
        mock_post.return_value = mock_response

        # Initialize client
        client = LLMClient(provider="vllm")

        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"},
        ]

        response = client.chat_completion(messages, temperature=0.7)

        # Check that the response is correct
        assert response == "This is a test response"
        # Check that vLLM API was called
        assert mock_post.called


@pytest.mark.unit
def test_llm_client_ollama_initialization(patch_config, test_env):
    """Test LLM client initialization with Ollama provider."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_response

        # Initialize client
        client = LLMClient(provider="ollama")

        # Check that the client was initialized correctly
        assert client.provider == "ollama"
        assert client.api_base is not None
        assert client.model is not None
        # Check that Ollama server was checked
        assert mock_get.called


@pytest.mark.unit
def test_llm_client_ollama_chat_completion(patch_config, test_env):
    """Test LLM client chat completion with Ollama provider."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock Ollama server check
        mock_check_response = MagicMock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_check_response

        # Mock Ollama API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a test response from Ollama"
        }
        mock_post.return_value = mock_response

        # Initialize client
        client = LLMClient(provider="ollama")

        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"},
        ]

        response = client.chat_completion(messages, temperature=0.7)

        # Check that the response is correct
        assert response == "This is a test response from Ollama"
        # Check that Ollama API was called
        assert mock_post.called
        
        # Check that the correct endpoint was called
        call_args = mock_post.call_args
        assert "/api/generate" in call_args[0][0]


@pytest.mark.unit
def test_ollama_message_formatting():
    """Test Ollama message formatting helper function."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_response

        client = LLMClient(provider="ollama")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"},
            {"role": "assistant", "content": "Synthetic data is artificially generated data."},
            {"role": "user", "content": "Can you explain more?"}
        ]
        
        formatted = client._format_messages_for_ollama(messages)
        
        expected = ("System: You are a helpful assistant.\n\n"
                   "User: What is synthetic data?\n\n"
                   "Assistant: Synthetic data is artificially generated data.\n\n"
                   "User: Can you explain more?\n\n"
                   "Assistant:")
        
        assert formatted == expected


@pytest.mark.unit
def test_llm_client_ollama_batch_completion(patch_config, test_env):
    """Test LLM client batch completion with Ollama provider."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock Ollama server check
        mock_check_response = MagicMock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_check_response

        # Mock Ollama API responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            {"response": "Response 1"},
            {"response": "Response 2"}
        ]
        mock_post.return_value = mock_response

        # Initialize client
        client = LLMClient(provider="ollama")

        # Test batch completion
        message_batches = [
            [{"role": "user", "content": "Hello 1"}],
            [{"role": "user", "content": "Hello 2"}]
        ]

        responses = client.batch_completion(message_batches, temperature=0.7, batch_size=1)

        # Check that the responses are correct
        assert len(responses) == 2
        assert responses[0] == "Response 1"
        assert responses[1] == "Response 2"
        # Check that Ollama API was called twice
        assert mock_post.call_count == 2

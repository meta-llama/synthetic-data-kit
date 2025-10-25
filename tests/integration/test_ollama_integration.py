"""Integration tests for Ollama provider."""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.models.llm_client import LLMClient


@pytest.mark.integration
def test_ollama_client_integration(patch_config, test_env):
    """Test Ollama client integration with mocked server responses."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock Ollama server check
        mock_check_response = MagicMock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.2:latest",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 4684876794
                }
            ]
        }
        mock_get.return_value = mock_check_response

        # Mock Ollama generation response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "response": "Synthetic data is artificially generated data that mimics real-world data patterns.",
            "done": True
        }
        mock_post.return_value = mock_response

        # Initialize Ollama client
        client = LLMClient(provider="ollama")

        # Test single chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"}
        ]

        response = client.chat_completion(messages, temperature=0.7, max_tokens=100)

        # Verify response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "synthetic data" in response.lower()

        # Verify API calls
        assert mock_get.called  # Server check
        assert mock_post.called  # Generation call

        # Verify the generation endpoint was called
        post_call_args = mock_post.call_args
        assert "/api/generate" in post_call_args[0][0]
        
        # Verify request payload structure
        request_data = json.loads(post_call_args[1]['data'])
        assert 'model' in request_data
        assert 'prompt' in request_data
        assert 'options' in request_data
        assert request_data['stream'] is False


@pytest.mark.integration
def test_ollama_batch_processing(patch_config, test_env):
    """Test Ollama client batch processing."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock Ollama server check
        mock_check_response = MagicMock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_check_response

        # Mock Ollama generation responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [
            {"response": "Response to question 1 about synthetic data."},
            {"response": "Response to question 2 about machine learning."},
            {"response": "Response to question 3 about data privacy."}
        ]
        mock_post.return_value = mock_response

        # Initialize Ollama client
        client = LLMClient(provider="ollama")

        # Test batch processing
        message_batches = [
            [{"role": "user", "content": "What is synthetic data?"}],
            [{"role": "user", "content": "How is machine learning used?"}], 
            [{"role": "user", "content": "Why is data privacy important?"}]
        ]

        responses = client.batch_completion(
            message_batches,
            temperature=0.7,
            max_tokens=100,
            batch_size=2
        )

        # Verify responses
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert isinstance(response, str)
            assert len(response) > 0
            assert f"Response to question {i+1}" in response

        # Verify API calls
        assert mock_get.called  # Server check
        assert mock_post.call_count == 3  # Three generation calls


@pytest.mark.integration
def test_ollama_message_formatting_edge_cases(patch_config, test_env):
    """Test Ollama message formatting with various edge cases."""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_response

        client = LLMClient(provider="ollama")

        # Test empty messages
        messages = []
        formatted = client._format_messages_for_ollama(messages)
        assert formatted == "Assistant:"

        # Test single system message
        messages = [{"role": "system", "content": "You are helpful."}]
        formatted = client._format_messages_for_ollama(messages)
        expected = "System: You are helpful.\n\nAssistant:"
        assert formatted == expected

        # Test unknown role
        messages = [{"role": "custom", "content": "Custom content"}]
        formatted = client._format_messages_for_ollama(messages)
        expected = "Custom: Custom content\n\nAssistant:"
        assert formatted == expected

        # Test empty content
        messages = [{"role": "user", "content": ""}]
        formatted = client._format_messages_for_ollama(messages)
        expected = "User: \n\nAssistant:"
        assert formatted == expected


@pytest.mark.integration
def test_ollama_error_handling(patch_config, test_env):
    """Test Ollama client error handling."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
        # Mock Ollama server check
        mock_check_response = MagicMock()
        mock_check_response.status_code = 200
        mock_check_response.json.return_value = {"models": [{"name": "llama3.2"}]}
        mock_get.return_value = mock_check_response

        # Initialize Ollama client
        client = LLMClient(provider="ollama")

        # Test HTTP error handling
        mock_post.side_effect = Exception("Connection error")

        messages = [{"role": "user", "content": "Test message"}]

        with pytest.raises(Exception) as exc_info:
            client.chat_completion(messages)

        assert "Failed to get Ollama completion" in str(exc_info.value)

        # Test malformed response
        mock_post.side_effect = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "response"}  # Missing 'response' field
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            client.chat_completion(messages)

        assert "Failed to get Ollama completion" in str(exc_info.value)


@pytest.mark.integration
def test_ollama_server_unavailable(patch_config, test_env):
    """Test Ollama client when server is unavailable."""
    with patch("requests.get") as mock_get:
        # Mock server unavailable
        mock_get.side_effect = Exception("Connection refused")

        # Should raise ConnectionError during initialization
        with pytest.raises(ConnectionError) as exc_info:
            LLMClient(provider="ollama")

        assert "Ollama server not available" in str(exc_info.value)
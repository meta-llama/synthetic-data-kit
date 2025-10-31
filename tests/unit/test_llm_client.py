"""Unit tests for LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synthetic_data_kit.models.llm_client import LLMClient


@pytest.mark.unit
def test_llm_client_initialization(patch_config, test_env):
    """LLM client should initialize the OpenAI provider using SDK clients."""
    with patch("synthetic_data_kit.models.openai_provider.OpenAI") as mock_openai, patch(
        "synthetic_data_kit.models.openai_provider.AsyncOpenAI"
    ) as mock_async_openai:
        mock_openai.return_value = MagicMock()
        mock_async_openai.return_value = MagicMock()

        client = LLMClient()

        assert client.provider == "openai-endpoint"
        assert client.model is not None
        assert client.api_base is not None
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()


@pytest.mark.unit
def test_llm_client_chat_completion(patch_config, test_env):
    """chat_completion should delegate to the OpenAI SDK and return extracted text."""
    with patch("synthetic_data_kit.models.openai_provider.OpenAI") as mock_openai, patch(
        "synthetic_data_kit.models.openai_provider.AsyncOpenAI"
    ) as mock_async_openai:
        mock_async_openai.return_value = MagicMock()

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock(content="This is a test response")
        mock_choice.message = mock_message
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion

        client = LLMClient()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is synthetic data?"},
        ]

        response = client.chat_completion(messages, temperature=0.7)

        assert response == "This is a test response"
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.unit
def test_llm_client_batch_completion(patch_config, test_env):
    """batch_completion should dispatch multiple async SDK calls."""
    with patch("synthetic_data_kit.models.openai_provider.OpenAI") as mock_openai, patch(
        "synthetic_data_kit.models.openai_provider.AsyncOpenAI"
    ) as mock_async_openai, patch(
        "synthetic_data_kit.models.openai_provider.time.sleep"
    ) as mock_sleep:
        mock_sleep.return_value = None

        mock_sync_client = MagicMock()
        mock_async_client = MagicMock()
        mock_openai.return_value = mock_sync_client
        mock_async_openai.return_value = mock_async_client

        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message = MagicMock(content="batch result")
        mock_response.choices = [mock_choice]

        async_create = AsyncMock(return_value=mock_response)
        mock_async_client.chat.completions.create = async_create

        client = LLMClient()

        batches = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "World"}],
        ]

        results = client.batch_completion(batches, temperature=0.1, max_tokens=16, top_p=0.9, batch_size=1)

        assert results == ["batch result", "batch result"]
        assert async_create.await_count == len(batches)


@pytest.mark.unit
def test_llm_client_unknown_provider_raises(patch_config):
    """Requesting an unsupported provider should fail fast."""
    with pytest.raises(ValueError):
        LLMClient(provider="unknown-provider")

#!/usr/bin/env python3
"""
Standalone test for the new LLM providers without pytest dependencies.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthetic_data_kit.models.llm_client import LLMClient

def test_openai_provider():
    """Test OpenAI provider initialization."""
    print("Testing OpenAI provider...")

    with patch("synthetic_data_kit.models.llm_client.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        try:
            client = LLMClient(provider="openai")
            assert client.provider == "openai"
            assert client.api_base == "https://api.openai.com/v1"
            assert client.model == "gpt-4o"
            print("✅ OpenAI provider test passed!")
            return True
        except Exception as e:
            print(f"❌ OpenAI provider test failed: {e}")
            return False

def test_ollama_provider():
    """Test Ollama provider initialization."""
    print("Testing Ollama provider...")

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": ["llama3.2:3b"]}
        mock_get.return_value = mock_response

        try:
            client = LLMClient(provider="ollama")
            assert client.provider == "ollama"
            assert client.api_base == "http://localhost:11434"
            assert client.model == "llama3.2:3b"
            print("✅ Ollama provider test passed!")
            return True
        except Exception as e:
            print(f"❌ Ollama provider test failed: {e}")
            return False

def test_provider_config_loading():
    """Test that provider configurations load correctly."""
    print("Testing provider configuration loading...")

    try:
        from synthetic_data_kit.utils.config import get_openai_direct_config, get_ollama_config

        openai_config = get_openai_direct_config({"openai": {"model": "gpt-4", "api_key": "test"}})
        ollama_config = get_ollama_config({"ollama": {"model": "llama3.2:3b"}})

        assert openai_config["model"] == "gpt-4"
        assert ollama_config["model"] == "llama3.2:3b"

        print("✅ Configuration loading test passed!")
        return True
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Standalone LLM Provider Tests")
    print("=" * 30)

    results = []
    results.append(test_provider_config_loading())
    results.append(test_openai_provider())
    results.append(test_ollama_provider())

    print("\n" + "=" * 30)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

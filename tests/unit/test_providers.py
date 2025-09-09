#!/usr/bin/env python3
"""
Test script for the new Ollama and OpenAI providers in synthetic-data-kit.
This script tests the LLM client functionality with different providers.
"""

import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synthetic_data_kit.models.llm_client import LLMClient

def test_ollama_provider():
    """Test Ollama provider initialization and basic functionality."""
    print("Testing Ollama provider...")

    try:
        # Test initialization
        client = LLMClient(provider="ollama")
        print(f"✓ Ollama client initialized successfully")
        print(f"  Provider: {client.provider}")
        print(f"  API Base: {client.api_base}")
        print(f"  Model: {client.model}")

        # Note: We can't test actual API calls without a running Ollama server
        # But we can test that the client is configured correctly
        return True

    except Exception as e:
        print(f"✗ Ollama provider test failed: {e}")
        return False

def test_openai_provider():
    """Test OpenAI provider initialization and basic functionality."""
    print("\nTesting OpenAI provider...")

    try:
        # Test initialization
        client = LLMClient(provider="openai")
        print(f"✓ OpenAI client initialized successfully")
        print(f"  Provider: {client.provider}")
        print(f"  API Base: {client.api_base}")
        print(f"  Model: {client.model}")

        # Note: We can't test actual API calls without a valid API key
        # But we can test that the client is configured correctly
        return True

    except Exception as e:
        print(f"✗ OpenAI provider test failed: {e}")
        return False

def test_provider_switching():
    """Test switching between different providers."""
    print("\nTesting provider switching...")

    providers = ["vllm", "api-endpoint", "openai", "ollama"]
    results = {}

    for provider in providers:
        try:
            client = LLMClient(provider=provider)
            results[provider] = True
            print(f"✓ {provider} provider initialized successfully")
        except Exception as e:
            results[provider] = False
            print(f"✗ {provider} provider failed: {e}")

    return results

def test_config_loading():
    """Test that the configuration loads correctly with new providers."""
    print("\nTesting configuration loading...")

    try:
        from synthetic_data_kit.utils.config import load_config, get_llm_provider, get_openai_direct_config, get_ollama_config

        config = load_config()
        provider = get_llm_provider(config)

        print(f"✓ Config loaded successfully")
        print(f"  Current provider: {provider}")

        # Test new config functions
        openai_config = get_openai_direct_config(config)
        ollama_config = get_ollama_config(config)

        print(f"✓ OpenAI config: {openai_config.get('model')}")
        print(f"✓ Ollama config: {ollama_config.get('model')}")

        return True

    except Exception as e:
        print(f"✗ Config loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Synthetic Data Kit - Provider Tests")
    print("=" * 40)

    # Test configuration
    config_ok = test_config_loading()

    if not config_ok:
        print("\n❌ Configuration tests failed. Cannot proceed with provider tests.")
        return 1

    # Test providers
    ollama_ok = test_ollama_provider()
    openai_ok = test_openai_provider()

    # Test provider switching
    switch_results = test_provider_switching()

    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"Configuration: {'✓' if config_ok else '✗'}")
    print(f"Ollama Provider: {'✓' if ollama_ok else '✗'}")
    print(f"OpenAI Provider: {'✓' if openai_ok else '✗'}")
    print("Provider Switching:")
    for provider, result in switch_results.items():
        print(f"  {provider}: {'✓' if result else '✗'}")

    # Overall result
    all_passed = config_ok and ollama_ok and openai_ok and all(switch_results.values())

    if all_passed:
        print("\n🎉 All tests passed! The new providers are working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

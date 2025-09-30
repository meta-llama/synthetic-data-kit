#!/usr/bin/env python3
"""
Demonstration script showing how to use the new Ollama and OpenAI providers
in the synthetic-data-kit pipeline.
"""

import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from synthetic_data_kit.models.llm_client import LLMClient

def demo_ollama_usage():
    """Demonstrate Ollama provider usage."""
    print("🚀 Demo: Using Ollama Provider")
    print("-" * 40)

    # Initialize Ollama client
    client = LLMClient(provider="ollama", model_name="llama3.2:3b")

    print(f"Provider: {client.provider}")
    print(f"Model: {client.model}")
    print(f"API Base: {client.api_base}")
    print()

    # Example messages for testing
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is synthetic data and why is it useful for AI training?"}
    ]

    print("Example API call (would work with running Ollama server):")
    print("messages = [")
    for msg in messages:
        print(f'    {{"role": "{msg["role"]}", "content": "{msg["content"]}"}}')
    print("]")
    print()
    print("response = client.chat_completion(messages)")
    print("# This would return the model's response")
    print()

def demo_openai_usage():
    """Demonstrate OpenAI provider usage."""
    print("🚀 Demo: Using OpenAI Provider")
    print("-" * 40)

    # Initialize OpenAI client
    client = LLMClient(provider="openai", model_name="gpt-4o")

    print(f"Provider: {client.provider}")
    print(f"Model: {client.model}")
    print(f"API Base: {client.api_base}")
    print()

    # Example messages for testing
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain the concept of chain-of-thought reasoning in AI."}
    ]

    print("Example API call (would work with valid OpenAI API key):")
    print("messages = [")
    for msg in messages:
        print(f'    {{"role": "{msg["role"]}", "content": "{msg["content"]}"}}')
    print("]")
    print()
    print("response = client.chat_completion(messages)")
    print("# This would return the model's response")
    print()

def demo_pipeline_integration():
    """Show how the new providers integrate with the full pipeline."""
    print("🔧 Pipeline Integration Examples")
    print("-" * 40)

    print("1. Using Ollama for QA pair generation:")
    print("   synthetic-data-kit create data/parsed/ --provider ollama --model llama3.2:3b --type qa")
    print()

    print("2. Using OpenAI for content curation:")
    print("   synthetic-data-kit curate data/generated/ --provider openai --model gpt-4o")
    print()

    print("3. Using API endpoint for batch processing:")
    print("   synthetic-data-kit create data/parsed/ --provider api-endpoint --model your-model")
    print()

    print("4. System check for different providers:")
    print("   synthetic-data-kit system-check --provider ollama")
    print("   synthetic-data-kit system-check --provider openai")
    print("   synthetic-data-kit system-check --provider api-endpoint")
    print()

def demo_configuration():
    """Show configuration examples."""
    print("⚙️  Configuration Examples")
    print("-" * 40)

    print("In your config.yaml or environment variables:")
    print()
    print("# For Ollama:")
    print("llm:")
    print("  provider: ollama")
    print("ollama:")
    print("  model: llama3.2:3b")
    print("  api_base: http://localhost:11434")
    print()

    print("# For OpenAI:")
    print("llm:")
    print("  provider: openai")
    print("openai:")
    print("  model: gpt-4o")
    print("  api_key: sk-your-openai-api-key  # or set OPENAI_API_KEY env var")
    print()

    print("# Environment variables:")
    print("export OPENAI_API_KEY='your-openai-api-key'")
    print("export API_ENDPOINT_KEY='your-api-endpoint-key'")
    print()

def main():
    """Run the demonstration."""
    print("Synthetic Data Kit - New Providers Demo")
    print("=" * 50)
    print()

    demo_ollama_usage()
    demo_openai_usage()
    demo_pipeline_integration()
    demo_configuration()

    print("✨ Summary:")
    print("- Ollama provider: Local LLM inference via Ollama API")
    print("- OpenAI provider: Direct OpenAI API integration")
    print("- API Endpoint provider: Compatible with OpenAI-compatible APIs")
    print("- VLLM provider: Local VLLM server integration")
    print()
    print("All providers support the full synthetic-data-kit pipeline:")
    print("ingest → create → curate → save-as")
    print()
    print("Choose the provider that best fits your needs!")

if __name__ == "__main__":
    main()

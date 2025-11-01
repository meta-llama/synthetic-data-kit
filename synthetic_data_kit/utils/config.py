# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Config Utilities
import yaml
import os
import warnings
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default config location relative to the package (original)
ORIGINAL_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "configs", "config.yaml")
)

# Add fallback location inside the package (recommended for installed packages)
PACKAGE_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
)

# Use internal package path as default
DEFAULT_CONFIG_PATH = PACKAGE_CONFIG_PATH

LEGACY_PROVIDER_ALIASES = {
    "vllm": "openai-endpoint",
    "api-endpoint": "openai-endpoint",
}

LEGACY_CONFIG_SECTION_ALIASES = {
    "openai-endpoint": ("api-endpoint", "vllm"),
}

DEFAULT_PROVIDER_CONFIGS = {
    "openai-endpoint": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "max_retries": 3,
        "retry_delay": 1.0,
        "sleep_time": 0.5,
        "http_request_timeout": 300,
        "max_concurrent_requests": 32,
        "api_key": None,
    },
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file"""
    if config_path is None:
        # Try each path in order until one exists
        for path in [PACKAGE_CONFIG_PATH, ORIGINAL_CONFIG_PATH]:
            if os.path.exists(path):
                config_path = path
                break
        else:
            # If none exists, use the default (which will likely fail, but with a clear error)
            config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Debug: Print LLM provider if it exists
    if 'llm' in config and 'provider' in config['llm']:
        print(f"Config has LLM provider set to: {config['llm']['provider']}")
    else:
        print("Config does not have LLM provider set")
    
    return config

def get_path_config(config: Dict[str, Any], path_type: str, file_type: Optional[str] = None) -> str:
    """Get path from configuration based on type and optionally file type"""
    paths = config.get('paths', {})
    
    if path_type == 'input':
        input_config = paths.get('input', 'data/input')
        # Handle both string and dict formats for input
        if isinstance(input_config, str):
            return input_config
        elif isinstance(input_config, dict):
            if file_type and file_type in input_config:
                return input_config[file_type]
            return input_config.get('default', 'data/input')
        else:
            return 'data/input'
    
    elif path_type == 'output':
        output_paths = paths.get('output', {})
        if file_type and file_type in output_paths:
            return output_paths[file_type]
        return output_paths.get('default', 'data/output')
    
    else:
        raise ValueError(f"Unknown path type: {path_type}")

def get_llm_provider(config: Dict[str, Any]) -> str:
    """Get the selected LLM provider."""
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'openai-endpoint')
    print(f"get_llm_provider returning: {provider}")
    return provider


def get_provider_config(config: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """Return provider configuration merged with defaults and legacy aliases."""
    base_config: Dict[str, Any] = DEFAULT_PROVIDER_CONFIGS.get(provider, {}).copy()

    explicit_config = config.get(provider, {})
    if isinstance(explicit_config, dict):
        base_config = merge_configs(base_config, explicit_config)

    providers_section = config.get('providers', {})
    if isinstance(providers_section, dict):
        nested_config = providers_section.get(provider, {})
        if isinstance(nested_config, dict):
            base_config = merge_configs(base_config, nested_config)

    for legacy_key in LEGACY_CONFIG_SECTION_ALIASES.get(provider, ()):
        legacy_config = config.get(legacy_key)
        if isinstance(legacy_config, dict):
            warnings.warn(
                f"Config section '{legacy_key}' is deprecated. Rename it to '{provider}'.",
                DeprecationWarning,
                stacklevel=3,
            )
            logger.warning(
                "Config section '%s' is deprecated. Treating it as '%s'.",
                legacy_key,
                provider,
            )
            base_config = merge_configs(base_config, legacy_config)

    return base_config

def get_vllm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get VLLM configuration (legacy helper)."""
    warnings.warn(
        "get_vllm_config is deprecated. Use get_provider_config(..., 'openai-endpoint') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    provider_config = get_provider_config(config, 'openai-endpoint')
    defaults = {
        'api_base': 'http://localhost:8000/v1',
        'port': 8000,
        'model': provider_config.get('model', 'meta-llama/Llama-3.3-70B-Instruct'),
        'max_retries': provider_config.get('max_retries', 3),
        'retry_delay': provider_config.get('retry_delay', 1.0),
        'sleep_time': provider_config.get('sleep_time', 0.1),
        'http_request_timeout': provider_config.get('http_request_timeout', 180),
    }
    return merge_configs(defaults, provider_config)

def get_openai_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get OpenAI endpoint configuration (legacy helper)."""
    warnings.warn(
        "get_openai_config is deprecated. Use get_provider_config(..., 'openai-endpoint') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    provider_config = get_provider_config(config, 'openai-endpoint')
    defaults = {
        'api_base': provider_config.get('api_base', 'https://api.openai.com/v1'),
        'api_key': provider_config.get('api_key'),
        'model': provider_config.get('model', 'gpt-4o'),
        'max_retries': provider_config.get('max_retries', 3),
        'retry_delay': provider_config.get('retry_delay', 1.0),
        'sleep_time': provider_config.get('sleep_time', 0.5),
        'http_request_timeout': provider_config.get('http_request_timeout', 300),
    }
    return merge_configs(defaults, provider_config)


def get_openai_endpoint_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Preferred helper for OpenAI-compatible endpoints."""
    return get_provider_config(config, 'openai-endpoint')

def get_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get generation configuration"""
    return config.get('generation', {
        'temperature': 0.7,
        'top_p': 0.95,
        'chunk_size': 4000,
        'overlap': 200,
        'max_tokens': 4096
    })

def get_curate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get curation configuration"""
    return config.get('curate', {
        'threshold': 7.0,
        'batch_size': 8,
        'temperature': 0.1
    })

def get_format_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get format configuration"""
    return config.get('format', {
        'default': 'jsonl',
        'include_metadata': True,
        'pretty_json': True
    })

def get_prompt(config: Dict[str, Any], prompt_name: str) -> str:
    """Get prompt by name"""
    prompts = config.get('prompts', {})
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found in configuration")
    return prompts[prompt_name]

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries"""
    result = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
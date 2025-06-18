# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Config Utilities
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

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
        input_paths = paths.get('input', {})
        if file_type and file_type in input_paths:
            return input_paths[file_type]
        return input_paths.get('default', 'data/input')
    
    elif path_type == 'output':
        output_paths = paths.get('output', {})
        if file_type and file_type in output_paths:
            return output_paths[file_type]
        return output_paths.get('default', 'data/output')
    
    else:
        raise ValueError(f"Unknown path type: {path_type}")

def get_llm_provider(config: Dict[str, Any]) -> str:
    """Get the selected LLM provider
    
    Returns:
        String with provider name: 'vllm' or 'api-endpoint'
    """
    llm_config = config.get('llm', {})
    provider = llm_config.get('provider', 'vllm')
    print(f"get_llm_provider returning: {provider}")
    if provider != 'api-endpoint' and 'llm' in config and 'provider' in config['llm'] and config['llm']['provider'] == 'api-endpoint':
        print(f"WARNING: Config has 'api-endpoint' but returning '{provider}'")
    return provider

def get_vllm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get VLLM configuration"""
    return config.get('vllm', {
        'api_base': 'http://localhost:8000/v1',
        'port': 8000,
        'model': 'meta-llama/Llama-3.3-70B-Instruct',
        'max_retries': 3,
        'retry_delay': 1.0
    })

def get_openai_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get API endpoint configuration"""
    return config.get('api-endpoint', {
        'api_base': None,  # None means use default API base URL
        'api_key': None,  # None means use environment variables
        'model': 'gpt-4o',
        'max_retries': 3,
        'retry_delay': 1.0
    })

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
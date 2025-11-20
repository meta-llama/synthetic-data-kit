# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Translation functionality for code/prompts

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_prompt


def read_json_file(file_path: str) -> list:
    """Read JSON file and return list of JSON objects"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both array of objects and single object
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [data]
    else:
        raise ValueError(f"JSON file {file_path} must contain an array of objects or a single object")


def write_json_file(file_path: str, items: list):
    """Write list of JSON objects to JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def get_translation_prompt(source_lang: str, target_lang: str, prompt: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Generate a translation prompt based on source and target languages
    
    Args:
        source_lang: Source language (e.g., "Rust", "Python")
        target_lang: Target language (e.g., "R", "Python")
        prompt: The code/prompt to translate
        config: Configuration dictionary (if None, will load default config)
    
    Returns:
        Formatted translation prompt
    """
    # Load config if not provided
    if config is None:
        from synthetic_data_kit.utils.config import load_config
        config = load_config()
    
    # Get translation prompt template from config
    try:
        template = get_prompt(config, "translation")
    except ValueError:
        # Fallback to generic template if not found in config
        template = """Translate the following {source_lang} code/prompt to {target_lang}.

Preserve the meaning and intent of the original code/prompt while adapting it to {target_lang} syntax and conventions.

{source_lang} Code/Prompt:

{prompt}

{target_lang} Code/Prompt:"""
    
    return template.format(
        source_lang=source_lang,
        target_lang=target_lang,
        prompt=prompt
    )


def translate_content(
    content: str,
    source_lang: str,
    target_lang: str,
    client: Optional[LLMClient] = None,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    verbose: bool = False
) -> str:
    """Translate content from source language to target language
    
    Args:
        content: The content to translate
        source_lang: Source language name
        target_lang: Target language name
        client: LLM client instance (if None, will create one)
        config_path: Path to configuration file
        api_base: API base URL
        model: Model name
        provider: LLM provider ('vllm' or 'api-endpoint')
        verbose: Whether to print verbose output
    
    Returns:
        Translated content
    """
    # Create LLM client if not provided
    if client is None:
        client = LLMClient(
            config_path=config_path,
            provider=provider,
            api_base=api_base,
            model_name=model
        )
    
    # Generate translation prompt using config
    translation_prompt = get_translation_prompt(source_lang, target_lang, content, client.config)
    
    # Create messages for LLM
    messages = [
        {"role": "user", "content": translation_prompt}
    ]
    
    if verbose:
        print(f"Translating from {source_lang} to {target_lang}...")
        print(f"Content length: {len(content)} characters")
    
    # Get translation from LLM
    translated = client.chat_completion(messages)
    
    if verbose:
        print(f"Translation completed. Output length: {len(translated)} characters")
    
    return translated.strip()


def process_file(
    file_path: str,
    output_dir: str,
    source_lang: str,
    target_lang: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """Process a file to translate its content
    
    Args:
        file_path: Path to the JSON file to translate
        output_dir: Directory to save translated content
        source_lang: Source language name
        target_lang: Target language name
        config_path: Path to configuration file
        api_base: API base URL
        model: Model name
        provider: LLM provider ('vllm' or 'api-endpoint')
        verbose: Whether to print verbose output
    
    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LLM client
    client = LLMClient(
        config_path=config_path,
        provider=provider,
        api_base=api_base,
        model_name=model
    )
    
    # Generate output filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Only support JSON files
    if not file_path.endswith('.json'):
        raise ValueError(f"Only JSON files are supported. Got: {file_path}")
    
    # Read JSON file
    items = read_json_file(file_path)
    
    if verbose:
        print(f"Found {len(items)} items in JSON file")
    
    # Translate each item
    translated_items = []
    for i, item in enumerate(items):
        if verbose:
            print(f"Translating item {i+1}/{len(items)}...")
        
        # Only look for 'prompt' field
        if 'prompt' not in item:
            raise ValueError(f"Could not find 'prompt' field in JSON item {i+1}. Available fields: {list(item.keys())}")
        
        if not isinstance(item['prompt'], str):
            raise ValueError(f"'prompt' field in JSON item {i+1} must be a string. Got: {type(item['prompt'])}")
        
        prompt_text = item['prompt']
        
        # Translate the prompt
        translated_prompt = translate_content(
            content=prompt_text,
            source_lang=source_lang,
            target_lang=target_lang,
            client=client,
            verbose=verbose
        )
        
        # Create new item with both original and translated prompt
        translated_item = item.copy()
        # Keep original prompt and add translated prompt
        translated_item['prompt_translated'] = translated_prompt
        
        translated_items.append(translated_item)
    
    # Write translated JSON file
    output_filename = f"{base_name}_translated_{target_lang}.json"
    output_path = os.path.join(output_dir, output_filename)
    write_json_file(output_path, translated_items)
    
    if verbose:
        print(f"Translated {len(translated_items)} items saved to: {output_path}")
    
    return output_path


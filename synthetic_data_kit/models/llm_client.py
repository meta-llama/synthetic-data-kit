# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List, Dict, Any, Optional
from pathlib import Path

from synthetic_data_kit.utils.config import (
    load_config, 
    get_backend_type,
    get_vllm_config, 
    get_ollama_config
)
from synthetic_data_kit.models.llm_backends import VLLMBackend, OllamaBackend

class LLMClient:
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 backend: Optional[str] = None,
                 api_base: Optional[str] = None, 
                 model_name: Optional[str] = None,
                 max_retries: Optional[int] = None,
                 retry_delay: Optional[float] = None):
        """Initialize an LLM client that supports multiple backends
        
        Args:
            config_path: Path to config file (if None, uses default)
            backend: Override backend type from config ("vllm" or "ollama")
            api_base: Override API base URL from config
            model_name: Override model name from config
            max_retries: Override max retries from config
            retry_delay: Override retry delay from config
        """
        # Load config
        self.config = load_config(config_path)
        
        # Determine backend type
        self.backend_type = backend or get_backend_type(self.config)
        
        # Initialize appropriate backend
        if self.backend_type == "vllm":
            vllm_config = get_vllm_config(self.config)
            self.backend = VLLMBackend(
                api_base=api_base or vllm_config.get('api_base'),
                model=model_name or vllm_config.get('model'),
                max_retries=max_retries or vllm_config.get('max_retries'),
                retry_delay=retry_delay or vllm_config.get('retry_delay')
            )
        elif self.backend_type == "ollama":
            ollama_config = get_ollama_config(self.config)
            self.backend = OllamaBackend(
                api_base=api_base or ollama_config.get('api_base'),
                model=model_name or ollama_config.get('model'),
                max_retries=max_retries or ollama_config.get('max_retries'),
                retry_delay=retry_delay or ollama_config.get('retry_delay')
            )
        else:
            raise ValueError(f"Unsupported backend type: {self.backend_type}")
        
        # Verify server is running
        available, info = self.backend.check_server()
        if not available:
            raise ConnectionError(f"{self.backend_type.upper()} server not available: {info}")
    
    def chat_completion(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = None, 
                      max_tokens: int = None,
                      top_p: float = None) -> str:
        """Generate a chat completion using the configured backend"""
        return self.backend.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
    
    def batch_completion(self, 
                       message_batches: List[List[Dict[str, str]]], 
                       temperature: float = None, 
                       max_tokens: int = None,
                       top_p: float = None,
                       batch_size: int = None) -> List[str]:
        """Process multiple message sets in batches using the configured backend"""
        return self.backend.batch_completion(
            message_batches=message_batches,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            batch_size=batch_size
        )
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models (Ollama only)"""
        if self.backend_type == "ollama":
            return self.backend.list_models()
        raise NotImplementedError("Model listing is only supported for Ollama backend")
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists (Ollama only)"""
        if self.backend_type == "ollama":
            return self.backend.check_model_exists(model_name)
        raise NotImplementedError("Model existence check is only supported for Ollama backend")
    
    @classmethod
    def from_config(cls, config_path: Path) -> 'LLMClient':
        """Create a client from configuration file"""
        return cls(config_path=config_path)
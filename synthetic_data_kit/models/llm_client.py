# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# vLLM logic: Will be expanded to ollama and Cerebras in future.
from typing import List, Dict, Any, Optional
import requests
import json
import time
import os
from pathlib import Path

from synthetic_data_kit.utils.config import load_config, get_vllm_config

class LLMClient:
    def __init__(self, 
                 config_path: Optional[Path] = None,
                 api_base: Optional[str] = None, 
                 model_name: Optional[str] = None,
                 max_retries: Optional[int] = None,
                 retry_delay: Optional[float] = None,
                 api_key: Optional[str] = None):  # added api_key parameter
        """Initialize an OpenAI-compatible client that connects to a VLLM server
        
        Args:
            config_path: Path to config file (if None, uses default)
            api_base: Override API base URL from config
            model_name: Override model name from config
            max_retries: Override max retries from config
            retry_delay: Override retry delay from config
            api_key: Override API key from config
        """
        # Load config
        self.config = load_config(config_path)
        vllm_config = get_vllm_config(self.config)
        
        # Set parameters, with CLI overrides taking precedence
        self.api_base = api_base or vllm_config.get('api_base')
        self.model = model_name or vllm_config.get('model')
        self.max_retries = max_retries or vllm_config.get('max_retries')
        self.retry_delay = retry_delay or vllm_config.get('retry_delay')
        self.api_key = api_key or vllm_config.get('api_key')  # save API key
        
        # Verify server is running
        available, info = self._check_server()
        if not available:
            raise ConnectionError(f"VLLM server not available at {self.api_base}: {info}")
    
    def _check_server(self) -> tuple:
        """Check if the VLLM server is running and accessible"""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.get(f"{self.api_base}/models", headers=headers, timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, f"Server returned status code: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Server connection error: {str(e)}"
    
    def chat_completion(self, 
                      messages: List[Dict[str, str]], 
                      temperature: float = None, 
                      max_tokens: int = None,
                      top_p: float = None) -> str:
        """Generate a chat completion using the VLLM OpenAI-compatible API"""
        # Get defaults from config if not provided
        generation_config = self.config.get('generation', {})
        temperature = temperature if temperature is not None else generation_config.get('temperature', 0.1)
        max_tokens = max_tokens if max_tokens is not None else generation_config.get('max_tokens', 4096)
        top_p = top_p if top_p is not None else generation_config.get('top_p', 0.95)
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        for attempt in range(self.max_retries):
            try:
                # Only print if verbose mode is enabled
                if os.environ.get('SDK_VERBOSE', 'false').lower() == 'true':
                    print(f"Sending request to model {self.model}...")
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    data=json.dumps(data),
                    timeout=180  # Increased timeout to 180 seconds
                )
                if os.environ.get('SDK_VERBOSE', 'false').lower() == 'true':
                    print(f"Received response with status code: {response.status_code}")
                
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to get completion after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
    
    def batch_completion(self, 
                       message_batches: List[List[Dict[str, str]]], 
                       temperature: float = None, 
                       max_tokens: int = None,
                       top_p: float = None,
                       batch_size: int = None) -> List[str]:
        """Process multiple message sets in batches
        
        Instead of sending requests one at a time, this method processes
        multiple prompts in batches to maximize throughput. It uses VLLM's
        ability to efficiently batch requests.
        """
        # Get defaults from config if not provided
        generation_config = self.config.get('generation', {})
        temperature = temperature if temperature is not None else generation_config.get('temperature', 0.1)
        max_tokens = max_tokens if max_tokens is not None else generation_config.get('max_tokens', 4096)
        top_p = top_p if top_p is not None else generation_config.get('top_p', 0.95)
        batch_size = batch_size if batch_size is not None else generation_config.get('batch_size', 32)
        
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        results = []
        
        # Process message batches in chunks to avoid overloading the server
        for i in range(0, len(message_batches), batch_size):
            batch_chunk = message_batches[i:i+batch_size]
            if verbose:
                print(f"Processing batch {i//batch_size + 1}/{(len(message_batches) + batch_size - 1) // batch_size} with {len(batch_chunk)} requests")
            
            # Create batch request payload for VLLM
            batch_requests = []
            for messages in batch_chunk:
                batch_requests.append({
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p
                })
            
            try:
                # For now, we run these in parallel with multiple requests
                batch_results = []
                for request_data in batch_requests:
                    # Only print if verbose mode is enabled
                    if verbose:
                        print(f"Sending batch request to model {self.model}...")
                    
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    response = requests.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        data=json.dumps(request_data),
                        timeout=180  # Increased timeout for batch processing
                    )
                    
                    if verbose:
                        print(f"Received response with status code: {response.status_code}")
                    
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                    batch_results.append(content)
                
                results.extend(batch_results)
                
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                raise Exception(f"Failed to process batch: {str(e)}")
            
            time.sleep(0.1)
        
        return results
    
    @classmethod
    def from_config(cls, config_path: Path) -> 'LLMClient':
        """Create a client from configuration file"""
        return cls(config_path=config_path)
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
import json
import time
import os
from pathlib import Path

class BaseLLMBackend(ABC):
    """Abstract base class for LLM inference backends"""
    
    @abstractmethod
    def check_server(self) -> tuple:
        """Check if the server is running and accessible"""
        pass
    
    @abstractmethod
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: float = None,
                       max_tokens: int = None,
                       top_p: float = None) -> str:
        """Generate a chat completion"""
        pass
    
    @abstractmethod
    def batch_completion(self,
                        message_batches: List[List[Dict[str, str]]],
                        temperature: float = None,
                        max_tokens: int = None,
                        top_p: float = None,
                        batch_size: int = None) -> List[str]:
        """Process multiple message sets in batches"""
        pass

class VLLMBackend(BaseLLMBackend):
    """VLLM backend implementation"""
    
    def __init__(self, api_base: str, model: str, max_retries: int = 3, retry_delay: float = 1.0):
        self.api_base = api_base
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def check_server(self) -> tuple:
        try:
            response = requests.get(f"{self.api_base}/models", timeout=5)
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
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.7,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "top_p": top_p if top_p is not None else 0.95
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data),
                    timeout=180
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to get completion after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
    
    def batch_completion(self,
                        message_batches: List[List[Dict[str, str]]],
                        temperature: float = None,
                        max_tokens: int = None,
                        top_p: float = None,
                        batch_size: int = None) -> List[str]:
        batch_size = batch_size if batch_size is not None else 32
        results = []
        
        for i in range(0, len(message_batches), batch_size):
            batch_chunk = message_batches[i:i+batch_size]
            batch_results = []
            
            for messages in batch_chunk:
                content = self.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                batch_results.append(content)
            
            results.extend(batch_results)
            time.sleep(0.1)
        
        return results

class OllamaBackend(BaseLLMBackend):
    """Ollama backend implementation"""
    
    def __init__(self, api_base: str = "http://localhost:11434/v1", model: str = "llama3.2", max_retries: int = 3, retry_delay: float = 1.0):
        self.api_base = api_base
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def check_server(self) -> tuple:
        try:
            response = requests.get(f"{self.api_base}/models", timeout=5)
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
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.7,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "top_p": top_p if top_p is not None else 0.95
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data),
                    timeout=180
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.exceptions.RequestException, KeyError, IndexError) as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to get completion after {self.max_retries} attempts: {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
    
    def batch_completion(self,
                        message_batches: List[List[Dict[str, str]]],
                        temperature: float = None,
                        max_tokens: int = None,
                        top_p: float = None,
                        batch_size: int = None) -> List[str]:
        batch_size = batch_size if batch_size is not None else 32
        results = []
        
        for i in range(0, len(message_batches), batch_size):
            batch_chunk = message_batches[i:i+batch_size]
            batch_results = []
            
            for messages in batch_chunk:
                content = self.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                batch_results.append(content)
            
            results.extend(batch_results)
            time.sleep(0.1)
        
        return results
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models"""
        response = requests.get(f"{self.api_base}/models")
        response.raise_for_status()
        return response.json()["models"]
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists"""
        models = self.list_models()
        return any(model["name"] == model_name for model in models) 
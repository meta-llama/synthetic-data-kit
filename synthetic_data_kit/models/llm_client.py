# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List, Dict, Any, Optional
import logging
import os

from synthetic_data_kit.models.base import BaseLLMProvider
from synthetic_data_kit.models.openai_provider import OpenAIEndpointProvider
from synthetic_data_kit.utils.config import (
    load_config,
    get_llm_provider,
    get_provider_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """High-level client that exposes a unified interface for chat and batch generation."""

    PROVIDER_REGISTRY = {
        "openai-endpoint": OpenAIEndpointProvider,
    }

    def __init__(
        self,
        config_path: Optional[os.PathLike] = None,
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        http_request_timeout: Optional[int] = None,
    ):
        self.config = load_config(config_path)

        active_provider = provider or get_llm_provider(self.config)
        self.provider = active_provider
        self.provider_config = get_provider_config(self.config, active_provider)

        overrides = {
            "api_base": api_base,
            "api_key": api_key,
            "model_name": model_name,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "http_request_timeout": http_request_timeout,
        }

        self._provider_impl = self._initialize_provider(active_provider, self.provider_config, overrides)

        self.model = self._provider_impl.model
        self.max_retries = self._provider_impl.max_retries
        self.retry_delay = self._provider_impl.retry_delay
        self.sleep_time = getattr(self._provider_impl, "sleep_time", None)
        self.api_base = getattr(self._provider_impl, "api_base", None)
        self.http_request_timeout = getattr(self._provider_impl, "http_request_timeout", None)

    def _initialize_provider(
        self,
        provider_name: str,
        provider_config: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> BaseLLMProvider:
        provider_class = self.PROVIDER_REGISTRY.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown LLM provider '{provider_name}'")
        return provider_class(provider_config, overrides)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        generation_config = self.config.get("generation", {})
        temperature = temperature if temperature is not None else generation_config.get("temperature", 0.1)
        max_tokens = max_tokens if max_tokens is not None else generation_config.get("max_tokens", 4096)
        top_p = top_p if top_p is not None else generation_config.get("top_p", 0.95)

        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"

        return self._provider_impl.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            verbose=verbose,
        )

    def batch_completion(
        self,
        message_batches: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        generation_config = self.config.get("generation", {})
        temperature = temperature if temperature is not None else generation_config.get("temperature", 0.1)
        max_tokens = max_tokens if max_tokens is not None else generation_config.get("max_tokens", 4096)
        top_p = top_p if top_p is not None else generation_config.get("top_p", 0.95)
        batch_size = batch_size if batch_size is not None else generation_config.get("batch_size", 32)

        verbose = os.environ.get("SDK_VERBOSE", "false").lower() == "true"

        return self._provider_impl.batch_completion(
            message_batches=message_batches,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            batch_size=batch_size,
            verbose=verbose,
        )

    @classmethod
    def from_config(cls, config_path: os.PathLike) -> "LLMClient":
        """Create a client from configuration file."""
        return cls(config_path=config_path)

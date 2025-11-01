# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import logging
import os
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseLLMProvider:
    """Abstract base class for all LLM providers."""

    def __init__(self, name: str, config: Dict[str, Any], overrides: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.debug_mode = os.environ.get("SDK_DEBUG", "false").lower() == "true"

        model_override = overrides.get("model_name")
        self.model = model_override or config.get("model")
        if not self.model:
            raise ValueError(f"Model must be specified for provider '{name}'")

        self.max_retries = overrides.get("max_retries") or config.get("max_retries", 3)
        self.retry_delay = overrides.get("retry_delay") or config.get("retry_delay", 1.0)
        self.sleep_time = config.get("sleep_time", 0.5)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        verbose: bool,
    ) -> str:
        raise NotImplementedError

    def batch_completion(
        self,
        message_batches: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        batch_size: int,
        verbose: bool,
    ) -> List[str]:
        raise NotImplementedError
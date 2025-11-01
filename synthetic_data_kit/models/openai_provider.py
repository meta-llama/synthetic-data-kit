# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import logging
import os
import threading
import time
from typing import Any, Dict, List

from openai import AsyncOpenAI, OpenAI, OpenAIError

from synthetic_data_kit.models.base import BaseLLMProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIEndpointProvider(BaseLLMProvider):
    """Provider that leverages the official OpenAI Python client library."""

    def __init__(self, config: Dict[str, Any], overrides: Dict[str, Any]):
        super().__init__("openai-endpoint", config, overrides)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.api_base = (
            overrides.get("api_base")
            or config.get("api_base")
        )

        env_keys = config.get("api_key_env_vars", ["OPENAI_API_KEY", "API_ENDPOINT_KEY"])
        env_api_key = next((os.environ.get(key) for key in env_keys if os.environ.get(key)), None)

        self.api_key = overrides.get("api_key") or env_api_key or config.get("api_key")

        headers = config.get("headers", {})
        self.extra_headers = (
            {str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else {}
        )

        query_params = config.get("query_params", {})
        self.extra_query = dict(query_params) if isinstance(query_params, dict) else {}

        payload_overrides = config.get("payload_overrides", {})
        self.payload_overrides = (
            dict(payload_overrides) if isinstance(payload_overrides, dict) else {}
        )

        self.http_request_timeout = (
            overrides.get("http_request_timeout")
            or config.get("http_request_timeout", 300)
        )

        self.max_concurrent_requests = config.get("max_concurrent_requests", 32) or 32

        client_kwargs: Dict[str, Any] = {
        }
        if self.api_base:
            client_kwargs["base_url"] = self.api_base
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.extra_headers:
            client_kwargs["default_headers"] = dict(self.extra_headers)

        self._client_kwargs = client_kwargs
        self._request_timeout = float(self.http_request_timeout) if self.http_request_timeout else None

        self._client = OpenAI(**self._client_kwargs)
        self._async_client = AsyncOpenAI(**self._client_kwargs)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        verbose: bool,
    ) -> str:
        kwargs = self._build_request_kwargs(messages, temperature, max_tokens, top_p)
        

        for attempt in range(self.max_retries):
            try:
                if verbose and attempt == 0:
                    self.logger.info("Sending request via OpenAI SDK to model %s...", self.model)

                response = self._client.chat.completions.create(**kwargs)

                if self.debug_mode:
                    self.logger.debug("Raw SDK response: %s", response)

                return self._extract_chat_content(response, verbose)

            except (OpenAIError, ValueError, KeyError) as exc:
                error_message = str(exc)
                if verbose:
                    self.logger.warning(
                        "OpenAI SDK error (attempt %s/%s): %s",
                        attempt + 1,
                        self.max_retries,
                        error_message,
                    )

                if attempt == self.max_retries - 1:
                    raise Exception(
                        f"Failed to get openai-endpoint completion after {self.max_retries} attempts: {error_message}"
                    ) from exc

                time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff

        raise RuntimeError("Exceeded retry attempts without raising")

    def batch_completion(
        self,
        message_batches: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        batch_size: int,
        verbose: bool,
    ) -> List[str]:
        if not message_batches:
            return []

        batch_size = max(1, batch_size)
        total_batches = (len(message_batches) + batch_size - 1) // batch_size
        results: List[str] = []

        for index in range(0, len(message_batches), batch_size):
            batch_chunk = message_batches[index:index + batch_size]
            batch_number = index // batch_size + 1

            if verbose:
                self.logger.info(
                    "Processing batch %s/%s with %s requests via OpenAI SDK",
                    batch_number,
                    total_batches,
                    len(batch_chunk),
                )

            chunk_results = self._run_coroutine(
                self._process_batch_async(batch_chunk, temperature, max_tokens, top_p, verbose)
            )
            results.extend(chunk_results)

            if index + batch_size < len(message_batches):
                time.sleep(self.sleep_time)

        return results

    def _build_request_kwargs(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if self.payload_overrides:
            kwargs.update(self.payload_overrides)
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self.extra_query:
            kwargs["extra_query"] = self.extra_query
        if self._request_timeout is not None:
            kwargs["timeout"] = self._request_timeout
        return kwargs

    async def _process_batch_async(
        self,
        batch_chunk: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        verbose: bool,
    ) -> List[str]:
        if not batch_chunk:
            return []

        max_concurrent = self.max_concurrent_requests or len(batch_chunk)
        max_concurrent = max(1, min(max_concurrent, len(batch_chunk)))

        # Create semaphore to limit concurrent connections (prevent overwhelming the server)
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            self._process_single_request(messages, semaphore, temperature, max_tokens, top_p, verbose)
            for messages in batch_chunk
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _process_single_request(
        self,
        messages: List[Dict[str, str]],
        semaphore: asyncio.Semaphore,
        temperature: float,
        max_tokens: int,
        top_p: float,
        verbose: bool,
    ) -> str:
        kwargs = self._build_request_kwargs(messages, temperature, max_tokens, top_p)
        debug_mode = os.environ.get("SDK_DEBUG", "false").lower() == "true"

        for attempt in range(self.max_retries):
            try:
                async with semaphore:
                    if verbose and attempt == 0:
                        self.logger.info(
                            "Sending async request via OpenAI SDK to model %s...",
                            self.model,
                        )

                    response = await self._async_client.chat.completions.create(**kwargs)

                    if debug_mode:
                        self.logger.debug("Raw async SDK response: %s", response)

                    return self._extract_chat_content(response, verbose)

            except (OpenAIError, asyncio.TimeoutError, ValueError, KeyError) as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries - 1:
                    if verbose:
                        self.logger.warning(
                            "OpenAI async request failed after %s attempts: %s",
                            self.max_retries,
                            error_message,
                        )
                    return f"ERROR: {error_message}"

                await asyncio.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff

        return "ERROR: exceeded retry attempts"

    def _run_coroutine(self, coroutine: asyncio.Future) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return self._run_coroutine_in_thread(coroutine)

        return asyncio.run(coroutine)

    def _run_coroutine_in_thread(self, coroutine: asyncio.Future) -> Any:
        result_container: Dict[str, Any] = {}

        def runner() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result_container["result"] = new_loop.run_until_complete(coroutine)
            except Exception as exc:
                result_container["error"] = exc
            finally:
                new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                new_loop.close()
                asyncio.set_event_loop(None)

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
    
        if "error" in result_container:
            raise result_container["error"]

        return result_container.get("result", [])
    
    def _extract_chat_content(self, response: Any, verbose: bool) -> str:
        """Extract primary message content from an OpenAI-style chat response."""
        if response is None:
            raise ValueError("Empty response from provider")

        # Method 1: Try standard OpenAI API response format
        try:
            if hasattr(response, 'choices') and response.choices is not None and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message is not None:
                    if hasattr(choice.message, 'content') and choice.message.content is not None:
                        return choice.message.content
        except Exception as e:
            if verbose:
                self.logger.info(f"Standard format extraction failed: {e}, trying alternative formats...")
        
        # Method 2: Llama API format
        try:
            if hasattr(response, 'completion_message') and response.completion_message is not None:
                completion = response.completion_message
                # Handle dictionary case
                if isinstance(completion, dict) and 'content' in completion:
                    content = completion['content']
                    # Different Llama API response formats
                    if isinstance(content, dict) and 'text' in content:
                        return content['text']
                    elif isinstance(content, str):
                        return content
        except Exception as e:
            if verbose:
                self.logger.info(f"Llama API format extraction failed: {e}, trying dictionary access...")
        
        # Method 3: Try dictionary access for both formats
        try:
            # Convert to dictionary if possible
            response_dict = None
            if hasattr(response, 'model_dump'):
                response_dict = response.model_dump()
            elif hasattr(response, 'dict'):
                response_dict = response.dict()
            elif hasattr(response, '__dict__'):
                response_dict = response.__dict__
            elif isinstance(response, dict):
                response_dict = response
            
            if response_dict is not None:
                # Try Llama API format
                if 'completion_message' in response_dict and response_dict['completion_message'] is not None:
                    comp = response_dict['completion_message']
                    if isinstance(comp, dict) and 'content' in comp:
                        content = comp['content']
                        if isinstance(content, dict) and 'text' in content:
                            return content['text']
                        elif isinstance(content, str):
                            return content
                
                # Try OpenAI format
                if 'choices' in response_dict and response_dict['choices'] is not None and len(response_dict['choices']) > 0:
                    choice = response_dict['choices'][0]
                    if isinstance(choice, dict) and 'message' in choice:
                        message = choice['message']
                        if isinstance(message, dict) and 'content' in message and message['content'] is not None:
                            return message['content']
        except Exception as e:
            if verbose:
                self.logger.info(f"Dictionary access failed: {e}")
        
        # Last resort: Try to print the full response for debugging
        if verbose or self.debug_mode:
            self.logger.error("Could not extract content from response using any known method")
            self.logger.error(f"Response: {response}")
            if isinstance(response, dict):
                for k, v in response.items():
                    self.logger.error(f"Key: {k}, Value type: {type(v)}, Value: {v}")
            # Try to find any content-like fields
            all_attrs = dir(response)
            content_fields = [attr for attr in all_attrs if 'content' in attr.lower() or 'text' in attr.lower() or 'message' in attr.lower()]
            for field in content_fields:
                try:
                    self.logger.error(f"Potential content field '{field}': {getattr(response, field, 'N/A')}")
                except:
                    pass
        
        raise ValueError(f"Could not extract content from response using any known method")

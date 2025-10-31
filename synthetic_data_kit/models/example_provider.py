# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import os
import threading
import time
from typing import Any, Dict, List

import aiohttp
import requests

from synthetic_data_kit.models.base import BaseLLMProvider


"""
This is just supposed to serve as an example of how to implement a new provider which
does not provide a dedicated client library and we instead have to use HTTP requests directly.
"""
class ExampleHTTPProvider(BaseLLMProvider):
    """Example provider using raw HTTP transport for OpenAI-compatible endpoints."""

    def __init__(self, config: Dict[str, Any], overrides: Dict[str, Any]):
        super().__init__("openai-http", config, overrides)

        self.api_base = (
            overrides.get("api_base")
            or config.get("api_base")
        )

        env_keys = config.get("api_key_env_vars", ["API_ENDPOINT_KEY", "OPENAI_API_KEY"])
        env_api_key = None
        for key in env_keys:
            value = os.environ.get(key)
            if value:
                env_api_key = value
                break

        self.api_key = overrides.get("api_key") or env_api_key or config.get("api_key")
        self.api_key_header = config.get("api_key_header", "Authorization")
        self.api_key_scheme = config.get("api_key_scheme", "Bearer")
        self.require_api_key = config.get("require_api_key", False)

        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        additional_headers = config.get("headers", {})
        if isinstance(additional_headers, dict):
            self.headers.update({str(k): str(v) for k, v in additional_headers.items()})

        if self.api_key:
            self._apply_api_key_header(self.headers)
        elif self.require_api_key:
            raise ValueError(
                "API key is required for openai-endpoint provider. Set it in config or via environment variables."
            )

        query_params = config.get("query_params", {})
        self.query_params = dict(query_params) if isinstance(query_params, dict) else {}

        payload_overrides = config.get("payload_overrides", {})
        self.payload_overrides = (
            dict(payload_overrides) if isinstance(payload_overrides, dict) else {}
        )

        self.http_request_timeout = (
            overrides.get("http_request_timeout") or config.get("http_request_timeout", 300)
        )

        self.max_concurrent_requests = config.get("max_concurrent_requests", 32) or 32

        endpoint_path = config.get("endpoint_path", "/chat/completions").format(model=self.model)
        if not endpoint_path.startswith("/"):
            endpoint_path = f"/{endpoint_path}"
        self.chat_url = f"{self.api_base.rstrip('/')}{endpoint_path}"
        self.models_url = f"{self.api_base.rstrip('/')}/models"

    def _apply_api_key_header(self, headers: Dict[str, str]) -> None:
        header_name = (self.api_key_header or "Authorization").strip()
        if header_name.lower() == "authorization":
            scheme = (self.api_key_scheme or "Bearer").strip()
            if scheme:
                headers["Authorization"] = f"{scheme} {self.api_key}"
            else:
                headers["Authorization"] = str(self.api_key)
        else:
            headers[header_name] = str(self.api_key)

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        verbose: bool,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, top_p)
        debug_mode = os.environ.get("SDK_DEBUG", "false").lower() == "true"

        for attempt in range(self.max_retries):
            try:
                if verbose and attempt == 0:
                    self.logger.info(
                        "Sending request to %s model %s...", self.name, self.model
                    )

                response = requests.post(
                    self.chat_url,
                    headers=self.headers,
                    params=self.query_params,
                    json=payload,
                    timeout=self.http_request_timeout,
                )

                if verbose:
                    self.logger.info(
                        "Received response with status code: %s", response.status_code
                    )

                response.raise_for_status()
                response_json = response.json()

                if debug_mode:
                    self.logger.debug("Raw response: %s", response_json)
                    
                return response_json["choices"][0]["message"]["content"]

            except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
                error_message = str(exc)
                if verbose:
                    self.logger.warning(
                        "%s API error (attempt %s/%s): %s",
                        self.name,
                        attempt + 1,
                        self.max_retries,
                        error_message,
                    )

                if attempt == self.max_retries - 1:
                    raise Exception(
                        f"Failed to get {self.name} completion after {self.max_retries} attempts: {error_message}"
                    ) from exc

                time.sleep(self.retry_delay * (attempt + 1))

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
                    "Processing batch %s/%s with %s requests",
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

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if self.payload_overrides:
            payload.update(self.payload_overrides)
        return payload

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
        connector = aiohttp.TCPConnector(
            limit=max_concurrent * 2,
            limit_per_host=max_concurrent,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.http_request_timeout)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=dict(self.headers),
        ) as session:
            semaphore = asyncio.Semaphore(max_concurrent)
            tasks = [
                self._process_single_request(
                    session,
                    messages,
                    semaphore,
                    temperature,
                    max_tokens,
                    top_p,
                    verbose,
                )
                for messages in batch_chunk
            ]
            return await asyncio.gather(*tasks, return_exceptions=False)

    async def _process_single_request(
        self,
        session: aiohttp.ClientSession,
        messages: List[Dict[str, str]],
        semaphore: asyncio.Semaphore,
        temperature: float,
        max_tokens: int,
        top_p: float,
        verbose: bool,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, top_p)
        debug_mode = os.environ.get("SDK_DEBUG", "false").lower() == "true"

        for attempt in range(self.max_retries):
            try:
                async with semaphore:  # Limit concurrent requests
                    if verbose and attempt == 0:
                        self.logger.info(
                            "Sending async request to %s model %s...", self.name, self.model
                        )

                    async with session.post(
                        self.chat_url,
                        params=self.query_params,
                        json=payload,
                    ) as response:
                        if verbose and attempt == 0:
                            self.logger.info(
                                "Received response with status code: %s", response.status
                            )

                        response.raise_for_status()
                        response_json = await response.json()

                        if debug_mode:
                            self.logger.debug("Raw async response: %s", response_json)

                        return response_json["choices"][0]["message"]["content"]

            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError) as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries - 1:
                    if verbose:
                        self.logger.warning(
                            "%s async request failed after %s attempts: %s",
                            self.name,
                            self.max_retries,
                            error_message,
                        )
                    return f"ERROR: {error_message}"

                await asyncio.sleep(self.retry_delay * (attempt + 1))

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


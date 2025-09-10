"""Functional tests for the --language option across providers and Arabic input."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app


@pytest.mark.functional
@pytest.mark.parametrize(
    "language,expected,provider",
    [
        ("english", "english", "api-endpoint"),
        ("english", "english", "openai"),
        ("english", "english", "vllm"),
        ("english", "english", "ollama"),
        ("source", "source", "api-endpoint"),
        ("source", "source", "openai"),
        ("source", "source", "vllm"),
        ("source", "source", "ollama"),
    ],
)
def test_create_language_option_providers(patch_config, test_env, language, expected, provider):
    """Verify CLI forwards --language to processing for all providers."""
    runner = CliRunner()

    # Create a temporary text file as input
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+", delete=False) as f:
        f.write("This is a sample document.")
        input_path = f.name

    captured_kwargs = {}

    try:
        # Mock server checks for vllm/ollama providers
        if provider in ("vllm", "ollama"):
            def _mock_get(url, timeout=2):
                m = MagicMock()
                m.status_code = 200
                if provider == "vllm":
                    m.json.return_value = ["mock-model"]
                else:
                    m.json.return_value = {"models": [{"name": "llama"}]}
                return m

            get_patch_ctx = patch("requests.get", side_effect=_mock_get)
        else:
            # Dummy context manager that does nothing
            class _Noop:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False
            get_patch_ctx = _Noop()

        with get_patch_ctx:
            # Patch core.create.process_file to capture language
            with patch("synthetic_data_kit.core.create.process_file") as mock_process:
                # Capture kwargs and return a fake output path
                def _mock_process(*args, **kwargs):
                    captured_kwargs.update(kwargs)
                    return os.path.join(os.path.dirname(input_path), "out.json")

                mock_process.side_effect = _mock_process

                result = runner.invoke(
                    app,
                    [
                        "create",
                        input_path,
                        "--type",
                        "qa",
                        "--language",
                        language,
                        "--provider",
                        provider,
                    ],
                )

        assert result.exit_code == 0
        # Ensure language forwarded correctly
        assert captured_kwargs.get("language") == expected

    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_create_with_specific_arabic_pdf(patch_config, test_env, tmp_path):
    """If the Arabic PDF exists locally, run a minimal generation and verify language flow.

    This test is skipped if the provided Arabic PDF path is not present.
    """
    arabic_pdf_path = "/home/w-ds-026/Documents/المؤشر الوطني للذكاء الاصطناعي_250727_135526-1.pdf"
    if not os.path.exists(arabic_pdf_path):
        pytest.skip("Arabic PDF not found on this machine; skipping.")

    # Patch network/server checks
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = ["mock-model"]
        mock_get.return_value = mock_resp

        # Capture messages passed to the client
        captured = {"messages": []}

        from synthetic_data_kit.models.llm_client import LLMClient

        def fake_chat_completion(self, messages, temperature=None, max_tokens=None, top_p=None):
            captured["messages"].append(messages)
            return "Summary"

        def fake_batch_completion(self, message_batches, temperature=None, max_tokens=None, top_p=None, batch_size=None):
            captured["messages"].extend(message_batches)
            # Return minimal valid JSON for QA pairs per batch item
            return ["[]" for _ in message_batches]

        with patch.object(LLMClient, "chat_completion", new=fake_chat_completion), \
             patch.object(LLMClient, "batch_completion", new=fake_batch_completion):
            from synthetic_data_kit.core.create import process_file

            out = process_file(
                file_path=arabic_pdf_path,
                output_dir=str(tmp_path),
                content_type="qa",
                num_pairs=2,
                provider="vllm",
                api_base="http://localhost:8000",
                model="mock-model",
                language="source",
                verbose=False,
            )

    assert os.path.exists(out)
    # Verify at least one prompt carried the 'source' language instruction
    found_instruction = any(
        any(isinstance(m, dict) and m.get("role") == "system" and "same language" in (m.get("content") or "") for m in msg_set)
        for msg_set in captured["messages"]
    )
    assert found_instruction

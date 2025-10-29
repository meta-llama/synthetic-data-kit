import os
import time
import json
import pytest
import requests

from synthetic_data_kit.models.llm_client import LLMClient

# ---------- Live server detection helpers ----------

def _live_base():
    return os.getenv("LIVE_VLLM_BASE", "http://localhost:8000/v1")

def _vllm_is_live(base: str, timeout=1.0) -> bool:
    try:
        r = requests.get(f"{base}/models", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

@pytest.fixture(scope="session")
def live_info():
    base = _live_base()
    return {"base": base, "up": _vllm_is_live(base)}

# ---------- Live path (no mocks) ----------

@pytest.mark.integration
def test_live_vllm_smoke(live_info):
    """
    If a real vLLM server is reachable, run a real call.
    Otherwise skip this test automatically.
    """
    if not live_info["up"]:
        pytest.skip(f"No live vLLM at {_live_base()} — skipping live smoke.")

    client = LLMClient(
        provider="vllm",
        api_base=live_info["base"],
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # must match the server's model
        max_retries=2,
        retry_delay=0.2,
    )
    out = client.chat_completion(
        messages=[{"role": "user", "content": "Say 'pong' and nothing else."}],
        temperature=0.0,
        max_tokens=16,
    )
    assert isinstance(out, str)
    assert "pong" in out.lower()

# ---------- Mock path (runs when live is NOT available) ----------

@pytest.fixture
def patch_config(monkeypatch):
    """
    Force config -> provider=vllm and point to the default base.
    """
    def fake_load_config(_path=None):
        return {
            "llm": {"provider": "vllm"},
            "vllm": {
                "api_base": _live_base(),
                "model": "fake-model",
                "max_retries": 2,
                "retry_delay": 0.05,
                "sleep_time": 0.01,
                "max_concurrency": 4,
            },
            "generation": {"temperature": 0.0, "max_tokens": 32, "top_p": 0.95, "batch_size": 8},
        }
    monkeypatch.setattr("synthetic_data_kit.models.llm_client.load_config", fake_load_config)
    return fake_load_config

@pytest.fixture
def test_env(monkeypatch, live_info):
    """
    If live vLLM is DOWN, we stub _check_vllm_server and requests.post.
    If live is UP, we do nothing (so the live smoke test can run).
    """
    if live_info["up"]:
        # Live server exists; let the live test handle it.
        return

    # --- stub health check to "available" ---
    def fake_check_vllm_server(self):
        return True, {"models": [{"id": "fake-model"}]}
    monkeypatch.setattr("synthetic_data_kit.models.llm_client.LLMClient._check_vllm_server", fake_check_vllm_server)

    # --- stub POST /chat/completions to return deterministic JSON ---
    def fake_post(url, headers=None, data=None, timeout=None):
        body = json.loads(data or "{}")
        # Simulate a small stagger so futures may complete out-of-order
        time.sleep(0.01 if "slow" not in (headers or {}) else 0.02)

        class Resp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                # Echo back the last user message content
                msgs = body.get("messages", [])
                content = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
                return {"choices": [{"message": {"content": f"echo:{content}"}}]}
        return Resp()

    monkeypatch.setattr("synthetic_data_kit.models.llm_client.requests.post", fake_post)


# --- Tests that run under mocks when live server is not available ---

@pytest.mark.unit
def test_vllm_batch_completion_preserves_order(patch_config, test_env, live_info):
    if live_info["up"]:
        pytest.skip("Live vLLM detected; order test is only for the mock path.")

    client = LLMClient(provider="vllm")  # config fixture supplies base/model
    batches = [
        [{"role": "user", "content": f"u{i}"}]
        for i in range(5)
    ]
    out = client._vllm_batch_completion(
        message_batches=batches,
        temperature=0.0,
        max_tokens=16,
        top_p=0.95,
        batch_size=5,
        verbose=False,
    )
    assert out == [f"echo:u{i}" for i in range(5)]

@pytest.mark.unit
def test_vllm_batch_completion_isolates_errors(monkeypatch, patch_config, test_env, live_info):
    if live_info["up"]:
        pytest.skip("Live vLLM detected; error-isolation test is only for the mock path.")

    # Patch one of the POSTs to raise
    real_post = requests.post

    def flaky_post(url, headers=None, data=None, timeout=None):
        body = json.loads(data or "{}")
        msgs = body.get("messages", [])
        content = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
        if content == "boom":
            raise requests.exceptions.ConnectionError("simulated failure")
        return real_post(url, headers=headers, data=data, timeout=timeout)

    monkeypatch.setattr("synthetic_data_kit.models.llm_client.requests.post", flaky_post)

    client = LLMClient(provider="vllm")
    batches = [
        [{"role": "user", "content": "ok1"}],
        [{"role": "user", "content": "boom"}],  # will fail
        [{"role": "user", "content": "ok2"}],
    ]
    out = client._vllm_batch_completion(
        message_batches=batches,
        temperature=0.0,
        max_tokens=16,
        top_p=0.95,
        batch_size=3,
        verbose=False,
    )
    assert out[0].startswith("echo:ok1")
    assert out[1].startswith("ERROR:")
    assert out[2].startswith("echo:ok2")

@pytest.mark.unit
def test_vllm_batch_respects_max_concurrency(monkeypatch, patch_config, test_env, live_info):
    if live_info["up"]:
        pytest.skip("Live vLLM detected; concurrency test is only for the mock path.")

    monkeypatch.setenv("SDK_VLLM_MAX_CONCURRENCY", "2")
    client = LLMClient(provider="vllm")
    assert client.max_concurrency == 2

@pytest.mark.unit
def test_vllm_post_once_happy_path(patch_config, test_env, live_info):
    if live_info["up"]:
        pytest.skip("Live vLLM detected; post-once test is only for the mock path.")

    client = LLMClient(provider="vllm")
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "xyz"}],
        "temperature": 0.0,
        "max_tokens": 8,
        "top_p": 0.95,
    }
    out = client._vllm_post_once(payload, verbose=False)
    assert out == "echo:xyz"


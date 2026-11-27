"""Unit tests for curating chain-of-thought files."""

import json

import pytest

from synthetic_data_kit.core import curate
from synthetic_data_kit.utils.config import load_config

REASONING = "Step 1: 120 L. Step 2: 8 L/min. Step 3: 120/8 = 15."


class FakeJudge:
    """Stand-in for LLMClient: rates every pair 9 using the schema of the shipped qa_rating prompt."""

    def __init__(self, *args, **kwargs):
        self.config = load_config(None)

    def batch_completion(self, messages, **kwargs):
        return ['[{"question":"Q1","answer":"15 minutes","rating":9}]' for _ in messages]


@pytest.mark.unit
def test_curate_keeps_reasoning_of_cot_examples(tmp_path, monkeypatch):
    """The judge only returns question, answer and rating; the kept CoT examples must still carry their reasoning."""
    monkeypatch.setattr(curate, "LLMClient", FakeJudge)
    source = tmp_path / "in.json"
    target = tmp_path / "out.json"
    source.write_text(json.dumps({
        "summary": "s",
        "cot_examples": [{"question": "Q1", "reasoning": REASONING, "answer": "15 minutes"}],
    }))

    curate.curate_qa_pairs(str(source), str(target), threshold=7.0)

    result = json.loads(target.read_text())
    assert result["cot_examples"][0]["reasoning"] == REASONING
    assert result["cot_examples"][0]["rating"] == 9
    assistant = result["conversations"][0][-1]["content"]
    assert REASONING in assistant
    assert assistant.endswith("So the answer is: 15 minutes")


@pytest.mark.unit
def test_curate_qa_pairs_output_is_unchanged(tmp_path, monkeypatch):
    """A plain QA file gets no reasoning field and the plain conversation format."""
    monkeypatch.setattr(curate, "LLMClient", FakeJudge)
    source = tmp_path / "in.json"
    target = tmp_path / "out.json"
    source.write_text(json.dumps({"summary": "s", "qa_pairs": [{"question": "Q1", "answer": "15 minutes"}]}))

    curate.curate_qa_pairs(str(source), str(target), threshold=7.0)

    result = json.loads(target.read_text())
    assert result["qa_pairs"] == [{"question": "Q1", "answer": "15 minutes", "rating": 9}]
    assert result["conversations"][0][-1]["content"] == "15 minutes"

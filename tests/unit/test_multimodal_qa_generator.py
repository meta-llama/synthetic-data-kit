"""Unit tests for Multimodal QA Generator count behavior."""

import json
from unittest.mock import MagicMock

import pytest

from synthetic_data_kit.generators.multimodal_qa_generator import MultimodalQAGenerator


@pytest.mark.unit
def test_multimodal_generate_qa_pairs_exact_count(patch_config):
    """Ensure the multimodal generator returns exactly the requested number of pairs."""
    mock_client = MagicMock()
    # Provide a concrete config dict since MultimodalQAGenerator uses client.config when no config_path is passed
    mock_client.config = {
        "generation": {
            "chunk_size": 4000,
            "overlap": 200,
            "batch_size": 32,
            "temperature": 0.7,
            "max_rounds": 2,
            "max_per_chunk_cap": 8,
        }
    }

    # Provide more pairs than requested in a single response
    pairs = [
        {"question": f"Q{i}?", "answer": f"A{i}."}
        for i in range(10)
    ]
    # The generator will call batch_completion once for one chunk in this test
    mock_client.batch_completion.return_value = [json.dumps(pairs)]

    generator = MultimodalQAGenerator(client=mock_client)

    documents = [{"text": "This is a short passage."}]  # no image
    result = generator.generate_qa_pairs(documents, num_pairs=5, verbose=False)

    assert len(result) == 5
    assert all("question" in p and "answer" in p for p in result)

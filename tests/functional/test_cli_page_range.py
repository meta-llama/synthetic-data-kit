import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app


@pytest.mark.functional
def test_ingest_forwards_page_range_single_file(patch_config, test_env):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        input_path = f.name
    try:
        captured = {}
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_proc:
            def _mock(*args, **kwargs):
                captured.update(kwargs)
                out_dir = os.path.dirname(input_path)
                return os.path.join(out_dir, "out.lance")
            mock_proc.side_effect = _mock
            result = runner.invoke(app, [
                "ingest",
                input_path,
                "--page-range",
                "[2,5]",
            ])
        assert result.exit_code == 0
        assert captured.get("page_range") == (2, 5)
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_create_forwards_page_range_single_file(patch_config, test_env):
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        input_path = f.name
    try:
        captured = {}
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = ["mock-model"]
            mock_get.return_value = mock_resp

            with patch("synthetic_data_kit.core.create.process_file") as mock_proc:
                def _mock(*args, **kwargs):
                    captured.update(kwargs)
                    out_dir = os.path.dirname(input_path)
                    return os.path.join(out_dir, "out.json")
                mock_proc.side_effect = _mock

                result = runner.invoke(app, [
                    "create",
                    input_path,
                    "--type",
                    "qa",
                    "--provider",
                    "vllm",
                    "--page-range",
                    "10-12",
                ])
        assert result.exit_code == 0
        assert captured.get("page_range") == (10, 12)
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

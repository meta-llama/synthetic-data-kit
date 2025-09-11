import os
import tempfile
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app


@pytest.mark.functional
def test_ingest_cli_forwards_page_range(patch_config):
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        input_path = f.name

    try:
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
            mock_process.return_value = os.path.join(os.path.dirname(input_path), "out.lance")

            result = runner.invoke(
                app,
                [
                    "ingest",
                    input_path,
                    "--page-range",
                    "[10,12]",
                ],
            )

            assert result.exit_code == 0
            # Ensure kwarg propagated
            assert mock_process.called
            assert mock_process.call_args.kwargs.get("page_range") == (10, 12)
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_create_cli_forwards_page_range_qa(patch_config, test_env):
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        input_path = f.name

    try:
        with patch("synthetic_data_kit.core.create.process_file") as mock_process:
            mock_process.return_value = os.path.join(os.path.dirname(input_path), "out.json")

            result = runner.invoke(
                app,
                [
                    "create",
                    input_path,
                    "--type",
                    "qa",
                    "--page-range",
                    "5-7",
                ],
            )

            assert result.exit_code == 0
            assert mock_process.called
            assert mock_process.call_args.kwargs.get("page_range") == (5, 7)
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)


@pytest.mark.functional
def test_create_cli_forwards_page_range_cot(patch_config, test_env):
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        input_path = f.name

    try:
        with patch("synthetic_data_kit.core.create.process_file") as mock_process:
            mock_process.return_value = os.path.join(os.path.dirname(input_path), "out.json")

            result = runner.invoke(
                app,
                [
                    "create",
                    input_path,
                    "--type",
                    "cot",
                    "--page-range",
                    "[1,1]",
                ],
            )

            assert result.exit_code == 0
            assert mock_process.called
            assert mock_process.call_args.kwargs.get("page_range") == (1, 1)
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

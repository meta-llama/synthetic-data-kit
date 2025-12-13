import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app

REAL_PDF = "/home/w-ds-026/Desktop/taqrir_halat_alsiyasat.pdf"


@pytest.mark.functional
@pytest.mark.skipif(not os.path.exists(REAL_PDF), reason="Real PDF path not available on this machine")
def test_ingest_with_real_pdf_and_page_range(patch_config):
    runner = CliRunner()

    with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
        mock_process.return_value = os.path.join(os.path.dirname(REAL_PDF), "out.lance")

        result = runner.invoke(
            app,
            [
                "ingest",
                REAL_PDF,
                "--page-range",
                "[100,115]",
            ],
        )

        assert result.exit_code == 0
        assert mock_process.called
        assert mock_process.call_args.kwargs.get("page_range") == (100, 115)

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from synthetic_data_kit.parsers.pdf_parser import PDFParser


@pytest.mark.unit
@patch("pdfminer.high_level.extract_text")
@patch("synthetic_data_kit.parsers.pdf_parser.requests.get")
def test_pdfparser_page_range_local(mock_requests_get, mock_extract_text):
    # Local file path
    parser = PDFParser()
    mock_extract_text.return_value = "page text"

    result = parser.parse("/tmp/example.pdf", page_range=(2, 5))

    # pdfminer expects 0-based page numbers, inclusive range 2..5 -> {1,2,3,4}
    mock_extract_text.assert_called_once()
    kwargs = mock_extract_text.call_args.kwargs
    assert kwargs.get("page_numbers") == set([1, 2, 3, 4])
    assert result and isinstance(result, list)


@pytest.mark.unit
@patch("pdfminer.high_level.extract_text")
@patch("synthetic_data_kit.parsers.pdf_parser.requests.get")
def test_pdfparser_page_range_url(mock_requests_get, mock_extract_text):
    # URL file path
    parser = PDFParser()
    mock_extract_text.return_value = "url text"

    # Mock download stream
    mock_resp = MagicMock()
    mock_resp.iter_content = lambda chunk_size: [b"%PDF-1.4", b"..."]
    mock_resp.raise_for_status = lambda: None
    mock_requests_get.return_value = mock_resp

    result = parser.parse("https://example.com/doc.pdf", page_range=(1, 3))

    # 1..3 -> {0,1,2}
    assert mock_extract_text.called
    kwargs = mock_extract_text.call_args.kwargs
    assert kwargs.get("page_numbers") == set([0, 1, 2])
    assert result and isinstance(result, list)


@pytest.mark.unit
def test_pdfparser_invalid_page_range():
    parser = PDFParser()
    with pytest.raises(ValueError):
        parser.parse("/tmp/example.pdf", page_range=(0, 2))
    with pytest.raises(ValueError):
        parser.parse("/tmp/example.pdf", page_range=(3, 2))


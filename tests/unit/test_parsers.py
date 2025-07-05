"""Unit tests for document parsers."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.parsers.html_parser import HTMLParser
from synthetic_data_kit.parsers.pdf_parser import PDFParser
from synthetic_data_kit.parsers.txt_parser import TXTParser


@pytest.mark.unit
def test_txt_parser():
    """Test TXT parser."""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
        f.write("This is sample text content for testing.")
        file_path = f.name

    try:
        # Initialize parser
        parser = TXTParser()

        # Parse the file
        content = parser.parse(file_path)

        # Check that content was extracted correctly
        assert content == "This is sample text content for testing."

        # Test saving content
        output_path = os.path.join(tempfile.gettempdir(), "output.txt")
        parser.save(content, output_path)

        # Check that the file was saved correctly
        with open(output_path) as f:
            saved_content = f.read()

        assert saved_content == content
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.unlink(file_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_html_parser():
    """Test HTML parser."""
    # Create a temporary HTML file
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
    </head>
    <body>
        <h1>Test Heading</h1>
        <p>This is sample HTML content for testing.</p>
    </body>
    </html>
    """
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".html", delete=False) as f:
        f.write(html_content)
        file_path = f.name

    output_path = os.path.join(tempfile.gettempdir(), "output.txt")

    try:
        # Mock bs4.BeautifulSoup (since it's imported inside the method)
        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()
            mock_soup.get_text.return_value = (
                "Test Heading\nThis is sample HTML content for testing."
            )
            mock_bs.return_value = mock_soup

            # Initialize parser
            parser = HTMLParser()

            # Parse the file
            content = parser.parse(file_path)

            # Check that BeautifulSoup was called
            mock_bs.assert_called_once()

            # Check that content extraction method was called
            mock_soup.get_text.assert_called_once()

            # Test saving content
            parser.save(content, output_path)

            # Check that the file was saved correctly
            with open(output_path) as f:
                saved_content = f.read()

            assert saved_content == content
    finally:
        # Clean up
        if os.path.exists(file_path):
            os.unlink(file_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


@pytest.mark.unit
def test_pdf_parser():
    """Test PDF parser."""
    # Mock pdfminer.high_level.extract_text (since it's imported inside the method)
    with patch("pdfminer.high_level.extract_text") as mock_extract:
        mock_extract.return_value = "This is sample PDF content for testing."

        # Create a dummy file path
        file_path = "/dummy/path/to/file.pdf"

        # Initialize parser
        parser = PDFParser()

        # Parse the file
        content = parser.parse(file_path)

        # Check that extract_text was called
        mock_extract.assert_called_once_with(file_path)

        # Check that content matches our mock return value
        assert content == "This is sample PDF content for testing."

        # Test saving content
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.txt")
            parser.save(content, output_path)

            # Check that the file was saved correctly
            with open(output_path) as f:
                saved_content = f.read()

            assert saved_content == content

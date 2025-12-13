# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PDF parser logic
import os
import tempfile
import requests
import logging
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse


class PDFParser:
    """Parser for PDF documents"""

    def parse(self, file_path: str, page_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """Parse a PDF file into plain text

        Args:
            file_path: Path to the PDF file
            page_range: Optional inclusive 1-based page range tuple (start, end)

        Returns:
            Extracted text from the PDF
        """
        # Suppress pdfminer warnings
        logging.getLogger('pdfminer').setLevel(logging.ERROR)
        
        try:
            from pdfminer.high_level import extract_text
        except ImportError:
            raise ImportError(
                "pdfminer.six is required for PDF parsing. Install it with: pip install pdfminer.six"
            )

        # Prepare page_numbers set for pdfminer (0-based, end exclusive in range)
        page_numbers = None
        if page_range is not None:
            try:
                start, end = page_range
                # Convert to 0-based, keep end inclusive by using range(..., end)
                # pdfminer expects a set of 0-based page numbers
                if start < 1 or end < 1 or end < start:
                    raise ValueError("Invalid page_range, expected [start,end] with start>=1 and end>=start")
                page_numbers = set(range(start - 1, end))
            except Exception as e:
                raise ValueError(f"Invalid page_range provided: {page_range}. Error: {e}")

        if file_path.startswith(("http://", "https://")):
            # Download PDF to temporary file
            response = requests.get(file_path, stream=True)
            response.raise_for_status()  # Raise error for bad status codes

            # Create temp file with .pdf extension to help with mime type detection
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_path = temp_file.name
                # Write PDF content to temp file
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)

            try:
                # Parse the downloaded PDF
                if page_numbers is not None:
                    text = extract_text(temp_path, page_numbers=page_numbers)
                else:
                    text = extract_text(temp_path)
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        else:
            # Handle local files as before
            if page_numbers is not None:
                text = extract_text(file_path, page_numbers=page_numbers)
            else:
                text = extract_text(file_path)
        return [{"text": text}]

    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a file

        Args:
            content: Extracted text content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

import os
import sys
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import importlib

from synthetic_data_kit.utils.config import get_path_config


def _check_pdf_url(url: str) -> bool:
    """Check if `url` points to PDF content

    Args:
        url: URL to check


    Returns:
        bool: True if the URL points to PDF content, False otherwise
    """
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get("Content-Type", "")
        return "application/pdf" in content_type
    except requests.RequestException:
        return False


def determine_parser(file_path: str, config: Dict[str, Any], multimodal: bool = False):
    """Determine the appropriate parser for a file or URL"""
    from synthetic_data_kit.parsers.pdf_parser import PDFParser
    from synthetic_data_kit.parsers.html_parser import HTMLParser
    from synthetic_data_kit.parsers.youtube_parser import YouTubeParser
    from synthetic_data_kit.parsers.docx_parser import DOCXParser
    from synthetic_data_kit.parsers.ppt_parser import PPTParser
    from synthetic_data_kit.parsers.txt_parser import TXTParser
    from synthetic_data_kit.parsers.multimodal_parser import MultimodalParser

    ext = os.path.splitext(file_path)[1].lower()
    if multimodal:
        if ext in [".pdf", ".docx", ".pptx"]:
            return MultimodalParser()
        else:
            raise ValueError(f"Unsupported file extension for multimodal parsing: {ext}")

    if ext == ".pdf":
        return PDFParser()

    # Check if it's a URL
    if file_path.startswith(("http://", "https://")):
        # YouTube URL
        if "youtube.com" in file_path or "youtu.be" in file_path:
            return YouTubeParser()
        # PDF URL
        elif _check_pdf_url(file_path):
            return MultimodalParser() if multimodal else PDFParser()
        # HTML URL
        else:
            return HTMLParser()

    # File path - determine by extension
    if os.path.exists(file_path):
        parsers = {
            ".html": HTMLParser(),
            ".htm": HTMLParser(),
            ".docx": DOCXParser(),
            ".pptx": PPTParser(),
            ".txt": TXTParser(),
        }

        if ext in parsers:
            return parsers[ext]
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    raise FileNotFoundError(f"File not found: {file_path}")


def process_file(
    file_path: str,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    multimodal: bool = False,
    page_range: Optional[Tuple[int, int]] = None,
) -> str:
    """Process a file using the appropriate parser

    Args:
        file_path: Path to the file or URL to parse
        output_dir: Directory to save parsed text (if None, uses config)
        output_name: Custom filename for output (if None, uses original name)
        config: Configuration dictionary (if None, uses default)
        multimodal: Whether to use the multimodal parser

    Returns:
        Path to the output file
    """
    from synthetic_data_kit.utils.lance_utils import create_lance_dataset
    import pyarrow as pa
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Determine parser based on file type
    parser = determine_parser(file_path, config, multimodal)

    # Parse the file
    # If PDF parser supports page_range, pass it through
    try:
        content = parser.parse(file_path, page_range=page_range)  # type: ignore[arg-type]
    except TypeError:
        content = parser.parse(file_path)

    # Generate output filename if not provided
    if not output_name:
        if file_path.startswith(("http://", "https://")):
            # Extract filename from URL
            if "youtube.com" in file_path or "youtu.be" in file_path:
                # Use video ID for YouTube URLs
                import re

                video_id = re.search(r"(?:v=|\.be/)([^&]+)", file_path).group(1)
                output_name = f"youtube_{video_id}"
            else:
                # Use domain for other URLs
                from urllib.parse import urlparse

                domain = urlparse(file_path).netloc.replace(".", "_")
                output_name = f"{domain}"
        else:
            # Use original filename
            base_name = os.path.basename(file_path)
            output_name = os.path.splitext(base_name)[0]

    output_name += ".lance"
    output_path = os.path.join(output_dir, output_name)

    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("image", pa.binary())
    ]) if multimodal else pa.schema([
        pa.field("text", pa.string())
    ])

    create_lance_dataset(content, output_path, schema=schema)


    return output_path

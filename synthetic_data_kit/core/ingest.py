# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Ingest different file formats

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import importlib

from synthetic_data_kit.utils.config import get_path_config

def determine_parser(file_path: str, config: Dict[str, Any]):
    """Determine the appropriate parser for a file or URL"""
    from synthetic_data_kit.parsers.pdf_parser import PDFParser
    from synthetic_data_kit.parsers.html_parser import HTMLParser
    from synthetic_data_kit.parsers.youtube_parser import YouTubeParser
    from synthetic_data_kit.parsers.docx_parser import DOCXParser
    from synthetic_data_kit.parsers.ppt_parser import PPTParser
    from synthetic_data_kit.parsers.txt_parser import TXTParser
    
    # Check if it's a URL
    if file_path.startswith(('http://', 'https://')):
        # YouTube URL
        if 'youtube.com' in file_path or 'youtu.be' in file_path:
            return YouTubeParser()
        # HTML URL
        else:
            return HTMLParser()
    
    # File path - determine by extension
    if os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        parsers = {
            '.pdf': PDFParser(),
            '.html': HTMLParser(),
            '.htm': HTMLParser(),
            '.docx': DOCXParser(),
            '.pptx': PPTParser(),
            '.txt': TXTParser(),
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
) -> str:
    """Process a file using the appropriate parser
    
    Args:
        file_path: Path to the file or URL to parse
        output_dir: Directory to save parsed text (if None, uses config)
        output_name: Custom filename for output (if None, uses original name)
        config: Configuration dictionary (if None, uses default)
    
    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine parser based on file type
    parser = determine_parser(file_path, config)
    
    # Parse the file
    content = parser.parse(file_path)
    
    # Generate output filename if not provided
    if not output_name:
        if file_path.startswith(('http://', 'https://')):
            # Extract filename from URL
            if 'youtube.com' in file_path or 'youtu.be' in file_path:
                # Use video ID for YouTube URLs
                import re
                video_id = re.search(r'(?:v=|\.be/)([^&]+)', file_path).group(1)
                output_name = f"youtube_{video_id}.lance"
            else:
                # Use domain for other URLs
                from urllib.parse import urlparse
                domain = urlparse(file_path).netloc.replace('.', '_')
                output_name = f"{domain}.lance"
        else:
            # Use original filename with .lance extension
            base_name = os.path.basename(file_path)
            output_name = os.path.splitext(base_name)[0] + '.lance'
    
    # Ensure .lance extension
    if not output_name.endswith('.lance'):
        output_name += '.lance'
    
    # Save the content
    output_path = os.path.join(output_dir, output_name)
    parser.save(content, output_path)
    
    return output_path

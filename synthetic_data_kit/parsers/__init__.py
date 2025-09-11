# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Document parsers for different file formats
from synthetic_data_kit.parsers.pdf_parser import PDFParser
from synthetic_data_kit.parsers.html_parser import HTMLParser
from synthetic_data_kit.parsers.youtube_parser import YouTubeParser
from synthetic_data_kit.parsers.docx_parser import DOCXParser
from synthetic_data_kit.parsers.ppt_parser import PPTParser
from synthetic_data_kit.parsers.txt_parser import TXTParser
from synthetic_data_kit.parsers.multimodal_parser import MultimodalParser
import os
from typing import List, Dict, Any, Optional, Tuple


def get_parser_for_file(file_path: str):
    """Get the appropriate parser for a file based on its extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        Parser instance for the file type
    """
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.pdf':
        return PDFParser()
    elif ext in ['.html', '.htm']:
        return HTMLParser()
    elif ext == '.txt':
        return TXTParser()
    elif ext in ['.docx', '.doc']:
        return DOCXParser()
    elif ext in ['.pptx', '.ppt']:
        return PPTParser()
    elif ext == '.lance':
        return MultimodalParser()
    else:
        # Default to TXT parser for unknown extensions
        return TXTParser()


def parse_file(file_path: str, page_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    """Parse a file using the appropriate parser
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of dictionaries containing parsed content
    """
    parser = get_parser_for_file(file_path)
    # Try passing page_range for parsers that support it (e.g., PDFParser)
    try:
        return parser.parse(file_path, page_range=page_range)  # type: ignore[call-arg]
    except TypeError:
        return parser.parse(file_path)
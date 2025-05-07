# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# HTML Parsers

import os
import requests
import lance
import pyarrow as pa
from typing import Dict, Any
from urllib.parse import urlparse

class HTMLParser:
    """Parser for HTML files and web pages"""
    
    def parse(self, file_path: str) -> str:
        """Parse an HTML file or URL into plain text
        
        Args:
            file_path: Path to the HTML file or URL
            
        Returns:
            Extracted text from the HTML
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML parsing. Install it with: pip install beautifulsoup4")
        
        # Determine if file_path is a URL or a local file
        if file_path.startswith(('http://', 'https://')):
            # It's a URL, fetch content
            response = requests.get(file_path)
            response.raise_for_status()
            html_content = response.text
        else:
            # It's a local file, read it
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a Lance file
        
        Args:
            content: Extracted text content
            output_path: Path to save the Lance file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Create a PyArrow table
        data = [pa.array([content])]
        names = ['text']
        table = pa.Table.from_arrays(data, names=names)
        
        # Write to Lance file
        lance.write_dataset(table, output_path, mode="overwrite")
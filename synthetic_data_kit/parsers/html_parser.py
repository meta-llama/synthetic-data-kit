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

import base64
from urllib.parse import urljoin
from pathlib import Path

class HTMLParser:
    """Parser for HTML files and web pages"""

    def parse(self, file_path: str, multimodal: bool = False) -> any:
        """Parse an HTML file or URL.
        
        Args:
            file_path: Path to the HTML file or URL.
            multimodal: If True, extract text chunks and associated images.
                        Otherwise, extract all text into a single string.
            
        Returns:
            If multimodal is False, returns a string with all extracted text.
            If multimodal is True, returns a list of dictionaries, where each
            dictionary has 'text' and 'image' keys.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML parsing. Install it with: pip install beautifulsoup4")

        base_url = None
        if file_path.startswith(('http://', 'https://')):
            base_url = file_path
            response = requests.get(file_path)
            response.raise_for_status()
            html_content = response.text
        else:
            # For local files, the base_url is the directory path
            base_url = Path(file_path).parent.as_uri() + '/' # Ensure it acts like a URL base
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')

        if not multimodal:
            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        else:
            data_entries = []
            
            # Text tags to consider
            text_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'span', 'blockquote']
            for tag_name in text_tags:
                for tag in soup.find_all(tag_name):
                    text = tag.get_text(separator=' ', strip=True)
                    if text:
                        data_entries.append({'text': text, 'image': None})
            
            # Image tags
            for img_tag in soup.find_all('img'):
                alt_text = img_tag.get('alt', '')
                src = img_tag.get('src')
                image_bytes = None

                if src:
                    try:
                        if src.startswith('data:image'):
                            # Data URI
                            header, encoded = src.split(',', 1)
                            image_bytes = base64.b64decode(encoded)
                        elif src.startswith(('http://', 'https://')):
                            # Absolute URL
                            img_response = requests.get(src, stream=True, timeout=5)
                            img_response.raise_for_status()
                            image_bytes = img_response.content
                        else:
                            # Relative URL
                            full_url = urljoin(base_url, src)
                            img_response = requests.get(full_url, stream=True, timeout=5)
                            img_response.raise_for_status()
                            image_bytes = img_response.content
                    except Exception: # Catch requests errors, base64 errors, etc.
                        image_bytes = None 
                
                data_entries.append({'text': alt_text, 'image': image_bytes})
            
            return data_entries

    def save(self, content: any, output_path: str) -> None:
        """Save the extracted content to a Lance file.
        
        Args:
            content: Extracted content (string or list of dicts)
            output_path: Path to save the Lance file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if isinstance(content, str):
            # Text-only mode
            data = [pa.array([content])]
            names = ['text']
            table = pa.Table.from_arrays(data, names=names)
        elif isinstance(content, list):
            # Multimodal mode
            texts = pa.array([item['text'] for item in content], type=pa.string())
            # Handle potential None values for images and ensure bytes for PyArrow binary type
            images_data = []
            for item in content:
                img_bytes = item.get('image')
                images_data.append(img_bytes if isinstance(img_bytes, bytes) else None)
            images = pa.array(images_data, type=pa.binary())
            
            schema = pa.schema([
                ('text', pa.string()),
                ('image', pa.binary())
            ])
            table = pa.Table.from_arrays([texts, images], schema=schema)
        else:
            raise ValueError("Unsupported content type for saving.")
            
        lance.write_dataset(table, output_path, mode="overwrite")
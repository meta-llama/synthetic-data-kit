# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PDF parser logic
import os
import lance
import pyarrow as pa
from typing import Dict, Any

import io
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams, LTImage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

class PDFParser:
    """Parser for PDF documents"""

    def parse(self, file_path: str, multimodal: bool = False) -> any:
        """Parse a PDF file.
        
        Args:
            file_path: Path to the PDF file
            multimodal: If True, extract text and images page by page. 
                        Otherwise, extract text only.
            
        Returns:
            If multimodal is False, returns a string with extracted text.
            If multimodal is True, returns a list of dictionaries, where each 
            dictionary has 'text' (text from a page) and 'image' (first image 
            bytes from that page, or None).
        """
        try:
            from pdfminer.high_level import extract_text, extract_pages
        except ImportError:
            raise ImportError("pdfminer.six is required for PDF parsing. Install it with: pip install pdfminer.six")

        if not multimodal:
            return extract_text(file_path)
        else:
            output_data = []
            resource_manager = PDFResourceManager(caching=True)
            
            for page_layout in extract_pages(file_path):
                # Extract text from the page
                text_output = io.StringIO()
                converter = TextConverter(resource_manager, text_output, laparams=LAParams())
                interpreter = PDFPageInterpreter(resource_manager, converter)
                interpreter.process_page(page_layout)
                page_text = text_output.getvalue()
                converter.close()
                text_output.close()

                # Extract first image from the page
                first_image_bytes = None
                for element in page_layout:
                    if isinstance(element, LTImage):
                        try:
                            # Accessing the stream and getting raw data
                            if hasattr(element, 'stream') and hasattr(element.stream, 'get_rawdata'):
                                first_image_bytes = element.stream.get_rawdata()
                                break # Found the first image
                        except Exception:
                            # Handle cases where image data might be corrupted or inaccessible
                            first_image_bytes = None # Ensure it's None if extraction fails
                            break 
                
                output_data.append({
                    'text': page_text,
                    'image': first_image_bytes
                })
            return output_data

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
            # Handle potential None values for images before creating the array
            images_data = []
            for item in content:
                img = item.get('image') # Use .get() for safety
                images_data.append(img if img is not None else None) # Explicitly append None
            images = pa.array(images_data, type=pa.binary())
            
            schema = pa.schema([
                ('text', pa.string()),
                ('image', pa.binary())
            ])
            table = pa.Table.from_arrays([texts, images], schema=schema)
        else:
            raise ValueError("Unsupported content type for saving.")
            
        lance.write_dataset(table, output_path, mode="overwrite")
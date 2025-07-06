# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PDF parser logic
import os
import lance
import pyarrow as pa
import tempfile
import requests
from typing import Dict, Any
from urllib.parse import urlparse


import io
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams, LTImage
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from PIL import Image

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
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF parsing. Install it with: pip install PyPDF2")

        try:
            reader = PdfReader(file_path)
            
            if not multimodal:
                # Text-only mode: combine all pages
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            else:
                # Multimodal mode: process page by page
                output_data = []
                
                for page in reader.pages:
                    # Extract text
                    page_text = page.extract_text()
                    
                    # Extract all images
                    images = []
                    if '/Resources' in page and '/XObject' in page['/Resources']:
                        xObject = page['/Resources']['/XObject'].get_object()
                        for obj in xObject:
                            if xObject[obj]['/Subtype'] == '/Image':
                                try:
                                    img_obj = xObject[obj]
                                    data = img_obj.get_data()
                                    if '/Filter' in img_obj:
                                        if img_obj['/Filter'] == '/DCTDecode':
                                            images.append(data)
                                        elif img_obj['/Filter'] == '/FlateDecode':
                                            img_bytes = self.extract_image_from_pdf_image_obj(img_obj)
                                            if img_bytes is not None:
                                                images.append(img_bytes)
                                        else:
                                            print(f"Unsupported image filter: {img_obj['/Filter']}")
                                    else:
                                        print("No filter found for image")
                                except Exception as e:
                                    print(f"Warning: Error extracting image: {str(e)}")
                                    continue
                    # If no images, still add a row with image=None
                    if images:
                        for img in images:
                            output_data.append({
                                'text': page_text,
                                'image': img
                            })
                    else:
                        output_data.append({
                            'text': page_text,
                            'image': None
                        })
                
                return output_data
                
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise

    def save(self, content: any, output_path: str) -> None:
        """Save the extracted content to a Lance file.
        
        Args:
            content: Extracted content (string or list of dicts)
            output_path: Path to save the Lance file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if isinstance(content, str):
            data = [pa.array([content])]
            names = ['text']
            table = pa.Table.from_arrays(data, names=names)
        elif isinstance(content, list):
            texts = pa.array([item['text'] for item in content], type=pa.string())
            # Handle potential None values for images before creating the array
            images_data = []
            for item in content:
                img = item.get('image')
                images_data.append(img if img is not None else None)
            images = pa.array(images_data, type=pa.binary())
            
            schema = pa.schema([
                ('text', pa.string()),
                ('image', pa.binary())
            ])
            table = pa.Table.from_arrays([texts, images], schema=schema)
        else:
            raise ValueError("Unsupported content type for saving.")
            
        lance.write_dataset(table, output_path, mode="overwrite")

    def extract_image_from_pdf_image_obj(self, img_obj):
        data = img_obj.get_data()
        width = img_obj['/Width']
        height = img_obj['/Height']
        color_space = img_obj['/ColorSpace']
        bits_per_component = img_obj['/BitsPerComponent']

        # Handle color space
        if isinstance(color_space, list) and color_space[0] == '/ICCBased':
            # Try to treat as RGB (most common)
            print("ICCBased color space detected, treating as RGB for extraction.")
            mode = "RGB"
        elif color_space == '/DeviceRGB':
            mode = "RGB"
        elif color_space == '/DeviceGray':
            mode = "L"
        else:
            print(f"Unsupported color space: {color_space}")
            return None

        # Create image from raw bytes
        try:
            image = Image.frombytes(mode, (width, height), data)
            # Save to PNG bytes
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
        except Exception as e:
            print(f"Error reconstructing image: {e}")
            return None

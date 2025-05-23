# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# DOCX parasers
import os
import lance
import pyarrow as pa
from typing import Dict, Any

class DOCXParser:
    """Parser for Microsoft Word documents"""

    def parse(self, file_path: str, multimodal: bool = False) -> any:
        """Parse a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            multimodal: If True, extract text and images. Otherwise, extract text only.
            
        Returns:
            If multimodal is False, returns a string with extracted text.
            If multimodal is True, returns a list of dictionaries, where each dictionary
            has 'text' and 'image' keys.
        """
        try:
            import docx
            from docx.document import Document
            from docx.image.image import Image as DocxImage # To avoid conflict with PIL.Image
        except ImportError:
            raise ImportError("python-docx is required for DOCX parsing. Install it with: pip install python-docx")

        doc: Document = docx.Document(file_path)
        
        if not multimodal:
            # Extract text from paragraphs
            paragraphs_text = [p.text for p in doc.paragraphs]
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        paragraphs_text.append(cell.text)
            
            return "\n\n".join(p for p in paragraphs_text if p)
        else:
            # Multimodal: Extract text and images
            output_data = []
            
            # Extract all images once (simple association: first image with all text blocks)
            first_image_bytes = None
            if doc.inline_shapes:
                for inline_shape in doc.inline_shapes:
                    if inline_shape.type == 3: # WD_INLINE_SHAPE.PICTURE
                        image: DocxImage = inline_shape.image
                        first_image_bytes = image.blob
                        break # Take the first image
            
            # Process paragraphs
            for p in doc.paragraphs:
                if p.text.strip(): # Only include non-empty paragraphs
                    output_data.append({
                        'text': p.text,
                        'image': first_image_bytes 
                    })

            # Process tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip(): # Only include non-empty cell text
                            output_data.append({
                                'text': cell.text,
                                'image': first_image_bytes
                            })
            
            # If no text was found but an image exists, add it with empty text
            if not output_data and first_image_bytes:
                 output_data.append({
                    'text': "",
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
            texts = pa.array([item['text'] for item in content])
            images = pa.array([item['image'] for item in content], type=pa.binary())
            
            schema = pa.schema([
                ('text', pa.string()),
                ('image', pa.binary())
            ])
            table = pa.Table.from_arrays([texts, images], schema=schema)
        else:
            raise ValueError("Unsupported content type for saving.")
            
        lance.write_dataset(table, output_path, mode="overwrite")
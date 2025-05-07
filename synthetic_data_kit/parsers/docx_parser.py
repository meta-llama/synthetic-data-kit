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
    
    def parse(self, file_path: str) -> str:
        """Parse a DOCX file into plain text
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text from the document
        """
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx is required for DOCX parsing. Install it with: pip install python-docx")
        
        doc = docx.Document(file_path)
        
        # Extract text from paragraphs
        paragraphs = [p.text for p in doc.paragraphs]
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    paragraphs.append(cell.text)
        
        return "\n\n".join(p for p in paragraphs if p)
    
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
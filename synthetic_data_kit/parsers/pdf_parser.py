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

class PDFParser:
    """Parser for PDF documents"""
    
    def parse(self, file_path: str) -> str:
        """Parse a PDF file into plain text
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        try:
            from pdfminer.high_level import extract_text
            return extract_text(file_path)
        except ImportError:
            raise ImportError("pdfminer.six is required for PDF parsing. Install it with: pip install pdfminer.six")
    
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
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# PDF parser logic
import os
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
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(filename=file_path)
            
            return "\n".join([str(elem) for elem in elements])
        except ImportError:
            raise ImportError("unstructured is required for PDF parsing. Install it with: pip install unstructured")
    
    def save(self, content: str, output_path: str) -> None:
        """Save the extracted text to a file
        
        Args:
            content: Extracted text content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
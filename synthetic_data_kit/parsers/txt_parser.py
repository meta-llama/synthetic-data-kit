# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# TXT parsering logic, probably the most minimal
import os
import lance
import pyarrow as pa
from typing import Dict, Any

class TXTParser:
    """Parser for plain text files"""
    
    def parse(self, file_path: str) -> str:
        """Parse a text file
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def save(self, content: str, output_path: str) -> None:
        """Save the text to a Lance file
        
        Args:
            content: Text content
            output_path: Path to save the Lance file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Create a PyArrow table
        data = [pa.array([content])]
        names = ['text']
        table = pa.Table.from_arrays(data, names=names)
        
        # Write to Lance file
        lance.write_dataset(table, output_path, mode="overwrite")
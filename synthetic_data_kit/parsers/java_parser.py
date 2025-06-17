# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Java parser logic using LangChain's LanguageParser

import os
from typing import Dict, Any

class JavaParser:
    """Parser for Java source code using langchain's LanguageParser"""
    
    def parse(self, file_path: str) -> str:
        """
        Parse a java file into structured segments
        
        Args:
            file_path: Path to the java file
        
        Returns:
            Extracts segments from the java file
        """
        try:
            from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
            from langchain_community.document_loaders.parsers.language.java import JavaSegmenter

            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            segmenter = JavaSegmenter(code)
            segments = segmenter.extract_functions_classes()

            return "\n\n".join(segments)
        except ImportError:
            raise ImportError("LangChain and its dependencies are required. Install them with: pip install langchain langchain-community tree_sitter tree_sitter_languages")
        except Exception as e:
            return f"Error parsing Java file: {e}"

    def save(self, content: str, output_path: str) -> None:
        """Save the extracted segments to a file

        Args:
            content: Extracted content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
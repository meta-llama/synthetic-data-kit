# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Text processing utilities
import re
import json
from typing import List, Dict, Any

def split_into_chunks(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """Split text into chunks with optional overlap using hierarchical approach"""
    
    # Ensure overlap is not larger than chunk_size
    overlap = min(overlap, chunk_size // 2)

    if "\n\n" in text:
        # Paragraph-based splitting
        segments = text.split("\n\n")
        join_str = "\n\n"
        overlap_split_str = '. '
    elif "\n" in text:
        # Line-based splitting  
        segments = text.split("\n")
        join_str = "\n"
        overlap_split_str = '. '
    else:
        # Sentence-based splitting - try to find sentences first
        sentences = re.split(r'([.!?])\s+(?=[A-Z])', text)
        segments = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                segments.append(sentence.strip())
        
        # Add remaining part if exists
        if len(sentences) % 2 == 1:
            segments.append(sentences[-1].strip())
        
        # If no proper sentences found, fall back to word splitting
        if len(segments) <= 1 and len(text) > chunk_size:
            segments = text.split(' ')
            join_str = " "
            overlap_split_str = ' '
        else:
            join_str = " "
            overlap_split_str = '. '
    
    chunks = []
    current_chunk = ""
    
    for segment in segments:
        potential_length = len(current_chunk) + (len(join_str) if current_chunk else 0) + len(segment)
        
        if potential_length > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create overlap for next chunk
            if overlap > 0:
                overlap_parts = current_chunk.split(overlap_split_str)
                if len(overlap_parts) > 1:
                    # Keep overlap amount of characters from the end
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    space_pos = overlap_text.find(' ')
                    if space_pos > 0:
                        overlap_text = overlap_text[space_pos + 1:]
                    current_chunk = overlap_text + join_str + segment
                else:
                    current_chunk = segment
            else:
                current_chunk = segment
        else:
            if current_chunk:
                current_chunk += join_str + segment
            else:
                current_chunk = segment
    
    # Add final chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Fallback: if only one chunk and text is longer than chunk_size, force character-based splitting
    if len(chunks) == 1 and len(text) > chunk_size:
        chunks = []
        step_size = max(1, chunk_size - overlap)
        
        for i in range(0, len(text), step_size):
            chunk_end = min(i + chunk_size, len(text))
            chunk = text[i:chunk_end]
            
            # Try to end at word boundary if not at end
            if chunk_end < len(text) and ' ' in chunk:
                last_space = chunk.rfind(' ')
                if last_space > len(chunk) * 0.7:  # Don't lose too much content
                    chunk = chunk[:last_space]
            
            if chunk.strip():
                chunks.append(chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text that might contain markdown or other content"""
    text = text.strip()
    
    # Try to parse as complete JSON
    if text.startswith('{') and text.endswith('}') or text.startswith('[') and text.endswith(']'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    
    # Look for JSON within Markdown code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # Try a more aggressive pattern
    json_pattern = r'\{[\s\S]*\}|\[[\s\S]*\]'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError("Could not extract valid JSON from the response")
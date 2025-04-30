# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for saving file format

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.format_converter import to_jsonl, to_alpaca, to_fine_tuning, to_chatml, to_hf_dataset
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format

def process_file(
    input_path: str,
    output_path: str,
    format_type: str,
    storage_format: str,
    config_path: Optional[Path] = None,
    llm_client: Optional[LLMClient] = None,
) -> str:
    """Convert data to different formats
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the output
        format_type: Output format (jsonl, alpaca, ft, chatml)
        storage_format: Storage format, either "json" or "hf" (Hugging Face dataset)
        config_path: Path to configuration file
        llm_client: Initialized LLMClient instance
    
    Returns:
        Path to the output file or directory
    """
    # Load input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract data based on known structures
    # Try to handle the case where we have QA pairs or conversations
    if "qa_pairs" in data:
        qa_pairs = data.get("qa_pairs", [])
    elif "filtered_pairs" in data:
        qa_pairs = data.get("filtered_pairs", [])
    elif "conversations" in data:
        conversations = data.get("conversations", [])
        qa_pairs = []
        for conv in conversations:
            if len(conv) >= 3 and conv[1]['role'] == 'user' and conv[2]['role'] == 'assistant':
                qa_pairs.append({
                    'question': conv[1]['content'],
                    'answer': conv[2]['content']
                })
    else:
        # If the file is just an array of objects, check if they look like QA pairs
        if isinstance(data, list):
            qa_pairs = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    qa_pairs.append(item)
        else:
            raise ValueError("Unrecognized data format - expected QA pairs or conversations")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # When using HF dataset storage format
    if storage_format == "hf":
        # For HF datasets, we need to prepare the data in the right structure
        if format_type == "jsonl":
            # For JSONL, just use the QA pairs directly
            formatted_pairs = qa_pairs
        elif format_type == "alpaca":
            # Format as Alpaca structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "instruction": pair["question"],
                    "input": "",
                    "output": pair["answer"]
                })
        elif format_type == "ft":
            # Format as OpenAI fine-tuning structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                })
        elif format_type == "chatml":
            # Format as ChatML structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]}
                    ]
                })
        else:
            raise ValueError(f"Unknown format type: {format_type}")
            
        # Save as HF dataset (Arrow format)
        return to_hf_dataset(formatted_pairs, output_path)
    
    # Standard JSON file storage format
    else:
        # Convert to the requested format using existing functions
        if format_type == "jsonl":
            return to_jsonl(qa_pairs, output_path)
        elif format_type == "alpaca":
            return to_alpaca(qa_pairs, output_path)
        elif format_type == "ft":
            return to_fine_tuning(qa_pairs, output_path)
        elif format_type == "chatml":
            return to_chatml(qa_pairs, output_path)
        else:
            raise ValueError(f"Unknown format type: {format_type}")
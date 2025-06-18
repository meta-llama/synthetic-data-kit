# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for saving file format

import os
import json
import lance
import datasets
from pathlib import Path
from typing import Optional, Dict, Any, List
import pyarrow as pa

from synthetic_data_kit.utils.format_converter import to_jsonl, to_alpaca, to_fine_tuning, to_chatml, to_hf_dataset
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format

def convert_format(
    input_path: str,
    output_path: str,
    format_type: str,
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
) -> str:
    """Convert data to different formats
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the output
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf" (Hugging Face dataset)
    
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
    
    # Format the data according to format_type
    if format_type == "jsonl":
        formatted_pairs = qa_pairs
    elif format_type == "alpaca":
        formatted_pairs = []
        for pair in qa_pairs:
            formatted_pairs.append({
                "instruction": pair["question"],
                "input": "",
                "output": pair["answer"]
            })
    elif format_type == "ft":
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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save in the specified storage format
    if storage_format == "lance":
        if not formatted_pairs:
            raise ValueError("No data to save in Lance format.")
        # Determine all keys
        all_keys = set()
        for item in formatted_pairs:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        all_keys = list(all_keys)
        # Build arrays for each key
        arrays = []
        for key in all_keys:
            values = [item.get(key, None) if isinstance(item, dict) else None for item in formatted_pairs]
            # Use string type for all fields except for 'rating' (float) and 'messages' (list/dict)
            if key == "rating":
                arrays.append(pa.array(values, type=pa.float64()))
            elif key == "messages":
                arrays.append(pa.array([json.dumps(v) if v is not None else None for v in values], type=pa.string()))
            else:
                arrays.append(pa.array(values, type=pa.string()))
        table = pa.Table.from_arrays(arrays, names=all_keys)
        if output_path.split("."):
            output_path = output_path.split(".")[0] + ".lance"
        else:
            output_path = output_path + ".lance"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lance.write_dataset(table, output_path, mode="overwrite")
        return output_path
    elif storage_format == "hf":
        return to_hf_dataset(formatted_pairs, output_path)
    else:
        # Standard JSON file storage format
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
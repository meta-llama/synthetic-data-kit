# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Utils for format conversions
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

def to_jsonl(data: List[Dict[str, Any]], output_path: str) -> str:
    """Convert data to JSONL format and save to a file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    return output_path

def to_alpaca(qa_pairs: List[Dict[str, str]], output_path: str) -> str:
    """Convert QA pairs to Alpaca format and save"""
    alpaca_data = []
    
    for pair in qa_pairs:
        alpaca_item = {
            "instruction": pair["question"],
            "input": "",
            "output": pair["answer"]
        }
        alpaca_data.append(alpaca_item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def to_fine_tuning(qa_pairs: List[Dict[str, str]], output_path: str) -> str:
    """Convert QA pairs to fine-tuning format and save"""
    ft_data = []
    
    for pair in qa_pairs:
        ft_item = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]}
            ]
        }
        ft_data.append(ft_item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ft_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def to_chatml(qa_pairs: List[Dict[str, str]], output_path: str) -> str:
    """Convert QA pairs to ChatML format and save as JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            chat = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]}
            ]
            f.write(json.dumps({"messages": chat}) + '\n')
    
    return output_path

def to_hf_dataset(qa_pairs: List[Dict[str, str]], output_path: str) -> str:
    """
    Convert QA pairs to a Hugging Face dataset and save in Arrow format.
    
    Args:
        qa_pairs: List of question-answer dictionaries
        output_path: Directory path to save the dataset
        
    Returns:
        Path to the saved dataset directory
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HF dataset format. "
            "Install it with: pip install datasets"
        )
    
    # Remove file extension if present
    if output_path.endswith(('.json', '.hf')):
        output_path = os.path.splitext(output_path)[0]
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert list of dicts to dict of lists for Dataset.from_dict()
    dict_of_lists = {}
    for key in qa_pairs[0].keys():
        dict_of_lists[key] = [item.get(key, "") for item in qa_pairs]
    
    # Create dataset
    dataset = Dataset.from_dict(dict_of_lists)
    
    # Save dataset in Arrow format
    dataset.save_to_disk(output_path)
    
    return output_path
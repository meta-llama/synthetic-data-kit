# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for generating CoT from scratch and also enhancing CoT (take existing format and add CoT)
import os
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import get_prompt, get_generation_config

class COTGenerator:
    """Generates chain-of-thought reasoning examples"""
    
    def __init__(self, client: LLMClient, config_path: Optional[Path] = None):
        """Initialize the CoT Generator with an LLM client and optional config"""
        self.client = client
        self.config = client.config
        self.generation_config = get_generation_config(self.config)
    
    def parse_json_output(self, output_text: str) -> Optional[List[Dict]]:
        """Parse JSON from LLM output text"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        output_text = output_text.strip()
        
        # Try to extract JSON array
        json_match = re.search(r"\[.*\]", output_text, re.DOTALL)
        if json_match:
            output_text = json_match.group(0)
        
        try:
            # Handle quoted JSON
            if output_text.startswith('"') and output_text.endswith('"'):
                output_text = json.loads(output_text)
            
            # Load the JSON
            result = json.loads(output_text)
            
            # Ensure it's a list
            if not isinstance(result, list):
                if verbose:
                    print("Warning: Expected a list but got another type")
                return None
            
            return result
        except json.JSONDecodeError as e:
            if verbose:
                print(f"Error parsing output: {e}")
            return None
    
    def generate_cot_examples(self, document_text: str, document_image: Optional[str] = None, num_examples: int = 5) -> List[Dict[str, Any]]:
        """Generate chain-of-thought reasoning examples"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Get the prompt template
        prompt_template = get_prompt(self.config, "cot_generation")
        
        # Format the prompt
        prompt = prompt_template.format(
            num_examples=num_examples,
            text=document_text
        )
        
        # Generate examples
        temperature = self.generation_config.get("temperature", 0.7)
        max_tokens = self.generation_config.get("max_tokens", 4096)
        
        if verbose:
            print(f"Generating {num_examples} CoT examples...")
        
        messages = [{"role": "system", "content": prompt}]
        
        # Add image if present
        if document_image:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": document_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{document_image}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": document_text})
        
        response = self.client.chat_completion(
            messages, 
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Parse response
        examples = self.parse_json_output(response)
        
        if examples is None:
            if verbose:
                print("Failed to parse CoT examples, returning empty list")
            return []
        
        if verbose:
            print(f"Successfully generated {len(examples)} CoT examples")
        
        return examples
    
    def enhance_with_cot(self, conversations: List[Dict], include_simple_steps: bool = False) -> List[Dict]:
        """Enhance existing conversations with CoT reasoning"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Get the prompt template
        prompt_template = get_prompt(self.config, "cot_enhancement")
        
        # Format the prompt
        conversation_str = json.dumps(conversations, ensure_ascii=False, indent=2)
        prompt = prompt_template.format(
            conversations=conversation_str,
            include_simple_steps=str(include_simple_steps).lower()
        )
        
        # Generate enhanced conversations
        temperature = self.generation_config.get("temperature", 0.2)
        max_tokens = self.generation_config.get("max_tokens", 4096)
        
        if verbose:
            print(f"Enhancing {len(conversations)} conversations with CoT...")
        
        messages = [{"role": "system", "content": prompt}]
        response = self.client.chat_completion(
            messages, 
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Parse response
        enhanced_conversations = self.parse_json_output(response)
        
        if enhanced_conversations is None:
            if verbose:
                print("Failed to parse enhanced conversations, returning original")
            return conversations
        
        if verbose:
            print(f"Successfully enhanced conversations with CoT")
        
        return enhanced_conversations
    
    def process_document(self, 
                        document_text: str, 
                        document_image: Optional[str] = None,
                        num_examples: int = 5, 
                        include_simple_steps: bool = False) -> Dict[str, Any]:
        """Process a document to generate CoT examples"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'
        
        # Generate summary first (helpful context)
        messages = [
            {"role": "system", "content": "Summarize this document in 2-3 sentences."}
        ]
        
        # Add image if present
        if document_image:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": document_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{document_image}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": document_text})
        
        summary = self.client.chat_completion(messages, temperature=0.1)
        
        # Generate CoT examples
        examples = self.generate_cot_examples(document_text, document_image, num_examples)
        
        # Format into simple conversation format as well
        conversations = []
        for example in examples:
            if "question" in example and "reasoning" in example and "answer" in example:
                conv = [
                    {"role": "system", "content": "You are a helpful assistant that provides detailed explanations."},
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": f"Let me think through this step by step:\n\n{example['reasoning']}\n\nSo the answer is: {example['answer']}"}
                ]
                conversations.append(conv)
        
        # Prepare result
        result = {
            "summary": summary,
            "cot_examples": examples,
            "conversations": conversations
        }
        
        # Print stats
        print(f"Generated {len(examples)} chain-of-thought examples")
        
        return result
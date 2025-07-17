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
    
    def generate_cot_examples(self, document_text: str, num_examples: int = None) -> List[Dict[str, Any]]:
        """Generate chain-of-thought reasoning examples using chunking for large documents"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Get default num_examples from config if not provided
        if num_examples is None:
            num_examples = self.generation_config.get("num_cot_examples", 5)
        
        # For small documents, use single call
        single_call_max_size = self.generation_config.get("single_call_max_size", 8000)
        if len(document_text) < single_call_max_size:
            return self._generate_single_call(document_text, num_examples)
        
        # For large documents, use chunking (same logic as QA generator)
        return self._generate_with_chunking(document_text, num_examples)
    
    def _generate_single_call(self, document_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Generate CoT examples in a single API call"""
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
            print(f"Generating {num_examples} CoT examples (single call)...")
        
        messages = [{"role": "system", "content": prompt}]
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
    
    def _generate_with_chunking(self, document_text: str, num_examples: int) -> List[Dict[str, Any]]:
        """Generate CoT examples using chunking strategy (copied from QA generator)"""
        from synthetic_data_kit.utils.text import split_into_chunks
        
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Get generation config
        chunk_size = self.generation_config.get("chunk_size", 4000)
        temperature = self.generation_config.get("temperature", 0.7)
        overlap = self.generation_config.get("overlap", 200)
        batch_size = self.generation_config.get("batch_size", 32)
        
        # Split text into chunks
        chunks = split_into_chunks(
            document_text, 
            chunk_size=chunk_size, 
            overlap=overlap
        )
        
        if verbose:
            print(f"Generating CoT examples using chunking...")
            print(f"Document split into {len(chunks)} chunks")
            print(f"Using batch size of {batch_size}")
        
        all_examples = []
        examples_per_chunk = max(1, round(num_examples / len(chunks)))
        
        # Get CoT generation prompt template
        cot_prompt_template = get_prompt(self.config, "cot_generation")
        
        # Prepare all message batches
        all_messages = []
        for i, chunk in enumerate(chunks):
            # Format the prompt with text
            cot_prompt = cot_prompt_template.format(
                num_examples=examples_per_chunk,
                text=chunk
            )
            
            messages = [
                {"role": "system", "content": cot_prompt}
            ]
            all_messages.append(messages)
        
        print(f"Processing {len(chunks)} chunks to generate CoT examples...")
        
        # Process in batches (same logic as QA generator)
        for batch_start in range(0, len(chunks), batch_size):
            # Check if we've already generated enough examples
            if len(all_examples) >= num_examples:
                if verbose:
                    print(f"Reached target of {num_examples} examples. Stopping processing.")
                break
                
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_messages = all_messages[batch_start:batch_end]
            current_batch_size = len(batch_messages)
            
            batch_num = batch_start//batch_size + 1
            total_batches = (len(chunks) + batch_size - 1)//batch_size
            
            # Simple progress indicator for non-verbose mode
            if not verbose:
                print(f"Processing batch {batch_num}/{total_batches}...", end="\r")
            else:
                print(f"Processing batch {batch_num}/{total_batches} with {current_batch_size} chunks")
            
            try:
                # Process the batch
                batch_responses = self.client.batch_completion(
                    batch_messages,
                    temperature=temperature,
                    batch_size=batch_size
                )
                
                # Process each response in the batch
                for j, response in enumerate(batch_responses):
                    # Check if we've reached the target before processing more
                    if len(all_examples) >= num_examples:
                        if verbose:
                            print(f"  Reached target of {num_examples} examples. Stopping batch processing.")
                        break
                        
                    chunk_index = batch_start + j
                    chunk_examples = self.parse_json_output(response)
                    
                    if chunk_examples:
                        # Only add examples up to the target limit
                        remaining_examples = num_examples - len(all_examples)
                        if remaining_examples > 0:
                            examples_to_add = chunk_examples[:remaining_examples]
                            all_examples.extend(examples_to_add)
                            
                            if verbose:
                                print(f"  Generated {len(examples_to_add)} examples from chunk {chunk_index+1} (total: {len(all_examples)}/{num_examples})")
                    
                    # Break if we've reached the target
                    if len(all_examples) >= num_examples:
                        break
                
                # Break outer loop if we've reached the target
                if len(all_examples) >= num_examples:
                    break
                
            except Exception as e:
                if verbose:
                    print(f"  Error processing batch {batch_num}: {str(e)}")
        
        # Clear the progress line in non-verbose mode
        if not verbose:
            print(" " * 80, end="\r")
            print("Batch processing complete.")
        
        # Always print summary information
        print(f"Generated {len(all_examples)} CoT examples total (requested: {num_examples})")
        return all_examples
    
    def enhance_with_cot(self, conversations: List[Dict], include_simple_steps: bool = False) -> List[Dict]:
        """Enhance existing conversations with CoT reasoning"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Get the prompt template
        prompt_template = get_prompt(self.config, "cot_enhancement")
        
        if verbose:
            print(f"Debug - Conversations to enhance structure: {type(conversations)}")
            print(f"Debug - First conversation: {json.dumps(conversations[0] if conversations else {}, indent=2)[:100]}...")
        
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
    
    def process_document(self, document_text: str, num_examples: int = None, include_simple_steps: bool = False) -> Dict[str, Any]:
        """Process a document to generate CoT examples"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'
        
        # Generate summary first (helpful context)
        max_context_length = self.generation_config.get("max_context_length", 8000)
        summary = self.client.chat_completion(
            [{"role": "system", "content": "Summarize this document in 2-3 sentences."},
             {"role": "user", "content": document_text[0:max_context_length]}], 
            temperature=0.1
        )
        
        # Generate CoT examples
        examples = self.generate_cot_examples(document_text, num_examples)
        
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

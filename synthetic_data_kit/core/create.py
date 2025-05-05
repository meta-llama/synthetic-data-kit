# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate the content: CoT/QA/Summary Datasets
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator
from synthetic_data_kit.utils.config import get_generation_config

def process_file(
    file_path: str,
    output_dir: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: str = "qa",
    num_pairs: Optional[int] = None,
    verbose: bool = False,
) -> str:
    """Process a file to generate content
    
    Args:
        file_path: Path to the text file to process
        output_dir: Directory to save generated content
        config_path: Path to configuration file
        api_base: VLLM API base URL
        model: Model to use
        content_type: Type of content to generate (qa, summary, cot)
        num_pairs: Target number of QA pairs to generate
        threshold: Quality threshold for filtering (1-10)
    
    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    # The reason for having this directory logic for now is explained in context.py
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Initialize LLM client
    client = LLMClient(
        config_path=config_path,
        api_base=api_base,
        model_name=model
    )
    
    # Generate base filename for output
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Generate content based on type
    if content_type == "qa":
        generator = QAGenerator(client, config_path)
        
        # Get num_pairs from args or config
        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs = generation_config.get("num_pairs", 25)
        
        # Process document
        result = generator.process_document(
            document_text,
            num_pairs=num_pairs,
            verbose=verbose
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
        print(f"Saving result to {output_path}")
        
        # First, let's save a basic test file to confirm the directory is writable
        test_path = os.path.join(output_dir, "test_write.json")
        try:
            with open(test_path, 'w', encoding='utf-8') as f:
                f.write('{"test": "data"}')
            print(f"Successfully wrote test file to {test_path}")
        except Exception as e:
            print(f"Error writing test file: {e}")
            
        # Now save the actual result
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"Successfully wrote result to {output_path}")
        except Exception as e:
            print(f"Error writing result file: {e}")
        
        return output_path
    
    elif content_type == "summary":
        generator = QAGenerator(client, config_path)
        
        # Generate just the summary
        summary = generator.generate_summary(document_text)
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_summary.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary}, f, indent=2)
        
        return output_path
    
    # So there are two separate categories of CoT
    # Simply CoT maps to "Hey I want CoT being generated"
    # CoT-enhance maps to "Please enhance my dataset with CoT"
    
    elif content_type == "cot":
        from synthetic_data_kit.generators.cot_generator import COTGenerator
        
        # Initialize the CoT generator
        generator = COTGenerator(client, config_path)
        
        # Get num_examples from args or config
        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs = generation_config.get("num_pairs", 5)
        
        # Process document to generate CoT examples
        result = generator.process_document(
            document_text,
            num_examples=num_pairs,
            include_simple_steps=verbose  # More detailed if verbose is enabled
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_cot_examples.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        if verbose:
            # Print some example content
            if result.get("cot_examples") and len(result.get("cot_examples", [])) > 0:
                first_example = result["cot_examples"][0]
                print("\nFirst CoT Example:")
                print(f"Question: {first_example.get('question', '')}")
                print(f"Reasoning (first 100 chars): {first_example.get('reasoning', '')[:100]}...")
                print(f"Answer: {first_example.get('answer', '')}")
        
        return output_path
        
    elif content_type == "cot-enhance":
        from synthetic_data_kit.generators.cot_generator import COTGenerator
        from tqdm import tqdm
        
        # Initialize the CoT generator
        generator = COTGenerator(client, config_path)
        
        # Instead of parsing as text, load the file as JSON with conversations
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different dataset formats
            if isinstance(data, dict) and "conversations" in data:
                # Single conversation with a conversations array
                conversations = [data]
                is_single_conversation = True
            elif isinstance(data, list) and all("conversations" in item for item in data if isinstance(item, dict)):
                # Array of conversation objects, each with a conversations array
                conversations = data
                is_single_conversation = False
            elif isinstance(data, list) and all(isinstance(msg, dict) and "from" in msg for msg in data):
                # Direct list of messages for a single conversation
                conversations = [{"conversations": data}]
                is_single_conversation = True
            else:
                # Try to handle as a generic list of conversations
                conversations = data
                is_single_conversation = False
            
            if verbose:
                print(f"Found {len(conversations)} conversation(s) to enhance")
            
            # Process each conversation
            enhanced_conversations = []
            
            for i, conversation in enumerate(tqdm(conversations, desc="Enhancing conversations")):
                # Check if this item has a conversations field
                if isinstance(conversation, dict) and "conversations" in conversation:
                    conv_messages = conversation["conversations"]
                    
                    # Validate messages format
                    if not isinstance(conv_messages, list):
                        print(f"Warning: conversations field is not a list in item {i}, skipping")
                        enhanced_conversations.append(conversation)  # Keep original
                        continue
                    
                    # Enhance this conversation's messages
                    enhanced_messages = generator.enhance_with_cot(conv_messages, include_simple_steps=verbose)
                    
                    # Create enhanced conversation with same structure
                    enhanced_conv = conversation.copy()
                    enhanced_conv["conversations"] = enhanced_messages
                    enhanced_conversations.append(enhanced_conv)
                else:
                    # Not the expected format, just keep original
                    enhanced_conversations.append(conversation)
            
            # Save enhanced conversations
            output_path = os.path.join(output_dir, f"{base_name}_enhanced.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if is_single_conversation and len(enhanced_conversations) == 1:
                    # Save the single conversation
                    json.dump(enhanced_conversations[0], f, indent=2)
                else:
                    # Save the array of conversations
                    json.dump(enhanced_conversations, f, indent=2)
            
            if verbose:
                print(f"Enhanced {len(enhanced_conversations)} conversation(s)")
                
            return output_path
            
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse {file_path} as JSON. For cot-enhance, input must be a valid JSON file.")
    
    else:
        raise ValueError(f"Unknown content type: {content_type}")

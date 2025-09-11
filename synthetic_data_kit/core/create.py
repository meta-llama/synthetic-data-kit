# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate the content: CoT/QA/Summary Datasets
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator
from synthetic_data_kit.generators.vqa_generator import VQAGenerator
from synthetic_data_kit.generators.multimodal_qa_generator import MultimodalQAGenerator

from synthetic_data_kit.utils.config import get_generation_config

from synthetic_data_kit.utils.lance_utils import load_lance_dataset

from synthetic_data_kit.parsers import parse_file


def process_file(
    file_path: str,
    output_dir: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: str = "qa",
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    rolling_summary: Optional[bool] = False,
    difficulty: Optional[str] = None,
    language: str = "english",
    page_range: Optional[Tuple[int, int]] = None,
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
    
    # Initialize LLM client
    client = LLMClient(
        config_path=config_path,
        provider=provider,
        api_base=api_base,
        model_name=model,
        verbose=verbose
    )
    
    # Override chunking config if provided
    if chunk_size is not None:
        client.config.setdefault('generation', {})['chunk_size'] = chunk_size
    if chunk_overlap is not None:
        client.config.setdefault('generation', {})['overlap'] = chunk_overlap
    
    # Generate base filename for output
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Generate content based on type
    if file_path.endswith(".lance"):
        dataset = load_lance_dataset(file_path)
        documents = dataset.to_table().to_pylist()
    else:
        documents = parse_file(file_path, page_range=page_range)

    # Determine target language behavior
    lang_mode = (language or "english").lower()
    target_language = "english" if lang_mode != "source" else "source"

    # Normalize difficulty early
    if difficulty:
        difficulty = difficulty.lower()
        if difficulty not in {"easy", "medium", "advanced"}:
            difficulty = None

    if content_type == "qa":
        generator = QAGenerator(client, config_path, target_language=target_language)

        # Get num_pairs from args or config
        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs = generation_config.get("num_pairs", 25)
        
        # Process document
        result = generator.process_documents(
            documents,
            num_pairs=num_pairs,
            verbose=verbose,
            rolling_summary=rolling_summary,
            difficulty=difficulty,
        )
        
        # Save output
        suffix = f"_{difficulty}" if difficulty else ""
        output_path = os.path.join(output_dir, f"{base_name}{suffix}_qa_pairs.json")
        print(f"Saving result to {output_path}")
            
        # Now save the actual result
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Successfully wrote result to {output_path}")
        except Exception as e:
            print(f"Error writing result file: {e}")
        
        return output_path
    
    elif content_type == "multimodal-qa":
        generator = MultimodalQAGenerator(client, config_path, target_language=target_language)
        output_path = generator.process_dataset(
            documents=documents,
            output_dir=output_dir,
            num_examples=num_pairs,
            verbose=verbose,
            base_name=base_name,
            difficulty=difficulty,
        )
        return output_path

    elif content_type == "vqa":
        generator = VQAGenerator(client, config_path)
        output_path = generator.process_dataset(
            documents=documents,
            output_dir=output_dir,
            num_examples=num_pairs,
            verbose=verbose
        )
        return output_path

    elif content_type == "summary":
        generator = QAGenerator(client, config_path, target_language=target_language)

        full_text = " ".join([doc["text"] for doc in documents])
        
        # Generate just the summary
        summary = generator.generate_summary(full_text)
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_summary.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary}, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    # So there are two separate categories of CoT
    # Simply CoT maps to "Hey I want CoT being generated"
    # CoT-enhance maps to "Please enhance my dataset with CoT"
    
    elif content_type == "cot":
        from synthetic_data_kit.generators.cot_generator import COTGenerator
        
        # Initialize the CoT generator
        generator = COTGenerator(client, config_path, target_language=target_language)

        full_text = " ".join([doc["text"] for doc in documents])
        
        # Get num_examples from args or config
        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs = generation_config.get("num_cot_examples", 5)
        
        # Process document to generate CoT examples
        result = generator.process_document(
            full_text,
            num_examples=num_pairs,
            include_simple_steps=verbose,  # More detailed if verbose is enabled
            difficulty=difficulty,
            verbose=verbose,
        )
        
        # Save output
        suffix = f"_{difficulty}" if difficulty else ""
        output_path = os.path.join(output_dir, f"{base_name}{suffix}_cot_examples.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
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
        generator = COTGenerator(client, config_path, target_language=target_language)

        parsed_content = parse_file(file_path)
        document_text = parsed_content[0]["text"] if parsed_content else ""
        
        # Get max_examples from args or config
        max_examples = None
        if num_pairs is not None:
            max_examples = num_pairs  # If user specified a number, use it
        else:
            config = client.config
            generation_config = get_generation_config(config)
            # Get the config value (will be None by default, meaning enhance all)
            max_examples = generation_config.get("num_cot_enhance_examples")
        
        # Instead of parsing as text, load the file as JSON with conversations
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different dataset formats
            # First, check for QA pairs format (the most common input format)
            if isinstance(data, dict) and "qa_pairs" in data:
                # QA pairs format from "create qa" command (make this the primary format)
                from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format
                
                qa_pairs = data.get("qa_pairs", [])
                if verbose:
                    print(f"Converting {len(qa_pairs)} QA pairs to conversation format")
                
                conv_list = convert_to_conversation_format(qa_pairs)
                # Wrap each conversation in the expected format
                conversations = [{"conversations": conv} for conv in conv_list]
                is_single_conversation = False
            # Then handle other conversation formats for backward compatibility
            elif isinstance(data, dict) and "conversations" in data:
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
            
            # Limit the number of conversations if needed
            if max_examples is not None and len(conversations) > max_examples:
                if verbose:
                    print(f"Limiting to {max_examples} conversations (from {len(conversations)} total)")
                conversations = conversations[:max_examples]
            
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
                    if verbose:
                        print(f"Debug - Conv_messages type: {type(conv_messages)}")
                        print(f"Debug - Conv_messages structure: {conv_messages[:1] if isinstance(conv_messages, list) else 'Not a list'}")
                    
                    # Always include simple steps when enhancing QA pairs
                    enhanced_messages = generator.enhance_with_cot(conv_messages, include_simple_steps=True)
                    
                    # Handle nested bug
                    if enhanced_messages and isinstance(enhanced_messages, list):
                        # Nested bug
                        if enhanced_messages and isinstance(enhanced_messages[0], list):
                            if verbose:
                                print(f"Debug - Flattening nested array response")
                            enhanced_messages = enhanced_messages[0]
                    
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
                    json.dump(enhanced_conversations[0], f, indent=2, ensure_ascii=False)
                else:
                    # Save the array of conversations
                    json.dump(enhanced_conversations, f, indent=2, ensure_ascii=False)
            
            if verbose:
                print(f"Enhanced {len(enhanced_conversations)} conversation(s)")
                
            return output_path
            
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse {file_path} as JSON. For cot-enhance, input must be a valid JSON file.")


    else:
        raise ValueError(f"Unknown content type: {content_type}")
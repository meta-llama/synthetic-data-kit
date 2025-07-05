# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Generate the content: CoT/QA/Summary Datasets
import os
import json
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import lance
import pyarrow as pa

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator
from synthetic_data_kit.generators.vqa_generator import VQAGenerator
from synthetic_data_kit.utils.config import get_generation_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _convert_image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Base64 encoded string of the image
    """
    if image_bytes is None:
        return None
    return base64.b64encode(image_bytes).decode('utf-8')
  
def read_json(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    return document_text


def process_file(
    file_path: str,
    output_dir: str,
    config_path: Optional[Path] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: str = "qa",
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    multimodal: bool = False,
    provider: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> str:
    """Process a file to generate content
    
    Args:
        file_path: Path to the file to process (txt or lance format)
        output_dir: Directory to save generated content
        config_path: Path to configuration file
        api_base: VLLM API base URL
        model: Model to use
        content_type: Type of content to generate (qa, summary, cot)
        num_pairs: Target number of QA pairs to generate
        verbose: Whether to print verbose output
        multimodal: Whether to process multimodal data (text + images)
    
    Returns:
        Path to the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine file type based on extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Initialize document_text variable
    document_text = None
    
    if file_extension == '.lance':
        import lance
        dataset = lance.dataset(file_path)
        # Check if this is a multimodal dataset
        is_multimodal = multimodal and 'text' in dataset.schema.names and 'image' in dataset.schema.names
        
        if is_multimodal:
            if verbose:
                logger.info(f"Processing multimodal dataset: {file_path}")
            
            # For multimodal datasets, process each text block separately
            batch = dataset.to_table().to_pandas()
            text_blocks = batch['text'].tolist()
            images = batch['image'].tolist()
            

            # Filter out empty text blocks
            valid_blocks = [(text, img) for text, img in zip(text_blocks, images) if text.strip()]

            print(f" num valid blocks {len(valid_blocks)}")
            
            if not valid_blocks:
                raise ValueError("No valid text blocks found in the multimodal dataset")
                
            if verbose:
                logger.info(f"Found {len(valid_blocks)} text blocks with images")
                logger.info(f"First block text length: {len(valid_blocks[0][0])} characters")
                logger.info(f"First block has image: {valid_blocks[0][1] is not None}")
            
            # Process each block and combine results
            all_results = []
            # Initialize LLM client for this block
            client = LLMClient(
                config_path=config_path,
                api_base=api_base,
                model_name=model
            )
            
            for i, (text, image) in enumerate(valid_blocks):
                if verbose:
                    logger.info(f"Processing block {i+1}/{len(valid_blocks)}")
                    logger.info(f"Block text length: {len(text)} characters")
                    logger.info(f"Block has image: {image is not None}")
                
                # Convert image to base64 if present
                image_base64 = _convert_image_to_base64(image) if image is not None else None
                
                # Generate content based on type
                if content_type == "qa":
                    generator = QAGenerator(client, config_path)
                    block_result = generator.process_document(
                        text,
                        image_base64,
                        num_pairs=num_pairs or 5, 
                        verbose=verbose
                    )
                    if verbose:
                        logger.info(f"Generated {len(block_result)} QA pairs for block {i+1}")
                    all_results.append(block_result)
                elif content_type == "summary":
                    generator = QAGenerator(client, config_path)
                    summary = generator.generate_summary(text)
                    if verbose:
                        logger.info(f"Generated summary of length {len(summary)} for block {i+1}")
                    all_results.append({"summary": summary})
                elif content_type == "cot":
                    from synthetic_data_kit.generators.cot_generator import COTGenerator
                    generator = COTGenerator(client, config_path)
                    block_result = generator.process_document(
                        text,
                        image_base64,
                        num_examples=num_pairs or 2, 
                        include_simple_steps=verbose
                    )
                    if verbose:
                        logger.info(f"Generated {len(block_result.get('cot_examples', []))} CoT examples for block {i+1}")
                    all_results.append(block_result.get("cot_examples", []))
                else:
                    raise ValueError(f"Unsupported content type for multimodal data: {content_type}")
            
            # Save combined results
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{content_type}_multimodal.json")
            
            if verbose:
                logger.info(f"Saving {len(all_results)} results to {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            
            return output_path
            
        else:
            # Text-only mode: combine all text into one document
            if not 'text' in dataset.schema.names:
                raise ValueError(f"Could not find text column in Lance dataset: {file_path}")

            batch = dataset.to_table(columns=['text']).to_pandas()
            document_text = '\n'.join(batch['text'].tolist())
            
            if verbose:
                logger.info(f"Processing text-only dataset: {file_path}")
                logger.info(f"Total text length: {len(document_text)} characters")

    else:
        # Default behavior for text files
        if os.path.isdir(file_path):
            raise ValueError(f"Cannot process directory as a text file: {file_path}. If this is a Lance dataset, ensure the extension is .lance. Otherwise, provide a valid text file.")
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        if verbose:
            logger.info(f"Processing text file: {file_path}")
            logger.info(f"Total text length: {len(document_text)} characters")
    
    # For text-only processing, continue with existing logic

    # Initialize LLM client
    client = LLMClient(
        config_path=config_path,
        provider=provider,
        api_base=api_base,
        model_name=model
    )
    
    # Override chunking config if provided
    if chunk_size is not None:
        client.config.setdefault('generation', {})['chunk_size'] = chunk_size
    if chunk_overlap is not None:
        client.config.setdefault('generation', {})['overlap'] = chunk_overlap
    
    # Debug: Print which provider is being used
    print(f"L Using {client.provider} provider")
    
    # Generate base filename for output
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Generate content based on type
    if content_type == "qa":
        generator = QAGenerator(client, config_path)

        # For text files, we need to read the content
        if file_extension != '.lance':
            document_text = read_json(file_path)
        
        # Get num_pairs from args or config
        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs = generation_config.get("num_pairs", 25)
        
        if verbose:
            logger.info(f"Generating {num_pairs} QA pairs")
        
        # Process document
        result = generator.process_document(
            document_text,
            num_pairs=num_pairs,
            verbose=verbose
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
        
        if verbose:
            logger.info(f"Saving {len(result)} QA pairs to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        return output_path
    
    elif content_type == "summary":
        generator = QAGenerator(client, config_path)

        # For text files, we need to read the content
        if file_extension != '.lance':
            document_text = read_json(file_path)
        
        if verbose:
            logger.info("Generating summary")
        
        # Generate just the summary
        summary = generator.generate_summary(document_text)
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_summary.json")
        
        if verbose:
            logger.info(f"Saving summary of length {len(summary)} to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"summary": summary}, f, indent=2)
        
        return output_path
    
    elif content_type == "cot":
        from synthetic_data_kit.generators.cot_generator import COTGenerator
        
        # Initialize the CoT generator
        generator = COTGenerator(client, config_path)

        # For text files, we need to read the content
        if file_extension != '.lance':
            document_text = read_json(file_path)
        
        # Get num_examples from args or config
        if num_pairs is None:
            config = client.config
            generation_config = get_generation_config(config)
            num_pairs = generation_config.get("num_cot_examples", 5)
        
        if verbose:
            logger.info(f"Generating {num_pairs} CoT examples")
        
        # Process document to generate CoT examples
        result = generator.process_document(
            document_text,
            num_examples=num_pairs,
            include_simple_steps=verbose
        )
        
        # Save output
        output_path = os.path.join(output_dir, f"{base_name}_cot_examples.json")
        
        if verbose:
            logger.info(f"Saving {len(result.get('cot_examples', []))} CoT examples to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        if verbose:
            # Print some example content
            if result.get("cot_examples") and len(result.get("cot_examples", [])) > 0:
                first_example = result["cot_examples"][0]
                logger.info("\nFirst CoT Example:")
                logger.info(f"Question: {first_example.get('question', '')}")
                logger.info(f"Reasoning (first 100 chars): {first_example.get('reasoning', '')[:100]}...")
                logger.info(f"Answer: {first_example.get('answer', '')}")
        
        return output_path
        
    elif content_type == "cot-enhance":
        from synthetic_data_kit.generators.cot_generator import COTGenerator
        from tqdm import tqdm
        
        # Initialize the CoT generator
        generator = COTGenerator(client, config_path)

        document_text = read_json(file_path)
        
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
                logger.info(f"Found {len(conversations)} conversation(s) to enhance")
            
            # Process each conversation
            enhanced_conversations = []
            
            for i, conversation in enumerate(tqdm(conversations, desc="Enhancing conversations")):
                # Check if this item has a conversations field
                if isinstance(conversation, dict) and "conversations" in conversation:
                    conv_messages = conversation["conversations"]
                    
                    # Validate messages format
                    if not isinstance(conv_messages, list):
                        logger.warning(f"conversations field is not a list in item {i}, skipping")
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
            
            if verbose:
                logger.info(f"Saving {len(enhanced_conversations)} enhanced conversations to {output_path}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if is_single_conversation and len(enhanced_conversations) == 1:
                    # Save the single conversation
                    json.dump(enhanced_conversations[0], f, indent=2)
                else:
                    # Save the array of conversations
                    json.dump(enhanced_conversations, f, indent=2)
            
            if verbose:
                logger.info(f"Enhanced {len(enhanced_conversations)} conversation(s)")
                
            return output_path
            
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse {file_path} as JSON. For cot-enhance, input must be a valid JSON file.")
    elif content_type == "vqa_add_reasoning":
        # Initialize the VQA generator
        generator = VQAGenerator(client, config_path)
        
        # Process the dataset
        output_path = generator.process_dataset(
            dataset_source=file_path,
            output_dir=output_dir,
            num_examples=num_pairs,
            verbose=verbose
        )
        
        return output_path

    else:
        raise ValueError(f"Unknown content type: {content_type}")

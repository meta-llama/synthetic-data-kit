# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Filter low quality examples

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.generators.qa_generator import QAGenerator
from synthetic_data_kit.utils.config import get_curate_config, get_prompt
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format, parse_ratings

def curate_qa_pairs(
    input_path: str,
    output_path: str,
    threshold: Optional[float] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[Path] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
) -> str:
    """Clean and filter QA pairs based on quality ratings
    
    Args:
        input_path: Path to the input file with QA pairs
        output_path: Path to save the cleaned output
        threshold: Quality threshold (1-10)
        api_base: VLLM API base URL
        model: Model to use
        config_path: Path to configuration file
        verbose: Show detailed output
    
    Returns:
        Path to the cleaned output file
    """
    # Set verbose either via CLI or via env variable. If its via CLI, set it to env variable
    if verbose:
        os.environ['SDK_VERBOSE'] = 'true'
    else:
        os.environ['SDK_VERBOSE'] = 'false'
    
    # Load input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract QA pairs or CoT examples
    qa_pairs = data.get("qa_pairs", [])
    cot_examples = data.get("cot_examples", [])
    summary = data.get("summary", "")
    
    # Determine which format we're working with
    is_cot_format = False
    if cot_examples and not qa_pairs:
        # Convert CoT examples to QA format for curation
        qa_pairs = []
        for example in cot_examples:
            qa_pair = {
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "reasoning": example.get("reasoning", "")  # Keep reasoning for reference
            }
            qa_pairs.append(qa_pair)
        is_cot_format = True
    
    # If there are no QA pairs or CoT examples
    if not qa_pairs:
        raise ValueError("No QA pairs or CoT examples found in the input file")
    
    # Initialize LLM client
    client = LLMClient(
        config_path=config_path,
        provider=provider,
        api_base=api_base,
        model_name=model
    )
    
    # Get threshold from args, then config, then default
    if threshold is None:
        config = client.config
        cleanup_config = get_curate_config(config)
        threshold = cleanup_config.get("threshold", 7.0)
    
    # Create QA generator
    generator = QAGenerator(client, config_path)
    
    # Get configuration
    curate_config = get_curate_config(client.config)
    
    # Allow environment variable to override batch size (for debugging)
    env_batch_size = os.environ.get('SDK_BATCH_SIZE')
    if env_batch_size and env_batch_size.isdigit():
        batch_size = int(env_batch_size)
        inference_batch = int(env_batch_size)
        if verbose:
            print(f"Using environment-specified batch size: {batch_size}")
    else:
        batch_size = curate_config.get("batch_size", 32)
        inference_batch = curate_config.get("inference_batch", 32)
        
    rating_temperature = curate_config.get("temperature", 0.1)
    
    # Get rating prompt template
    rating_prompt_template = get_prompt(client.config, "qa_rating")
    
    # Split QA pairs into batches
    batches = []
    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i:i+batch_size]
        batches.append(batch)
    
    # Prepare all message batches for rating
    all_messages = []
    for batch in batches:
        batch_json = json.dumps(batch, indent=2)
        rating_prompt = rating_prompt_template.format(pairs=batch_json)
        messages = [{"role": "system", "content": rating_prompt}]
        all_messages.append(messages)
    
    # Initialize counters and result containers
    filtered_pairs = []
    total_score = 0
    total_evaluated = 0
    total_passed = 0
    
    # Process batches with simple progress indicator rather than a detailed bar
    # This avoids conflicts with other output messages
    print(f"Processing {len(batches)} batches of QA pairs...")
    
    # Only use detailed progress bar in verbose mode
    if verbose:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
        
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        
        progress_ctx = Progress(*progress_columns)
        rate_task = progress_ctx.add_task(f"Rating QA pairs", total=len(batches))
        progress_ctx.start()
    else:
        progress_ctx = None
        rate_task = None
    
    # Process in inference batches
    for batch_start in range(0, len(all_messages), inference_batch):
        batch_end = min(batch_start + inference_batch, len(all_messages))
        current_batch = all_messages[batch_start:batch_end]
        current_batch_size = len(current_batch)
        
        batch_num = batch_start//inference_batch + 1
        total_batches = (len(all_messages) + inference_batch - 1)//inference_batch
        
        # Simple progress indicator for non-verbose mode
        if not verbose:
            print(f"Processing batch {batch_num}/{total_batches}...", end="\r")
        else:
            print(f"Processing batch {batch_num}/{total_batches}")
        
        try:
            # Get ratings for the batch
            if verbose:
                print(f"Sending batch request with {len(current_batch)} items")
                
            batch_responses = client.batch_completion(
                current_batch,
                temperature=rating_temperature,
                batch_size=inference_batch
            )
            
            if verbose:
                print(f"Received {len(batch_responses)} responses")
                for i, resp in enumerate(batch_responses):
                    print(f"Response {i+1}: {resp[:100]}...")
            
            # Process each response
            for j, response in enumerate(batch_responses):
                original_batch_index = batch_start + j
                if original_batch_index < len(batches):
                    original_batch = batches[original_batch_index]
                    
                    # Parse the ratings with original batch for fallback
                    try:
                        if verbose:
                            print(f"Processing batch {original_batch_index+1}")
                            
                        rated_batch = parse_ratings(response, original_batch)
                        
                        # Process the rated batch
                        for pair in rated_batch:
                            if "rating" in pair:
                                rating = pair["rating"]
                                total_score += rating
                                total_evaluated += 1
                                
                                if rating >= threshold:
                                    filtered_pairs.append(pair)
                                    total_passed += 1
                    except Exception as e:
                        if verbose:
                            print(f"Error processing batch {original_batch_index+1}: {str(e)}")
                            print(f"First 100 chars of response: {response[:100]}")
                        
                        # Try processing one pair at a time as a fallback
                        try:
                            if verbose:
                                print("Attempting to process items individually...")
                            
                            for item in original_batch:
                                item_json = json.dumps(item, indent=2)
                                rating_prompt = rating_prompt_template.format(pairs=item_json)
                                item_response = client.chat_completion(
                                    [{"role": "system", "content": rating_prompt}],
                                    temperature=rating_temperature
                                )
                                try:
                                    # This should be a single item
                                    rated_item = parse_ratings(item_response, [item])
                                    if rated_item and len(rated_item) > 0:
                                        pair = rated_item[0]
                                        if "rating" in pair:
                                            rating = pair["rating"]
                                            total_score += rating
                                            total_evaluated += 1
                                            
                                            if rating >= threshold:
                                                filtered_pairs.append(pair)
                                                total_passed += 1
                                                if verbose:
                                                    print(f"Successfully processed individual item with rating {rating}")
                                except Exception as inner_e:
                                    if verbose:
                                        print(f"Failed to process individual item: {str(inner_e)}")
                        except Exception as fallback_e:
                            if verbose:
                                print(f"Fallback processing failed: {str(fallback_e)}")
                            
                        # Continue processing other batches rather than failing completely
                        pass
            
            # Update progress bar if in verbose mode
            if progress_ctx and rate_task:
                progress_ctx.update(rate_task, advance=current_batch_size)
            
        except Exception as e:
            if verbose:
                print(f"Error processing inference batch {batch_num}: {str(e)}")
            
            # Update progress bar if in verbose mode
            if progress_ctx and rate_task:
                progress_ctx.update(rate_task, advance=current_batch_size)
    
    # Stop progress bar if in verbose mode
    if progress_ctx:
        progress_ctx.stop()
    
    # Clear the progress line in non-verbose mode
    if not verbose:
        print(" " * 80, end="\r")
        print("Batch processing complete.")
    
    # Calculate metrics
    metrics = {
        "total": len(qa_pairs),
        "filtered": len(filtered_pairs),
        "retention_rate": round(len(filtered_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
        "avg_score": round(total_score / total_evaluated, 1) if total_evaluated else 0
    }
    
    # Always print basic stats, even in non-verbose mode
    print(f"Rated {total_evaluated} QA pairs")
    print(f"Retained {total_passed} pairs (threshold: {threshold})")
    print(f"Average score: {metrics['avg_score']}")
    
    # Convert to conversation format
    conversations = convert_to_conversation_format(filtered_pairs)
    
    # Create result with filtered pairs in the appropriate format
    if is_cot_format:
        # Convert back to CoT format
        filtered_cot_examples = []
        for pair in filtered_pairs:
            cot_example = {
                "question": pair.get("question", ""),
                "reasoning": pair.get("reasoning", ""),
                "answer": pair.get("answer", "")
            }
            # Keep rating if it exists
            if "rating" in pair:
                cot_example["rating"] = pair["rating"]
            filtered_cot_examples.append(cot_example)
        
        result = {
            "summary": summary,
            "cot_examples": filtered_cot_examples,
            "conversations": conversations,
            "metrics": metrics
        }
    else:
        result = {
            "summary": summary,
            "qa_pairs": filtered_pairs,
            "conversations": conversations,
            "metrics": metrics
        }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save result
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    return output_path
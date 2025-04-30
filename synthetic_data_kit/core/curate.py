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

def process_file(
    input_path: str,
    output_path: str,
    config_path: Optional[Path] = None,
    llm_client: Optional[LLMClient] = None,
    threshold: Optional[float] = None,
    verbose: bool = False,
) -> str:
    """Clean and filter QA pairs based on quality ratings
    
    Args:
        input_path: Path to the input file with QA pairs
        output_path: Path to save the cleaned output
        config_path: Path to configuration file
        llm_client: Initialized LLMClient instance
        threshold: Quality threshold (1-10)
        verbose: Show detailed output
    
    Returns:
        Path to the cleaned output file
    """
    # Set verbose either via CLI or via env variable
    if verbose:
        os.environ['SDK_VERBOSE'] = 'true'
    else:
        os.environ['SDK_VERBOSE'] = 'false'
    
    # Load input file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract QA pairs
    qa_pairs = data.get("qa_pairs", [])
    summary = data.get("summary", "")
    
    # If there are no QA pairs or they're already filtered
    if not qa_pairs:
        raise ValueError("No QA pairs found in the input file")
    
    # Use provided client or create new one
    client = llm_client or LLMClient(config_path=config_path)
    
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
    
    if threshold is None:
        threshold = curate_config.get("threshold", 7.0)
    
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
    
    # Process batches with simple progress indicator
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
                            for pair in original_batch:
                                single_batch = [pair]
                                single_json = json.dumps(single_batch, indent=2)
                                single_prompt = rating_prompt_template.format(pairs=single_json)
                                single_messages = [{"role": "system", "content": single_prompt}]
                                
                                single_response = client.chat_completion(
                                    single_messages,
                                    temperature=rating_temperature
                                )
                                
                                rated_pair = parse_ratings(single_response, single_batch)[0]
                                if "rating" in rated_pair:
                                    rating = rated_pair["rating"]
                                    total_score += rating
                                    total_evaluated += 1
                                    
                                    if rating >= threshold:
                                        filtered_pairs.append(rated_pair)
                                        total_passed += 1
                        except Exception as e2:
                            if verbose:
                                print(f"Error in fallback processing: {str(e2)}")
            
            # Update progress
            if progress_ctx and rate_task:
                progress_ctx.update(rate_task, advance=current_batch_size)
        
        except Exception as e:
            if verbose:
                print(f"Error processing batch: {str(e)}")
            continue
    
    # Close progress bar if used
    if progress_ctx:
        progress_ctx.stop()
    
    # Print summary
    if total_evaluated > 0:
        avg_score = total_score / total_evaluated
        pass_rate = (total_passed / total_evaluated) * 100
        print(f"\nProcessed {total_evaluated} QA pairs")
        print(f"Average quality score: {avg_score:.2f}")
        print(f"Pass rate: {pass_rate:.1f}% ({total_passed} pairs above threshold {threshold})")
    
    # Save filtered pairs
    output_data = {
        "qa_pairs": filtered_pairs,
        "summary": summary,
        "metadata": {
            "total_evaluated": total_evaluated,
            "total_passed": total_passed,
            "average_score": avg_score if total_evaluated > 0 else 0,
            "threshold": threshold
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    return output_path
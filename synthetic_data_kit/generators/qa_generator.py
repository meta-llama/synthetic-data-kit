# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

from typing import Dict, List, Any, Optional, Tuple
import json
import time
import os
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.text import split_into_chunks
from synthetic_data_kit.utils.llm_processing import parse_qa_pairs, parse_ratings, convert_to_conversation_format
from synthetic_data_kit.utils.config import load_config, get_generation_config, get_curate_config, get_prompt

class QAGenerator:
    def __init__(self, 
                 client: LLMClient,
                 config_path: Optional[Path] = None):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client
        
        # Load config
        self.config = load_config(config_path)
        
        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)
    
    def generate_summary(self, document_text: str) -> str:
        """Generate a summary of the document"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        if verbose:
            print("Generating document summary...")
        
        # Get summary prompt from config
        prompt = get_prompt(self.config, "summary")
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": document_text}
        ]
        
        summary = self.client.chat_completion(
            messages, 
            temperature=0.1  # Use lower temperature for summaries
        )
        
        if verbose:
            print(f"Summary generated ({len(summary)} chars)")
        return summary
    
    def generate_qa_pairs(self, 
                        document_text: str, 
                        summary: str, 
                        num_pairs: int = 25) -> List[Dict[str, str]]:
        """Generate QA pairs from the document using batched processing"""
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
            print(f"Generating QA pairs...")
            print(f"Document split into {len(chunks)} chunks")
            print(f"Using batch size of {batch_size}")
        
        all_qa_pairs = []
        pairs_per_chunk = max(1, round(num_pairs / len(chunks)))
        
        # Get QA generation prompt template
        qa_prompt_template = get_prompt(self.config, "qa_generation")
        
        # Prepare all message batches
        all_messages = []
        for i, chunk in enumerate(chunks):
            # Format the prompt with summary and text
            qa_prompt = qa_prompt_template.format(
                num_pairs=pairs_per_chunk,
                summary=summary[:100],
                text=chunk
            )
            
            messages = [
                {"role": "system", "content": qa_prompt}
            ]
            all_messages.append(messages)
        
        print(f"Processing {len(chunks)} chunks to generate QA pairs...")
        
        # Set up progress tracking based on verbose mode
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
            generate_task = progress_ctx.add_task(f"Generating QA pairs", total=len(chunks))
            progress_ctx.start()
        else:
            progress_ctx = None
            generate_task = None
        
        # Process in batches
        for batch_start in range(0, len(chunks), batch_size):
            # Check if we've already generated enough pairs
            if len(all_qa_pairs) >= num_pairs:
                if verbose:
                    print(f"Reached target of {num_pairs} pairs. Stopping processing.")
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
                    if len(all_qa_pairs) >= num_pairs:
                        if verbose:
                            print(f"  Reached target of {num_pairs} pairs. Stopping batch processing.")
                        break
                        
                    chunk_index = batch_start + j
                    chunk_pairs = parse_qa_pairs(response)
                    
                    # Only add pairs up to the target limit
                    remaining_pairs = num_pairs - len(all_qa_pairs)
                    if remaining_pairs > 0:
                        pairs_to_add = chunk_pairs[:remaining_pairs]
                        all_qa_pairs.extend(pairs_to_add)
                        
                        if verbose:
                            print(f"  Generated {len(pairs_to_add)} pairs from chunk {chunk_index+1} (total: {len(all_qa_pairs)}/{num_pairs})")
                    
                    # Break if we've reached the target
                    if len(all_qa_pairs) >= num_pairs:
                        break
                
                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)
                
                # Break outer loop if we've reached the target
                if len(all_qa_pairs) >= num_pairs:
                    break
                
            except Exception as e:
                if verbose:
                    print(f"  Error processing batch {batch_num}: {str(e)}")
                
                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)
        
        # Stop progress bar if in verbose mode
        if progress_ctx:
            progress_ctx.stop()
        
        # Clear the progress line in non-verbose mode
        if not verbose:
            print(" " * 80, end="\r")
            print("Batch processing complete.")
        
        # Always print summary information, even in non-verbose mode
        print(f"Generated {len(all_qa_pairs)} QA pairs total (requested: {num_pairs})")
        return all_qa_pairs
    
    def rate_qa_pairs(self, 
                    qa_pairs: List[Dict[str, str]], 
                    summary: str, 
                    threshold: Optional[float] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rate and filter QA pairs by quality"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        if not qa_pairs:
            return [], {"total": 0, "filtered": 0, "retention_rate": 0, "avg_score": 0}
        
        # Get threshold from args, then config, then default
        if threshold is None:
            threshold = self.curate_config.get("threshold", 7.0)
            
        if verbose:
            print(f"Evaluating {len(qa_pairs)} pairs...")
        
        # Get rating config
        batch_size = self.curate_config.get("batch_size", 8)
        temperature = self.curate_config.get("temperature", 0.1)
        
        # Get rating prompt template
        rating_prompt_template = get_prompt(self.config, "qa_rating")
        
        # Process in batches
        batches = [qa_pairs[i:i+batch_size] for i in range(0, len(qa_pairs), batch_size)]
        
        rated_pairs = []
        total_score = 0
        
        # Create progress bar
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        
        with Progress(*progress_columns) as progress:
            rating_task = progress.add_task(f"Rating QA pairs", total=len(batches))
            
            for i, batch in enumerate(batches):
                if verbose:
                    print(f"Rating batch {i+1}/{len(batches)}...")
                batch_json = json.dumps(batch, indent=2)
                
                # Format the rating prompt with pairs
                rating_prompt = rating_prompt_template.format(pairs=batch_json)
                
                messages = [
                    {"role": "system", "content": rating_prompt}
                ]
                
                try:
                    response = self.client.chat_completion(
                        messages, 
                        temperature=temperature
                    )
                    
                    rated_batch = parse_ratings(response)
                    
                    for pair in rated_batch:
                        if "rating" in pair:
                            total_score += pair["rating"]
                            if pair["rating"] >= threshold:
                                rated_pairs.append(pair)
                
                except Exception as e:
                    if verbose:
                        print(f"Error rating batch {i+1}: {str(e)}")
                
                time.sleep(0.5)  # Avoid rate limits
                progress.update(rating_task, advance=1)
        
        # Calculate metrics
        metrics = {
            "total": len(qa_pairs),
            "filtered": len(rated_pairs),
            "retention_rate": round(len(rated_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
            "avg_score": round(total_score / len(qa_pairs), 1) if qa_pairs else 0
        }
        
        # Always print summary information, even in non-verbose mode
        print(f"Keeping {len(rated_pairs)} out of {len(qa_pairs)} pairs (threshold: {threshold})")
        print(f"Average score: {metrics['avg_score']}")
        return rated_pairs, metrics
    
    def process_document(self, 
                       document_text: str, 
                       num_pairs: int = 25, 
                       verbose: bool = False) -> Dict[str, Any]:
        """Process a document to generate QA pairs without rating"""
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'
        
        # Generate summary
        max_context_length = self.generation_config.get("max_context_length", 8000)
        summary = self.generate_summary(document_text[0:max_context_length])
        
        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(document_text, summary, num_pairs=num_pairs)
        
        # Prepare result - no rating at this stage
        result = {
            "summary": summary,
            "qa_pairs": qa_pairs
        }
        
        return result
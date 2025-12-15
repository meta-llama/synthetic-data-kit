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
    
    def generate_summary(self, 
                         document_text: str, 
                         rolling_summary: Optional[bool] = False) -> str:
        """Generate a summary of the document"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        if verbose:
            print("Generating document summary...")
        
        # Get summary prompt and params from config
        prompt = get_prompt(self.config, "summary")
        max_context_length = self.generation_config.get("max_context_length", 8000)
        summary_overlap = self.generation_config.get("summary_overlap", 0)

        if rolling_summary:
            summary_per_chunk = []
            #split text into long chunks for summarization
            chunks = split_into_chunks(document_text,
                                       chunk_size=max_context_length,
                                       overlap=summary_overlap)

            for chunk in chunks:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk}
                ]
                new_summary = self.client.chat_completion(
                    messages, 
                    temperature=0.1  # Use lower temperature for summaries
                )
                summary_per_chunk.append(new_summary)

            summary = " .".join(summary_per_chunk)
            # Summarize again to reduce overall length and redundancy
            summary = self.generate_summary(summary,
                                            rolling_summary=False)
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": document_text[0:max_context_length]}
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
                        num_pairs: int = 25,
                        num_pairs_per_chunk: Optional[int] = None) -> List[Dict[str, str]]:
        """Generate QA pairs from the document using batched processing
        
        Args:
            document_text: The text to generate QA pairs from
            summary: Summary of the document
            num_pairs: Total number of QA pairs to generate (used if num_pairs_per_chunk is None)
            num_pairs_per_chunk: Number of QA pairs to generate per chunk (takes precedence over num_pairs)
        
        Returns:
            List of QA pair dictionaries
        """
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
        
        # Determine generation mode and calculate targets
        if num_pairs_per_chunk is not None:
            # Per-chunk mode: scale with document size
            pairs_per_chunk = num_pairs_per_chunk
            total_target = num_pairs_per_chunk * len(chunks)
            mode = "per-chunk"
        else:
            # Total pairs mode: divide across chunks (original behavior)
            pairs_per_chunk = max(1, round(num_pairs / len(chunks)))
            total_target = num_pairs
            mode = "total"
        
        if verbose:
            print(f"Generating QA pairs...")
            print(f"Document split into {len(chunks)} chunks")
            print(f"Mode: {mode} (pairs per chunk: {pairs_per_chunk}, target total: {total_target})")
            print(f"Using batch size of {batch_size}")
        
        all_qa_pairs = []
        
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
            if len(all_qa_pairs) >= total_target:
                if verbose:
                    print(f"Reached target of {total_target} pairs. Stopping processing.")
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
                    if len(all_qa_pairs) >= total_target:
                        if verbose:
                            print(f"  Reached target of {total_target} pairs. Stopping batch processing.")
                        break
                        
                    chunk_index = batch_start + j
                    chunk_pairs = parse_qa_pairs(response)
                    
                    # Only add pairs up to the target limit
                    remaining_pairs = total_target - len(all_qa_pairs)
                    if remaining_pairs > 0:
                        pairs_to_add = chunk_pairs[:remaining_pairs]
                        all_qa_pairs.extend(pairs_to_add)
                        
                        if verbose:
                            print(f"  Generated {len(pairs_to_add)} pairs from chunk {chunk_index+1} (total: {len(all_qa_pairs)}/{total_target})")
                    
                    # Break if we've reached the target
                    if len(all_qa_pairs) >= total_target:
                        break
                
                # Update progress bar if in verbose mode
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)
                
                # Break outer loop if we've reached the target
                if len(all_qa_pairs) >= total_target:
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
        print(f"Generated {len(all_qa_pairs)} QA pairs total (target: {total_target}, mode: {mode})")
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
    
    def process_documents(self,
                        documents: List[Dict[str, Any]],
                        num_pairs: int = 25,
                        num_pairs_per_chunk: Optional[int] = None,
                        verbose: bool = False,
                        rolling_summary: Optional[bool] = False) -> Dict[str, Any]:
        """Process a list of documents to generate QA pairs without rating
        
        Args:
            documents: List of document dictionaries with 'text' field
            num_pairs: Total number of QA pairs to generate (used if num_pairs_per_chunk is None)
            num_pairs_per_chunk: Number of QA pairs per chunk (takes precedence over num_pairs)
            verbose: Whether to show detailed output
            rolling_summary: Whether to use rolling summary for long documents
        
        Returns:
            Dictionary with summary and qa_pairs
        """
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'

        all_qa_pairs = []
        full_text = " ".join([doc["text"] for doc in documents])

        # Generate summary
        summary = self.generate_summary(full_text, rolling_summary=rolling_summary)

        # Generate QA pairs
        qa_pairs = self.generate_qa_pairs(full_text, summary, num_pairs=num_pairs, num_pairs_per_chunk=num_pairs_per_chunk)

        all_qa_pairs.extend(qa_pairs)

        # Prepare result - no rating at this stage
        result = {
            "summary": summary,
            "qa_pairs": all_qa_pairs
        }

        return result
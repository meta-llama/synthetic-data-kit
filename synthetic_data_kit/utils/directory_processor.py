# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Directory processing utilities for batch operations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# Supported file extensions for each command
INGEST_EXTENSIONS = ['.pdf', '.html', '.htm', '.docx', '.pptx', '.txt']
CREATE_EXTENSIONS = ['.txt']
CURATE_EXTENSIONS = ['.json']
SAVE_AS_EXTENSIONS = ['.json']

def is_directory(path: str) -> bool:
    """Check if path is a directory"""
    return os.path.isdir(path)

def get_supported_files(directory: str, extensions: List[str]) -> List[str]:
    """Get all files with supported extensions in directory (non-recursive)
    
    Args:
        directory: Directory path to scan
        extensions: List of supported file extensions (e.g., ['.pdf', '.txt'])
    
    Returns:
        List of full file paths with supported extensions
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not os.path.isdir(directory):
        raise ValueError(f"Path is not a directory: {directory}")
    
    supported_files = []
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip directories, only process files
            if os.path.isfile(file_path):
                # Check if file has supported extension
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in extensions:
                    supported_files.append(file_path)
    
    except PermissionError:
        raise PermissionError(f"Permission denied accessing directory: {directory}")
    
    return sorted(supported_files)  # Sort for consistent processing order

def process_directory_ingest(
    directory: str,
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Process all supported files in directory for ingestion
    
    Args:
        directory: Directory containing files to process
        output_dir: Directory to save processed files
        config: Configuration dictionary
        verbose: Show detailed progress
    
    Returns:
        Dictionary with processing results
    """
    from synthetic_data_kit.core.ingest import process_file
    
    # Get all supported files
    supported_files = get_supported_files(directory, INGEST_EXTENSIONS)
    
    if not supported_files:
        console.print(f"No supported files found in {directory}", style="yellow")
        console.print(f"Supported extensions: {', '.join(INGEST_EXTENSIONS)}", style="yellow")
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
            "errors": []
        }
    
    console.print(f"Found {len(supported_files)} supported files to process", style="blue")
    
    # Initialize results tracking
    results = {
        "total_files": len(supported_files),
        "successful": 0,
        "failed": 0,
        "results": [],
        "errors": []
    }
    
    # Process files with progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose
    ) as progress:
        
        task = progress.add_task("Processing files", total=len(supported_files))
        
        for file_path in supported_files:
            filename = os.path.basename(file_path)
            
            try:
                # Process individual file
                output_path = process_file(file_path, output_dir, None, config)
                
                # Record success
                results["successful"] += 1
                results["results"].append({
                    "input_file": file_path,
                    "output_file": output_path,
                    "status": "success"
                })
                
                if verbose:
                    console.print(f"✓ Processed {filename} -> {os.path.basename(output_path)}", style="green")
                else:
                    console.print(f"✓ {filename}", style="green")
                
            except Exception as e:
                # Record failure
                results["failed"] += 1
                results["errors"].append({
                    "input_file": file_path,
                    "error": str(e),
                    "status": "failed"
                })
                
                if verbose:
                    console.print(f"✗ Failed to process {filename}: {e}", style="red")
                else:
                    console.print(f"✗ {filename}: {e}", style="red")
            
            progress.update(task, advance=1)
    
    # Show summary
    console.print("\n" + "="*50, style="bold")
    console.print(f"Processing Summary:", style="bold blue")
    console.print(f"Total files: {results['total_files']}")
    console.print(f"Successful: {results['successful']}", style="green")
    console.print(f"Failed: {results['failed']}", style="red" if results['failed'] > 0 else "green")
    console.print("="*50, style="bold")
    
    return results

def get_directory_stats(directory: str, extensions: List[str]) -> Dict[str, Any]:
    """Get statistics about supported files in directory
    
    Args:
        directory: Directory to analyze
        extensions: List of supported extensions
    
    Returns:
        Dictionary with file statistics
    """
    if not os.path.exists(directory):
        return {"error": f"Directory not found: {directory}"}
    
    if not os.path.isdir(directory):
        return {"error": f"Path is not a directory: {directory}"}
    
    stats = {
        "total_files": 0,
        "supported_files": 0,
        "unsupported_files": 0,
        "by_extension": {},
        "file_list": []
    }
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                stats["total_files"] += 1
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in extensions:
                    stats["supported_files"] += 1
                    stats["file_list"].append(filename)
                    
                    # Count by extension
                    if file_ext not in stats["by_extension"]:
                        stats["by_extension"][file_ext] = 0
                    stats["by_extension"][file_ext] += 1
                else:
                    stats["unsupported_files"] += 1
    
    except PermissionError:
        return {"error": f"Permission denied accessing directory: {directory}"}
    
    return stats

def process_directory_create(
    directory: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    content_type: str = "qa",
    num_pairs: Optional[int] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    multimodal: Optional[bool] = False
) -> Dict[str, Any]:
    """Process all supported files in directory for content creation
    
    Args:
        directory: Directory containing .txt files to process
        output_dir: Directory to save generated content
        config_path: Path to configuration file
        api_base: API base URL
        model: Model to use
        content_type: Type of content to generate (qa, summary, cot, cot-enhance)
        num_pairs: Target number of QA pairs or examples
        verbose: Show detailed progress
        provider: LLM provider to use
    
    Returns:
        Dictionary with processing results
    """
    from synthetic_data_kit.core.create import process_file
    
    # For create command, we process .txt files (output from ingest)
    # For cot-enhance, we process .json files instead
    if content_type == "cot-enhance":
        extensions = ['.json']
    else:
        extensions = CREATE_EXTENSIONS  # ['.txt']
    
    # Get all supported files
    supported_files = get_supported_files(directory, extensions)
    
    if not supported_files:
        console.print(f"No supported files found in {directory}", style="yellow")
        if content_type == "cot-enhance":
            console.print(f"For cot-enhance: looking for .json files", style="yellow")
        else:
            console.print(f"For {content_type}: looking for .txt files", style="yellow")
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
            "errors": []
        }
    
    console.print(f"Found {len(supported_files)} {content_type} files to process", style="blue")
    
    # Initialize results tracking
    results = {
        "total_files": len(supported_files),
        "successful": 0,
        "failed": 0,
        "results": [],
        "errors": []
    }
    
    # Process files with progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose
    ) as progress:
        
        task = progress.add_task(f"Generating {content_type} content", total=len(supported_files))
        
        for file_path in supported_files:
            filename = os.path.basename(file_path)
            
            try:
                # Process individual file
                output_path = process_file(
                    file_path,
                    output_dir,
                    config_path,
                    api_base,
                    model,
                    content_type,
                    num_pairs,
                    verbose,
                    provider=provider,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    multimodal=multimodal
                )
                
                # Record success
                results["successful"] += 1
                results["results"].append({
                    "input_file": file_path,
                    "output_file": output_path,
                    "content_type": content_type,
                    "status": "success"
                })
                
                if verbose:
                    console.print(f"✓ Generated {content_type} from {filename} -> {os.path.basename(output_path)}", style="green")
                else:
                    console.print(f"✓ {filename}", style="green")
                
            except Exception as e:
                # Record failure
                results["failed"] += 1
                results["errors"].append({
                    "input_file": file_path,
                    "error": str(e),
                    "content_type": content_type,
                    "status": "failed"
                })
                
                if verbose:
                    console.print(f"✗ Failed to process {filename}: {e}", style="red")
                else:
                    console.print(f"✗ {filename}: {e}", style="red")
            
            progress.update(task, advance=1)
    
    # Show summary
    console.print("\n" + "="*50, style="bold")
    console.print(f"Content Generation Summary ({content_type}):", style="bold blue")
    console.print(f"Total files: {results['total_files']}")
    console.print(f"Successful: {results['successful']}", style="green")
    console.print(f"Failed: {results['failed']}", style="red" if results['failed'] > 0 else "green")
    console.print("="*50, style="bold")
    
    return results

def process_directory_curate(
    directory: str,
    output_dir: Optional[str] = None,
    threshold: Optional[float] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    config_path: Optional[str] = None,
    verbose: bool = False,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Process all supported files in directory for content curation
    
    Args:
        directory: Directory containing .json files to curate
        output_dir: Directory to save curated content (if None, uses input directory structure)
        threshold: Quality threshold (1-10)
        api_base: API base URL
        model: Model to use
        config_path: Path to configuration file
        verbose: Show detailed progress
        provider: LLM provider to use
    
    Returns:
        Dictionary with processing results
    """
    from synthetic_data_kit.core.curate import curate_qa_pairs
    
    # For curate command, we process .json files (output from create)
    supported_files = get_supported_files(directory, CURATE_EXTENSIONS)  # ['.json']
    
    if not supported_files:
        console.print(f"No supported files found in {directory}", style="yellow")
        console.print(f"For curate: looking for .json files with QA pairs", style="yellow")
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
            "errors": []
        }
    
    console.print(f"Found {len(supported_files)} JSON files to curate", style="blue")
    
    # Initialize results tracking
    results = {
        "total_files": len(supported_files),
        "successful": 0,
        "failed": 0,
        "results": [],
        "errors": []
    }
    
    # If no output_dir specified, default to cleaned directory
    if output_dir is None:
        from synthetic_data_kit.utils.config import load_config, get_path_config
        config = load_config(config_path)
        output_dir = get_path_config(config, "output", "curated")
    
    # Process files with progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose
    ) as progress:
        
        task = progress.add_task("Curating QA pairs", total=len(supported_files))
        
        for file_path in supported_files:
            filename = os.path.basename(file_path)
            
            try:
                # Generate output path for this file
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}_cleaned.json")
                
                # Process individual file
                result_path = curate_qa_pairs(
                    file_path,
                    output_path,
                    threshold,
                    api_base,
                    model,
                    config_path,
                    verbose,
                    provider=provider
                )
                
                # Record success
                results["successful"] += 1
                results["results"].append({
                    "input_file": file_path,
                    "output_file": result_path,
                    "threshold": threshold,
                    "status": "success"
                })
                
                if verbose:
                    console.print(f"✓ Curated {filename} -> {os.path.basename(result_path)}", style="green")
                else:
                    console.print(f"✓ {filename}", style="green")
                
            except Exception as e:
                # Record failure
                results["failed"] += 1
                results["errors"].append({
                    "input_file": file_path,
                    "error": str(e),
                    "threshold": threshold,
                    "status": "failed"
                })
                
                if verbose:
                    console.print(f"✗ Failed to curate {filename}: {e}", style="red")
                else:
                    console.print(f"✗ {filename}: {e}", style="red")
            
            progress.update(task, advance=1)
    
    # Show summary
    console.print("\n" + "="*50, style="bold")
    console.print(f"Curation Summary (threshold: {threshold}):", style="bold blue")
    console.print(f"Total files: {results['total_files']}")
    console.print(f"Successful: {results['successful']}", style="green")
    console.print(f"Failed: {results['failed']}", style="red" if results['failed'] > 0 else "green")
    console.print("="*50, style="bold")
    
    return results

def process_directory_save_as(
    directory: str,
    output_dir: Optional[str] = None,
    format: str = "jsonl",
    storage_format: str = "json",
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process all supported files in directory for format conversion
    
    Args:
        directory: Directory containing .json files to convert
        output_dir: Directory to save converted files (if None, uses input directory structure)
        format: Output format (jsonl, alpaca, ft, chatml)
        storage_format: Storage format (json, hf)
        config: Configuration dictionary
        verbose: Show detailed progress
    
    Returns:
        Dictionary with processing results
    """
    from synthetic_data_kit.core.save_as import convert_format
    
    # For save-as command, we process .json files (output from curate)
    supported_files = get_supported_files(directory, SAVE_AS_EXTENSIONS)  # ['.json']
    
    if not supported_files:
        console.print(f"No supported files found in {directory}", style="yellow")
        console.print(f"For save-as: looking for .json files with cleaned QA pairs", style="yellow")
        return {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
            "errors": []
        }
    
    console.print(f"Found {len(supported_files)} JSON files to convert to {format} format", style="blue")
    
    # Initialize results tracking
    results = {
        "total_files": len(supported_files),
        "successful": 0,
        "failed": 0,
        "results": [],
        "errors": []
    }
    
    # If no output_dir specified, default to final directory
    if output_dir is None:
        from synthetic_data_kit.utils.config import load_config, get_path_config
        if config is None:
            config = load_config()
        output_dir = get_path_config(config, "output", "final")
    
    # Process files with progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        disable=not verbose
    ) as progress:
        
        task = progress.add_task(f"Converting to {format} format", total=len(supported_files))
        
        for file_path in supported_files:
            filename = os.path.basename(file_path)
            
            try:
                # Generate output path for this file
                base_name = os.path.splitext(filename)[0]
                
                if storage_format == "hf":
                    # For HF datasets, use a directory name
                    output_path = os.path.join(output_dir, f"{base_name}_{format}_hf")
                else:
                    # For JSON files, use appropriate extension
                    if format == "jsonl":
                        output_path = os.path.join(output_dir, f"{base_name}.jsonl")
                    else:
                        output_path = os.path.join(output_dir, f"{base_name}_{format}.json")
                
                # Process individual file
                result_path = convert_format(
                    file_path,
                    output_path,
                    format,
                    config,
                    storage_format=storage_format
                )
                
                # Record success
                results["successful"] += 1
                results["results"].append({
                    "input_file": file_path,
                    "output_file": result_path,
                    "format": format,
                    "storage": storage_format,
                    "status": "success"
                })
                
                if verbose:
                    console.print(f"✓ Converted {filename} -> {os.path.basename(result_path)} ({format}, {storage_format})", style="green")
                else:
                    console.print(f"✓ {filename}", style="green")
                
            except Exception as e:
                # Record failure
                results["failed"] += 1
                results["errors"].append({
                    "input_file": file_path,
                    "error": str(e),
                    "format": format,
                    "storage": storage_format,
                    "status": "failed"
                })
                
                if verbose:
                    console.print(f"✗ Failed to convert {filename}: {e}", style="red")
                else:
                    console.print(f"✗ {filename}: {e}", style="red")
            
            progress.update(task, advance=1)
    
    # Show summary
    console.print("\n" + "="*50, style="bold")
    console.print(f"Format Conversion Summary ({format}, {storage_format}):", style="bold blue")
    console.print(f"Total files: {results['total_files']}")
    console.print(f"Successful: {results['successful']}", style="green")
    console.print(f"Failed: {results['failed']}", style="red" if results['failed'] > 0 else "green")
    console.print("="*50, style="bold")
    
    return results
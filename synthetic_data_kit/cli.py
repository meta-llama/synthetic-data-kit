# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# CLI Logic for synthetic-data-kit

import os
import typer
from pathlib import Path
from typing import Optional
import requests
from rich.console import Console
from rich.table import Table

from synthetic_data_kit.utils.config import load_config, get_vllm_config, get_path_config
from synthetic_data_kit.core.context import AppContext

# Initialize Typer app
app = typer.Typer(
    name="synthetic-data-kit",
    help="A toolkit for preparing synthetic datasets for fine-tuning LLMs",
    add_completion=True,
)
console = Console()

# Create app context
ctx = AppContext()

# Define global options
@app.callback()
def callback(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
):
    """
    Global options for the Synthetic Data Kit CLI
    """
    if config:
        ctx.config_path = config
    ctx.config = load_config(ctx.config_path)


@app.command("system-check")
def system_check(
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL to check"
    )
):
    """
    Check if the VLLM server is running.
    """
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    
    with console.status(f"Checking VLLM server at {api_base}..."):
        try:
            response = requests.get(f"{api_base}/models", timeout=2)
            if response.status_code == 200:
                console.print(f" VLLM server is running at {api_base}", style="green")
                console.print(f"Available models: {response.json()}")
                return 0
            else:
                console.print(f"L VLLM server is not available at {api_base}", style="red")
                console.print(f"Error: Server returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            console.print(f"L VLLM server is not available at {api_base}", style="red")
            console.print(f"Error: {str(e)}")
            
        # Show instruction to start the server
        model = vllm_config.get("model")
        port = vllm_config.get("port", 8000)
        console.print("\nTo start the server, run:", style="yellow")
        console.print(f"vllm serve {model} --port {port}", style="bold blue")
        return 1


@app.command()
def ingest(
    input: str = typer.Argument(..., help="File or URL to parse"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Custom output filename"
    ),
    multimodal: bool = typer.Option(
        False, "--multimodal", help="Enable multimodal processing to include images."
    ),
):
    """
    Parse documents (PDF, HTML, YouTube, DOCX, PPT, TXT) into clean text.
    """
    from synthetic_data_kit.core.ingest import process_file
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "parsed")
    
    try:
        with console.status(f"Processing {input}..."):
            output_path = process_file(input, output_dir, name, ctx.config, multimodal)
        console.print(f" Text successfully extracted to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command()
def create(
    input: str = typer.Argument(..., help="File to process"),
    content_type: str = typer.Option(
        "qa", "--type", help="Type of content to generate [qa|summary|cot|cot-enhance]"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    num_pairs: Optional[int] = typer.Option(
        None, "--num-pairs", "-n", help="Target number of QA pairs to generate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
    multimodal: bool = typer.Option(
        False, "--multimodal", help="Process multimodal data (text + images)"
    ),
):
    """
    Generate content from text using local LLM inference.
    
    Content types:
    - qa: Generate question-answer pairs from text
    - summary: Generate a summary of the text
    - cot: Generate Chain of Thought reasoning examples from text
    - cot-enhance: Enhance existing tool-use conversations with Chain of Thought reasoning
      (for cot-enhance, the input must be a JSON file with either:
       - A single conversation in 'conversations' field
       - An array of conversation objects, each with a 'conversations' field
       - A direct array of conversation messages)
    """
    from synthetic_data_kit.core.create import process_file
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    model = model or vllm_config.get("model")
    
    # Check server first
    try:
        response = requests.get(f"{api_base}/models", timeout=2)
        if response.status_code != 200:
            console.print(f"L Error: VLLM server not available at {api_base}", style="red")
            console.print("Please start the VLLM server with:", style="yellow")
            console.print(f"vllm serve {model}", style="bold blue")
            return 1
    except requests.exceptions.RequestException:
        console.print(f"L Error: VLLM server not available at {api_base}", style="red")
        console.print("Please start the VLLM server with:", style="yellow")
        console.print(f"vllm serve {model}", style="bold blue")
        return 1
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "generated")
    
    try:
        with console.status(f"Generating {content_type} content from {input}..."):
            output_path = process_file(
                input,
                output_dir,
                ctx.config_path,
                api_base,
                model,
                content_type,
                num_pairs,
                verbose,
                multimodal
            )
        if output_path:
            console.print(f" Content saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command("curate")
def curate(
    input: str = typer.Argument(..., help="Input file to clean"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    threshold: Optional[float] = typer.Option(
        None, "--threshold", "-t", help="Quality threshold (1-10)"
    ),
    api_base: Optional[str] = typer.Option(
        None, "--api-base", help="VLLM API base URL"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Clean and filter content based on quality.
    """
    from synthetic_data_kit.core.curate import curate_qa_pairs
    
    # Get VLLM server details from args or config
    vllm_config = get_vllm_config(ctx.config)
    api_base = api_base or vllm_config.get("api_base")
    model = model or vllm_config.get("model")
    
    # Check server first
    try:
        response = requests.get(f"{api_base}/models", timeout=2)
        if response.status_code != 200:
            console.print(f"L Error: VLLM server not available at {api_base}", style="red")
            console.print("Please start the VLLM server with:", style="yellow")
            console.print(f"vllm serve {model}", style="bold blue")
            return 1
    except requests.exceptions.RequestException:
        console.print(f"L Error: VLLM server not available at {api_base}", style="red")
        console.print("Please start the VLLM server with:", style="yellow")
        console.print(f"vllm serve {model}", style="bold blue")
        return 1
    
    # Get default output path from config if not provided
    if not output:
        cleaned_dir = get_path_config(ctx.config, "output", "cleaned")
        os.makedirs(cleaned_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input))[0]
        output = os.path.join(cleaned_dir, f"{base_name}_cleaned.json")
    
    try:
        with console.status(f"Cleaning content from {input}..."):
            result_path = curate_qa_pairs(
                input,
                output,
                threshold,
                api_base,
                model,
                ctx.config_path,
                verbose
            )
        console.print(f" Cleaned content saved to [bold]{result_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


@app.command("save-as")
def save_as(
    input: str = typer.Argument(..., help="Input file to convert"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Output format [jsonl|alpaca|ft|chatml]"
    ),
    storage: str = typer.Option(
        "json", "--storage", help="Storage format [json|hf]",
        show_default=True
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
):
    """
    Convert to different formats for fine-tuning.
    
    The --format option controls the content format (how the data is structured).
    The --storage option controls how the data is stored (JSON file or HF dataset).
    
    When using --storage hf, the output will be a directory containing a Hugging Face 
    dataset in Arrow format, which is optimized for machine learning workflows.
    """
    from synthetic_data_kit.core.save_as import convert_format
    
    # Get format from args or config
    if not format:
        format_config = ctx.config.get("format", {})
        format = format_config.get("default", "jsonl")
    
    # Set default output path if not provided
    if not output:
        final_dir = get_path_config(ctx.config, "output", "final")
        os.makedirs(final_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input))[0]
        
        if storage == "hf":
            # For HF datasets, use a directory name
            output = os.path.join(final_dir, f"{base_name}_{format}_hf")
        else:
            # For JSON files, use appropriate extension
            if format == "jsonl":
                output = os.path.join(final_dir, f"{base_name}.jsonl")
            else:
                output = os.path.join(final_dir, f"{base_name}_{format}.json")
    
    try:
        with console.status(f"Converting {input} to {format} format with {storage} storage..."):
            output_path = convert_format(
                input,
                output,
                format,
                ctx.config,
                storage_format=storage
            )
        
        if storage == "hf":
            console.print(f" Converted to {format} format and saved as HF dataset to [bold]{output_path}[/bold]", style="green")
        else:
            console.print(f" Converted to {format} format and saved to [bold]{output_path}[/bold]", style="green")
        return 0
    except Exception as e:
        console.print(f"L Error: {e}", style="red")
        return 1


if __name__ == "__main__":
    app()

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

from synthetic_data_kit.utils.config import load_config, get_vllm_config, get_endpoint_config, get_llm_provider, get_path_config
from synthetic_data_kit.core.context import AppContext
from synthetic_data_kit.server.app import run_server

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
        None, "--api-base", help="API base URL to check"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="Provider to check ('vllm' or 'api-endpoint')"
    )
):
    """
    Check if the selected LLM provider's server is running.
    """
    # Check for API_ENDPOINT_KEY directly from environment
    console.print("Environment variable check:", style="bold blue")
    llama_key = os.environ.get('API_ENDPOINT_KEY')
    console.print(f"API_ENDPOINT_KEY: {'Present' if llama_key else 'Not found'}")
    # Debugging sanity test:
    # if llama_key:
        # console.print(f"  Value starts with: {llama_key[:10]}...")
    
    # To check the rename bug:
    #console.print("Available environment variables:", style="bold blue")
    #env_vars = [key for key in os.environ.keys() if 'API' in key or 'KEY' in key or 'TOKEN' in key]
    #for var in env_vars:
    #    console.print(f"  {var}")
    #console.print("")
    # Get provider from args or config
    selected_provider = provider or get_llm_provider(ctx.config)
    
    if selected_provider == "api-endpoint":
        # Get API endpoint config
        api_endpoint_config = get_endpoint_config(ctx.config)
        api_base = api_base or api_endpoint_config.get("api_base")
        
        # Check for environment variables
        api_endpoint_key = os.environ.get('API_ENDPOINT_KEY')
        console.print(f"API_ENDPOINT_KEY environment variable: {'Found' if api_endpoint_key else 'Not found'}")
        
        # Set API key with priority: env var > config
        api_key = api_endpoint_key or api_endpoint_config.get("api_key")
        if api_key:
            console.print(f"API key source: {'Environment variable' if api_endpoint_key else 'Config file'}")
        
        model = api_endpoint_config.get("model")
        
        # Check API endpoint access
        with console.status(f"Checking API endpoint access..."):
            try:
                # Try to import OpenAI
                try:
                    from openai import OpenAI
                    from together import Together
                except ImportError:
                    console.print("L API endpoint package not installed", style="red")
                    console.print("Install with: pip install openai>=1.0.0 or together>=1.5.17", style="yellow")
                    return 1
                
                # Create client
                client_kwargs = {}
                if api_key:
                    client_kwargs['api_key'] = api_key
                if api_base:
                    client_kwargs['base_url'] = api_base
                
                # Check API access
                try:
                    match api_endpoint_config.get("type"):
                        case "openai":
                            client = OpenAI(**client_kwargs)
                        case "together":
                            client = Together(**client_kwargs)
                        case _:
                            raise Exception("Unsupported API endpoint. Only support OpenAI and Together")
                    # Try a simple models list request to check connectivity
                    models = client.models.list()
                    console.print(f" API endpoint access confirmed", style="green")
                    if api_base:
                        console.print(f"Using custom API base: {api_base}", style="green")
                    console.print(f"Default model: {model}", style="green")
                    return 0
                except Exception as e:
                    console.print(f"L Error connecting to API endpoint: {str(e)}", style="red")
                    if api_base:
                        console.print(f"Using custom API base: {api_base}", style="yellow")
                    if not api_key and not api_base:
                        console.print("API key is required. Set in config.yaml or as API_ENDPOINT_KEY env var", style="yellow")
                    return 1
            except Exception as e:
                console.print(f"L Error: {str(e)}", style="red")
                return 1
    else:
        # Default to vLLM
        # Get vLLM server details
        vllm_config = get_vllm_config(ctx.config)
        api_base = api_base or vllm_config.get("api_base")
        model = vllm_config.get("model")
        port = vllm_config.get("port", 8000)
        
        with console.status(f"Checking vLLM server at {api_base}..."):
            try:
                response = requests.get(f"{api_base}/models", timeout=2)
                if response.status_code == 200:
                    console.print(f" vLLM server is running at {api_base}", style="green")
                    console.print(f"Available models: {response.json()}")
                    return 0
                else:
                    console.print(f"L vLLM server is not available at {api_base}", style="red")
                    console.print(f"Error: Server returned status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                console.print(f"L vLLM server is not available at {api_base}", style="red")
                console.print(f"Error: {str(e)}")
                
            # Show instruction to start the server
            console.print("\nTo start the server, run:", style="yellow")
            console.print(f"vllm serve {model} --port {port}", style="bold blue")
            return 1


@app.command()
def ingest(
    input: str = typer.Argument(..., help="File, URL, or directory to parse"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Where to save the output"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Custom output filename (only for single files)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed progress (for directories)"
    ),
    preview: bool = typer.Option(
        False, "--preview", help="Preview files to be processed without actually processing them"
    ),
):
    """
    Parse documents (PDF, HTML, YouTube, DOCX, PPT, TXT) into clean text.
    
    Can process:
    - Single file: synthetic-data-kit ingest document.pdf
    - Directory: synthetic-data-kit ingest ./documents/
    - URL: synthetic-data-kit ingest https://example.com/page.html
    """
    import os
    from synthetic_data_kit.core.ingest import process_file
    from synthetic_data_kit.utils.directory_processor import is_directory, process_directory_ingest
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "parsed")
    
    try:
        # Check if input is a directory
        if is_directory(input):
            # Process directory
            if name is not None:
                console.print("Warning: --name option is ignored when processing directories", style="yellow")
            
            # Preview mode - show files without processing
            if preview:
                from synthetic_data_kit.utils.directory_processor import get_directory_stats, INGEST_EXTENSIONS
                
                console.print(f"Preview: scanning directory [bold]{input}[/bold]", style="blue")
                stats = get_directory_stats(input, INGEST_EXTENSIONS)
                
                if "error" in stats:
                    console.print(f"❌ {stats['error']}", style="red")
                    return 1
                
                console.print(f"\n📁 Directory: {input}")
                console.print(f"📄 Total files: {stats['total_files']}")
                console.print(f"✅ Supported files: {stats['supported_files']}")
                console.print(f"❌ Unsupported files: {stats['unsupported_files']}")
                
                if stats['supported_files'] > 0:
                    console.print(f"\n📋 Files that would be processed:")
                    for ext, count in stats['by_extension'].items():
                        console.print(f"  {ext}: {count} file(s)")
                    
                    console.print(f"\n📝 File list:")
                    for filename in stats['file_list']:
                        console.print(f"  • {filename}")
                    
                    console.print(f"\n💡 To process these files, run:")
                    console.print(f"   synthetic-data-kit ingest {input} --output-dir {output_dir}", style="bold blue")
                else:
                    console.print(f"\n⚠️  No supported files found.", style="yellow")
                    console.print(f"   Supported extensions: {', '.join(INGEST_EXTENSIONS)}", style="yellow")
                
                return 0
            
            console.print(f"Processing directory: [bold]{input}[/bold]", style="blue")
            results = process_directory_ingest(
                directory=input,
                output_dir=output_dir,
                config=ctx.config,
                verbose=verbose
            )
            
            # Return appropriate exit code
            if results["failed"] > 0:
                console.print(f"⚠️  Completed with {results['failed']} errors", style="yellow")
                return 1
            else:
                console.print("✅ All files processed successfully!", style="green")
                return 0
        else:
            # Process single file (existing logic)
            if preview:
                console.print("Preview mode is only available for directories. Processing single file...", style="yellow")
            
            with console.status(f"Processing {input}..."):
                output_path = process_file(input, output_dir, name, ctx.config)
            console.print(f"✅ Text successfully extracted to [bold]{output_path}[/bold]", style="green")
            return 0
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        return 1


@app.command()
def create(
    input: str = typer.Argument(..., help="File or directory to process"),
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
        None, "--num-pairs", "-n", help="Target number of QA pairs or CoT examples to generate"
    ),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", help="Size of text chunks for processing large documents (default: 4000)"
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None, "--chunk-overlap", help="Overlap between chunks in characters (default: 200)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
    preview: bool = typer.Option(
        False, "--preview", help="Preview files to be processed without actually processing them"
    ),
):
    """
    Generate content from text using local LLM inference.
    
    Can process:
    - Single file: synthetic-data-kit create document.txt --type qa
    - Directory: synthetic-data-kit create ./processed-text/ --type qa
    
    Content types:
    - qa: Generate question-answer pairs from .txt files (use --num-pairs to specify how many)
    - summary: Generate summaries from .txt files
    - cot: Generate Chain of Thought reasoning examples from .txt files (use --num-pairs to specify how many)
    - cot-enhance: Enhance existing conversations with Chain of Thought reasoning from .json files
      (use --num-pairs to limit the number of conversations to enhance, default is to enhance all)
      (for cot-enhance, the input must be a JSON file with either:
       - A single conversation in 'conversations' field
       - An array of conversation objects, each with a 'conversations' field
       - A direct array of conversation messages)
    """
    import os
    from synthetic_data_kit.core.create import process_file
    from synthetic_data_kit.utils.directory_processor import is_directory, process_directory_create, get_directory_stats, CREATE_EXTENSIONS
    
    # Check the LLM provider from config
    provider = get_llm_provider(ctx.config)
    console.print(f"🔗 Using {provider} provider", style="green")
    
    if provider == "api-endpoint":
        # Use API endpoint config
        api_endpoint_config = get_endpoint_config(ctx.config)
        api_base = api_base or api_endpoint_config.get("api_base")
        model = model or api_endpoint_config.get("model")
        # No server check needed for API endpoint
    else:
        # Use vLLM config
        vllm_config = get_vllm_config(ctx.config)
        api_base = api_base or vllm_config.get("api_base")
        model = model or vllm_config.get("model")
        
        # Check vLLM server availability
        try:
            response = requests.get(f"{api_base}/models", timeout=2)
            if response.status_code != 200:
                console.print(f"❌ Error: VLLM server not available at {api_base}", style="red")
                console.print("Please start the VLLM server with:", style="yellow")
                console.print(f"vllm serve {model}", style="bold blue")
                return 1
        except requests.exceptions.RequestException:
            console.print(f"❌ Error: VLLM server not available at {api_base}", style="red")
            console.print("Please start the VLLM server with:", style="yellow")
            console.print(f"vllm serve {model}", style="bold blue")
            return 1
    
    # Get output directory from args, then config, then default
    if output_dir is None:
        output_dir = get_path_config(ctx.config, "output", "generated")
    
    try:
        # Check if input is a directory
        if is_directory(input):
            # Preview mode - show files without processing
            if preview:
                # For cot-enhance, look for .json files, otherwise .txt files
                extensions = ['.json'] if content_type == "cot-enhance" else CREATE_EXTENSIONS
                
                console.print(f"Preview: scanning directory [bold]{input}[/bold] for {content_type} processing", style="blue")
                stats = get_directory_stats(input, extensions)
                
                if "error" in stats:
                    console.print(f"❌ {stats['error']}", style="red")
                    return 1
                
                console.print(f"\n📁 Directory: {input}")
                console.print(f"📄 Total files: {stats['total_files']}")
                console.print(f"✅ Supported files: {stats['supported_files']}")
                console.print(f"❌ Unsupported files: {stats['unsupported_files']}")
                
                if stats['supported_files'] > 0:
                    console.print(f"\n📋 Files that would be processed for {content_type}:")
                    for ext, count in stats['by_extension'].items():
                        console.print(f"  {ext}: {count} file(s)")
                    
                    console.print(f"\n📝 File list:")
                    for filename in stats['file_list']:
                        console.print(f"  • {filename}")
                    
                    console.print(f"\n💡 To process these files, run:")
                    console.print(f"   synthetic-data-kit create {input} --type {content_type} --output-dir {output_dir}", style="bold blue")
                else:
                    console.print(f"\n⚠️  No supported files found for {content_type}.", style="yellow")
                    if content_type == "cot-enhance":
                        console.print(f"   Looking for: .json files", style="yellow")
                    else:
                        console.print(f"   Looking for: .txt files", style="yellow")
                
                return 0
            
            console.print(f"Processing directory: [bold]{input}[/bold] for {content_type} generation", style="blue")
            results = process_directory_create(
                directory=input,
                output_dir=output_dir,
                config_path=ctx.config_path,
                api_base=api_base,
                model=model,
                content_type=content_type,
                num_pairs=num_pairs,
                verbose=verbose,
                provider=provider,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Return appropriate exit code
            if results["failed"] > 0:
                console.print(f"⚠️  Completed with {results['failed']} errors", style="yellow")
                return 1
            else:
                console.print("✅ All files processed successfully!", style="green")
                return 0
        else:
            # Process single file (existing logic)
            if preview:
                console.print("Preview mode is only available for directories. Processing single file...", style="yellow")
            
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
                    provider=provider,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            if output_path:
                console.print(f"✅ Content saved to [bold]{output_path}[/bold]", style="green")
            return 0
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        return 1


@app.command("curate")
def curate(
    input: str = typer.Argument(..., help="Input file or directory to clean"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (for single files) or directory (for directories)"
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
    preview: bool = typer.Option(
        False, "--preview", help="Preview files to be processed without actually processing them"
    ),
):
    """
    Clean and filter content based on quality.
    
    Can process:
    - Single file: synthetic-data-kit curate qa_pairs.json --threshold 8.0
    - Directory: synthetic-data-kit curate ./generated/ --threshold 8.0
    
    Processes .json files containing QA pairs and filters them based on quality ratings.
    """
    import os
    from synthetic_data_kit.core.curate import curate_qa_pairs
    from synthetic_data_kit.utils.directory_processor import is_directory, process_directory_curate, get_directory_stats, CURATE_EXTENSIONS
    
    # Check the LLM provider from config
    provider = get_llm_provider(ctx.config)
    
    console.print(f"🔗 Using {provider} provider", style="green")
    
    if provider == "api-endpoint":
        # Use API endpoint config
        api_endpoint_config = get_endpoint_config(ctx.config)
        api_base = api_base or api_endpoint_config.get("api_base")
        model = model or api_endpoint_config.get("model")
        # No server check needed for API endpoint
    else:
        # Use vLLM config
        vllm_config = get_vllm_config(ctx.config)
        api_base = api_base or vllm_config.get("api_base")
        model = model or vllm_config.get("model")

        # Check vLLM server availability
        try:
            response = requests.get(f"{api_base}/models", timeout=2)
            if response.status_code != 200:
                console.print(f"❌ Error: VLLM server not available at {api_base}", style="red")
                console.print("Please start the VLLM server with:", style="yellow")
                console.print(f"vllm serve {model}", style="bold blue")
                return 1
        except requests.exceptions.RequestException:
            console.print(f"❌ Error: VLLM server not available at {api_base}", style="red")
            console.print("Please start the VLLM server with:", style="yellow")
            console.print(f"vllm serve {model}", style="bold blue")
            return 1
    
    try:
        # Check if input is a directory
        if is_directory(input):
            # Preview mode - show files without processing
            if preview:
                console.print(f"Preview: scanning directory [bold]{input}[/bold] for curation", style="blue")
                stats = get_directory_stats(input, CURATE_EXTENSIONS)
                
                if "error" in stats:
                    console.print(f"❌ {stats['error']}", style="red")
                    return 1
                
                console.print(f"\n📁 Directory: {input}")
                console.print(f"📄 Total files: {stats['total_files']}")
                console.print(f"✅ Supported files: {stats['supported_files']}")
                console.print(f"❌ Unsupported files: {stats['unsupported_files']}")
                
                if stats['supported_files'] > 0:
                    console.print(f"\n📋 Files that would be curated:")
                    for ext, count in stats['by_extension'].items():
                        console.print(f"  {ext}: {count} file(s)")
                    
                    console.print(f"\n📝 File list:")
                    for filename in stats['file_list']:
                        console.print(f"  • {filename}")
                    
                    default_output = get_path_config(ctx.config, "output", "curated")
                    console.print(f"\n💡 To process these files, run:")
                    console.print(f"   synthetic-data-kit curate {input} --threshold {threshold or 7.0} --output {output or default_output}", style="bold blue")
                else:
                    console.print(f"\n⚠️  No supported files found for curation.", style="yellow")
                    console.print(f"   Looking for: .json files with QA pairs", style="yellow")
                
                return 0
            
            # Get default output directory if not provided
            if not output:
                output = get_path_config(ctx.config, "output", "curated")
            
            console.print(f"Processing directory: [bold]{input}[/bold] for curation", style="blue")
            results = process_directory_curate(
                directory=input,
                output_dir=output,
                threshold=threshold,
                api_base=api_base,
                model=model,
                config_path=ctx.config_path,
                verbose=verbose,
                provider=provider
            )
            
            # Return appropriate exit code
            if results["failed"] > 0:
                console.print(f"⚠️  Completed with {results['failed']} errors", style="yellow")
                return 1
            else:
                console.print("✅ All files processed successfully!", style="green")
                return 0
        else:
            # Process single file (existing logic)
            if preview:
                console.print("Preview mode is only available for directories. Processing single file...", style="yellow")
            
            # Get default output path from config if not provided
            if not output:
                curated_dir = get_path_config(ctx.config, "output", "curated")
                os.makedirs(curated_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(input))[0]
                output = os.path.join(curated_dir, f"{base_name}_cleaned.json")
            
            with console.status(f"Cleaning content from {input}..."):
                result_path = curate_qa_pairs(
                    input,
                    output,
                    threshold,
                    api_base,
                    model,
                    ctx.config_path,
                    verbose,
                    provider=provider
                )
            console.print(f"✅ Cleaned content saved to [bold]{result_path}[/bold]", style="green")
            return 0
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        return 1


@app.command("save-as")
def save_as(
    input: str = typer.Argument(..., help="Input file or directory to convert"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Output format [jsonl|alpaca|ft|chatml]"
    ),
    storage: str = typer.Option(
        "json", "--storage", help="Storage format [json|hf]",
        show_default=True
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (for single files) or directory (for directories)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
    preview: bool = typer.Option(
        False, "--preview", help="Preview files to be processed without actually processing them"
    ),
):
    """
    Convert to different formats for fine-tuning.
    
    Can process:
    - Single file: synthetic-data-kit save-as curated_file.json --format alpaca
    - Directory: synthetic-data-kit save-as ./curated/ --format alpaca
    
    The --format option controls the content format (how the data is structured).
    The --storage option controls how the data is stored (JSON file or HF dataset).
    
    When using --storage hf, the output will be a directory containing a Hugging Face 
    dataset in Arrow format, which is optimized for machine learning workflows.
    
    Processes .json files containing curated QA pairs and converts them to training formats.
    """
    import os
    from synthetic_data_kit.core.save_as import convert_format
    from synthetic_data_kit.utils.directory_processor import is_directory, process_directory_save_as, get_directory_stats, SAVE_AS_EXTENSIONS
    
    # Get format from args or config
    if not format:
        format_config = ctx.config.get("format", {})
        format = format_config.get("default", "jsonl")
    
    try:
        # Check if input is a directory
        if is_directory(input):
            # Preview mode - show files without processing
            if preview:
                console.print(f"Preview: scanning directory [bold]{input}[/bold] for format conversion", style="blue")
                stats = get_directory_stats(input, SAVE_AS_EXTENSIONS)
                
                if "error" in stats:
                    console.print(f"❌ {stats['error']}", style="red")
                    return 1
                
                console.print(f"\n📁 Directory: {input}")
                console.print(f"📄 Total files: {stats['total_files']}")
                console.print(f"✅ Supported files: {stats['supported_files']}")
                console.print(f"❌ Unsupported files: {stats['unsupported_files']}")
                
                if stats['supported_files'] > 0:
                    console.print(f"\n📋 Files that would be converted to {format} format:")
                    for ext, count in stats['by_extension'].items():
                        console.print(f"  {ext}: {count} file(s)")
                    
                    console.print(f"\n📝 File list:")
                    for filename in stats['file_list']:
                        console.print(f"  • {filename}")
                    
                    default_output = get_path_config(ctx.config, "output", "final")
                    console.print(f"\n💡 To process these files, run:")
                    console.print(f"   synthetic-data-kit save-as {input} --format {format} --storage {storage} --output {output or default_output}", style="bold blue")
                else:
                    console.print(f"\n⚠️  No supported files found for format conversion.", style="yellow")
                    console.print(f"   Looking for: .json files with curated QA pairs", style="yellow")
                
                return 0
            
            # Get default output directory if not provided
            if not output:
                output = get_path_config(ctx.config, "output", "final")
            
            console.print(f"Processing directory: [bold]{input}[/bold] for format conversion to {format}", style="blue")
            results = process_directory_save_as(
                directory=input,
                output_dir=output,
                format=format,
                storage_format=storage,
                config=ctx.config,
                verbose=verbose
            )
            
            # Return appropriate exit code
            if results["failed"] > 0:
                console.print(f"⚠️  Completed with {results['failed']} errors", style="yellow")
                return 1
            else:
                console.print("✅ All files converted successfully!", style="green")
                return 0
        else:
            # Process single file (existing logic)
            if preview:
                console.print("Preview mode is only available for directories. Processing single file...", style="yellow")
            
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
            
            with console.status(f"Converting {input} to {format} format with {storage} storage..."):
                output_path = convert_format(
                    input,
                    output,
                    format,
                    ctx.config,
                    storage_format=storage
                )
            
            if storage == "hf":
                console.print(f"✅ Converted to {format} format and saved as HF dataset to [bold]{output_path}[/bold]", style="green")
            else:
                console.print(f"✅ Converted to {format} format and saved to [bold]{output_path}[/bold]", style="green")
            return 0
            
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        return 1


@app.command("server")
def server(
    host: str = typer.Option(
        "127.0.0.1", "--host", help="Host address to bind the server to"
    ),
    port: int = typer.Option(
        5000, "--port", "-p", help="Port to run the server on"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Run the server in debug mode"
    ),
):
    """
    Start a web interface for the Synthetic Data Kit.
    
    This launches a web server that provides a UI for all SDK functionality,
    including generating and curating QA pairs, as well as viewing
    and managing generated files.
    """
    provider = get_llm_provider(ctx.config)
    console.print(f"Starting web server with {provider} provider...", style="green")
    console.print(f"Web interface available at: http://{host}:{port}", style="bold green")
    console.print("Press CTRL+C to stop the server.", style="italic")
    
    # Run the Flask server
    run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    app()

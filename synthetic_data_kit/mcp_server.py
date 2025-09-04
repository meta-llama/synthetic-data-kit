"""
MCP (Model Context Protocol) server for the Synthetic Data Kit.
This server allows MCP-compatible clients to interact with the Synthetic Data Kit.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult, Prompt, PromptArgument, PromptMessage, TextContent,
    Tool, ListRootsResult, Root
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("synthetic-data-kit-mcp")

# Register tool definitions
@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools in the Synthetic Data Kit."""
    return [
        Tool(
            name="sdk_ingest",
            description="Parse documents (PDF, HTML, YouTube, DOCX, PPT, TXT) into clean text",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "File, URL, or directory to parse"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Where to save the output"
                    },
                    "name": {
                        "type": "string",
                        "description": "Custom output filename (only for single files)"
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="sdk_create",
            description="Generate content from text using local LLM inference",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "File or directory to process"
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["qa", "summary", "cot", "cot-enhance", "multimodal-qa"],
                        "description": "Type of content to generate"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Where to save the output"
                    },
                    "num_pairs": {
                        "type": "integer",
                        "description": "Target number of QA pairs or CoT examples to generate"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use"
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="sdk_curate",
            description="Clean and filter content based on quality",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input file or directory to clean"
                    },
                    "output": {
                        "type": "string",
                        "description": "Output file path (for single files) or directory (for directories)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Quality threshold (1-10)"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use"
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="sdk_save_as",
            description="Convert to different formats for fine-tuning",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input file or directory to convert"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["jsonl", "alpaca", "ft", "chatml"],
                        "description": "Output format"
                    },
                    "storage": {
                        "type": "string",
                        "enum": ["json", "hf"],
                        "description": "Storage format"
                    },
                    "output": {
                        "type": "string",
                        "description": "Output file path (for single files) or directory (for directories)"
                    }
                },
                "required": ["input"]
            }
        ),
        Tool(
            name="sdk_system_check",
            description="Check if the selected LLM provider's server is running",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "enum": ["vllm", "api-endpoint"],
                        "description": "Provider to check"
                    },
                    "api_base": {
                        "type": "string",
                        "description": "API base URL to check"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute a tool from the Synthetic Data Kit."""
    logger.info(f"Calling tool: {name} with arguments: {arguments}")
    
    try:
        # Map tool names to CLI commands
        if name == "sdk_ingest":
            cmd = ["synthetic-data-kit", "ingest"]
            if "input" in arguments:
                cmd.append(arguments["input"])
            if "output_dir" in arguments:
                cmd.extend(["--output-dir", arguments["output_dir"]])
            if "name" in arguments:
                cmd.extend(["--name", arguments["name"]])
                
        elif name == "sdk_create":
            cmd = ["synthetic-data-kit", "create"]
            if "input" in arguments:
                cmd.append(arguments["input"])
            if "content_type" in arguments:
                cmd.extend(["--type", arguments["content_type"]])
            if "output_dir" in arguments:
                cmd.extend(["--output-dir", arguments["output_dir"]])
            if "num_pairs" in arguments:
                cmd.extend(["--num-pairs", str(arguments["num_pairs"])])
            if "model" in arguments:
                cmd.extend(["--model", arguments["model"]])
                
        elif name == "sdk_curate":
            cmd = ["synthetic-data-kit", "curate"]
            if "input" in arguments:
                cmd.append(arguments["input"])
            if "output" in arguments:
                cmd.extend(["--output", arguments["output"]])
            if "threshold" in arguments:
                cmd.extend(["--threshold", str(arguments["threshold"])])
            if "model" in arguments:
                cmd.extend(["--model", arguments["model"]])
                
        elif name == "sdk_save_as":
            cmd = ["synthetic-data-kit", "save-as"]
            if "input" in arguments:
                cmd.append(arguments["input"])
            if "format" in arguments:
                cmd.extend(["--format", arguments["format"]])
            if "storage" in arguments:
                cmd.extend(["--storage", arguments["storage"]])
            if "output" in arguments:
                cmd.extend(["--output", arguments["output"]])
                
        elif name == "sdk_system_check":
            cmd = ["synthetic-data-kit", "system-check"]
            if "provider" in arguments:
                cmd.extend(["--provider", arguments["provider"]])
            if "api_base" in arguments:
                cmd.extend(["--api-base", arguments["api_base"]])
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        # Execute the command
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Format the output
        output = f"Command: {' '.join(cmd)}\n"
        output += f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"Stdout:\n{result.stdout}\n"
        if result.stderr:
            output += f"Stderr:\n{result.stderr}\n"
            
        return [TextContent(type="text", text=output)]
        
    except Exception as e:
        logger.error(f"Error executing tool {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")]

@server.list_prompts()
async def list_prompts() -> List[Prompt]:
    """List all available prompts."""
    return [
        Prompt(
            name="sdk-workflow",
            description="A complete workflow for generating synthetic data",
            arguments=[
                PromptArgument(
                    name="document_path",
                    description="Path to the document to process",
                    required=True
                ),
                PromptArgument(
                    name="content_type",
                    description="Type of content to generate (qa, summary, cot)",
                    required=False
                ),
                PromptArgument(
                    name="num_pairs",
                    description="Number of QA pairs to generate",
                    required=False
                )
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: Dict[str, Any]) -> GetPromptResult:
    """Get a prompt by name."""
    if name == "sdk-workflow":
        document_path = arguments.get("document_path", "")
        content_type = arguments.get("content_type", "qa")
        num_pairs = arguments.get("num_pairs", 25)
        
        # Create a prompt that describes the workflow
        messages = [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Please help me process the document at '{document_path}' using the Synthetic Data Kit.\n"
                         f"I want to generate {content_type} content with {num_pairs} pairs/examples.\n\n"
                         f"Here's the workflow I'd like to follow:\n"
                         f"1. Ingest the document to extract clean text\n"
                         f"2. Create {content_type} content from the text\n"
                         f"3. Curate the generated content for quality\n"
                         f"4. Save the final result in an appropriate format\n\n"
                         f"Can you generate the appropriate commands for this workflow?"
                )
            )
        ]
        return GetPromptResult(messages=messages)
    
    raise ValueError(f"Unknown prompt: {name}")

async def main():
    """Main entry point for the MCP server."""
    # Create the server session
    async with stdio_server() as (read_stream, write_stream):
        # Create initialization options
        from mcp.server import InitializationOptions
        from mcp.types import ServerCapabilities, ToolsCapability, PromptsCapability
        
        capabilities = ServerCapabilities(
            tools=ToolsCapability(),
            prompts=PromptsCapability()
        )
        
        initialization_options = InitializationOptions(
            server_name="synthetic-data-kit",
            server_version="0.1.0",
            capabilities=capabilities
        )
        
        await server.run(read_stream, write_stream, initialization_options)

if __name__ == "__main__":
    asyncio.run(main())
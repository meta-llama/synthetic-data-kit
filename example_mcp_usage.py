#!/usr/bin/env python3
"""
Example script demonstrating how to use the Synthetic Data Kit MCP server.
This script shows how to process a document using the MCP server.
"""

import asyncio
import json
import sys
import os
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


async def process_document_example():
    """Example of processing a document using the MCP server."""
    # Get the current working directory
    cwd = os.getcwd()
    
    # Start the MCP server as a subprocess
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "synthetic_data_kit.mcp_server"],
        cwd=cwd
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # Create a sample text file to process
            sample_text = """
            Artificial Intelligence (AI) is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. 
            This can include learning from experience, understanding natural language, solving problems, and recognizing patterns.
            
            Machine Learning (ML) is a subset of AI that focuses on algorithms and statistical models that enable computers to improve at tasks 
            with experience. Deep Learning is a further subset of ML that uses neural networks with multiple layers.
            
            Natural Language Processing (NLP) is another important area of AI that deals with the interaction between computers and humans 
            using natural language. It involves tasks like language translation, sentiment analysis, and text summarization.
            
            Computer Vision is yet another field that enables computers to interpret and understand visual information from the world, 
            including image and video recognition.
            """
            
            # Write the sample text to a file
            with open("sample_document.txt", "w") as f:
                f.write(sample_text)
            
            print("=== Processing Sample Document ===")
            print("Sample document created: sample_document.txt")
            
            # 1. Use the ingest tool to process the document
            print("\n1. Ingesting document...")
            try:
                result = await session.call_tool(
                    "sdk_ingest",
                    {
                        "input": "sample_document.txt",
                        "output_dir": "data/parsed"
                    }
                )
                print(f"Ingest result: {result}")
            except Exception as e:
                print(f"Error during ingestion: {e}")
            
            # 2. Use the create tool to generate QA pairs
            print("\n2. Creating QA pairs...")
            try:
                result = await session.call_tool(
                    "sdk_create",
                    {
                        "input": "data/parsed/sample_document.lance",
                        "content_type": "qa",
                        "num_pairs": 5
                    }
                )
                print(f"Create result: {result}")
            except Exception as e:
                print(f"Error during creation: {e}")
            
            # 3. Use the curate tool to filter content
            print("\n3. Curating content...")
            try:
                result = await session.call_tool(
                    "sdk_curate",
                    {
                        "input": "data/generated/sample_document_qa_pairs.json",
                        "threshold": 7.0
                    }
                )
                print(f"Curate result: {result}")
            except Exception as e:
                print(f"Error during curation: {e}")
            
            # 4. Use the save-as tool to convert format
            print("\n4. Saving in final format...")
            try:
                result = await session.call_tool(
                    "sdk_save_as",
                    {
                        "input": "data/curated/sample_document_cleaned.json",
                        "format": "alpaca"
                    }
                )
                print(f"Save-as result: {result}")
            except Exception as e:
                print(f"Error during format conversion: {e}")
            
            print("\n=== Document Processing Complete ===")
            
            # Clean up sample files
            try:
                os.remove("sample_document.txt")
                print("Cleaned up sample_document.txt")
            except:
                pass


if __name__ == "__main__":
    asyncio.run(process_document_example())
#!/usr/bin/env python3
"""
Test script for the Synthetic Data Kit MCP server.
This script demonstrates how to interact with the MCP server.
"""

import asyncio
import json
import sys
import os
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


async def test_mcp_server():
    """Test the Synthetic Data Kit MCP server."""
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
            
            # Test 1: List tools
            print("=== Testing tool listing ===")
            tools = await session.list_tools()
            print(f"Available tools: {tools}")
            
            # Test 2: List prompts
            print("\n=== Testing prompt listing ===")
            prompts = await session.list_prompts()
            print(f"Available prompts: {prompts}")
            
            print("\n=== MCP Server Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())

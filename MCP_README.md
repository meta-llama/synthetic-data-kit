# Synthetic Data Kit MCP Server

This directory contains an MCP (Model Context Protocol) server implementation for the Synthetic Data Kit. This allows MCP-compatible clients to interact with the Synthetic Data Kit through a standardized protocol.

## Features

The MCP server provides access to all major functionalities of the Synthetic Data Kit:

1. **Document Ingestion** - Parse documents (PDF, HTML, YouTube, DOCX, PPT, TXT) into clean text
2. **Content Creation** - Generate QA pairs, summaries, and Chain of Thought examples
3. **Content Curation** - Clean and filter content based on quality
4. **Format Conversion** - Convert to different formats for fine-tuning
5. **System Check** - Verify LLM provider connectivity

## Installation

The MCP server is automatically installed when you install the Synthetic Data Kit in development mode:

```bash
cd /path/to/synthetic-data-kit
pip install -e .
```

## Usage

### Starting the MCP Server

To start the MCP server directly:

```bash
synthetic-data-kit-mcp
```

This will start the server and listen for MCP connections over stdio.

### Using with an MCP Client

The server can be used with any MCP-compatible client. For example, if you're using Claude Desktop or another MCP client, you can configure it to connect to this server.

### Tools Available

The server exposes the following tools:

1. `sdk_ingest` - Parse documents into clean text
2. `sdk_create` - Generate content from text 
3. `sdk_curate` - Clean and filter content based on quality
4. `sdk_save_as` - Convert to different formats for fine-tuning
5. `sdk_system_check` - Check if the selected LLM provider's server is running

Each tool maps directly to the corresponding CLI command in the Synthetic Data Kit.

### Prompts Available

The server also provides prompts:

1. `sdk-workflow` - A complete workflow for generating synthetic data

## Example Usage

Here's an example of how an MCP client might interact with the server:

1. Client requests tool list → Server responds with available tools
2. Client calls `sdk_ingest` with a document path → Server runs `synthetic-data-kit ingest`
3. Client calls `sdk_create` with parameters → Server runs `synthetic-data-kit create`
4. Client calls `sdk_curate` to filter results → Server runs `synthetic-data-kit curate`
5. Client calls `sdk_save_as` to convert formats → Server runs `synthetic-data-kit save-as`

## Development

To test the MCP server:

```bash
cd /path/to/synthetic-data-kit
python test_mcp_server.py
```

To run a complete example:

```bash
cd /path/to/synthetic-data-kit
python example_mcp_usage.py
```

## Architecture

The MCP server acts as a bridge between MCP clients and the Synthetic Data Kit CLI:

```
MCP Client ↔ MCP Server ↔ Synthetic Data Kit CLI
```

All commands are executed as subprocess calls to the CLI, ensuring full compatibility with existing functionality.

## Configuration

The MCP server uses the same configuration as the Synthetic Data Kit CLI. Make sure your `config.yaml` is properly set up before using the server.

## Troubleshooting

If you encounter issues:

1. Ensure the Synthetic Data Kit is properly installed: `pip install -e .`
2. Verify the CLI works: `synthetic-data-kit --help`
3. Check that required dependencies are installed
4. Ensure your LLM provider (vLLM or API endpoint) is properly configured and running

Note: For API endpoints, you'll need to set the appropriate API keys in your environment or configuration file.
"""Integration tests for backward compatibility with single-file processing."""

import os
import tempfile
import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.mark.integration
def test_single_file_ingest_still_works(patch_config):
    """Test that single file ingest processing works unchanged."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is test content for single file processing.")
        input_file = f.name
        
    output_dir = tempfile.mkdtemp()
    
    try:
        # Mock the core process_file function
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
            expected_output = os.path.join(output_dir, "test_output.txt")
            mock_process.return_value = expected_output
            
            # Test single file processing through CLI
            from synthetic_data_kit.cli import app
            from typer.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(app, ['ingest', input_file, '--output-dir', output_dir])
            
            # Should process single file successfully
            assert result.exit_code == 0
            assert "successfully extracted" in result.stdout
            
            # Should call process_file once
            mock_process.assert_called_once()
            call_args = mock_process.call_args[0]
            assert call_args[0] == input_file  # input file path
            assert str(call_args[1]) == output_dir  # output directory (Path object converted to string)
            
    finally:
        os.unlink(input_file)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_single_file_create_still_works(patch_config):
    """Test that single file create processing works unchanged."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is test content for QA generation.")
        input_file = f.name
        
    output_dir = tempfile.mkdtemp()
    
    try:
        # Mock the core process_file function and LLM components
        with patch("synthetic_data_kit.core.create.process_file") as mock_process:
            expected_output = os.path.join(output_dir, "test_qa.json")
            mock_process.return_value = expected_output
            
            # Test single file processing through CLI
            from synthetic_data_kit.cli import app
            from typer.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(app, ['create', input_file, '--type', 'qa', '--output-dir', output_dir])
            
            # Should process single file successfully
            assert result.exit_code == 0
            assert "Content saved to" in result.stdout
            
            # Should call process_file once
            mock_process.assert_called_once()
            
    finally:
        os.unlink(input_file)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_single_file_curate_still_works(patch_config):
    """Test that single file curate processing works unchanged."""
    # Create test QA pairs file
    qa_pairs = {
        "qa_pairs": [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(qa_pairs, f)
        input_file = f.name
        
    output_dir = tempfile.mkdtemp()
    
    try:
        # Mock the core curate function
        with patch("synthetic_data_kit.core.curate.curate_qa_pairs") as mock_curate:
            expected_output = os.path.join(output_dir, "curated.json")
            mock_curate.return_value = expected_output
            
            # Test single file processing through CLI
            from synthetic_data_kit.cli import app
            from typer.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(app, ['curate', input_file, '--threshold', '7.0', '--output', expected_output])
            
            # Should process single file successfully
            assert result.exit_code == 0
            assert "Cleaned content saved to" in result.stdout
            
            # Should call curate_qa_pairs once
            mock_curate.assert_called_once()
            
    finally:
        os.unlink(input_file)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_single_file_save_as_still_works(patch_config):
    """Test that single file save-as processing works unchanged."""
    # Create test QA pairs file
    qa_pairs = {
        "qa_pairs": [
            {"question": "What is AI?", "answer": "Artificial Intelligence"},
            {"question": "What is ML?", "answer": "Machine Learning"}
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(qa_pairs, f)
        input_file = f.name
        
    output_dir = tempfile.mkdtemp()
    
    try:
        # Mock the core convert_format function
        with patch("synthetic_data_kit.core.save_as.convert_format") as mock_convert:
            expected_output = os.path.join(output_dir, "converted.jsonl")
            mock_convert.return_value = expected_output
            
            # Test single file processing through CLI
            from synthetic_data_kit.cli import app
            from typer.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(app, ['save-as', input_file, '--format', 'jsonl', '--output', expected_output])
            
            # Should process single file successfully
            assert result.exit_code == 0
            assert "Converted to jsonl format" in result.stdout
            
            # Should call convert_format once
            mock_convert.assert_called_once()
            
    finally:
        os.unlink(input_file)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_single_file_with_name_option():
    """Test that --name option still works for single files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content with custom name.")
        input_file = f.name
        
    output_dir = tempfile.mkdtemp()
    custom_name = "custom_output_name"
    
    try:
        # Mock the core process_file function
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
            expected_output = os.path.join(output_dir, f"{custom_name}.txt")
            mock_process.return_value = expected_output
            
            # Test single file processing with custom name
            from synthetic_data_kit.cli import app
            from typer.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(app, ['ingest', input_file, '--output-dir', output_dir, '--name', custom_name])
            
            # Should process single file successfully
            assert result.exit_code == 0
            
            # Should call process_file with custom name
            mock_process.assert_called_once()
            call_args = mock_process.call_args[0]
            assert call_args[2] == custom_name  # name parameter
            
    finally:
        os.unlink(input_file)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_single_file_error_handling_unchanged():
    """Test that single file error handling works as before."""
    # Test with non-existent file by calling the CLI function directly
    from synthetic_data_kit.cli import ingest
    
    # Mock the context to avoid directory creation issues
    with patch("synthetic_data_kit.cli.ctx") as mock_ctx:
        mock_ctx.config = {}
        
        # Call ingest directly with non-existent file
        try:
            result = ingest('/path/that/does/not/exist.txt')
            # Should return 1 for error
            assert result == 1
        except SystemExit as e:
            # Or should raise SystemExit with code 1
            assert e.code == 1


@pytest.mark.integration
def test_directory_name_option_ignored():
    """Test that --name option is ignored for directories with warning."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create a test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Test content")
            
        # Mock the directory processor
        with patch("synthetic_data_kit.utils.directory_processor.process_directory_ingest") as mock_process:
            mock_process.return_value = {"total_files": 1, "successful": 1, "failed": 0}
            
            # Test directory processing with --name option
            from synthetic_data_kit.cli import app
            from typer.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(app, ['ingest', temp_dir, '--name', 'ignored_name'])
            
            # Should show warning about ignored --name option
            assert result.exit_code == 0
            assert "Warning: --name option is ignored when processing directories" in result.stdout
            
    finally:
        os.unlink(test_file)
        os.rmdir(temp_dir)
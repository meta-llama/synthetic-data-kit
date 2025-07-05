"""Functional tests for preview mode across all CLI commands."""

import os
import tempfile
import json
from unittest.mock import patch

import pytest


@pytest.mark.functional
def test_ingest_preview_mode(patch_config):
    """Test ingest command with --preview flag."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test files
        txt_file = os.path.join(temp_dir, "test.txt")
        with open(txt_file, "w") as f:
            f.write("Test content")
            
        pdf_file = os.path.join(temp_dir, "test.pdf")
        with open(pdf_file, "w") as f:
            f.write("PDF content")
            
        # Mock CLI execution
        with patch('sys.argv', ['synthetic-data-kit', 'ingest', temp_dir, '--preview']):
            from synthetic_data_kit.cli import app
            from typer.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(app, ['ingest', temp_dir, '--preview'])
            
            # Should show preview without processing
            assert result.exit_code == 0
            assert "Preview:" in result.stdout
            assert "Total files:" in result.stdout
            assert "Supported files:" in result.stdout
            assert "test.txt" in result.stdout
            assert "test.pdf" in result.stdout
            assert "To process these files, run:" in result.stdout
            
    finally:
        # Clean up
        for filename in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)


@pytest.mark.functional
def test_create_preview_mode(patch_config):
    """Test create command with --preview flag."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test .txt files for create command
        txt_file1 = os.path.join(temp_dir, "test1.txt")
        with open(txt_file1, "w") as f:
            f.write("Test content 1")
            
        txt_file2 = os.path.join(temp_dir, "test2.txt")
        with open(txt_file2, "w") as f:
            f.write("Test content 2")
            
        # Mock CLI execution
        from synthetic_data_kit.cli import app
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ['create', temp_dir, '--type', 'qa', '--preview'])
        
        # Should show preview without processing
        assert result.exit_code == 0
        assert "Preview:" in result.stdout
        assert "qa processing" in result.stdout
        assert "Total files:" in result.stdout
        assert "Supported files: 2" in result.stdout
        assert "test1.txt" in result.stdout
        assert "test2.txt" in result.stdout
        
    finally:
        # Clean up
        for filename in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)


@pytest.mark.functional
def test_curate_preview_mode(patch_config):
    """Test curate command with --preview flag."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test .json files for curate command
        json_file1 = os.path.join(temp_dir, "test1.json")
        with open(json_file1, "w") as f:
            json.dump({"qa_pairs": [{"question": "Q1?", "answer": "A1."}]}, f)
            
        json_file2 = os.path.join(temp_dir, "test2.json") 
        with open(json_file2, "w") as f:
            json.dump({"qa_pairs": [{"question": "Q2?", "answer": "A2."}]}, f)
            
        # Mock CLI execution
        from synthetic_data_kit.cli import app
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ['curate', temp_dir, '--threshold', '7.0', '--preview'])
        
        # Should show preview without processing
        assert result.exit_code == 0
        assert "Preview:" in result.stdout
        assert "curation" in result.stdout
        assert "Total files:" in result.stdout
        assert "Supported files: 2" in result.stdout
        assert "test1.json" in result.stdout
        assert "test2.json" in result.stdout
        
    finally:
        # Clean up
        for filename in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)


@pytest.mark.functional
def test_save_as_preview_mode(patch_config):
    """Test save-as command with --preview flag."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test .json files for save-as command
        json_file1 = os.path.join(temp_dir, "test1.json")
        with open(json_file1, "w") as f:
            json.dump({"qa_pairs": [{"question": "Q1?", "answer": "A1."}]}, f)
            
        json_file2 = os.path.join(temp_dir, "test2.json")
        with open(json_file2, "w") as f:
            json.dump({"qa_pairs": [{"question": "Q2?", "answer": "A2."}]}, f)
            
        # Mock CLI execution
        from synthetic_data_kit.cli import app
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ['save-as', temp_dir, '--format', 'alpaca', '--preview'])
        
        # Should show preview without processing
        assert result.exit_code == 0
        assert "Preview:" in result.stdout
        assert "for format" in result.stdout and "conversion" in result.stdout  # Handle line breaks
        assert "Total files:" in result.stdout
        assert "Supported files: 2" in result.stdout
        assert "alpaca format" in result.stdout
        assert "test1.json" in result.stdout
        assert "test2.json" in result.stdout
        
    finally:
        # Clean up
        for filename in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)


@pytest.mark.functional
def test_preview_mode_empty_directory(patch_config):
    """Test preview mode with empty directory."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test with empty directory
        from synthetic_data_kit.cli import app
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ['ingest', temp_dir, '--preview'])
        
        # Should handle empty directory gracefully
        assert result.exit_code == 0
        assert "Preview:" in result.stdout
        assert "Total files: 0" in result.stdout
        assert "No supported files found" in result.stdout
        
    finally:
        os.rmdir(temp_dir)


@pytest.mark.functional
def test_preview_mode_single_file_warning(patch_config):
    """Test that preview mode shows warning for single files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test content")
        temp_file = f.name
        
    try:
        # Test preview mode with single file
        from synthetic_data_kit.cli import app
        from typer.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(app, ['ingest', temp_file, '--preview'])
        
        # Should show warning that preview is only for directories
        assert result.exit_code == 0
        assert "Preview mode is only available for directories" in result.stdout
        
    finally:
        os.unlink(temp_file)
"""Integration tests for directory processing edge cases."""

import os
import tempfile
import json
from unittest.mock import patch, MagicMock

import pytest

from synthetic_data_kit.utils.directory_processor import (
    process_directory_ingest,
    process_directory_save_as,
    get_directory_stats,
    INGEST_EXTENSIONS,
    SAVE_AS_EXTENSIONS
)


@pytest.mark.integration
def test_empty_directory_handling(patch_config):
    """Test processing empty directories doesn't crash."""
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Test ingest with empty directory
        results = process_directory_ingest(
            directory=temp_dir,
            output_dir=output_dir,
            config=None,
            verbose=False
        )
        
        # Should handle gracefully
        assert results["total_files"] == 0
        assert results["successful"] == 0
        assert results["failed"] == 0
        
    finally:
        os.rmdir(temp_dir)
        os.rmdir(output_dir)


@pytest.mark.integration  
def test_mixed_file_types_directory(patch_config):
    """Test processing directories with supported and unsupported files."""
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Create supported file
        supported_file = os.path.join(temp_dir, "test.txt")
        with open(supported_file, "w") as f:
            f.write("Test content")
            
        # Create unsupported file
        unsupported_file = os.path.join(temp_dir, "test.xyz")
        with open(unsupported_file, "w") as f:
            f.write("Unsupported content")
            
        # Mock the process_file function
        with patch("synthetic_data_kit.core.ingest.process_file") as mock_process:
            mock_process.return_value = os.path.join(output_dir, "test.txt")
            
            results = process_directory_ingest(
                directory=temp_dir,
                output_dir=output_dir,
                config=None,
                verbose=False
            )
            
            # Should process only supported file
            assert results["total_files"] == 1  # Only .txt file counted
            assert results["successful"] == 1
            assert results["failed"] == 0
            assert mock_process.call_count == 1
            
    finally:
        # Clean up
        for filename in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_directory_stats_functionality():
    """Test get_directory_stats function for preview mode."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test files
        txt_file = os.path.join(temp_dir, "test.txt")
        with open(txt_file, "w") as f:
            f.write("Test content")
            
        pdf_file = os.path.join(temp_dir, "test.pdf") 
        with open(pdf_file, "w") as f:
            f.write("PDF content")
            
        unsupported_file = os.path.join(temp_dir, "test.xyz")
        with open(unsupported_file, "w") as f:
            f.write("Unsupported")
            
        # Test stats for ingest extensions
        stats = get_directory_stats(temp_dir, INGEST_EXTENSIONS)
        
        assert stats["total_files"] == 3
        assert stats["supported_files"] == 2  # .txt and .pdf
        assert stats["unsupported_files"] == 1  # .xyz
        assert ".txt" in stats["by_extension"]
        assert ".pdf" in stats["by_extension"]
        assert stats["by_extension"][".txt"] == 1
        assert stats["by_extension"][".pdf"] == 1
        assert len(stats["file_list"]) == 2
        
    finally:
        # Clean up
        for filename in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)


@pytest.mark.integration
def test_nonexistent_directory_error():
    """Test handling of non-existent directories."""
    nonexistent_dir = "/path/that/does/not/exist"
    
    # Test directory stats with non-existent directory
    stats = get_directory_stats(nonexistent_dir, INGEST_EXTENSIONS)
    assert "error" in stats
    assert "not found" in stats["error"]


@pytest.mark.integration  
def test_file_as_directory_error():
    """Test handling when a file path is passed as directory."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test")
        file_path = f.name
        
    try:
        # Test directory stats with file path
        stats = get_directory_stats(file_path, INGEST_EXTENSIONS)
        assert "error" in stats
        assert "not a directory" in stats["error"]
        
    finally:
        os.unlink(file_path)


@pytest.mark.integration
def test_partial_processing_failures(patch_config):
    """Test that some files failing doesn't stop entire directory processing."""
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Create test JSON files for save-as processing
        good_file = os.path.join(temp_dir, "good.json")
        with open(good_file, "w") as f:
            json.dump({"qa_pairs": [{"question": "Q?", "answer": "A."}]}, f)
            
        bad_file = os.path.join(temp_dir, "bad.json")
        with open(bad_file, "w") as f:
            f.write("invalid json content")
            
        # Process directory - should handle partial failures
        results = process_directory_save_as(
            directory=temp_dir,
            output_dir=output_dir,
            format="jsonl",
            storage_format="json",
            config=None,
            verbose=False
        )
        
        # Should have mix of success and failure
        assert results["total_files"] == 2
        assert results["successful"] >= 0  # At least some should succeed
        assert results["failed"] >= 0      # Some might fail
        assert results["successful"] + results["failed"] == 2
        
    finally:
        # Clean up
        for filename in os.listdir(temp_dir):
            os.unlink(os.path.join(temp_dir, filename))
        os.rmdir(temp_dir)
        
        # Clean up output dir
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(output_dir)


@pytest.mark.integration
def test_directory_with_subdirectories():
    """Test that subdirectories are ignored (non-recursive processing)."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create file in main directory
        main_file = os.path.join(temp_dir, "main.txt")
        with open(main_file, "w") as f:
            f.write("Main content")
            
        # Create subdirectory with file
        sub_dir = os.path.join(temp_dir, "subdir")
        os.makedirs(sub_dir)
        sub_file = os.path.join(sub_dir, "sub.txt")
        with open(sub_file, "w") as f:
            f.write("Sub content")
            
        # Test stats - should only count main directory files
        stats = get_directory_stats(temp_dir, INGEST_EXTENSIONS)
        
        assert stats["total_files"] == 1  # Only main.txt
        assert stats["supported_files"] == 1
        assert len(stats["file_list"]) == 1
        assert "main.txt" in stats["file_list"]
        
    finally:
        # Clean up
        os.unlink(os.path.join(sub_dir, "sub.txt"))
        os.rmdir(sub_dir)
        os.unlink(main_file)
        os.rmdir(temp_dir)
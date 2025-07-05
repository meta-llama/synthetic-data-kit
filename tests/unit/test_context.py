# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from synthetic_data_kit.core.context import AppContext


class TestAppContext:
    """Test cases for AppContext class"""

    def test_app_context_initialization_default_config(self):
        """Test AppContext initialization with default config path"""
        with patch("synthetic_data_kit.core.context.DEFAULT_CONFIG_PATH", "/fake/default/path"):
            with patch.object(AppContext, "_ensure_data_dirs"):
                context = AppContext()
                assert context.config_path == "/fake/default/path"
                assert context.config == {}

    def test_app_context_initialization_custom_config(self):
        """Test AppContext initialization with custom config path"""
        custom_path = Path("/custom/config/path")
        with patch.object(AppContext, "_ensure_data_dirs"):
            context = AppContext(config_path=custom_path)
            assert context.config_path == custom_path
            assert context.config == {}

    @patch("synthetic_data_kit.core.context.os.makedirs")
    @patch("synthetic_data_kit.core.context.load_config")
    def test_ensure_data_dirs(self, mock_load_config, mock_makedirs):
        """Test that _ensure_data_dirs creates all required directories"""
        # Mock config with default directory structure
        mock_config = {
            'paths': {
                'input': 'data/input',
                'output': {
                    'parsed': 'data/parsed',
                    'generated': 'data/generated', 
                    'curated': 'data/curated',
                    'final': 'data/final'
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch("synthetic_data_kit.core.context.DEFAULT_CONFIG_PATH", "/fake/path"):
            AppContext()

            # Verify all expected directories are created
            expected_dirs = [
                "data/input",
                "data/parsed",
                "data/generated",
                "data/curated",
                "data/final",
            ]

            assert mock_makedirs.call_count == len(expected_dirs)

            # Check that each expected directory was created with exist_ok=True
            for expected_dir in expected_dirs:
                mock_makedirs.assert_any_call(expected_dir, exist_ok=True)

    @patch("synthetic_data_kit.core.context.load_config")
    def test_ensure_data_dirs_integration(self, mock_load_config):
        """Integration test for directory creation"""
        # Mock config with default directory structure
        mock_config = {
            'paths': {
                'input': 'data/input',
                'output': {
                    'parsed': 'data/parsed',
                    'generated': 'data/generated', 
                    'curated': 'data/curated',
                    'final': 'data/final'
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for this test
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)

                with patch("synthetic_data_kit.core.context.DEFAULT_CONFIG_PATH", "/fake/path"):
                    AppContext()

                # Verify all directories were actually created
                expected_dirs = [
                    "data/input",
                    "data/parsed",
                    "data/generated",
                    "data/curated",
                    "data/final",
                ]

                for dir_path in expected_dirs:
                    full_path = os.path.join(temp_dir, dir_path)
                    assert os.path.exists(full_path), f"Directory {dir_path} was not created"
                    assert os.path.isdir(full_path), f"{dir_path} exists but is not a directory"

            finally:
                os.chdir(original_cwd)

    def test_config_attribute_mutable(self):
        """Test that config attribute can be modified"""
        with patch.object(AppContext, "_ensure_data_dirs"):
            context = AppContext()

            # Initially empty
            assert context.config == {}

            # Should be able to modify
            context.config["test_key"] = "test_value"
            assert context.config["test_key"] == "test_value"

            # Should be able to add nested structure
            context.config["nested"] = {"inner_key": "inner_value"}
            assert context.config["nested"]["inner_key"] == "inner_value"

    def test_multiple_instances_independent(self):
        """Test that multiple AppContext instances are independent"""
        with patch.object(AppContext, "_ensure_data_dirs"):
            context1 = AppContext(config_path=Path("/path1"))
            context2 = AppContext(config_path=Path("/path2"))

            # Different config paths
            assert context1.config_path != context2.config_path

            # Independent config dictionaries
            context1.config["key1"] = "value1"
            context2.config["key2"] = "value2"

            assert "key1" in context1.config
            assert "key1" not in context2.config
            assert "key2" in context2.config
            assert "key2" not in context1.config

    @patch("synthetic_data_kit.core.context.os.makedirs")
    @patch("synthetic_data_kit.core.context.load_config")
    def test_ensure_data_dirs_exception_handling(self, mock_load_config, mock_makedirs):
        """Test that AppContext handles directory creation errors gracefully"""
        # Mock config with default directory structure
        mock_config = {
            'paths': {
                'input': 'data/input',
                'output': {
                    'parsed': 'data/parsed',
                    'generated': 'data/generated', 
                    'curated': 'data/curated',
                    'final': 'data/final'
                }
            }
        }
        mock_load_config.return_value = mock_config
        
        # Mock makedirs to raise an exception
        mock_makedirs.side_effect = OSError("Permission denied")

        with patch("synthetic_data_kit.core.context.DEFAULT_CONFIG_PATH", "/fake/path"):
            # Should raise the OSError since _ensure_data_dirs doesn't catch exceptions
            with pytest.raises(OSError, match="Permission denied"):
                AppContext()

    def test_config_path_type_handling(self):
        """Test that config_path handles different path types correctly"""
        with patch.object(AppContext, "_ensure_data_dirs"):
            # Test with string path
            string_path = "/string/path"
            context1 = AppContext(config_path=string_path)
            assert context1.config_path == string_path

            # Test with Path object
            path_obj = Path("/path/object")
            context2 = AppContext(config_path=path_obj)
            assert context2.config_path == path_obj
            assert isinstance(context2.config_path, Path)

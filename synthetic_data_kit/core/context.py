# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Context Manager
from pathlib import Path
from typing import Optional, Dict, Any
import os

from synthetic_data_kit.utils.config import DEFAULT_CONFIG_PATH, load_config, get_path_config

class AppContext:
    """Context manager for global app state"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize app context"""
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
        
        # Ensure data directories exist
        self._ensure_data_dirs()
        
    # Why have separeate folders? Yes ideally you should just be able to ingest an input folder and have everything being ingested and converted BUT
    # Managing context window is hard and there are more edge cases which needs to be handled carefully
    # it's also easier to debug in alpha if we have multiple files. 
    def _ensure_data_dirs(self):
        """Ensure data directories exist based on configuration"""
        # Load config to get proper paths
        config = load_config(self.config_path)
        paths_config = config.get('paths', {})
        
        # Create input directory - handle new config format where input is a string
        input_dir = paths_config.get('input', 'data/input')
        os.makedirs(input_dir, exist_ok=True)
        
        # Create output directories based on config
        output_config = paths_config.get('output', {})
        output_dirs = [
            output_config.get('parsed', 'data/parsed'),
            output_config.get('generated', 'data/generated'), 
            output_config.get('curated', 'data/curated'),
            output_config.get('final', 'data/final'),
        ]
        
        for dir_path in output_dirs:
            os.makedirs(dir_path, exist_ok=True)
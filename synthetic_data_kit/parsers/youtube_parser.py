# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Download and save the transcript

import os
from typing import Dict, Any

class YouTubeParser:
    """Parser for YouTube transcripts"""
    
    def parse(self, url: str) -> str:
        """Parse a YouTube video transcript
        
        Args:
            url: YouTube video URL
            
        Returns:
            Transcript text
        """
        try:
            from pytubefix import YouTube
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError(
                "pytube and youtube-transcript-api are required for YouTube parsing. "
                "Install them with: pip install pytube youtube-transcript-api"
            )
        
        # Extract video ID from URL
        yt = YouTube(url)
        video_id = yt.video_id
        
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine transcript segments
        combined_text = []
        for segment in transcript:
            combined_text.append(segment['text'])
        
        # Add video metadata
        metadata = (
            f"Title: {yt.title}\n"
            f"Author: {yt.author}\n"
            f"Length: {yt.length} seconds\n"
            f"URL: {url}\n\n"
            f"Transcript:\n"
        )
        
        return metadata + "\n".join(combined_text)
    
    def save(self, content: str, output_path: str) -> None:
        """Save the transcript to a file
        
        Args:
            content: Transcript content
            output_path: Path to save the text
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
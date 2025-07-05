# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.parsers.docx_parser import DOCXParser
from synthetic_data_kit.parsers.ppt_parser import PPTParser
from synthetic_data_kit.parsers.youtube_parser import YouTubeParser


class TestDOCXParser:
    """Test cases for DOCX parser"""

    def test_docx_parser_initialization(self):
        """Test that DOCX parser can be initialized"""
        parser = DOCXParser()
        assert parser is not None

    @patch("builtins.__import__")
    def test_docx_parse_success(self, mock_import):
        """Test successful DOCX parsing with paragraphs and tables"""
        # Setup mock docx module
        mock_docx = MagicMock()
        mock_doc = MagicMock()
        mock_docx.Document.return_value = mock_doc
        mock_import.return_value = mock_docx

        # Mock paragraphs
        mock_paragraph1 = MagicMock()
        mock_paragraph1.text = "First paragraph"
        mock_paragraph2 = MagicMock()
        mock_paragraph2.text = "Second paragraph"
        mock_paragraph3 = MagicMock()
        mock_paragraph3.text = ""  # Empty paragraph should be filtered

        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3]

        # Mock tables
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell 2"

        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2]

        mock_table = MagicMock()
        mock_table.rows = [mock_row]

        mock_doc.tables = [mock_table]

        # Test parsing
        parser = DOCXParser()
        result = parser.parse("/fake/path.docx")

        expected = "First paragraph\n\nSecond paragraph\n\nCell 1\n\nCell 2"
        assert result == expected
        # Verify docx import was called (first argument of any call should be 'docx')
        import_calls = [call[0][0] for call in mock_import.call_args_list if call[0]]
        assert "docx" in import_calls
        mock_docx.Document.assert_called_once_with("/fake/path.docx")

    @patch("builtins.__import__")
    def test_docx_parse_import_error(self, mock_import):
        """Test ImportError when docx module is not available"""
        mock_import.side_effect = ImportError("No module named 'docx'")
        parser = DOCXParser()
        with pytest.raises(ImportError, match="python-docx is required"):
            parser.parse("/fake/path.docx")

    def test_docx_save(self):
        """Test saving DOCX content to file"""
        parser = DOCXParser()
        content = "Test content"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "output.txt")
            parser.save(content, output_path)

            # Verify file was created and content is correct
            assert os.path.exists(output_path)
            with open(output_path, encoding="utf-8") as f:
                assert f.read() == content


class TestPPTParser:
    """Test cases for PowerPoint parser"""

    def test_ppt_parser_initialization(self):
        """Test that PPT parser can be initialized"""
        parser = PPTParser()
        assert parser is not None

    @patch("builtins.__import__")
    def test_ppt_parse_success(self, mock_import):
        """Test successful PPTX parsing with slides and shapes"""
        # Setup mock pptx module
        mock_pptx = MagicMock()
        mock_presentation_class = MagicMock()
        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs
        mock_pptx.Presentation = mock_presentation_class
        mock_import.return_value = mock_pptx

        # Mock slide 1
        mock_slide1 = MagicMock()
        mock_title1 = MagicMock()
        mock_title1.text = "Slide 1 Title"

        # Setup shapes collection with title attribute
        mock_shapes1 = MagicMock()
        mock_shapes1.title = mock_title1
        mock_shapes1.__iter__ = lambda self: iter([mock_title1, MagicMock(text="Slide 1 content")])
        mock_slide1.shapes = mock_shapes1

        # Mock slide 2 (no title)
        mock_slide2 = MagicMock()
        mock_shapes2 = MagicMock()
        mock_shapes2.title = None
        mock_shape2 = MagicMock()
        mock_shape2.text = "Slide 2 content"
        mock_shapes2.__iter__ = lambda self: iter([mock_shape2])
        mock_slide2.shapes = mock_shapes2

        mock_prs.slides = [mock_slide1, mock_slide2]

        # Test parsing
        parser = PPTParser()
        result = parser.parse("/fake/path.pptx")

        expected_lines = [
            "--- Slide 1 ---",
            "Title: Slide 1 Title",
            "Slide 1 Title",
            "Slide 1 content",
            "",  # Empty line between slides
            "--- Slide 2 ---",
            "Slide 2 content",
        ]
        expected = "\n\n".join(["\n".join(expected_lines[:4]), "\n".join(expected_lines[5:])])
        assert result == expected
        # Verify pptx import was called
        import_calls = [call[0][0] for call in mock_import.call_args_list if call[0]]
        assert "pptx" in import_calls
        mock_presentation_class.assert_called_once_with("/fake/path.pptx")

    @patch("builtins.__import__")
    def test_ppt_parse_import_error(self, mock_import):
        """Test ImportError when pptx module is not available"""
        mock_import.side_effect = ImportError("No module named 'pptx'")
        parser = PPTParser()
        with pytest.raises(ImportError, match="python-pptx is required"):
            parser.parse("/fake/path.pptx")

    def test_ppt_save(self):
        """Test saving PPT content to file"""
        parser = PPTParser()
        content = "Test presentation content"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "output.txt")
            parser.save(content, output_path)

            # Verify file was created and content is correct
            assert os.path.exists(output_path)
            with open(output_path, encoding="utf-8") as f:
                assert f.read() == content


class TestYouTubeParser:
    """Test cases for YouTube parser"""

    def test_youtube_parser_initialization(self):
        """Test that YouTube parser can be initialized"""
        parser = YouTubeParser()
        assert parser is not None

    @patch("builtins.__import__")
    def test_youtube_parse_success(self, mock_import):
        """Test successful YouTube transcript parsing"""

        # Setup mock modules
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "pytubefix":
                mock_pytubefix = MagicMock()
                mock_yt = MagicMock()
                mock_yt.video_id = "test_video_id"
                mock_yt.title = "Test Video Title"
                mock_yt.author = "Test Author"
                mock_yt.length = 120
                mock_pytubefix.YouTube.return_value = mock_yt
                return mock_pytubefix
            elif name == "youtube_transcript_api":
                mock_transcript_api = MagicMock()
                mock_transcript = [
                    {"text": "Hello everyone"},
                    {"text": "Welcome to this video"},
                    {"text": "Today we'll learn about testing"},
                ]
                mock_transcript_api.YouTubeTranscriptApi.get_transcript.return_value = (
                    mock_transcript
                )
                return mock_transcript_api
            return MagicMock()  # fallback for other imports

        mock_import.side_effect = mock_import_side_effect

        # Test parsing
        parser = YouTubeParser()
        test_url = "https://www.youtube.com/watch?v=test_video_id"
        result = parser.parse(test_url)

        # Verify the result structure
        assert "Title: Test Video Title" in result
        assert "Author: Test Author" in result
        assert "Length: 120 seconds" in result
        assert test_url in result
        assert "Transcript:" in result
        assert "Hello everyone" in result
        assert "Welcome to this video" in result
        assert "Today we'll learn about testing" in result

        # Verify imports were called
        import_calls = [call[0][0] for call in mock_import.call_args_list]
        assert "pytubefix" in import_calls
        assert "youtube_transcript_api" in import_calls

    @patch("builtins.__import__")
    def test_youtube_parse_import_error_pytube(self, mock_import):
        """Test ImportError when YouTube module is not available"""

        def import_side_effect(name, *args, **kwargs):
            if name == "pytubefix":
                raise ImportError("No module named 'pytubefix'")
            return MagicMock()

        mock_import.side_effect = import_side_effect
        parser = YouTubeParser()
        with pytest.raises(ImportError, match="pytube and youtube-transcript-api are required"):
            parser.parse("https://www.youtube.com/watch?v=test")

    @patch("builtins.__import__")
    def test_youtube_parse_import_error_transcript(self, mock_import):
        """Test ImportError when transcript API is not available"""

        def import_side_effect(name, *args, **kwargs):
            if name == "youtube_transcript_api":
                raise ImportError("No module named 'youtube_transcript_api'")
            elif name == "pytubefix":
                return MagicMock()  # pytubefix succeeds
            return MagicMock()

        mock_import.side_effect = import_side_effect
        parser = YouTubeParser()
        with pytest.raises(ImportError, match="pytube and youtube-transcript-api are required"):
            parser.parse("https://www.youtube.com/watch?v=test")

    def test_youtube_save(self):
        """Test saving YouTube transcript to file"""
        parser = YouTubeParser()
        content = "Test transcript content"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "output.txt")
            parser.save(content, output_path)

            # Verify file was created and content is correct
            assert os.path.exists(output_path)
            with open(output_path, encoding="utf-8") as f:
                assert f.read() == content

    @patch("builtins.__import__")
    def test_youtube_parse_empty_transcript(self, mock_import):
        """Test parsing with empty transcript"""

        # Setup mock modules with empty transcript
        def mock_import_side_effect(name, *args, **kwargs):
            if name == "pytubefix":
                mock_pytubefix = MagicMock()
                mock_yt = MagicMock()
                mock_yt.video_id = "test_video_id"
                mock_yt.title = "Empty Video"
                mock_yt.author = "Test Author"
                mock_yt.length = 0
                mock_pytubefix.YouTube.return_value = mock_yt
                return mock_pytubefix
            elif name == "youtube_transcript_api":
                mock_transcript_api = MagicMock()
                mock_transcript_api.YouTubeTranscriptApi.get_transcript.return_value = []
                return mock_transcript_api
            return MagicMock()

        mock_import.side_effect = mock_import_side_effect

        # Test parsing
        parser = YouTubeParser()
        test_url = "https://www.youtube.com/watch?v=test_video_id"
        result = parser.parse(test_url)

        # Should still have metadata even with empty transcript
        assert "Title: Empty Video" in result
        assert "Author: Test Author" in result
        assert "Length: 0 seconds" in result
        assert "Transcript:" in result

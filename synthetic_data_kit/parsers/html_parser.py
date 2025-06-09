# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# HTML Parsers

import os
import tempfile
import logging
from pathlib import Path
from .pdf_parser import PDFParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HTMLParser:
    """Parser for HTML files and web pages using PDF conversion"""

    def parse(self, file_path: str, multimodal: bool = False) -> any:
        """Parse an HTML file or URL by converting to PDF first.
        
        Args:
            file_path: Path to the HTML file or URL.
            multimodal: If True, extract text chunks and associated images.
                        Otherwise, extract all text into a single string.
            
        Returns:
            If multimodal is False, returns a string with all extracted text.
            If multimodal is True, returns a list of dictionaries, where each
            dictionary has 'text' and 'image' keys.
            
        Raises:
            ValueError: If the page contains client-side errors.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError("playwright is required for HTML parsing. Install it with: pip install playwright && playwright install chromium")

        logger.info(f"Starting to parse: {file_path}")
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name
            
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                
                # Set viewport to a large size to ensure all content is visible
                page.set_viewport_size({"width": 1920, "height": 1080})
                
                if file_path.startswith(('http://', 'https://')):
                    logger.info("Loading URL...")
                    page.goto(file_path, wait_until='networkidle')
                else:
                    logger.info("Loading local file...")
                    file_url = Path(file_path).absolute().as_uri()
                    page.goto(file_url, wait_until='networkidle')
                
                # Check for client-side error
                error_text = page.evaluate("""
                    () => {
                        const bodyText = document.body.textContent;
                        if (bodyText.includes('Application error: a client-side exception has occurred')) {
                            return bodyText;
                        }
                        return null;
                    }
                """)
                
                if error_text:
                    raise ValueError("Page contains client-side error")
                
                # Wait for images to load
                logger.info("Waiting for images to load...")
                page.wait_for_load_state('domcontentloaded')
                
                # TODO - Get rid of this section -- redundant 
                logger.info("Scrolling through page to trigger lazy loading...")
                page.evaluate("""
                    () => {
                        return new Promise((resolve) => {
                            let totalHeight = 0;
                            const distance = 100;
                            const timer = setInterval(() => {
                                const scrollHeight = document.body.scrollHeight;
                                window.scrollBy(0, distance);
                                totalHeight += distance;
                                
                                if(totalHeight >= scrollHeight){
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 100);
                        });
                    }
                """)
                
                page.wait_for_timeout(2000)  # Wait 2 seconds for any remaining images
                
                logger.info("Converting to PDF...")
                # Generate PDF with better quality settings
                page.pdf(
                    path=temp_pdf_path,
                    format='A4',
                    print_background=True,
                    prefer_css_page_size=True,
                    scale=1.0,  # Ensure no scaling
                    margin={
                        'top': '20px',
                        'right': '20px',
                        'bottom': '20px',
                        'left': '20px'
                    }
                )
                browser.close()
                
            logger.info("PDF conversion completed, now parsing PDF...")
            # Use the PDF parser to process the converted file
            pdf_parser = PDFParser()
            result = pdf_parser.parse(temp_pdf_path, multimodal=multimodal)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing HTML: {str(e)}")
            raise
        finally:
            # Clean up temporary PDF file
            try:
                os.unlink(temp_pdf_path)
                logger.info("Temporary PDF file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary PDF file: {str(e)}")

    def save(self, content: any, output_path: str) -> None:
        """Save the extracted content to a Lance file.
        
        Args:
            content: Extracted content (string or list of dicts)
            output_path: Path to save the Lance file
        """
        logger.info(f"Saving content to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use the PDF parser's save method since we're using the same data structure
        pdf_parser = PDFParser()
        pdf_parser.save(content, output_path)
        logger.info("Save completed successfully")
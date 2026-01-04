"""
PDF Text Layer Extraction Module

This module handles extraction of text from the native text layer of PDFs.
It uses a two-library approach for maximum compatibility:
1. pdfplumber (primary) - excellent for complex layouts
2. PyMuPDF/fitz (fallback) - faster, handles some edge cases better

Why multiple libraries?
Real-world PDFs are created by dozens of different tools (Word, InDesign, 
LaTeX, scanners, etc.), each embedding text differently. No single library
handles all cases well. We try pdfplumber first because it preserves
layout better, then fall back to PyMuPDF if needed.

Performance considerations:
- pdfplumber is slower but more accurate for tables
- PyMuPDF is 3-5x faster but can mangle complex layouts
- For large batches, consider using PyMuPDF first, pdfplumber for failures
"""

from pathlib import Path
from typing import Optional, Generator
import io

from loguru import logger

from .utils import (
    ExtractionResult, 
    ExtractionMethod, 
    PageAnalysis,
    normalize_text,
    calculate_text_confidence,
    detect_page_orientation,
    is_likely_scanned,
)


class PDFTextExtractor:
    """
    Extracts text from PDF text layers using multiple backends.
    
    Usage:
        extractor = PDFTextExtractor()
        result = extractor.extract_text(Path("invoice.pdf"))
        print(result.text)
        print(f"Confidence: {result.confidence}")
    """
    
    def __init__(self, primary_backend: str = "pdfplumber"):
        """
        Initialize the text extractor.
        
        Args:
            primary_backend: Which library to try first ("pdfplumber" or "pymupdf")
        """
        self.primary_backend = primary_backend
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Verify required libraries are available."""
        self.has_pdfplumber = False
        self.has_pymupdf = False
        
        try:
            import pdfplumber
            self.has_pdfplumber = True
        except ImportError:
            logger.warning("pdfplumber not installed - some features may be limited")
        
        try:
            import fitz  # PyMuPDF
            self.has_pymupdf = True
        except ImportError:
            logger.warning("PyMuPDF not installed - some features may be limited")
        
        if not self.has_pdfplumber and not self.has_pymupdf:
            raise RuntimeError(
                "No PDF text extraction library available. "
                "Install pdfplumber or PyMuPDF: pip install pdfplumber PyMuPDF"
            )
    
    def analyze_pages(self, pdf_path: Path) -> list[PageAnalysis]:
        """
        Analyze each page of a PDF to determine extraction strategy.
        
        This is a lightweight pass that determines:
        - Which pages have text layers
        - Which pages appear to be scanned
        - Which pages contain tables
        
        Returns:
            List of PageAnalysis objects, one per page
        """
        analyses = []
        
        if self.has_pdfplumber:
            analyses = self._analyze_with_pdfplumber(pdf_path)
        elif self.has_pymupdf:
            analyses = self._analyze_with_pymupdf(pdf_path)
        
        return analyses
    
    def _analyze_with_pdfplumber(self, pdf_path: Path) -> list[PageAnalysis]:
        """Analyze pages using pdfplumber."""
        import pdfplumber
        
        analyses = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        # Extract text to count characters
                        text = page.extract_text() or ""
                        text_char_count = len(text.strip())
                        
                        # Analyze images
                        images = page.images or []
                        has_images = len(images) > 0
                        
                        # Calculate image coverage
                        page_area = page.width * page.height
                        image_area = sum(
                            (img.get('width', 0) or (img.get('x1', 0) - img.get('x0', 0))) * 
                            (img.get('height', 0) or (img.get('y1', 0) - img.get('y0', 0)))
                            for img in images
                        )
                        image_coverage = min(image_area / page_area, 1.0) if page_area > 0 else 0
                        
                        # Check for tables
                        tables = page.find_tables()
                        has_tables = len(tables) > 0 if tables else False
                        
                        # Determine if likely scanned
                        page_info = {
                            'image_count': len(images),
                            'image_area_ratio': image_coverage,
                            'text_char_count': text_char_count
                        }
                        likely_scanned = is_likely_scanned(page_info)
                        
                        analysis = PageAnalysis(
                            page_number=i,
                            has_text_layer=text_char_count > 0,
                            text_char_count=text_char_count,
                            has_images=has_images,
                            image_coverage=round(image_coverage, 3),
                            likely_scanned=likely_scanned,
                            has_tables=has_tables,
                            orientation=detect_page_orientation(page.width, page.height)
                        )
                        analyses.append(analysis)
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze page {i}: {e}")
                        # Add a default analysis for failed pages
                        analyses.append(PageAnalysis(
                            page_number=i,
                            has_text_layer=False,
                            text_char_count=0,
                            has_images=True,  # Assume it might need OCR
                            image_coverage=1.0,
                            likely_scanned=True,
                            has_tables=False
                        ))
                        
        except Exception as e:
            logger.error(f"Failed to open PDF for analysis: {e}")
            raise
        
        return analyses
    
    def _analyze_with_pymupdf(self, pdf_path: Path) -> list[PageAnalysis]:
        """Analyze pages using PyMuPDF."""
        import fitz
        
        analyses = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for i, page in enumerate(doc, start=1):
                try:
                    # Extract text
                    text = page.get_text() or ""
                    text_char_count = len(text.strip())
                    
                    # Get images
                    images = page.get_images()
                    has_images = len(images) > 0
                    
                    # Estimate image coverage (PyMuPDF doesn't give sizes easily)
                    # Use a heuristic based on number and text presence
                    if has_images and text_char_count < 100:
                        image_coverage = 0.9
                    elif has_images:
                        image_coverage = 0.3
                    else:
                        image_coverage = 0.0
                    
                    # Check for tables using text blocks structure
                    blocks = page.get_text("blocks")
                    # Simple heuristic: multiple aligned blocks might be tables
                    has_tables = len(blocks) > 5  # Very rough estimate
                    
                    page_info = {
                        'image_count': len(images),
                        'image_area_ratio': image_coverage,
                        'text_char_count': text_char_count
                    }
                    likely_scanned = is_likely_scanned(page_info)
                    
                    rect = page.rect
                    analysis = PageAnalysis(
                        page_number=i,
                        has_text_layer=text_char_count > 0,
                        text_char_count=text_char_count,
                        has_images=has_images,
                        image_coverage=image_coverage,
                        likely_scanned=likely_scanned,
                        has_tables=has_tables,
                        orientation=detect_page_orientation(rect.width, rect.height)
                    )
                    analyses.append(analysis)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze page {i}: {e}")
                    analyses.append(PageAnalysis(
                        page_number=i,
                        has_text_layer=False,
                        text_char_count=0,
                        has_images=True,
                        image_coverage=1.0,
                        likely_scanned=True,
                        has_tables=False
                    ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Failed to open PDF for analysis: {e}")
            raise
        
        return analyses
    
    def extract_text(
        self, 
        pdf_path: Path, 
        pages: Optional[list[int]] = None,
        normalize: bool = True
    ) -> ExtractionResult:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            pages: Specific pages to extract (1-indexed). None = all pages.
            normalize: Whether to normalize the extracted text
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        logger.info(f"Extracting text from: {pdf_path}")
        
        # Try primary backend first
        if self.primary_backend == "pdfplumber" and self.has_pdfplumber:
            result = self._extract_with_pdfplumber(pdf_path, pages)
            if result.text.strip():
                if normalize:
                    result.text = normalize_text(result.text, aggressive=True)
                return result
            logger.debug("pdfplumber returned empty, trying PyMuPDF fallback")
        
        if self.primary_backend == "pymupdf" and self.has_pymupdf:
            result = self._extract_with_pymupdf(pdf_path, pages)
            if result.text.strip():
                if normalize:
                    result.text = normalize_text(result.text, aggressive=True)
                return result
            logger.debug("PyMuPDF returned empty, trying pdfplumber fallback")
        
        # Try fallback
        if self.has_pdfplumber and self.primary_backend != "pdfplumber":
            result = self._extract_with_pdfplumber(pdf_path, pages)
        elif self.has_pymupdf and self.primary_backend != "pymupdf":
            result = self._extract_with_pymupdf(pdf_path, pages)
        else:
            result = ExtractionResult(
                text="",
                method=ExtractionMethod.TEXT_LAYER,
                confidence=0.0,
                warnings=["No text could be extracted from the PDF"]
            )
        
        if normalize and result.text:
            result.text = normalize_text(result.text, aggressive=True)
        
        return result
    
    def _extract_with_pdfplumber(
        self, 
        pdf_path: Path, 
        pages: Optional[list[int]] = None
    ) -> ExtractionResult:
        """Extract text using pdfplumber."""
        import pdfplumber
        
        texts = []
        warnings = []
        total_pages = 0
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                # Determine which pages to process
                if pages:
                    page_indices = [p - 1 for p in pages if 0 < p <= total_pages]
                else:
                    page_indices = range(total_pages)
                
                for idx in page_indices:
                    page = pdf.pages[idx]
                    try:
                        # Extract with layout preservation settings
                        text = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=False,  # Don't try to preserve exact layout
                            x_density=7.25,
                            y_density=13
                        )
                        
                        if text:
                            texts.append(text)
                        else:
                            warnings.append(f"Page {idx + 1} returned no text")
                            
                    except Exception as e:
                        warnings.append(f"Page {idx + 1} extraction failed: {str(e)}")
                        logger.warning(f"Failed to extract page {idx + 1}: {e}")
                        
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ExtractionResult(
                text="",
                method=ExtractionMethod.TEXT_LAYER,
                confidence=0.0,
                warnings=[f"Extraction failed: {str(e)}"]
            )
        
        combined_text = "\n\n".join(texts)
        confidence = calculate_text_confidence(combined_text, ExtractionMethod.TEXT_LAYER)
        
        return ExtractionResult(
            text=combined_text,
            method=ExtractionMethod.TEXT_LAYER,
            confidence=confidence,
            metadata={
                'total_pages': total_pages,
                'pages_extracted': len(texts),
                'backend': 'pdfplumber'
            },
            warnings=warnings
        )
    
    def _extract_with_pymupdf(
        self, 
        pdf_path: Path, 
        pages: Optional[list[int]] = None
    ) -> ExtractionResult:
        """Extract text using PyMuPDF (fitz)."""
        import fitz
        
        texts = []
        warnings = []
        total_pages = 0
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Determine which pages to process
            if pages:
                page_indices = [p - 1 for p in pages if 0 < p <= total_pages]
            else:
                page_indices = range(total_pages)
            
            for idx in page_indices:
                try:
                    page = doc[idx]
                    
                    # Extract text with better handling of blocks
                    text = page.get_text("text")
                    
                    if text and text.strip():
                        texts.append(text)
                    else:
                        warnings.append(f"Page {idx + 1} returned no text")
                        
                except Exception as e:
                    warnings.append(f"Page {idx + 1} extraction failed: {str(e)}")
                    logger.warning(f"Failed to extract page {idx + 1}: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return ExtractionResult(
                text="",
                method=ExtractionMethod.TEXT_LAYER,
                confidence=0.0,
                warnings=[f"Extraction failed: {str(e)}"]
            )
        
        combined_text = "\n\n".join(texts)
        confidence = calculate_text_confidence(combined_text, ExtractionMethod.TEXT_LAYER)
        
        return ExtractionResult(
            text=combined_text,
            method=ExtractionMethod.TEXT_LAYER,
            confidence=confidence,
            metadata={
                'total_pages': total_pages,
                'pages_extracted': len(texts),
                'backend': 'pymupdf'
            },
            warnings=warnings
        )
    
    def extract_text_by_page(
        self, 
        pdf_path: Path
    ) -> Generator[ExtractionResult, None, None]:
        """
        Generator that yields extracted text page by page.
        
        Useful for:
        - Processing large PDFs without loading all text into memory
        - Applying different strategies to different pages
        - Progress tracking
        
        Yields:
            ExtractionResult for each page
        """
        if self.has_pdfplumber:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text() or ""
                        confidence = calculate_text_confidence(text, ExtractionMethod.TEXT_LAYER)
                        
                        yield ExtractionResult(
                            text=normalize_text(text, aggressive=True),
                            method=ExtractionMethod.TEXT_LAYER,
                            confidence=confidence,
                            page_number=i,
                            metadata={'backend': 'pdfplumber'}
                        )
                    except Exception as e:
                        logger.warning(f"Page {i} extraction failed: {e}")
                        yield ExtractionResult(
                            text="",
                            method=ExtractionMethod.TEXT_LAYER,
                            confidence=0.0,
                            page_number=i,
                            warnings=[str(e)]
                        )
        
        elif self.has_pymupdf:
            import fitz
            
            doc = fitz.open(pdf_path)
            for i, page in enumerate(doc, start=1):
                try:
                    text = page.get_text() or ""
                    confidence = calculate_text_confidence(text, ExtractionMethod.TEXT_LAYER)
                    
                    yield ExtractionResult(
                        text=normalize_text(text, aggressive=True),
                        method=ExtractionMethod.TEXT_LAYER,
                        confidence=confidence,
                        page_number=i,
                        metadata={'backend': 'pymupdf'}
                    )
                except Exception as e:
                    logger.warning(f"Page {i} extraction failed: {e}")
                    yield ExtractionResult(
                        text="",
                        method=ExtractionMethod.TEXT_LAYER,
                        confidence=0.0,
                        page_number=i,
                        warnings=[str(e)]
                    )
            
            doc.close()


def extract_text_simple(pdf_path: Path) -> str:
    """
    Simple convenience function for basic text extraction.
    
    Use this for quick one-off extractions where you don't need
    confidence scores or metadata.
    """
    extractor = PDFTextExtractor()
    result = extractor.extract_text(pdf_path)
    return result.text

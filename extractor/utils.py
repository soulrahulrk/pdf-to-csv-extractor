"""
Utility functions shared across the PDF extraction pipeline.

This module provides common functionality needed by multiple extraction strategies:
- Text preprocessing and normalization
- Confidence scoring
- Page analysis helpers
- Common data structures

Why this exists:
Real-world PDFs have inconsistent encoding, mixed character sets, and formatting
artifacts. These utilities handle the messy reality of PDF text extraction.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pathlib import Path

from loguru import logger


class ExtractionMethod(Enum):
    """
    Tracks which method was used to extract content.
    This is crucial for confidence scoring - OCR text is inherently less reliable.
    """
    TEXT_LAYER = "text_layer"       # Direct text extraction (most reliable)
    OCR = "ocr"                      # Optical character recognition (less reliable)
    TABLE = "table"                  # Table extraction library
    HYBRID = "hybrid"                # Combined methods
    UNKNOWN = "unknown"


@dataclass
class ExtractionResult:
    """
    Standardized container for extraction results.
    
    Every extraction method returns this structure so downstream processing
    doesn't need to know which method was used.
    """
    text: str                                           # The extracted text content
    method: ExtractionMethod                            # How it was extracted
    confidence: float = 1.0                             # 0.0 to 1.0 confidence score
    page_number: Optional[int] = None                   # Source page (1-indexed)
    bounding_box: Optional[tuple] = None                # (x0, y0, x1, y1) if available
    metadata: dict = field(default_factory=dict)        # Additional extraction info
    warnings: list = field(default_factory=list)        # Non-fatal issues encountered
    
    def __post_init__(self):
        """Ensure confidence is within valid range."""
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class PageAnalysis:
    """
    Analysis results for a single PDF page.
    Used to decide which extraction strategy to apply.
    """
    page_number: int
    has_text_layer: bool                    # Does the page have extractable text?
    text_char_count: int                    # How many characters were extracted?
    has_images: bool                        # Are there embedded images?
    image_coverage: float                   # What % of page is covered by images?
    likely_scanned: bool                    # Is this probably a scanned page?
    has_tables: bool                        # Were table structures detected?
    orientation: str = "portrait"           # portrait or landscape
    
    def needs_ocr(self, text_threshold: int = 50) -> bool:
        """
        Determine if this page needs OCR processing.
        
        OCR is expensive and error-prone, so we only use it when necessary:
        - Page has images but very little text
        - Page appears to be a scan
        
        Args:
            text_threshold: Minimum character count to consider page "has text"
        """
        if self.text_char_count >= text_threshold:
            return False
        return self.likely_scanned or (self.has_images and self.image_coverage > 0.5)


def normalize_text(text: str, aggressive: bool = False) -> str:
    """
    Normalize extracted text for consistent processing.
    
    Why this matters:
    - PDFs use various Unicode representations (é vs e + combining accent)
    - Ligatures (ﬁ, ﬂ) break regex matching
    - Multiple space types (non-breaking, em-space) cause matching issues
    - Smart quotes vs straight quotes
    
    Args:
        text: Raw extracted text
        aggressive: If True, also collapse whitespace and strip control chars
        
    Returns:
        Normalized text suitable for pattern matching
    """
    if not text:
        return ""
    
    # Step 1: Unicode normalization (NFC form - composed characters)
    # This converts "e + combining acute" to "é" 
    text = unicodedata.normalize('NFC', text)
    
    # Step 2: Replace common ligatures with their component characters
    # Many PDFs use ligatures that break word matching
    ligature_map = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        'ﬃ': 'ffi',
        'ﬄ': 'ffl',
        'ﬅ': 'st',
        'ﬆ': 'st',
        'Ꜳ': 'AA',
        'ꜳ': 'aa',
        'Æ': 'AE',
        'æ': 'ae',
        'Œ': 'OE',
        'œ': 'oe',
    }
    for ligature, replacement in ligature_map.items():
        text = text.replace(ligature, replacement)
    
    # Step 3: Normalize quotes and dashes
    # Smart quotes and em-dashes from Word/InDesign cause regex failures
    quote_map = {
        '"': '"', '"': '"',  # Smart double quotes
        ''': "'", ''': "'",  # Smart single quotes
        '–': '-', '—': '-',  # En-dash and em-dash
        '…': '...',          # Ellipsis
        '\u00a0': ' ',       # Non-breaking space
        '\u2003': ' ',       # Em space
        '\u2002': ' ',       # En space
        '\u2009': ' ',       # Thin space
    }
    for char, replacement in quote_map.items():
        text = text.replace(char, replacement)
    
    if aggressive:
        # Step 4: Remove control characters (except newline/tab)
        text = ''.join(char for char in text if char == '\n' or char == '\t' or not unicodedata.category(char).startswith('C'))
        
        # Step 5: Collapse multiple whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 6: Strip leading/trailing whitespace from each line
        text = '\n'.join(line.strip() for line in text.split('\n'))
    
    return text


def remove_noise(text: str, noise_patterns: list[str]) -> str:
    """
    Remove common noise patterns from extracted text.
    
    PDFs often contain repeated headers, footers, watermarks, and page numbers
    that pollute the extracted content. This function removes them.
    
    Args:
        text: Extracted text
        noise_patterns: List of regex patterns to remove
        
    Returns:
        Cleaned text with noise removed
    """
    if not text:
        return ""
    
    cleaned = text
    for pattern in noise_patterns:
        try:
            # Use IGNORECASE and MULTILINE for broader matching
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            logger.warning(f"Invalid noise pattern '{pattern}': {e}")
    
    # Clean up any resulting multiple blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()


def calculate_text_confidence(text: str, method: ExtractionMethod) -> float:
    """
    Calculate a confidence score for extracted text.
    
    This is a heuristic that considers:
    - Extraction method (OCR is less reliable)
    - Text quality indicators (garbage characters, proper word structure)
    - Character distribution
    
    Args:
        text: Extracted text to evaluate
        method: How the text was extracted
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    # Base confidence by method
    base_confidence = {
        ExtractionMethod.TEXT_LAYER: 0.95,
        ExtractionMethod.TABLE: 0.90,
        ExtractionMethod.OCR: 0.70,
        ExtractionMethod.HYBRID: 0.80,
        ExtractionMethod.UNKNOWN: 0.50,
    }.get(method, 0.50)
    
    # Analyze text quality
    penalties = 0.0
    
    # Penalty for high ratio of non-printable/garbage characters
    printable_ratio = sum(1 for c in text if c.isprintable() or c in '\n\t') / len(text)
    if printable_ratio < 0.9:
        penalties += (0.9 - printable_ratio) * 0.5
    
    # Penalty for lack of proper words (sequences of letters)
    words = re.findall(r'[a-zA-Z]{2,}', text)
    if len(text) > 50 and len(words) < 3:
        penalties += 0.2
    
    # Penalty for too many special characters (suggests OCR errors)
    special_ratio = sum(1 for c in text if c in '@#$%^&*{}[]|\\<>') / max(len(text), 1)
    if special_ratio > 0.1:
        penalties += special_ratio * 0.3
    
    # Penalty for extremely long "words" (suggests missing spaces, common in OCR)
    long_words = re.findall(r'[a-zA-Z]{20,}', text)
    if long_words:
        penalties += 0.1 * min(len(long_words), 3)
    
    final_confidence = max(0.0, base_confidence - penalties)
    return round(final_confidence, 3)


def detect_page_orientation(width: float, height: float) -> str:
    """Simple orientation detection based on dimensions."""
    return "landscape" if width > height else "portrait"


def is_likely_scanned(page_analysis: dict) -> bool:
    """
    Heuristic to detect if a page is likely a scanned document.
    
    Indicators of a scan:
    - Large image covering most of the page
    - Very little or no extractable text
    - Single full-page image
    
    Args:
        page_analysis: Dict containing image_count, image_area_ratio, text_char_count
    """
    image_count = page_analysis.get('image_count', 0)
    image_area_ratio = page_analysis.get('image_area_ratio', 0)
    text_chars = page_analysis.get('text_char_count', 0)
    
    # Classic scan: one big image, almost no text
    if image_count == 1 and image_area_ratio > 0.8 and text_chars < 100:
        return True
    
    # Multiple images covering most of page with little text
    if image_area_ratio > 0.9 and text_chars < 50:
        return True
    
    return False


def safe_filename(name: str) -> str:
    """
    Convert a string to a safe filename.
    Removes/replaces characters that are problematic on various filesystems.
    """
    # Remove or replace problematic characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe.strip('._')
    
    # Limit length
    if len(safe) > 200:
        safe = safe[:200]
    
    return safe or "unnamed"


def merge_extraction_results(results: list[ExtractionResult]) -> ExtractionResult:
    """
    Merge multiple extraction results into one.
    
    Used when combining results from different extraction methods
    or from multiple pages.
    
    The confidence of the merged result is the weighted average
    based on text length.
    """
    if not results:
        return ExtractionResult(text="", method=ExtractionMethod.UNKNOWN, confidence=0.0)
    
    if len(results) == 1:
        return results[0]
    
    # Combine text
    combined_text = "\n\n".join(r.text for r in results if r.text)
    
    # Weighted average confidence
    total_weight = sum(len(r.text) for r in results if r.text)
    if total_weight > 0:
        weighted_confidence = sum(r.confidence * len(r.text) for r in results if r.text) / total_weight
    else:
        weighted_confidence = sum(r.confidence for r in results) / len(results)
    
    # Collect all warnings
    all_warnings = []
    for r in results:
        all_warnings.extend(r.warnings)
    
    # Determine method
    methods = set(r.method for r in results)
    if len(methods) == 1:
        final_method = methods.pop()
    else:
        final_method = ExtractionMethod.HYBRID
    
    return ExtractionResult(
        text=combined_text,
        method=final_method,
        confidence=round(weighted_confidence, 3),
        metadata={'source_count': len(results)},
        warnings=all_warnings
    )


def chunk_text(text: str, max_chunk_size: int = 10000, overlap: int = 500) -> list[str]:
    """
    Split long text into overlapping chunks for processing.
    
    Useful when text is too long for certain operations.
    Overlap ensures we don't miss content that spans chunk boundaries.
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start + max_chunk_size // 2, end)
            if para_break > start:
                end = para_break
            else:
                # Look for sentence end
                sentence_break = text.rfind('. ', start + max_chunk_size // 2, end)
                if sentence_break > start:
                    end = sentence_break + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    
    return chunks


def get_pdf_info(pdf_path: Path) -> dict[str, Any]:
    """
    Extract basic metadata from a PDF file.
    
    This is a lightweight check that doesn't fully parse the PDF.
    Used for initial validation and logging.
    """
    info = {
        'path': str(pdf_path),
        'filename': pdf_path.name,
        'size_bytes': 0,
        'exists': False,
        'readable': False,
    }
    
    try:
        if pdf_path.exists():
            info['exists'] = True
            info['size_bytes'] = pdf_path.stat().st_size
            info['size_mb'] = round(info['size_bytes'] / (1024 * 1024), 2)
            
            # Quick read test
            with open(pdf_path, 'rb') as f:
                header = f.read(8)
                info['readable'] = header.startswith(b'%PDF')
                if not info['readable']:
                    info['error'] = 'File does not appear to be a valid PDF'
    except Exception as e:
        info['error'] = str(e)
    
    return info

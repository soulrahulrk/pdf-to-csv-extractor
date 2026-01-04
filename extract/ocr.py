"""
OCR extraction for scanned PDF pages.
Only triggered when page has insufficient text content.
"""

from dataclasses import dataclass
from typing import List, Optional
import io

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


@dataclass
class OCRBlock:
    """OCR-extracted text block."""
    content: str
    block_type: str = 'ocr_text'
    top: float = 0
    bottom: float = 0
    page_num: int = 0
    confidence: float = 0.0


# Threshold: if page has fewer than this many characters, trigger OCR
DEFAULT_OCR_THRESHOLD = 50


def should_ocr(char_count: int, threshold: int = DEFAULT_OCR_THRESHOLD) -> bool:
    """
    Determine if OCR should be triggered for a page.
    
    Args:
        char_count: Number of characters extracted from text layer
        threshold: Minimum characters before OCR is skipped
        
    Returns:
        True if OCR should be performed
    """
    return char_count < threshold


def extract_ocr_text(page, page_num: int, language: str = 'eng') -> List[OCRBlock]:
    """
    Perform OCR on a PDF page.
    
    Args:
        page: pdfplumber page object
        page_num: 1-based page number
        language: Tesseract language code
        
    Returns:
        List of OCRBlock objects
    """
    if not OCR_AVAILABLE:
        return [OCRBlock(
            content='[OCR unavailable - pytesseract not installed]',
            block_type='ocr_text',
            page_num=page_num
        )]
    
    try:
        # Convert page to image
        # pdfplumber can render pages to images
        img = page.to_image(resolution=300)
        
        # Convert to PIL Image
        pil_image = img.original
        
        # Perform OCR
        ocr_text = pytesseract.image_to_string(
            pil_image,
            lang=language,
            config='--psm 1'  # Automatic page segmentation with OSD
        )
        
        # Clean and return
        ocr_text = ocr_text.strip()
        
        if not ocr_text:
            return [OCRBlock(
                content='',
                block_type='ocr_text',
                page_num=page_num,
                confidence=0.0
            )]
        
        # Get confidence data
        try:
            data = pytesseract.image_to_data(
                pil_image,
                lang=language,
                output_type=pytesseract.Output.DICT
            )
            confidences = [c for c in data['conf'] if c > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except Exception:
            avg_confidence = 0.0
        
        return [OCRBlock(
            content=ocr_text,
            block_type='ocr_text',
            page_num=page_num,
            confidence=avg_confidence / 100.0  # Normalize to 0-1
        )]
        
    except Exception as e:
        return [OCRBlock(
            content=f'[OCR error: {str(e)}]',
            block_type='ocr_text',
            page_num=page_num,
            confidence=0.0
        )]


def is_ocr_available() -> bool:
    """Check if OCR is available."""
    return OCR_AVAILABLE

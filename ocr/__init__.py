"""
Region-Based OCR Package

This package provides intelligent, region-based OCR that:
- Detects text-poor regions that need OCR
- Ignores logos, signatures, and noise
- Applies DPI-aware preprocessing
- Supports auto language detection
- OCRs only candidate regions (not full page)

This approach is more accurate and efficient than naive full-page OCR.

Components:
- RegionDetector: Identifies regions needing OCR
- OCRPreprocessor: Image preprocessing for better accuracy
- SmartOCREngine: Coordinates the OCR pipeline

Usage:
    from ocr import SmartOCREngine, RegionDetector
    
    engine = SmartOCREngine()
    result = engine.ocr_document(pdf_path)
    
    # Or detect regions first for analysis
    detector = RegionDetector()
    regions = detector.find_ocr_candidates(page_image)
"""

from .region_detector import (
    RegionDetector,
    OCRCandidate,
    RegionAnalysis,
    ImageQualityMetrics,
)
from .preprocessor import (
    OCRPreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    enhance_for_ocr,
)
from .ocr_engine import (
    SmartOCREngine,
    OCRConfig,
    OCRResult,
    PageOCRResult,
    DocumentOCRResult,
)

__all__ = [
    # Region detection
    'RegionDetector',
    'OCRCandidate',
    'RegionAnalysis',
    'ImageQualityMetrics',
    
    # Preprocessing
    'OCRPreprocessor',
    'PreprocessingConfig',
    'PreprocessingResult',
    'enhance_for_ocr',
    
    # OCR engine
    'SmartOCREngine',
    'OCRConfig',
    'OCRResult',
    'PageOCRResult',
    'DocumentOCRResult',
]

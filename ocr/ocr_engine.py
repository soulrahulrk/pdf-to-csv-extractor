"""
Smart OCR Engine

This module provides intelligent OCR that:
- Only OCRs regions that need it (not full pages)
- Applies appropriate preprocessing per region
- Supports multiple languages with auto-detection
- Returns results with bounding boxes
- Provides confidence scores

Why Smart OCR:
- Full-page OCR is slow and expensive
- Preprocessing improves accuracy significantly
- Region-based OCR allows mixing with text layer
- Confidence scores enable quality control
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

from .region_detector import (
    RegionDetector,
    OCRCandidate,
    RegionAnalysis,
    RegionCategory,
)
from .preprocessor import (
    OCRPreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
)

# Import layout types for result format
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from layout.box import BoundingBox, TextBlock, BlockType

logger = logging.getLogger(__name__)

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available - OCR disabled")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class OCRConfig:
    """
    Configuration for OCR processing.
    """
    # Language settings
    language: str = 'eng'
    auto_detect_language: bool = False
    fallback_languages: List[str] = field(default_factory=lambda: ['eng'])
    
    # Processing settings
    enable_preprocessing: bool = True
    target_dpi: int = 300
    
    # Tesseract settings
    tesseract_config: str = '--oem 3 --psm 6'
    
    # Region settings
    use_region_detection: bool = True
    min_confidence: float = 0.3
    
    # Performance settings
    max_workers: int = 4
    timeout_per_page: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'language': self.language,
            'auto_detect_language': self.auto_detect_language,
            'enable_preprocessing': self.enable_preprocessing,
            'target_dpi': self.target_dpi,
            'use_region_detection': self.use_region_detection,
            'min_confidence': self.min_confidence,
        }


@dataclass
class OCRResult:
    """
    Result of OCR on a single region or text block.
    """
    text: str
    bbox: BoundingBox
    confidence: float
    language: str
    source: str = 'ocr'
    word_confidences: List[Tuple[str, float]] = field(default_factory=list)
    preprocessing_applied: List[str] = field(default_factory=list)
    
    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()
    
    @property
    def word_count(self) -> int:
        return len(self.text.split()) if self.text else 0
    
    @property
    def avg_word_confidence(self) -> float:
        if not self.word_confidences:
            return self.confidence
        return sum(c for _, c in self.word_confidences) / len(self.word_confidences)
    
    def to_text_block(self, page_number: int = 0) -> TextBlock:
        """Convert to TextBlock for layout integration."""
        return TextBlock(
            text=self.text,
            bbox=self.bbox,
            block_type=BlockType.TEXT,
            confidence=self.confidence,
            page_number=page_number,
            source='ocr',
            metadata={
                'language': self.language,
                'preprocessing': self.preprocessing_applied,
            }
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'bbox': self.bbox.to_dict(),
            'confidence': round(self.confidence, 3),
            'language': self.language,
            'word_count': self.word_count,
            'avg_word_confidence': round(self.avg_word_confidence, 3),
        }


@dataclass
class PageOCRResult:
    """
    OCR results for a single page.
    """
    page_number: int
    results: List[OCRResult]
    region_analysis: Optional[RegionAnalysis] = None
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def text(self) -> str:
        """Get all text from page."""
        return '\n'.join(r.text for r in self.results if r.text.strip())
    
    @property
    def block_count(self) -> int:
        return len(self.results)
    
    @property
    def avg_confidence(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.confidence for r in self.results) / len(self.results)
    
    @property
    def has_low_confidence(self) -> bool:
        return self.avg_confidence < 0.5
    
    def get_text_blocks(self) -> List[TextBlock]:
        """Convert to TextBlock list."""
        return [r.to_text_block(self.page_number) for r in self.results]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'page_number': self.page_number,
            'block_count': self.block_count,
            'avg_confidence': round(self.avg_confidence, 3),
            'processing_time': round(self.processing_time, 2),
            'warnings': self.warnings,
            'results': [r.to_dict() for r in self.results],
        }


@dataclass
class DocumentOCRResult:
    """
    OCR results for entire document.
    """
    pages: List[PageOCRResult]
    source_path: Optional[Path] = None
    config: Optional[OCRConfig] = None
    total_processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    @property
    def page_count(self) -> int:
        return len(self.pages)
    
    @property
    def total_blocks(self) -> int:
        return sum(p.block_count for p in self.pages)
    
    @property
    def text(self) -> str:
        """Get all text from document."""
        return '\n\n'.join(p.text for p in self.pages)
    
    @property
    def avg_confidence(self) -> float:
        if not self.pages:
            return 0.0
        total_results = sum(len(p.results) for p in self.pages)
        if total_results == 0:
            return 0.0
        total_confidence = sum(
            r.confidence for p in self.pages for r in p.results
        )
        return total_confidence / total_results
    
    def get_page(self, page_number: int) -> Optional[PageOCRResult]:
        """Get results for specific page."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def get_all_text_blocks(self) -> List[TextBlock]:
        """Get all text blocks from all pages."""
        blocks = []
        for page in self.pages:
            blocks.extend(page.get_text_blocks())
        return blocks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_path': str(self.source_path) if self.source_path else None,
            'page_count': self.page_count,
            'total_blocks': self.total_blocks,
            'avg_confidence': round(self.avg_confidence, 3),
            'total_processing_time': round(self.total_processing_time, 2),
            'warnings': self.warnings,
            'pages': [p.to_dict() for p in self.pages],
        }


class SmartOCREngine:
    """
    Intelligent OCR engine with region detection and preprocessing.
    
    Features:
    - Region-based OCR (not full page)
    - Automatic preprocessing
    - Multi-language support
    - Confidence scoring
    - Parallel processing
    
    Usage:
        engine = SmartOCREngine()
        
        # OCR a document
        result = engine.ocr_document(pdf_path)
        
        # OCR specific pages
        result = engine.ocr_document(pdf_path, pages=[0, 2, 3])
        
        # OCR a single image
        page_result = engine.ocr_image(image, page_number=0)
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize OCR engine.
        
        Args:
            config: OCR configuration
        """
        self.config = config or OCRConfig()
        
        # Initialize components
        self.region_detector = RegionDetector()
        self.preprocessor = OCRPreprocessor()
        
        # Check Tesseract
        self._tesseract_checked = False
        self._tesseract_available = False
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available."""
        if self._tesseract_checked:
            return self._tesseract_available
        
        self._tesseract_checked = True
        
        if not TESSERACT_AVAILABLE:
            logger.warning("pytesseract not installed")
            self._tesseract_available = False
            return False
        
        try:
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self._tesseract_available = False
            return False
    
    def ocr_document(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        config: Optional[OCRConfig] = None,
    ) -> DocumentOCRResult:
        """
        OCR a PDF document.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to OCR (0-indexed), or None for all
            config: Override configuration
            
        Returns:
            DocumentOCRResult with OCR results
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        config = config or self.config
        warnings = []
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not self._check_tesseract():
            warnings.append("Tesseract not available - OCR disabled")
            return DocumentOCRResult(
                pages=[],
                source_path=pdf_path,
                config=config,
                warnings=warnings,
            )
        
        if not PYMUPDF_AVAILABLE:
            warnings.append("PyMuPDF not available - cannot render pages")
            return DocumentOCRResult(
                pages=[],
                source_path=pdf_path,
                config=config,
                warnings=warnings,
            )
        
        # Open PDF and process pages
        page_results = []
        
        doc = fitz.open(pdf_path)
        try:
            page_nums = pages if pages else range(len(doc))
            
            for page_num in page_nums:
                if page_num >= len(doc):
                    warnings.append(f"Page {page_num} not found")
                    continue
                
                page = doc[page_num]
                
                # Render page to image
                mat = fitz.Matrix(config.target_dpi / 72, config.target_dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # OCR the page
                page_result = self.ocr_image(
                    img,
                    page_number=page_num,
                    config=config,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                )
                
                page_results.append(page_result)
        finally:
            doc.close()
        
        total_time = time.time() - start_time
        
        return DocumentOCRResult(
            pages=page_results,
            source_path=pdf_path,
            config=config,
            total_processing_time=total_time,
            warnings=warnings,
        )
    
    def ocr_image(
        self,
        image: Any,
        page_number: int = 0,
        config: Optional[OCRConfig] = None,
        page_width: float = 0,
        page_height: float = 0,
    ) -> PageOCRResult:
        """
        OCR a single page image.
        
        Args:
            image: Page image (PIL.Image or numpy array)
            page_number: Page number for reference
            config: Override configuration
            page_width: Original PDF page width (for coordinate mapping)
            page_height: Original PDF page height
            
        Returns:
            PageOCRResult with OCR results
        """
        import time
        start_time = time.time()
        
        config = config or self.config
        warnings = []
        results = []
        
        if not self._check_tesseract():
            return PageOCRResult(
                page_number=page_number,
                results=[],
                warnings=["Tesseract not available"],
            )
        
        # Convert to PIL if needed
        if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_width, img_height = image.size
        
        # Calculate scale factor for coordinate mapping
        scale_x = page_width / img_width if page_width > 0 else 1.0
        scale_y = page_height / img_height if page_height > 0 else 1.0
        
        # Detect regions if enabled
        region_analysis = None
        if config.use_region_detection:
            region_analysis = self.region_detector.analyze_page(
                image, page_number
            )
        
        # Decide what to OCR
        if region_analysis and region_analysis.ocr_candidates:
            # OCR specific regions
            for candidate in region_analysis.ocr_candidates:
                region_result = self._ocr_region(
                    image,
                    candidate,
                    config,
                    scale_x,
                    scale_y,
                    page_height,
                )
                if region_result and not region_result.is_empty:
                    results.append(region_result)
        else:
            # OCR full page
            full_result = self._ocr_full_image(
                image,
                config,
                scale_x,
                scale_y,
                page_height,
            )
            results.extend(full_result)
        
        processing_time = time.time() - start_time
        
        return PageOCRResult(
            page_number=page_number,
            results=results,
            region_analysis=region_analysis,
            processing_time=processing_time,
            warnings=warnings,
        )
    
    def _ocr_region(
        self,
        image: 'Image.Image',
        candidate: OCRCandidate,
        config: OCRConfig,
        scale_x: float,
        scale_y: float,
        page_height: float,
    ) -> Optional[OCRResult]:
        """OCR a specific region of the image."""
        # Crop region
        region_img = image.crop((
            candidate.x0,
            candidate.y0,
            candidate.x1,
            candidate.y1,
        ))
        
        # Preprocess if enabled
        preprocessing_applied = []
        if config.enable_preprocessing:
            preprocess_result = self.preprocessor.process(region_img)
            region_img = preprocess_result.image
            preprocessing_applied = preprocess_result.operations_applied
            
            # Convert back to PIL if needed
            if NUMPY_AVAILABLE and isinstance(region_img, np.ndarray):
                region_img = Image.fromarray(region_img)
        
        # Run OCR
        try:
            ocr_data = pytesseract.image_to_data(
                region_img,
                lang=config.language,
                config=config.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
            
            # Parse results
            text_parts = []
            word_confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if not word.strip():
                    continue
                
                conf = ocr_data['conf'][i]
                if conf > config.min_confidence * 100:
                    text_parts.append(word)
                    word_confidences.append((word, conf / 100.0))
            
            if not text_parts:
                return None
            
            text = ' '.join(text_parts)
            avg_conf = sum(c for _, c in word_confidences) / len(word_confidences)
            
            # Map coordinates back to PDF space
            bbox = BoundingBox(
                x0=candidate.x0 * scale_x,
                y0=page_height - (candidate.y1 * scale_y),  # Convert to PDF coords
                x1=candidate.x1 * scale_x,
                y1=page_height - (candidate.y0 * scale_y),
            )
            
            return OCRResult(
                text=text,
                bbox=bbox,
                confidence=avg_conf,
                language=config.language,
                word_confidences=word_confidences,
                preprocessing_applied=preprocessing_applied,
            )
            
        except Exception as e:
            logger.warning(f"OCR failed for region: {e}")
            return None
    
    def _ocr_full_image(
        self,
        image: 'Image.Image',
        config: OCRConfig,
        scale_x: float,
        scale_y: float,
        page_height: float,
    ) -> List[OCRResult]:
        """OCR the full image and return block-level results."""
        results = []
        
        # Preprocess if enabled
        preprocessing_applied = []
        if config.enable_preprocessing:
            preprocess_result = self.preprocessor.process(image)
            image = preprocess_result.image
            preprocessing_applied = preprocess_result.operations_applied
            
            if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
                image = Image.fromarray(image)
        
        try:
            # Get detailed OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(
                image,
                lang=config.language,
                config=config.tesseract_config,
                output_type=pytesseract.Output.DICT,
            )
            
            # Group by block
            blocks: Dict[int, Dict] = {}
            
            for i in range(len(ocr_data['text'])):
                block_num = ocr_data['block_num'][i]
                word = ocr_data['text'][i]
                conf = ocr_data['conf'][i]
                
                if not word.strip():
                    continue
                
                if conf < config.min_confidence * 100:
                    continue
                
                if block_num not in blocks:
                    blocks[block_num] = {
                        'words': [],
                        'confidences': [],
                        'x0': float('inf'),
                        'y0': float('inf'),
                        'x1': 0,
                        'y1': 0,
                    }
                
                blocks[block_num]['words'].append(word)
                blocks[block_num]['confidences'].append(conf / 100.0)
                
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                blocks[block_num]['x0'] = min(blocks[block_num]['x0'], x)
                blocks[block_num]['y0'] = min(blocks[block_num]['y0'], y)
                blocks[block_num]['x1'] = max(blocks[block_num]['x1'], x + w)
                blocks[block_num]['y1'] = max(blocks[block_num]['y1'], y + h)
            
            # Create OCRResult for each block
            for block_data in blocks.values():
                if not block_data['words']:
                    continue
                
                text = ' '.join(block_data['words'])
                avg_conf = sum(block_data['confidences']) / len(block_data['confidences'])
                
                # Map coordinates
                x0 = block_data['x0'] * scale_x
                y0 = page_height - (block_data['y1'] * scale_y)
                x1 = block_data['x1'] * scale_x
                y1 = page_height - (block_data['y0'] * scale_y)
                
                bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)
                
                word_confs = list(zip(block_data['words'], block_data['confidences']))
                
                results.append(OCRResult(
                    text=text,
                    bbox=bbox,
                    confidence=avg_conf,
                    language=config.language,
                    word_confidences=word_confs,
                    preprocessing_applied=preprocessing_applied,
                ))
            
        except Exception as e:
            logger.error(f"Full image OCR failed: {e}")
        
        return results
    
    def detect_language(self, image: Any) -> str:
        """
        Detect the language of text in an image.
        
        Args:
            image: Image to analyze
            
        Returns:
            ISO 639-3 language code
        """
        if not self._check_tesseract():
            return 'eng'
        
        try:
            # Use Tesseract's OSD (Orientation and Script Detection)
            osd = pytesseract.image_to_osd(image)
            
            # Parse script from OSD output
            for line in osd.split('\n'):
                if 'Script:' in line:
                    script = line.split(':')[1].strip()
                    # Map script to language (simplified)
                    script_to_lang = {
                        'Latin': 'eng',
                        'Cyrillic': 'rus',
                        'Arabic': 'ara',
                        'Han': 'chi_sim',
                        'Japanese': 'jpn',
                        'Korean': 'kor',
                        'Devanagari': 'hin',
                    }
                    return script_to_lang.get(script, 'eng')
            
            return 'eng'
            
        except Exception:
            return 'eng'


def is_tesseract_available() -> bool:
    """Check if Tesseract OCR is available."""
    engine = SmartOCREngine()
    return engine._check_tesseract()

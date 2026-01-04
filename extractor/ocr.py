"""
OCR (Optical Character Recognition) Module

This module handles text extraction from scanned PDFs and images using Tesseract OCR.
It includes preprocessing steps to improve OCR accuracy on real-world documents.

Why OCR is tricky:
- Scanned documents have varying quality (resolution, lighting, skew)
- Tesseract works best on clean, high-contrast, properly oriented images
- Without preprocessing, OCR accuracy can be <50% on poor quality scans
- Preprocessing can improve accuracy to 85-95% in many cases

Dependencies:
- pytesseract: Python wrapper for Tesseract OCR engine
- Pillow: Image loading and basic manipulation
- OpenCV (optional but recommended): Advanced preprocessing

IMPORTANT: Tesseract must be installed separately:
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: apt-get install tesseract-ocr
- macOS: brew install tesseract
"""

from pathlib import Path
from typing import Optional, Union
import io
import tempfile
import os

from loguru import logger

from .utils import (
    ExtractionResult,
    ExtractionMethod,
    normalize_text,
    calculate_text_confidence,
)


class OCRExtractor:
    """
    Extracts text from images and scanned PDF pages using Tesseract OCR.
    
    Features:
    - Automatic image preprocessing for better accuracy
    - Support for multiple languages
    - Configurable Tesseract parameters
    - Graceful degradation when OpenCV isn't available
    
    Usage:
        ocr = OCRExtractor(language='eng')
        result = ocr.extract_from_pdf(Path("scanned.pdf"))
        print(result.text)
    """
    
    def __init__(
        self,
        language: str = 'eng',
        tesseract_cmd: Optional[str] = None,
        enable_preprocessing: bool = True,
        dpi: int = 300
    ):
        """
        Initialize OCR extractor.
        
        Args:
            language: Tesseract language code (e.g., 'eng', 'fra', 'deu', 'eng+fra')
            tesseract_cmd: Path to tesseract executable (auto-detected if None)
            enable_preprocessing: Whether to preprocess images for better accuracy
            dpi: DPI for PDF to image conversion (higher = better quality but slower)
        """
        self.language = language
        self.enable_preprocessing = enable_preprocessing
        self.dpi = dpi
        
        self._check_dependencies(tesseract_cmd)
    
    def _check_dependencies(self, tesseract_cmd: Optional[str]):
        """Verify OCR dependencies are available."""
        # Check pytesseract
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
            # Set custom tesseract path if provided
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            # Verify tesseract is actually installed and accessible
            try:
                version = pytesseract.get_tesseract_version()
                logger.debug(f"Tesseract version: {version}")
            except Exception as e:
                raise RuntimeError(
                    f"Tesseract is not installed or not in PATH: {e}\n"
                    "Install Tesseract:\n"
                    "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "  Linux: apt-get install tesseract-ocr\n"
                    "  macOS: brew install tesseract"
                )
                
        except ImportError:
            raise RuntimeError(
                "pytesseract not installed. Run: pip install pytesseract"
            )
        
        # Check Pillow
        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            raise RuntimeError(
                "Pillow not installed. Run: pip install Pillow"
            )
        
        # Check OpenCV (optional but recommended)
        self.has_opencv = False
        try:
            import cv2
            self.cv2 = cv2
            self.has_opencv = True
            logger.debug("OpenCV available for image preprocessing")
        except ImportError:
            logger.warning(
                "OpenCV not installed. OCR will work but accuracy may be reduced. "
                "Install with: pip install opencv-python"
            )
        
        # Check PyMuPDF for PDF rendering
        self.has_pymupdf = False
        try:
            import fitz
            self.fitz = fitz
            self.has_pymupdf = True
        except ImportError:
            logger.warning("PyMuPDF not available for PDF rendering")
        
        # Check pdf2image as fallback
        self.has_pdf2image = False
        try:
            import pdf2image
            self.pdf2image = pdf2image
            self.has_pdf2image = True
        except ImportError:
            if not self.has_pymupdf:
                logger.warning(
                    "Neither PyMuPDF nor pdf2image available. "
                    "PDF OCR will not work. Install: pip install PyMuPDF"
                )
    
    def preprocess_image(self, image) -> 'Image':
        """
        Preprocess an image to improve OCR accuracy.
        
        Steps applied:
        1. Convert to grayscale (reduces noise)
        2. Resize if too small (OCR works best at ~300 DPI)
        3. Deskew (correct rotation)
        4. Binarization (convert to black and white)
        5. Denoise (remove speckles)
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        if not self.enable_preprocessing:
            return image
        
        import numpy as np
        
        # Convert PIL to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Convert to grayscale if color
        if len(img_array.shape) == 3:
            if self.has_opencv:
                gray = self.cv2.cvtColor(img_array, self.cv2.COLOR_RGB2GRAY)
            else:
                # Fallback: simple grayscale conversion
                gray = np.mean(img_array, axis=2).astype(np.uint8)
        else:
            gray = img_array
        
        if self.has_opencv:
            # Apply OpenCV preprocessing pipeline
            processed = self._opencv_preprocess(gray)
        else:
            # Basic preprocessing without OpenCV
            processed = self._basic_preprocess(gray)
        
        # Convert back to PIL Image
        return self.Image.fromarray(processed)
    
    def _opencv_preprocess(self, gray_image) -> 'np.ndarray':
        """
        Apply OpenCV preprocessing for better OCR results.
        
        This pipeline is tuned for document images (invoices, forms, etc.)
        """
        import numpy as np
        cv2 = self.cv2
        
        # Step 1: Resize if image is too small
        # Tesseract works best with text that's at least 12px tall
        height, width = gray_image.shape
        if height < 1000:
            scale = 1000 / height
            gray_image = cv2.resize(
                gray_image, 
                None, 
                fx=scale, 
                fy=scale, 
                interpolation=cv2.INTER_CUBIC
            )
        
        # Step 2: Noise reduction
        # Bilateral filter preserves edges while smoothing
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # Step 3: Increase contrast using CLAHE
        # (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 4: Binarization using Otsu's method
        # This automatically finds the optimal threshold
        _, binary = cv2.threshold(
            enhanced, 
            0, 
            255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Step 5: Morphological operations to clean up
        # Remove small noise and fill small holes
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Step 6: Deskew if needed
        binary = self._deskew(binary)
        
        return binary
    
    def _deskew(self, image) -> 'np.ndarray':
        """
        Correct image skew (rotation) which hurts OCR accuracy.
        
        Uses the projection profile method to detect skew angle.
        """
        import numpy as np
        cv2 = self.cv2
        
        # Find all non-zero pixels (text pixels)
        coords = np.column_stack(np.where(image < 128))
        
        if len(coords) < 100:
            # Not enough text to determine skew
            return image
        
        # Get minimum area rectangle
        try:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            
            # Normalize angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            
            # Only correct if skew is significant but not too extreme
            if abs(angle) > 0.5 and abs(angle) < 10:
                (h, w) = image.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, 
                    M, 
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
        except Exception as e:
            logger.debug(f"Deskew failed: {e}")
        
        return image
    
    def _basic_preprocess(self, gray_image) -> 'np.ndarray':
        """
        Basic preprocessing when OpenCV is not available.
        Limited but better than nothing.
        """
        import numpy as np
        
        # Simple thresholding
        threshold = np.mean(gray_image)
        binary = np.where(gray_image > threshold, 255, 0).astype(np.uint8)
        
        return binary
    
    def extract_from_image(
        self, 
        image_path: Union[Path, str],
        config: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract text from an image file.
        
        Args:
            image_path: Path to image file (PNG, JPG, TIFF, etc.)
            config: Custom Tesseract config string
            
        Returns:
            ExtractionResult with extracted text
        """
        logger.info(f"OCR extracting from image: {image_path}")
        
        try:
            # Load image
            image = self.Image.open(image_path)
            
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Build config
            if config is None:
                # PSM 3 = Fully automatic page segmentation (default)
                # PSM 6 = Assume a single uniform block of text
                config = '--psm 3 --oem 3'
            
            # Run OCR
            text = self.pytesseract.image_to_string(
                processed,
                lang=self.language,
                config=config
            )
            
            # Get detailed data for confidence calculation
            data = self.pytesseract.image_to_data(
                processed,
                lang=self.language,
                config=config,
                output_type=self.pytesseract.Output.DICT
            )
            
            # Calculate average confidence from word confidences
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.5
            
            # Adjust confidence based on text quality heuristics
            text_confidence = calculate_text_confidence(text, ExtractionMethod.OCR)
            final_confidence = (avg_confidence + text_confidence) / 2
            
            return ExtractionResult(
                text=normalize_text(text, aggressive=True),
                method=ExtractionMethod.OCR,
                confidence=round(final_confidence, 3),
                metadata={
                    'language': self.language,
                    'word_count': len([w for w in data['text'] if w.strip()]),
                    'avg_word_confidence': round(avg_confidence, 3)
                }
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ExtractionResult(
                text="",
                method=ExtractionMethod.OCR,
                confidence=0.0,
                warnings=[f"OCR failed: {str(e)}"]
            )
    
    def extract_from_pdf(
        self,
        pdf_path: Path,
        pages: Optional[list[int]] = None
    ) -> ExtractionResult:
        """
        Extract text from a PDF using OCR.
        
        This converts PDF pages to images, then runs OCR on each image.
        Use this for scanned PDFs or pages with no text layer.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to OCR (1-indexed). None = all pages.
            
        Returns:
            ExtractionResult with combined text from all pages
        """
        logger.info(f"OCR extracting from PDF: {pdf_path}")
        
        if self.has_pymupdf:
            return self._extract_pdf_pymupdf(pdf_path, pages)
        elif self.has_pdf2image:
            return self._extract_pdf_pdf2image(pdf_path, pages)
        else:
            return ExtractionResult(
                text="",
                method=ExtractionMethod.OCR,
                confidence=0.0,
                warnings=["No PDF rendering library available (need PyMuPDF or pdf2image)"]
            )
    
    def _extract_pdf_pymupdf(
        self,
        pdf_path: Path,
        pages: Optional[list[int]] = None
    ) -> ExtractionResult:
        """Extract from PDF using PyMuPDF for rendering."""
        texts = []
        warnings = []
        confidences = []
        
        try:
            doc = self.fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Determine which pages to process
            if pages:
                page_indices = [p - 1 for p in pages if 0 < p <= total_pages]
            else:
                page_indices = range(total_pages)
            
            for idx in page_indices:
                try:
                    page = doc[idx]
                    
                    # Render page to image at specified DPI
                    # zoom = DPI / 72 (default PDF resolution)
                    zoom = self.dpi / 72
                    mat = self.fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = self.Image.open(io.BytesIO(img_data))
                    
                    # Preprocess
                    processed = self.preprocess_image(image)
                    
                    # OCR
                    text = self.pytesseract.image_to_string(
                        processed,
                        lang=self.language,
                        config='--psm 3 --oem 3'
                    )
                    
                    if text.strip():
                        texts.append(text)
                        conf = calculate_text_confidence(text, ExtractionMethod.OCR)
                        confidences.append(conf)
                    else:
                        warnings.append(f"Page {idx + 1} OCR returned no text")
                        
                except Exception as e:
                    warnings.append(f"Page {idx + 1} OCR failed: {str(e)}")
                    logger.warning(f"OCR failed for page {idx + 1}: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PDF OCR extraction failed: {e}")
            return ExtractionResult(
                text="",
                method=ExtractionMethod.OCR,
                confidence=0.0,
                warnings=[f"PDF OCR failed: {str(e)}"]
            )
        
        combined_text = "\n\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ExtractionResult(
            text=normalize_text(combined_text, aggressive=True),
            method=ExtractionMethod.OCR,
            confidence=round(avg_confidence, 3),
            metadata={
                'total_pages': total_pages,
                'pages_processed': len(texts),
                'dpi': self.dpi,
                'language': self.language
            },
            warnings=warnings
        )
    
    def _extract_pdf_pdf2image(
        self,
        pdf_path: Path,
        pages: Optional[list[int]] = None
    ) -> ExtractionResult:
        """Extract from PDF using pdf2image for rendering."""
        texts = []
        warnings = []
        confidences = []
        
        try:
            # Convert PDF to images
            if pages:
                # pdf2image uses 1-indexed pages
                images = self.pdf2image.convert_from_path(
                    pdf_path,
                    dpi=self.dpi,
                    first_page=min(pages),
                    last_page=max(pages)
                )
            else:
                images = self.pdf2image.convert_from_path(pdf_path, dpi=self.dpi)
            
            for i, image in enumerate(images, start=1):
                try:
                    # Preprocess
                    processed = self.preprocess_image(image)
                    
                    # OCR
                    text = self.pytesseract.image_to_string(
                        processed,
                        lang=self.language,
                        config='--psm 3 --oem 3'
                    )
                    
                    if text.strip():
                        texts.append(text)
                        conf = calculate_text_confidence(text, ExtractionMethod.OCR)
                        confidences.append(conf)
                    else:
                        warnings.append(f"Page {i} OCR returned no text")
                        
                except Exception as e:
                    warnings.append(f"Page {i} OCR failed: {str(e)}")
                    logger.warning(f"OCR failed for page {i}: {e}")
                    
        except Exception as e:
            logger.error(f"PDF OCR extraction failed: {e}")
            return ExtractionResult(
                text="",
                method=ExtractionMethod.OCR,
                confidence=0.0,
                warnings=[f"PDF OCR failed: {str(e)}"]
            )
        
        combined_text = "\n\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return ExtractionResult(
            text=normalize_text(combined_text, aggressive=True),
            method=ExtractionMethod.OCR,
            confidence=round(avg_confidence, 3),
            metadata={
                'pages_processed': len(texts),
                'dpi': self.dpi,
                'language': self.language
            },
            warnings=warnings
        )
    
    def extract_from_pdf_page(
        self,
        pdf_path: Path,
        page_number: int
    ) -> ExtractionResult:
        """
        Extract text from a single PDF page using OCR.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page to extract (1-indexed)
            
        Returns:
            ExtractionResult for the single page
        """
        result = self.extract_from_pdf(pdf_path, pages=[page_number])
        result.page_number = page_number
        return result


def check_tesseract_installed() -> bool:
    """Quick check if Tesseract is installed and accessible."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

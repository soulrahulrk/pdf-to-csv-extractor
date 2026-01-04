"""
OCR Image Preprocessor

This module provides image preprocessing to improve OCR accuracy.
Different preprocessing techniques are applied based on image quality.

Preprocessing Steps:
1. Deskewing - Correct rotation
2. Binarization - Convert to binary image
3. Noise removal - Remove speckles
4. Contrast enhancement - Improve text visibility
5. Resolution scaling - Ensure adequate DPI
6. Border removal - Remove black borders

Why Preprocessing Matters:
- OCR engines assume clean, high-contrast images
- Real-world scans are often skewed, noisy, low-contrast
- Proper preprocessing can improve accuracy by 20-50%
- But over-processing can hurt accuracy too
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any
import io

logger = logging.getLogger(__name__)

# Try to import image processing libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class BinarizationMethod(Enum):
    """Methods for converting to binary image."""
    OTSU = auto()               # Otsu's thresholding
    ADAPTIVE_MEAN = auto()      # Adaptive mean thresholding
    ADAPTIVE_GAUSSIAN = auto()  # Adaptive Gaussian thresholding
    SAUVOLA = auto()           # Sauvola local thresholding
    NONE = auto()              # No binarization


class DenoiseMethod(Enum):
    """Methods for noise removal."""
    GAUSSIAN = auto()          # Gaussian blur
    MEDIAN = auto()            # Median filter
    BILATERAL = auto()         # Bilateral filter (edge-preserving)
    MORPHOLOGICAL = auto()     # Morphological operations
    NONE = auto()              # No denoising


@dataclass
class PreprocessingConfig:
    """
    Configuration for image preprocessing.
    
    Adjust these parameters based on document type and quality.
    """
    # Resolution
    target_dpi: int = 300
    scale_if_below_dpi: int = 200
    
    # Deskewing
    deskew: bool = True
    max_skew_angle: float = 15.0
    
    # Binarization
    binarization: BinarizationMethod = BinarizationMethod.ADAPTIVE_GAUSSIAN
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    
    # Denoising
    denoise: DenoiseMethod = DenoiseMethod.MEDIAN
    denoise_strength: int = 3
    
    # Contrast
    enhance_contrast: bool = True
    contrast_factor: float = 1.5
    
    # Border removal
    remove_borders: bool = True
    border_threshold: int = 10
    
    # Inversion handling
    auto_invert: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'target_dpi': self.target_dpi,
            'scale_if_below_dpi': self.scale_if_below_dpi,
            'deskew': self.deskew,
            'max_skew_angle': self.max_skew_angle,
            'binarization': self.binarization.name,
            'denoise': self.denoise.name,
            'enhance_contrast': self.enhance_contrast,
            'remove_borders': self.remove_borders,
        }


@dataclass
class PreprocessingResult:
    """
    Result of image preprocessing.
    
    Contains the processed image and metadata about applied operations.
    """
    image: Any  # numpy array or PIL Image
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    operations_applied: List[str]
    scale_factor: float = 1.0
    skew_corrected: float = 0.0
    was_inverted: bool = False
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without image)."""
        return {
            'original_size': self.original_size,
            'processed_size': self.processed_size,
            'operations_applied': self.operations_applied,
            'scale_factor': round(self.scale_factor, 2),
            'skew_corrected': round(self.skew_corrected, 2),
            'was_inverted': self.was_inverted,
            'warnings': self.warnings,
        }


class OCRPreprocessor:
    """
    Image preprocessor for OCR.
    
    Applies a pipeline of preprocessing operations to improve
    OCR accuracy on various document types.
    
    Usage:
        preprocessor = OCRPreprocessor()
        
        # Process with default settings
        result = preprocessor.process(image)
        
        # Process with custom config
        config = PreprocessingConfig(
            target_dpi=400,
            binarization=BinarizationMethod.OTSU,
        )
        result = preprocessor.process(image, config)
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Default preprocessing configuration
        """
        self.default_config = config or PreprocessingConfig()
        
        # Check dependencies
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - preprocessing limited")
        if not PIL_AVAILABLE:
            logger.warning("PIL not available - preprocessing limited")
    
    def process(
        self,
        image: Any,
        config: Optional[PreprocessingConfig] = None,
        current_dpi: int = 150,
    ) -> PreprocessingResult:
        """
        Apply preprocessing pipeline to an image.
        
        Args:
            image: Input image (PIL.Image or numpy array)
            config: Preprocessing configuration
            current_dpi: Estimated current DPI of image
            
        Returns:
            PreprocessingResult with processed image
        """
        config = config or self.default_config
        operations = []
        warnings = []
        
        # Convert to numpy array
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            img = np.array(image)
            original_format = 'pil'
        elif NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            img = image.copy()
            original_format = 'numpy'
        else:
            warnings.append("Cannot process - unsupported image format")
            return PreprocessingResult(
                image=image,
                original_size=(0, 0),
                processed_size=(0, 0),
                operations_applied=[],
                warnings=warnings,
            )
        
        original_size = (img.shape[1], img.shape[0])  # width, height
        
        # Convert to grayscale if color
        if len(img.shape) == 3:
            if CV2_AVAILABLE:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = img[:, :, 0]  # Take first channel
            operations.append('grayscale')
        
        # Check for inverted image
        was_inverted = False
        if config.auto_invert:
            if np.mean(img) < 127:  # Dark image
                img = 255 - img
                was_inverted = True
                operations.append('invert')
        
        # Scale if needed
        scale_factor = 1.0
        if current_dpi < config.scale_if_below_dpi:
            scale_factor = config.target_dpi / current_dpi
            if CV2_AVAILABLE and scale_factor > 1.0:
                new_width = int(img.shape[1] * scale_factor)
                new_height = int(img.shape[0] * scale_factor)
                img = cv2.resize(
                    img,
                    (new_width, new_height),
                    interpolation=cv2.INTER_CUBIC
                )
                operations.append(f'scale_{scale_factor:.1f}x')
        
        # Deskew
        skew_corrected = 0.0
        if config.deskew and CV2_AVAILABLE:
            skew_angle = self._detect_skew(img)
            if abs(skew_angle) > 0.5 and abs(skew_angle) < config.max_skew_angle:
                img = self._rotate_image(img, skew_angle)
                skew_corrected = skew_angle
                operations.append(f'deskew_{skew_angle:.1f}deg')
        
        # Remove borders
        if config.remove_borders and CV2_AVAILABLE:
            img, border_removed = self._remove_borders(img, config.border_threshold)
            if border_removed:
                operations.append('remove_borders')
        
        # Denoise
        if config.denoise != DenoiseMethod.NONE and CV2_AVAILABLE:
            img = self._denoise(img, config.denoise, config.denoise_strength)
            operations.append(f'denoise_{config.denoise.name}')
        
        # Enhance contrast
        if config.enhance_contrast and CV2_AVAILABLE:
            img = self._enhance_contrast(img, config.contrast_factor)
            operations.append('enhance_contrast')
        
        # Binarize
        if config.binarization != BinarizationMethod.NONE and CV2_AVAILABLE:
            img = self._binarize(
                img,
                config.binarization,
                config.adaptive_block_size,
                config.adaptive_c,
            )
            operations.append(f'binarize_{config.binarization.name}')
        
        processed_size = (img.shape[1], img.shape[0])
        
        # Convert back to original format if needed
        if original_format == 'pil' and PIL_AVAILABLE:
            img = Image.fromarray(img)
        
        return PreprocessingResult(
            image=img,
            original_size=original_size,
            processed_size=processed_size,
            operations_applied=operations,
            scale_factor=scale_factor,
            skew_corrected=skew_corrected,
            was_inverted=was_inverted,
            warnings=warnings,
        )
    
    def _detect_skew(self, img: 'np.ndarray') -> float:
        """Detect skew angle using Hough transform."""
        try:
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is None:
                return 0.0
            
            angles = []
            for line in lines[:50]:
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:
                    angles.append(angle)
            
            if not angles:
                return 0.0
            
            return float(np.median(angles))
        except Exception:
            return 0.0
    
    def _rotate_image(
        self,
        img: 'np.ndarray',
        angle: float,
    ) -> 'np.ndarray':
        """Rotate image by given angle."""
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_width = int(height * sin + width * cos)
        new_height = int(height * cos + width * sin)
        
        # Adjust the rotation matrix
        rotation_matrix[0, 2] += (new_width - width) / 2
        rotation_matrix[1, 2] += (new_height - height) / 2
        
        # Perform rotation
        rotated = cv2.warpAffine(
            img,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        
        return rotated
    
    def _remove_borders(
        self,
        img: 'np.ndarray',
        threshold: int,
    ) -> Tuple['np.ndarray', bool]:
        """Remove black borders from image."""
        if img.ndim != 2:
            return img, False
        
        # Find non-black rows and columns
        row_sums = np.sum(img, axis=1)
        col_sums = np.sum(img, axis=0)
        
        row_threshold = img.shape[1] * threshold
        col_threshold = img.shape[0] * threshold
        
        # Find content boundaries
        rows = np.where(row_sums > row_threshold)[0]
        cols = np.where(col_sums > col_threshold)[0]
        
        if len(rows) == 0 or len(cols) == 0:
            return img, False
        
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]
        
        # Only crop if significant border found
        original_area = img.shape[0] * img.shape[1]
        content_area = (bottom - top) * (right - left)
        
        if content_area / original_area < 0.9:  # More than 10% was border
            return img[top:bottom+1, left:right+1], True
        
        return img, False
    
    def _denoise(
        self,
        img: 'np.ndarray',
        method: DenoiseMethod,
        strength: int,
    ) -> 'np.ndarray':
        """Apply denoising to image."""
        if method == DenoiseMethod.GAUSSIAN:
            return cv2.GaussianBlur(img, (strength, strength), 0)
        
        elif method == DenoiseMethod.MEDIAN:
            kernel_size = strength if strength % 2 == 1 else strength + 1
            return cv2.medianBlur(img, kernel_size)
        
        elif method == DenoiseMethod.BILATERAL:
            return cv2.bilateralFilter(img, strength * 2 + 1, 75, 75)
        
        elif method == DenoiseMethod.MORPHOLOGICAL:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return img
    
    def _enhance_contrast(
        self,
        img: 'np.ndarray',
        factor: float,
    ) -> 'np.ndarray':
        """Enhance image contrast using CLAHE."""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def _binarize(
        self,
        img: 'np.ndarray',
        method: BinarizationMethod,
        block_size: int,
        c: int,
    ) -> 'np.ndarray':
        """Convert image to binary."""
        if method == BinarizationMethod.OTSU:
            _, binary = cv2.threshold(
                img, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary
        
        elif method == BinarizationMethod.ADAPTIVE_MEAN:
            return cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size, c
            )
        
        elif method == BinarizationMethod.ADAPTIVE_GAUSSIAN:
            return cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, c
            )
        
        elif method == BinarizationMethod.SAUVOLA:
            return self._sauvola_threshold(img)
        
        return img
    
    def _sauvola_threshold(
        self,
        img: 'np.ndarray',
        window_size: int = 25,
        k: float = 0.2,
        r: float = 128,
    ) -> 'np.ndarray':
        """Apply Sauvola thresholding."""
        # Calculate local mean and std
        mean = cv2.blur(img.astype(float), (window_size, window_size))
        
        mean_sq = cv2.blur(
            (img.astype(float) ** 2),
            (window_size, window_size)
        )
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
        
        # Sauvola threshold
        threshold = mean * (1 + k * ((std / r) - 1))
        
        binary = np.zeros_like(img)
        binary[img > threshold] = 255
        
        return binary.astype(np.uint8)


def enhance_for_ocr(
    image: Any,
    target_dpi: int = 300,
    deskew: bool = True,
) -> Any:
    """
    Convenience function for basic OCR enhancement.
    
    Args:
        image: Input image
        target_dpi: Target DPI for scaling
        deskew: Whether to correct skew
        
    Returns:
        Enhanced image
    """
    config = PreprocessingConfig(
        target_dpi=target_dpi,
        deskew=deskew,
        binarization=BinarizationMethod.ADAPTIVE_GAUSSIAN,
        denoise=DenoiseMethod.MEDIAN,
        enhance_contrast=True,
    )
    
    preprocessor = OCRPreprocessor(config)
    result = preprocessor.process(image, config)
    
    return result.image

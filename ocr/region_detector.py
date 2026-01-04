"""
Region Detector for Intelligent OCR

This module identifies which regions of a page need OCR processing.
It analyzes page images to find:
- Text-poor regions (likely scanned content)
- Areas with graphics/logos to exclude
- Signature regions to exclude
- Regions with sufficient contrast for OCR

Why Region Detection Matters:
- Full-page OCR is expensive and slow
- OCRing logos/graphics produces garbage
- Focused OCR on text regions is faster and more accurate
- Allows mixing text-layer extraction with OCR on same page

Detection Strategy:
1. Convert page to image
2. Analyze image histogram and contrast
3. Detect text-like regions using edge detection
4. Filter out likely non-text regions (logos, photos)
5. Return candidate regions with confidence scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
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
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class RegionCategory(Enum):
    """Categories of detected regions."""
    TEXT = auto()           # Contains text, good for OCR
    TABLE = auto()          # Table region, needs special handling
    GRAPHIC = auto()        # Photo/graphic, skip OCR
    LOGO = auto()           # Logo, skip OCR
    SIGNATURE = auto()      # Signature, skip OCR
    BARCODE = auto()        # Barcode/QR code
    BLANK = auto()          # Empty region
    NOISE = auto()          # Artifacts, skip OCR
    UNKNOWN = auto()        # Unclassified


@dataclass
class ImageQualityMetrics:
    """
    Quality metrics for an image or region.
    
    Used to decide if OCR is likely to succeed.
    """
    dpi: int = 0
    contrast: float = 0.0           # 0-1, higher is better
    sharpness: float = 0.0          # 0-1, higher is better
    brightness: float = 0.0         # 0-1, 0.5 is ideal
    noise_level: float = 0.0        # 0-1, lower is better
    text_density: float = 0.0       # Estimated text density
    skew_angle: float = 0.0         # Degrees of rotation
    is_inverted: bool = False       # White text on black?
    
    @property
    def ocr_suitability(self) -> float:
        """
        Overall suitability score for OCR (0-1).
        
        Considers all metrics to produce a single score.
        Higher is better.
        """
        # Weight factors
        score = 0.0
        
        # DPI contribution (300 DPI is ideal)
        if self.dpi >= 300:
            score += 0.3
        elif self.dpi >= 200:
            score += 0.2
        elif self.dpi >= 150:
            score += 0.1
        
        # Contrast contribution
        score += self.contrast * 0.25
        
        # Sharpness contribution
        score += self.sharpness * 0.2
        
        # Noise penalty
        score -= self.noise_level * 0.15
        
        # Brightness penalty (penalize extreme values)
        brightness_penalty = abs(self.brightness - 0.5) * 0.2
        score -= brightness_penalty
        
        # Skew penalty
        skew_penalty = min(abs(self.skew_angle) / 45.0, 1.0) * 0.1
        score -= skew_penalty
        
        return max(0.0, min(1.0, score))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'dpi': self.dpi,
            'contrast': round(self.contrast, 3),
            'sharpness': round(self.sharpness, 3),
            'brightness': round(self.brightness, 3),
            'noise_level': round(self.noise_level, 3),
            'text_density': round(self.text_density, 3),
            'skew_angle': round(self.skew_angle, 2),
            'is_inverted': self.is_inverted,
            'ocr_suitability': round(self.ocr_suitability, 3),
        }


@dataclass
class OCRCandidate:
    """
    A region identified as a candidate for OCR.
    
    Contains the region bounds, category, and confidence.
    """
    x0: int
    y0: int
    x1: int
    y1: int
    category: RegionCategory
    confidence: float = 0.5
    quality_metrics: Optional[ImageQualityMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def width(self) -> int:
        return self.x1 - self.x0
    
    @property
    def height(self) -> int:
        return self.y1 - self.y0
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def should_ocr(self) -> bool:
        """Whether this region should be OCRed."""
        return self.category in (RegionCategory.TEXT, RegionCategory.TABLE)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Return as (x0, y0, x1, y1) tuple."""
        return (self.x0, self.y0, self.x1, self.y1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bounds': self.to_tuple(),
            'category': self.category.name,
            'confidence': round(self.confidence, 3),
            'width': self.width,
            'height': self.height,
            'should_ocr': self.should_ocr,
            'quality': self.quality_metrics.to_dict() if self.quality_metrics else None,
        }


@dataclass
class RegionAnalysis:
    """
    Complete analysis of a page for OCR.
    
    Contains all detected regions and page-level metrics.
    """
    page_number: int
    page_width: int
    page_height: int
    candidates: List[OCRCandidate]
    quality_metrics: ImageQualityMetrics
    needs_ocr: bool
    ocr_coverage: float          # % of page that needs OCR
    warnings: List[str] = field(default_factory=list)
    
    @property
    def ocr_candidates(self) -> List[OCRCandidate]:
        """Get only regions that should be OCRed."""
        return [c for c in self.candidates if c.should_ocr]
    
    @property
    def skip_regions(self) -> List[OCRCandidate]:
        """Get regions to skip."""
        return [c for c in self.candidates if not c.should_ocr]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'page_number': self.page_number,
            'page_size': (self.page_width, self.page_height),
            'needs_ocr': self.needs_ocr,
            'ocr_coverage': round(self.ocr_coverage, 3),
            'quality': self.quality_metrics.to_dict(),
            'candidates': [c.to_dict() for c in self.candidates],
            'warnings': self.warnings,
        }


class RegionDetector:
    """
    Detects regions that need OCR processing.
    
    Uses computer vision techniques to analyze page images
    and identify text-like regions vs. graphics/noise.
    
    Features:
    - Text region detection using edge analysis
    - Logo/graphic detection using contour analysis
    - Image quality assessment
    - Automatic DPI detection
    - Skew detection
    
    Usage:
        detector = RegionDetector()
        
        # Analyze a page image
        analysis = detector.analyze_page(page_image, page_number=0)
        
        if analysis.needs_ocr:
            for candidate in analysis.ocr_candidates:
                # Process region with OCR
                ...
    """
    
    def __init__(
        self,
        min_region_area: int = 1000,
        text_contrast_threshold: float = 0.3,
        text_density_threshold: float = 0.1,
        logo_aspect_ratio_threshold: float = 2.0,
    ):
        """
        Initialize region detector.
        
        Args:
            min_region_area: Minimum pixel area for a region
            text_contrast_threshold: Minimum contrast for text regions
            text_density_threshold: Minimum text density
            logo_aspect_ratio_threshold: Max aspect ratio before assuming logo
        """
        self.min_region_area = min_region_area
        self.text_contrast_threshold = text_contrast_threshold
        self.text_density_threshold = text_density_threshold
        self.logo_aspect_ratio_threshold = logo_aspect_ratio_threshold
        
        # Verify dependencies
        if not NUMPY_AVAILABLE:
            logger.warning("numpy not available - region detection limited")
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - region detection limited")
        if not PIL_AVAILABLE:
            logger.warning("PIL not available - region detection limited")
    
    def analyze_page(
        self,
        image: Any,  # PIL.Image or numpy array
        page_number: int = 0,
        existing_text_coverage: float = 0.0,
    ) -> RegionAnalysis:
        """
        Analyze a page image for OCR candidates.
        
        Args:
            image: Page image (PIL.Image or numpy array)
            page_number: Page number for reference
            existing_text_coverage: % of page already covered by text layer
            
        Returns:
            RegionAnalysis with detected candidates
        """
        warnings = []
        
        # Convert to numpy array if needed
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            img_array = np.array(image)
        elif NUMPY_AVAILABLE and isinstance(image, np.ndarray):
            img_array = image
        else:
            # Can't process - return empty analysis
            warnings.append("Cannot process image - missing dependencies")
            return RegionAnalysis(
                page_number=page_number,
                page_width=0,
                page_height=0,
                candidates=[],
                quality_metrics=ImageQualityMetrics(),
                needs_ocr=False,
                ocr_coverage=0.0,
                warnings=warnings,
            )
        
        # Get dimensions
        if len(img_array.shape) == 3:
            height, width, channels = img_array.shape
        else:
            height, width = img_array.shape
            channels = 1
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(img_array)
        
        # Detect regions
        candidates = []
        if CV2_AVAILABLE:
            candidates = self._detect_regions_cv2(img_array)
        else:
            # Fallback: treat entire page as one candidate
            candidates = [OCRCandidate(
                x0=0,
                y0=0,
                x1=width,
                y1=height,
                category=RegionCategory.TEXT,
                confidence=0.5,
            )]
            warnings.append("OpenCV not available - using full page")
        
        # Calculate OCR coverage
        total_ocr_area = sum(c.area for c in candidates if c.should_ocr)
        page_area = width * height
        ocr_coverage = total_ocr_area / page_area if page_area > 0 else 0.0
        
        # Determine if OCR is needed
        # OCR needed if: low text coverage AND some text-like regions detected
        needs_ocr = (
            existing_text_coverage < 0.5 and  # Less than 50% text layer coverage
            quality_metrics.text_density > self.text_density_threshold and
            any(c.should_ocr for c in candidates)
        )
        
        return RegionAnalysis(
            page_number=page_number,
            page_width=width,
            page_height=height,
            candidates=candidates,
            quality_metrics=quality_metrics,
            needs_ocr=needs_ocr,
            ocr_coverage=ocr_coverage,
            warnings=warnings,
        )
    
    def _calculate_quality_metrics(
        self,
        img_array: 'np.ndarray',
    ) -> ImageQualityMetrics:
        """Calculate image quality metrics."""
        if not NUMPY_AVAILABLE:
            return ImageQualityMetrics()
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else img_array[:, :, 0]
        else:
            gray = img_array
        
        # Brightness (mean pixel value)
        brightness = np.mean(gray) / 255.0
        
        # Contrast (standard deviation)
        contrast = np.std(gray) / 128.0  # Normalize to ~0-1
        contrast = min(contrast, 1.0)
        
        # Check for inverted image
        is_inverted = brightness < 0.3
        
        # Sharpness (using Laplacian variance)
        sharpness = 0.5
        if CV2_AVAILABLE:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = min(np.var(laplacian) / 1000.0, 1.0)
        
        # Noise level (estimated from high-frequency content)
        noise_level = 0.0
        if CV2_AVAILABLE:
            # Use median filter to estimate noise
            denoised = cv2.medianBlur(gray, 3)
            noise = np.abs(gray.astype(float) - denoised.astype(float))
            noise_level = min(np.mean(noise) / 50.0, 1.0)
        
        # Text density (using edge detection)
        text_density = 0.0
        if CV2_AVAILABLE:
            edges = cv2.Canny(gray, 50, 150)
            text_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Skew detection
        skew_angle = 0.0
        if CV2_AVAILABLE:
            skew_angle = self._detect_skew(gray)
        
        # Estimate DPI (rough heuristic based on text size)
        # This is a rough estimate - actual DPI requires page dimensions
        dpi = 150  # Default assumption
        if CV2_AVAILABLE:
            # Look for typical text line heights
            dpi = self._estimate_dpi(gray)
        
        return ImageQualityMetrics(
            dpi=dpi,
            contrast=contrast,
            sharpness=sharpness,
            brightness=brightness,
            noise_level=noise_level,
            text_density=text_density,
            skew_angle=skew_angle,
            is_inverted=is_inverted,
        )
    
    def _detect_regions_cv2(
        self,
        img_array: 'np.ndarray',
    ) -> List[OCRCandidate]:
        """Detect regions using OpenCV."""
        candidates = []
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        height, width = gray.shape
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Group nearby contours into regions
        # Using morphological operations to merge close text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        region_contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in region_contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Skip small regions
            if area < self.min_region_area:
                continue
            
            # Classify region
            category = self._classify_region(
                gray[y:y+h, x:x+w],
                w, h
            )
            
            # Calculate confidence based on region characteristics
            confidence = self._calculate_region_confidence(
                gray[y:y+h, x:x+w],
                category
            )
            
            candidates.append(OCRCandidate(
                x0=x,
                y0=y,
                x1=x + w,
                y1=y + h,
                category=category,
                confidence=confidence,
            ))
        
        # If no regions found, treat page as single region
        if not candidates:
            candidates.append(OCRCandidate(
                x0=0,
                y0=0,
                x1=width,
                y1=height,
                category=RegionCategory.TEXT,
                confidence=0.5,
            ))
        
        return candidates
    
    def _classify_region(
        self,
        region: 'np.ndarray',
        width: int,
        height: int,
    ) -> RegionCategory:
        """Classify a region based on its content."""
        if not CV2_AVAILABLE:
            return RegionCategory.UNKNOWN
        
        # Check aspect ratio for logo detection
        aspect_ratio = width / max(height, 1)
        if aspect_ratio > self.logo_aspect_ratio_threshold:
            return RegionCategory.LOGO
        
        # Calculate edge density
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Text regions have moderate edge density
        if 0.05 < edge_density < 0.4:
            return RegionCategory.TEXT
        
        # Very high edge density might be a barcode
        if edge_density > 0.6:
            return RegionCategory.BARCODE
        
        # Very low edge density might be blank or graphic
        if edge_density < 0.02:
            # Check if it's mostly uniform (blank)
            std_dev = np.std(region)
            if std_dev < 20:
                return RegionCategory.BLANK
            else:
                return RegionCategory.GRAPHIC
        
        # Check for signature-like patterns (many curves)
        contours, _ = cv2.findContours(
            cv2.adaptiveThreshold(
                region, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            ),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) > 50 and edge_density < 0.15:
            return RegionCategory.SIGNATURE
        
        return RegionCategory.TEXT
    
    def _calculate_region_confidence(
        self,
        region: 'np.ndarray',
        category: RegionCategory,
    ) -> float:
        """Calculate confidence score for region classification."""
        if category == RegionCategory.TEXT:
            # Higher confidence for good contrast
            contrast = np.std(region) / 128.0
            return min(0.5 + contrast * 0.5, 1.0)
        elif category in (RegionCategory.LOGO, RegionCategory.GRAPHIC):
            return 0.7  # Moderate confidence for non-text
        elif category == RegionCategory.BLANK:
            return 0.9  # High confidence for blank detection
        else:
            return 0.5
    
    def _detect_skew(self, gray: 'np.ndarray') -> float:
        """Detect page skew angle in degrees."""
        if not CV2_AVAILABLE:
            return 0.0
        
        try:
            # Use Hough transform to detect lines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is None or len(lines) == 0:
                return 0.0
            
            # Calculate angle distribution
            angles = []
            for line in lines[:50]:  # Limit to first 50 lines
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:  # Filter reasonable angles
                    angles.append(angle)
            
            if not angles:
                return 0.0
            
            # Return median angle
            return float(np.median(angles))
        except Exception:
            return 0.0
    
    def _estimate_dpi(self, gray: 'np.ndarray') -> int:
        """Estimate DPI from image characteristics."""
        if not CV2_AVAILABLE:
            return 150
        
        try:
            # Detect text-like structures
            # Typical body text at 300 DPI has line height ~30-50 pixels
            # At 150 DPI, it would be ~15-25 pixels
            
            # Use horizontal projection to find line heights
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )
            
            # Sum along rows
            projection = np.sum(binary, axis=1)
            
            # Find gaps between text lines
            threshold = np.mean(projection) * 0.5
            is_text = projection > threshold
            
            # Find runs of text
            runs = []
            in_run = False
            run_start = 0
            
            for i, val in enumerate(is_text):
                if val and not in_run:
                    run_start = i
                    in_run = True
                elif not val and in_run:
                    runs.append(i - run_start)
                    in_run = False
            
            if not runs:
                return 150
            
            median_height = np.median(runs)
            
            # Estimate DPI based on typical text heights
            # 12pt text at 72 DPI = 12 pixels
            # 12pt text at 300 DPI = 50 pixels
            if median_height > 35:
                return 300
            elif median_height > 20:
                return 200
            else:
                return 150
        except Exception:
            return 150
    
    def find_text_sparse_regions(
        self,
        page_image: Any,
        text_blocks: List[Tuple[int, int, int, int]],
        page_number: int = 0,
    ) -> RegionAnalysis:
        """
        Find regions that have no text layer coverage.
        
        Compares detected text blocks with the page image to find
        areas that might need OCR.
        
        Args:
            page_image: Page image
            text_blocks: List of (x0, y0, x1, y1) text layer bounding boxes
            page_number: Page number
            
        Returns:
            RegionAnalysis for uncovered regions
        """
        # Get base analysis
        analysis = self.analyze_page(page_image, page_number)
        
        if not text_blocks:
            # No text layer - return as-is
            return analysis
        
        # Create mask of text-covered areas
        width = analysis.page_width
        height = analysis.page_height
        
        if not NUMPY_AVAILABLE:
            return analysis
        
        coverage_mask = np.zeros((height, width), dtype=np.uint8)
        
        for x0, y0, x1, y1 in text_blocks:
            # Clamp coordinates
            x0 = max(0, min(x0, width))
            y0 = max(0, min(y0, height))
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            coverage_mask[y0:y1, x0:x1] = 255
        
        # Calculate coverage
        text_coverage = np.sum(coverage_mask > 0) / (width * height)
        
        # Filter candidates to only uncovered regions
        filtered_candidates = []
        
        for candidate in analysis.candidates:
            # Check overlap with text layer
            region_mask = np.zeros((height, width), dtype=np.uint8)
            region_mask[candidate.y0:candidate.y1, candidate.x0:candidate.x1] = 255
            
            overlap = np.sum((coverage_mask > 0) & (region_mask > 0))
            region_area = candidate.area
            
            overlap_ratio = overlap / region_area if region_area > 0 else 0
            
            # Keep if less than 50% covered by text layer
            if overlap_ratio < 0.5:
                filtered_candidates.append(candidate)
        
        # Update analysis
        analysis.candidates = filtered_candidates
        analysis.needs_ocr = (
            text_coverage < 0.5 and
            any(c.should_ocr for c in filtered_candidates)
        )
        
        total_ocr_area = sum(c.area for c in filtered_candidates if c.should_ocr)
        analysis.ocr_coverage = total_ocr_area / (width * height)
        
        return analysis

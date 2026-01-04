"""
Document Intelligence Pipeline

Main orchestration module that integrates all components into a unified pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator, Callable
from pathlib import Path
from enum import Enum
import logging
import time
import json

# Layout components
from .layout.box import BoundingBox, TextBlock, BlockType
from .layout.spatial_index import SpatialIndex
from .layout.layout_graph import LayoutGraph
from .layout.analyzer import LayoutAnalyzer, DocumentLayout

# OCR components
from .ocr.region_detector import RegionDetector, OCRCandidate
from .ocr.preprocessor import OCRPreprocessor
from .ocr.ocr_engine import SmartOCREngine, OCRConfig

# Validation components
from .validation.semantic_rules import SemanticValidator
from .validation.arithmetic_checks import ArithmeticChecker

# Decision system
from .decision.decision_engine import (
    Decision, DecisionEngine, FieldDecision, DocumentDecision
)

# Table processing
from .tables.table_detector import TableDetector
from .tables.continuation_detector import ContinuationDetector
from .tables.table_stitcher import TableStitcher, StitchedTable

# Review system
from .review.review_data import ReviewSession, ReviewField, ReviewStatus
from .review.html_preview import HTMLPreviewGenerator
from .review.json_export import ReviewExporter

# Performance
from .performance.streaming import StreamingProcessor
from .performance.worker_pool import WorkerPool, Task
from .performance.retry import RetryPolicy, with_retry

# Document types
from .doctypes.document_type import DocumentType
from .doctypes.registry import DocumentTypeRegistry, detect_document_type
from .doctypes.extractors import DocumentExtractor, ExtractionResult, TextBlock as ExtTextBlock

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Document processing mode."""
    FAST = "fast"           # Speed priority, minimal validation
    STANDARD = "standard"   # Balanced speed and accuracy
    THOROUGH = "thorough"   # Maximum accuracy, full validation
    REVIEW = "review"       # Generate human review output


@dataclass
class PipelineConfig:
    """Configuration for the document intelligence pipeline."""
    
    # Processing mode
    mode: ProcessingMode = ProcessingMode.STANDARD
    
    # OCR settings
    enable_ocr: bool = True
    ocr_language: str = 'eng'
    ocr_dpi: int = 300
    
    # Layout analysis
    enable_layout_analysis: bool = True
    layout_algorithm: str = 'spatial_graph'
    
    # Table detection
    enable_table_detection: bool = True
    stitch_multi_page_tables: bool = True
    
    # Validation
    enable_semantic_validation: bool = True
    enable_arithmetic_validation: bool = True
    
    # Decision thresholds
    verified_threshold: float = 0.95
    likely_threshold: float = 0.80
    review_threshold: float = 0.60
    
    # Performance
    max_workers: int = 4
    batch_size: int = 10
    enable_streaming: bool = True
    checkpoint_interval: int = 100
    
    # Output
    output_format: str = 'json'  # json, csv, review
    include_bounding_boxes: bool = True
    include_confidence: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'mode': self.mode.value,
            'enable_ocr': self.enable_ocr,
            'ocr_language': self.ocr_language,
            'ocr_dpi': self.ocr_dpi,
            'enable_layout_analysis': self.enable_layout_analysis,
            'layout_algorithm': self.layout_algorithm,
            'enable_table_detection': self.enable_table_detection,
            'stitch_multi_page_tables': self.stitch_multi_page_tables,
            'enable_semantic_validation': self.enable_semantic_validation,
            'enable_arithmetic_validation': self.enable_arithmetic_validation,
            'verified_threshold': self.verified_threshold,
            'likely_threshold': self.likely_threshold,
            'review_threshold': self.review_threshold,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'enable_streaming': self.enable_streaming,
            'checkpoint_interval': self.checkpoint_interval,
            'output_format': self.output_format,
            'include_bounding_boxes': self.include_bounding_boxes,
            'include_confidence': self.include_confidence,
        }


@dataclass
class ProcessingMetrics:
    """Metrics from document processing."""
    
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    
    total_pages: int = 0
    total_fields_extracted: int = 0
    
    # Decision breakdown
    verified_fields: int = 0
    likely_fields: int = 0
    review_required_fields: int = 0
    rejected_fields: int = 0
    
    # Timing
    total_time_ms: float = 0
    avg_time_per_doc_ms: float = 0
    avg_time_per_page_ms: float = 0
    
    # Quality
    extraction_rate: float = 0.0
    validation_pass_rate: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_documents': self.total_documents,
            'successful_documents': self.successful_documents,
            'failed_documents': self.failed_documents,
            'total_pages': self.total_pages,
            'total_fields_extracted': self.total_fields_extracted,
            'verified_fields': self.verified_fields,
            'likely_fields': self.likely_fields,
            'review_required_fields': self.review_required_fields,
            'rejected_fields': self.rejected_fields,
            'total_time_ms': self.total_time_ms,
            'avg_time_per_doc_ms': self.avg_time_per_doc_ms,
            'avg_time_per_page_ms': self.avg_time_per_page_ms,
            'extraction_rate': self.extraction_rate,
            'validation_pass_rate': self.validation_pass_rate,
        }


@dataclass
class DocumentResult:
    """Result from processing a single document."""
    
    # Source info
    source_path: str
    document_type: str
    page_count: int
    
    # Extracted data
    fields: Dict[str, Any]
    tables: List[Dict[str, Any]]
    
    # Decisions
    field_decisions: Dict[str, FieldDecision]
    document_decision: DocumentDecision
    
    # Validation
    validation_errors: List[str]
    arithmetic_errors: List[str]
    
    # Metadata
    processing_time_ms: float
    extraction_result: Optional[ExtractionResult] = None
    review_session: Optional[ReviewSession] = None
    
    @property
    def is_verified(self) -> bool:
        """Whether document passed verification."""
        return self.document_decision.decision == Decision.VERIFIED
    
    @property
    def needs_review(self) -> bool:
        """Whether document needs human review."""
        return self.document_decision.decision == Decision.REVIEW_REQUIRED
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'source_path': self.source_path,
            'document_type': self.document_type,
            'page_count': self.page_count,
            'fields': self.fields,
            'tables': self.tables,
            'field_decisions': {
                k: v.to_dict() for k, v in self.field_decisions.items()
            },
            'document_decision': self.document_decision.to_dict(),
            'validation_errors': self.validation_errors,
            'arithmetic_errors': self.arithmetic_errors,
            'processing_time_ms': self.processing_time_ms,
            'is_verified': self.is_verified,
            'needs_review': self.needs_review,
        }
    
    def to_flat_dict(self) -> dict:
        """Convert to flat dictionary for CSV export."""
        result = {
            'source_path': self.source_path,
            'document_type': self.document_type,
            'page_count': self.page_count,
            'decision': self.document_decision.decision.value,
            'confidence': self.document_decision.overall_confidence,
            'processing_time_ms': self.processing_time_ms,
        }
        
        for field_name, value in self.fields.items():
            result[field_name] = value
            if field_name in self.field_decisions:
                fd = self.field_decisions[field_name]
                result[f'{field_name}_decision'] = fd.decision.value
                result[f'{field_name}_confidence'] = fd.confidence
        
        return result


class DocumentIntelligencePipeline:
    """
    Main orchestration class for document intelligence processing.
    
    Integrates:
    - Layout analysis
    - OCR with region detection
    - Field extraction
    - Semantic validation
    - Decision making
    - Human review generation
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        document_type: Optional[DocumentType] = None,
    ):
        self.config = config or PipelineConfig()
        self.document_type = document_type
        
        # Initialize components
        self._init_components()
        
        # Metrics
        self.metrics = ProcessingMetrics()
    
    def _init_components(self) -> None:
        """Initialize pipeline components."""
        
        # Layout analyzer
        self.layout_analyzer = LayoutAnalyzer(
            enable_ocr_fallback=self.config.enable_ocr,
        )
        
        # OCR components
        if self.config.enable_ocr:
            self.region_detector = RegionDetector()
            self.ocr_preprocessor = OCRPreprocessor()
            self.ocr_engine = SmartOCREngine(
                OCRConfig(
                    language=self.config.ocr_language,
                    dpi=self.config.ocr_dpi,
                )
            )
        
        # Table processing
        if self.config.enable_table_detection:
            self.table_detector = TableDetector()
            self.continuation_detector = ContinuationDetector()
            self.table_stitcher = TableStitcher()
        
        # Validation
        if self.config.enable_semantic_validation:
            self.semantic_validator = SemanticValidator()
        
        if self.config.enable_arithmetic_validation:
            self.arithmetic_checker = ArithmeticChecker()
        
        # Decision engine
        self.decision_engine = DecisionEngine(
            verified_threshold=self.config.verified_threshold,
            likely_threshold=self.config.likely_threshold,
            review_threshold=self.config.review_threshold,
        )
        
        # Document type registry
        self.type_registry = DocumentTypeRegistry()
    
    def process_document(
        self,
        pdf_path: str,
        document_type: Optional[str] = None,
    ) -> DocumentResult:
        """
        Process a single PDF document.
        
        Args:
            pdf_path: Path to PDF file
            document_type: Optional document type name (auto-detect if not provided)
        
        Returns:
            DocumentResult with all extracted data and decisions
        """
        start_time = time.time()
        
        try:
            # Load PDF
            pdf_content = self._load_pdf(pdf_path)
            
            # Analyze layout
            if self.config.enable_layout_analysis:
                doc_layout = self.layout_analyzer.analyze_document(pdf_content)
            else:
                doc_layout = self._basic_layout(pdf_content)
            
            # Get text blocks
            text_blocks = self._extract_text_blocks(doc_layout)
            
            # Detect or use specified document type
            if document_type:
                doc_type = self.type_registry.get(document_type)
            else:
                doc_type = detect_document_type(
                    ' '.join(b.text for b in text_blocks[:50])
                )
            
            if not doc_type:
                doc_type = self.document_type or self._get_default_type()
            
            # Create extractor
            extractor = DocumentExtractor(doc_type)
            
            # Extract tables
            tables = []
            if self.config.enable_table_detection:
                tables = self._extract_tables(doc_layout)
            
            # Extract fields
            ext_blocks = [
                ExtTextBlock(
                    id=str(i),
                    text=b.text,
                    page=b.page_number if hasattr(b, 'page_number') else 0,
                    bbox=(b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1),
                    confidence=b.confidence if hasattr(b, 'confidence') else 1.0,
                )
                for i, b in enumerate(text_blocks)
            ]
            
            extraction_result = extractor.extract(ext_blocks, tables)
            
            # Run validation
            validation_errors = []
            arithmetic_errors = []
            
            if self.config.enable_semantic_validation:
                validation_errors = self._run_semantic_validation(
                    extraction_result, doc_type
                )
            
            if self.config.enable_arithmetic_validation:
                arithmetic_errors = self._run_arithmetic_validation(
                    extraction_result
                )
            
            # Make decisions
            field_decisions = {}
            for field_name, extracted in extraction_result.fields.items():
                decision = self.decision_engine.decide_field(
                    field_name=field_name,
                    value=extracted.value,
                    confidence=extracted.confidence,
                    extraction_method=extracted.method.value,
                    validation_passed=extracted.is_valid,
                    cross_validation_score=1.0 if not extracted.validation_errors else 0.5,
                )
                field_decisions[field_name] = decision
            
            # Overall document decision
            doc_decision = self.decision_engine.decide_document(
                field_decisions=field_decisions,
                required_fields=[
                    f.name for f in doc_type.fields if f.required
                ],
                validation_errors=validation_errors + arithmetic_errors,
            )
            
            # Build result
            processing_time = (time.time() - start_time) * 1000
            
            result = DocumentResult(
                source_path=pdf_path,
                document_type=doc_type.name,
                page_count=len(doc_layout.pages) if hasattr(doc_layout, 'pages') else 1,
                fields={k: v.value for k, v in extraction_result.fields.items()},
                tables=tables,
                field_decisions=field_decisions,
                document_decision=doc_decision,
                validation_errors=validation_errors,
                arithmetic_errors=arithmetic_errors,
                processing_time_ms=processing_time,
                extraction_result=extraction_result,
            )
            
            # Generate review session if needed
            if self.config.mode == ProcessingMode.REVIEW or doc_decision.decision == Decision.REVIEW_REQUIRED:
                result.review_session = self._create_review_session(
                    result, doc_layout
                )
            
            # Update metrics
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            self.metrics.failed_documents += 1
            raise
    
    def process_batch(
        self,
        pdf_paths: List[str],
        document_type: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[DocumentResult]:
        """
        Process multiple PDF documents.
        
        Args:
            pdf_paths: List of PDF file paths
            document_type: Optional document type (auto-detect if not provided)
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of DocumentResult objects
        """
        results = []
        total = len(pdf_paths)
        
        for i, pdf_path in enumerate(pdf_paths):
            try:
                result = self.process_document(pdf_path, document_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def process_streaming(
        self,
        pdf_paths: Iterator[str],
        document_type: Optional[str] = None,
    ) -> Iterator[DocumentResult]:
        """
        Process documents in streaming mode.
        
        Yields results as they complete, useful for large batches.
        """
        for pdf_path in pdf_paths:
            try:
                yield self.process_document(pdf_path, document_type)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
    
    def _load_pdf(self, pdf_path: str) -> Any:
        """Load PDF file using available backend."""
        try:
            import pdfplumber
            return pdfplumber.open(pdf_path)
        except ImportError:
            pass
        
        try:
            import fitz  # PyMuPDF
            return fitz.open(pdf_path)
        except ImportError:
            pass
        
        raise ImportError("No PDF library available. Install pdfplumber or PyMuPDF.")
    
    def _basic_layout(self, pdf_content: Any) -> Any:
        """Create basic layout without full analysis."""
        # Minimal layout extraction
        return pdf_content
    
    def _extract_text_blocks(self, doc_layout: Any) -> List[TextBlock]:
        """Extract text blocks from document layout."""
        blocks = []
        
        # Handle different layout types
        if hasattr(doc_layout, 'pages'):
            for page_layout in doc_layout.pages:
                if hasattr(page_layout, 'blocks'):
                    blocks.extend(page_layout.blocks)
        elif hasattr(doc_layout, 'blocks'):
            blocks.extend(doc_layout.blocks)
        
        return blocks
    
    def _extract_tables(self, doc_layout: Any) -> List[Dict[str, Any]]:
        """Extract and stitch tables from document."""
        tables = []
        
        if not self.config.enable_table_detection:
            return tables
        
        # Detect tables on each page
        page_tables = []
        if hasattr(doc_layout, 'pages'):
            for page in doc_layout.pages:
                detected = self.table_detector.detect(page)
                page_tables.extend(detected)
        
        # Stitch multi-page tables
        if self.config.stitch_multi_page_tables and len(page_tables) > 1:
            stitched = self.table_stitcher.stitch_tables(page_tables)
            tables = [t.to_dict() for t in stitched]
        else:
            tables = page_tables
        
        return tables
    
    def _run_semantic_validation(
        self,
        extraction_result: ExtractionResult,
        doc_type: DocumentType,
    ) -> List[str]:
        """Run semantic validation on extracted data."""
        errors = []
        
        for field_name, extracted in extraction_result.fields.items():
            field_def = doc_type.get_field(field_name)
            if not field_def:
                continue
            
            # Validate based on field type
            issues = self.semantic_validator.validate_field(
                field_name=field_name,
                value=extracted.value,
                field_type=field_def.field_type.value,
            )
            
            errors.extend([str(issue) for issue in issues])
        
        return errors
    
    def _run_arithmetic_validation(
        self,
        extraction_result: ExtractionResult,
    ) -> List[str]:
        """Run arithmetic validation on extracted data."""
        errors = []
        
        # Get all numeric fields
        values = {}
        for field_name, extracted in extraction_result.fields.items():
            values[field_name] = extracted.value
        
        # Check common arithmetic relationships
        issues = self.arithmetic_checker.check_all(values)
        errors.extend([str(issue) for issue in issues])
        
        return errors
    
    def _create_review_session(
        self,
        result: DocumentResult,
        doc_layout: Any,
    ) -> ReviewSession:
        """Create a review session for human-in-the-loop."""
        
        fields = []
        for field_name, value in result.fields.items():
            decision = result.field_decisions.get(field_name)
            
            field = ReviewField(
                field_name=field_name,
                extracted_value=value,
                confidence=decision.confidence if decision else 0.0,
                decision=decision.decision.value if decision else 'unknown',
                needs_review=decision.decision in [Decision.REVIEW_REQUIRED, Decision.REJECTED] if decision else True,
            )
            fields.append(field)
        
        return ReviewSession(
            document_id=result.source_path,
            fields=fields,
            status=ReviewStatus.PENDING,
        )
    
    def _get_default_type(self) -> DocumentType:
        """Get default generic document type."""
        from .doctypes.document_type import create_field, FieldType
        
        return DocumentType(
            name='generic',
            display_name='Generic Document',
            description='Generic document with common fields',
            version='1.0',
            fields=[
                create_field('date', FieldType.DATE, ['Date']),
                create_field('total', FieldType.CURRENCY, ['Total', 'Amount']),
                create_field('reference', FieldType.IDENTIFIER, ['Ref', 'Reference', '#']),
            ],
        )
    
    def _update_metrics(self, result: DocumentResult) -> None:
        """Update processing metrics."""
        self.metrics.total_documents += 1
        self.metrics.successful_documents += 1
        self.metrics.total_pages += result.page_count
        self.metrics.total_fields_extracted += len(result.fields)
        self.metrics.total_time_ms += result.processing_time_ms
        
        # Decision counts
        for fd in result.field_decisions.values():
            if fd.decision == Decision.VERIFIED:
                self.metrics.verified_fields += 1
            elif fd.decision == Decision.LIKELY:
                self.metrics.likely_fields += 1
            elif fd.decision == Decision.REVIEW_REQUIRED:
                self.metrics.review_required_fields += 1
            elif fd.decision == Decision.REJECTED:
                self.metrics.rejected_fields += 1
        
        # Averages
        if self.metrics.total_documents > 0:
            self.metrics.avg_time_per_doc_ms = (
                self.metrics.total_time_ms / self.metrics.total_documents
            )
        
        if self.metrics.total_pages > 0:
            self.metrics.avg_time_per_page_ms = (
                self.metrics.total_time_ms / self.metrics.total_pages
            )
    
    def export_results(
        self,
        results: List[DocumentResult],
        output_path: str,
        format: str = 'json',
    ) -> None:
        """Export processing results to file."""
        
        if format == 'json':
            self._export_json(results, output_path)
        elif format == 'csv':
            self._export_csv(results, output_path)
        elif format == 'review':
            self._export_review(results, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_json(self, results: List[DocumentResult], output_path: str) -> None:
        """Export to JSON."""
        data = {
            'results': [r.to_dict() for r in results],
            'metrics': self.metrics.to_dict(),
            'config': self.config.to_dict(),
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _export_csv(self, results: List[DocumentResult], output_path: str) -> None:
        """Export to CSV."""
        import csv
        
        if not results:
            return
        
        # Get all field names
        all_fields = set()
        for r in results:
            all_fields.update(r.fields.keys())
        
        fieldnames = [
            'source_path', 'document_type', 'page_count',
            'decision', 'confidence', 'processing_time_ms',
        ] + sorted(all_fields)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for result in results:
                row = result.to_flat_dict()
                writer.writerow(row)
    
    def _export_review(self, results: List[DocumentResult], output_path: str) -> None:
        """Export review files for human-in-the-loop."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generator = HTMLPreviewGenerator()
        exporter = ReviewExporter()
        
        for result in results:
            if result.review_session:
                # Generate HTML preview
                html_path = output_dir / f"{Path(result.source_path).stem}_review.html"
                html_content = generator.generate(
                    result.review_session,
                    result.source_path,
                )
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Generate JSON for corrections
                json_path = output_dir / f"{Path(result.source_path).stem}_review.json"
                exporter.export_json(result.review_session, str(json_path))
    
    def get_metrics(self) -> ProcessingMetrics:
        """Get current processing metrics."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset processing metrics."""
        self.metrics = ProcessingMetrics()


# Convenience function for quick processing
def process_pdf(
    pdf_path: str,
    document_type: Optional[str] = None,
    mode: ProcessingMode = ProcessingMode.STANDARD,
) -> DocumentResult:
    """
    Quick function to process a single PDF.
    
    Args:
        pdf_path: Path to PDF file
        document_type: Optional document type name
        mode: Processing mode
    
    Returns:
        DocumentResult with extracted data
    """
    config = PipelineConfig(mode=mode)
    pipeline = DocumentIntelligencePipeline(config)
    return pipeline.process_document(pdf_path, document_type)


def process_batch(
    pdf_paths: List[str],
    output_path: str,
    document_type: Optional[str] = None,
    format: str = 'json',
) -> ProcessingMetrics:
    """
    Quick function to process multiple PDFs and export results.
    
    Args:
        pdf_paths: List of PDF file paths
        output_path: Path for output file
        document_type: Optional document type
        format: Output format (json, csv, review)
    
    Returns:
        Processing metrics
    """
    pipeline = DocumentIntelligencePipeline()
    results = pipeline.process_batch(pdf_paths, document_type)
    pipeline.export_results(results, output_path, format)
    return pipeline.get_metrics()

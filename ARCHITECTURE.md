# Document Intelligence Pipeline - Architecture Guide

This document describes the architecture of the upgraded Document Intelligence Pipeline.

## Overview

The Document Intelligence Pipeline is a production-grade system for extracting structured data from PDF documents. It goes beyond simple regex-based extraction to use spatial reasoning, semantic validation, and categorical decision making.

## Key Upgrades from v1

1. **Layout-Aware Extraction** - Uses R-tree spatial indexing and layout graphs
2. **Region-Based OCR** - Only OCRs regions that need it
3. **Semantic Validation** - Business logic beyond format checks
4. **Decision System** - VERIFIED/LIKELY/REVIEW_REQUIRED/REJECTED
5. **Multi-Page Tables** - Detects and stitches tables across pages
6. **Human-in-the-Loop** - Interactive review with HTML preview
7. **Performance Hardening** - Streaming, worker pools, retries
8. **Pluggable Types** - Pre-built and custom document types
9. **Dataset Export** - COCO, Label Studio, HuggingFace, YOLO formats

## Module Architecture

### Layout Analysis (`layout/`)

The layout module provides spatial intelligence:

- **BoundingBox**: Immutable box with geometry operations (union, intersection, IoU)
- **TextBlock**: Text content with position, type classification, reading order
- **SpatialIndex**: R-tree index for fast spatial queries (falls back to naive search)
- **LayoutGraph**: Graph representation of spatial relationships between blocks
- **LayoutAnalyzer**: Orchestrates layout analysis across pages

```python
from pdf_to_csv.layout import LayoutAnalyzer, BoundingBox, Direction

analyzer = LayoutAnalyzer()
layout = analyzer.analyze_document(pdf)

# Find blocks near a point
nearby = layout.spatial_index.find_near(x=100, y=200, radius=50)

# Get blocks to the right of another block
right_blocks = layout.spatial_index.find_direction(block.bbox, Direction.RIGHT)
```

### OCR Engine (`ocr/`)

Region-based OCR that only processes what needs it:

- **RegionDetector**: Identifies regions needing OCR based on text quality
- **OCRPreprocessor**: Adaptive image preprocessing (binarization, denoising)
- **SmartOCREngine**: Multi-pass OCR with confidence tracking

```python
from pdf_to_csv.ocr import SmartOCREngine, OCRConfig

engine = SmartOCREngine(OCRConfig(language='eng', dpi=300))
result = engine.process_region(image, bbox)
print(result.text, result.confidence)
```

### Validation (`validation/`)

Two-tier validation system:

- **SemanticValidator**: Domain-specific rules (date validity, currency format)
- **ArithmeticChecker**: Mathematical relationships (subtotal + tax = total)

```python
from pdf_to_csv.validation import SemanticValidator, ArithmeticChecker

validator = SemanticValidator()
issues = validator.validate_field('invoice_date', '2024-13-45', 'date')

checker = ArithmeticChecker()
issues = checker.check_all({
    'subtotal': '100.00',
    'tax': '8.00',
    'total': '109.00',  # Should be 108.00!
})
```

### Decision Engine (`decision/`)

Replaces numeric confidence with actionable categories:

```python
from pdf_to_csv.decision import Decision, DecisionEngine

engine = DecisionEngine(
    verified_threshold=0.95,
    likely_threshold=0.80,
    review_threshold=0.60,
)

decision = engine.decide_field(
    field_name='total',
    value='$1,234.56',
    confidence=0.92,
    extraction_method='label_match',
    validation_passed=True,
)

# decision.decision is one of:
# - Decision.VERIFIED: High confidence, auto-process
# - Decision.LIKELY: Good confidence, flag for review
# - Decision.REVIEW_REQUIRED: Needs human verification
# - Decision.REJECTED: Failed validation
```

### Table Processing (`tables/`)

Handles complex table scenarios:

- **TableDetector**: Finds tables using structural and visual cues
- **ContinuationDetector**: Identifies tables that span pages
- **TableStitcher**: Merges multi-page tables into single structures

```python
from pdf_to_csv.tables import TableDetector, TableStitcher

detector = TableDetector()
tables = [detector.detect(page) for page in pages]

stitcher = TableStitcher()
stitched = stitcher.stitch_tables(tables)
```

### Review System (`review/`)

Human-in-the-loop support:

- **ReviewSession**: Tracks review state for a document
- **HTMLPreviewGenerator**: Creates interactive HTML with bounding boxes
- **ReviewExporter**: Exports to JSON/CSV/COCO/Label Studio

```python
from pdf_to_csv.review import HTMLPreviewGenerator, ReviewExporter

generator = HTMLPreviewGenerator()
html = generator.generate(review_session, pdf_path)

exporter = ReviewExporter()
exporter.export_json(review_session, 'review.json')
```

### Performance (`performance/`)

Scale hardening components:

- **StreamingProcessor**: Memory-efficient batch processing
- **WorkerPool**: Thread/process pool with task management
- **RetryPolicy**: Configurable retry with exponential backoff

```python
from pdf_to_csv.performance import StreamingProcessor, WorkerPool

processor = StreamingProcessor(checkpoint_dir='./checkpoints')
for result in processor.process(pdf_paths, process_func):
    handle_result(result)
```

### Document Types (`doctypes/`)

Pluggable document type system:

- **DocumentType**: Definition of expected fields and validation rules
- **DocumentTypeRegistry**: Global registry with auto-detection
- **DocumentExtractor**: Extraction using type definitions
- **builtin_types**: Pre-built types (invoice, bank statement, receipt, resume)

```python
from pdf_to_csv.doctypes import (
    DocumentType, FieldType, create_field,
    register_document_type, detect_document_type,
)

# Create custom type
po_type = DocumentType(
    name='purchase_order',
    fields=[
        create_field('po_number', FieldType.IDENTIFIER, ['PO #']),
        create_field('total', FieldType.CURRENCY, ['Total']),
    ],
)
register_document_type(po_type)

# Auto-detect type from content
doc_type = detect_document_type(document_text)
```

## Pipeline Flow

```
┌──────────────────────────────────────────────────────────────┐
│                         PDF INPUT                             │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      LAYOUT ANALYSIS                          │
│  • Extract text with bounding boxes                          │
│  • Build spatial index (R-tree)                              │
│  • Classify block types (header, body, table, etc.)          │
│  • Create layout graph with relationships                     │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      REGION-BASED OCR                         │
│  • Identify low-confidence regions                           │
│  • Apply adaptive preprocessing                              │
│  • Multi-pass OCR with confidence tracking                   │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    DOCUMENT TYPE DETECTION                    │
│  • Match keywords and patterns                               │
│  • Score against registered types                            │
│  • Select best-matching type or use default                  │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     FIELD EXTRACTION                          │
│  • Label-based: Match labels to adjacent values              │
│  • Pattern-based: Apply regex patterns                       │
│  • Position-based: Use spatial hints                         │
│  • Relation-based: Use spatial relationships                 │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    TABLE PROCESSING                           │
│  • Detect table boundaries                                   │
│  • Identify continuation signals                             │
│  • Stitch multi-page tables                                  │
│  • Extract rows and cells                                    │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      VALIDATION                               │
│  • Semantic validation (dates, currencies, patterns)         │
│  • Arithmetic validation (totals, calculations)              │
│  • Cross-field validation (business rules)                   │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    DECISION ENGINE                            │
│  • Per-field decisions based on confidence + validation      │
│  • Document-level decision aggregation                       │
│  • Categories: VERIFIED | LIKELY | REVIEW_REQUIRED | REJECTED│
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        OUTPUT                                 │
│  • JSON with full provenance                                 │
│  • CSV for tabular data                                      │
│  • HTML preview for human review                             │
│  • ML dataset formats (COCO, YOLO, Label Studio)             │
└──────────────────────────────────────────────────────────────┘
```

## Configuration

### Pipeline Config

```python
from pdf_to_csv import PipelineConfig, ProcessingMode

config = PipelineConfig(
    # Processing mode
    mode=ProcessingMode.STANDARD,  # FAST, STANDARD, THOROUGH, REVIEW
    
    # OCR
    enable_ocr=True,
    ocr_language='eng',
    ocr_dpi=300,
    
    # Layout
    enable_layout_analysis=True,
    
    # Tables
    enable_table_detection=True,
    stitch_multi_page_tables=True,
    
    # Validation
    enable_semantic_validation=True,
    enable_arithmetic_validation=True,
    
    # Decision thresholds
    verified_threshold=0.95,
    likely_threshold=0.80,
    review_threshold=0.60,
    
    # Performance
    max_workers=4,
    batch_size=10,
    enable_streaming=True,
)
```

### Custom Document Type

```python
from pdf_to_csv import (
    DocumentType, FieldType, create_field,
    register_document_type,
)

def validate_po_totals(data: dict) -> list:
    """Custom cross-field validation."""
    errors = []
    # Add validation logic
    return errors

po_type = DocumentType(
    name='purchase_order',
    display_name='Purchase Order',
    description='Corporate purchase order',
    version='1.0',
    
    fields=[
        create_field(
            name='po_number',
            field_type=FieldType.IDENTIFIER,
            labels=['PO #', 'Purchase Order', 'Order Number'],
            patterns=[r'PO[-/]?\d{4,}'],
            required=True,
            group='header',
        ),
        create_field(
            name='vendor',
            field_type=FieldType.TEXT,
            labels=['Vendor', 'Supplier'],
            group='vendor',
        ),
        create_field(
            name='total',
            field_type=FieldType.CURRENCY,
            labels=['Total', 'Grand Total'],
            required=True,
            group='totals',
        ),
        create_field(
            name='line_items',
            field_type=FieldType.TABLE,
            labels=['Item', 'Description'],
            group='items',
        ),
    ],
    
    identification_patterns=[r'purchase\s*order', r'po\s*#'],
    identification_keywords=['purchase order', 'requisition', 'procurement'],
    
    table_fields=['line_items'],
    table_column_mappings={
        'line_items': ['item', 'description', 'qty', 'price', 'total'],
    },
    
    cross_field_rules=[validate_po_totals],
    
    category='procurement',
    tags=['po', 'purchase', 'order'],
)

register_document_type(po_type)
```

## Usage Examples

### Basic Processing

```python
from pdf_to_csv import process_pdf, Decision

result = process_pdf('invoice.pdf')

print(f"Type: {result.document_type}")
print(f"Decision: {result.document_decision.decision.value}")

for name, value in result.fields.items():
    decision = result.field_decisions.get(name)
    print(f"  {name}: {value} [{decision.decision.value}]")
```

### Batch Processing

```python
from pdf_to_csv import DocumentIntelligencePipeline, PipelineConfig

config = PipelineConfig(max_workers=8)
pipeline = DocumentIntelligencePipeline(config)

results = pipeline.process_batch([
    'doc1.pdf', 'doc2.pdf', 'doc3.pdf'
])

# Export to CSV
pipeline.export_results(results, 'results.csv', format='csv')

# Print metrics
metrics = pipeline.get_metrics()
print(f"Processed: {metrics.total_documents}")
print(f"Verified: {metrics.verified_fields}")
```

### ML Dataset Export

```python
from pdf_to_csv import (
    DatasetExporter, ExportFormat, DatasetSplit,
    create_dataset_from_results,
)

# Create annotations from results
annotations = create_dataset_from_results(results, include_boxes=True)

# Export with train/val/test split
exporter = DatasetExporter('./ml_data/')
split = DatasetSplit(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

exporter.export(annotations, ExportFormat.COCO, split)
exporter.export(annotations, ExportFormat.YOLO, split)
exporter.export(annotations, ExportFormat.LABEL_STUDIO, split)
```

### Human Review Workflow

```python
from pdf_to_csv import (
    DocumentIntelligencePipeline,
    PipelineConfig,
    ProcessingMode,
    Decision,
)

# Process with review mode
config = PipelineConfig(mode=ProcessingMode.REVIEW)
pipeline = DocumentIntelligencePipeline(config)

result = pipeline.process_document('invoice.pdf')

# Check if review needed
if result.document_decision.decision == Decision.REVIEW_REQUIRED:
    # Export review files
    pipeline.export_results([result], './review/', format='review')
    print("Review files generated in ./review/")
```

## Dependencies

Required:
- Python 3.9+
- pdfplumber or PyMuPDF (PDF extraction)

Optional:
- pytesseract + Tesseract (OCR)
- opencv-python (image preprocessing)
- rtree (fast spatial indexing)
- pyarrow (Parquet export)

```bash
pip install pdfplumber pytesseract opencv-python rtree pyarrow
```

## Error Handling

The pipeline uses graceful degradation:

1. If R-tree unavailable, falls back to naive spatial search
2. If OCR fails, continues with text layer only
3. If document type unknown, uses generic extraction
4. All failures logged with context for debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logging during processing
result = process_pdf('invoice.pdf')
```

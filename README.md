# PDF to CSV Extractor

A robust, production-grade Python module for extracting structured data from PDF documents and converting them to clean CSV files. Built to handle messy, real-world PDFs, not just clean demo files.

## 🎯 Features

- **Multi-strategy extraction**: Layered approach using text layer, OCR, and table extraction
- **Configurable fields**: Define what to extract via YAML configuration, no code changes needed
- **Robust parsing**: Regex patterns + positional heuristics + keyword proximity
- **Data validation**: Pydantic-based validation with type coercion
- **Normalization**: Consistent date, currency, and text formatting
- **Error handling**: Graceful degradation, detailed logging, never silently fails
- **Line item extraction**: Automatic table detection and parsing
- **Batch processing**: Process entire directories with progress tracking

## 📁 Project Structure

```
pdf_to_csv/
├── extractor/                 # PDF content extraction
│   ├── __init__.py
│   ├── pdf_text.py           # Text layer extraction (pdfplumber/PyMuPDF)
│   ├── ocr.py                # OCR for scanned documents (Tesseract)
│   ├── tables.py             # Table extraction (Camelot/Tabula)
│   └── utils.py              # Shared utilities and data structures
├── parser/                    # Data parsing and validation
│   ├── __init__.py
│   ├── field_mapper.py       # Field extraction from text
│   ├── validators.py         # Value validation
│   └── normalizers.py        # Value normalization
├── output/                    # Output generation
│   ├── __init__.py
│   └── csv_writer.py         # CSV file writing
├── config/
│   └── fields.yaml           # Field configuration
├── main.py                   # CLI and orchestration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🚀 Installation

### 1. Clone and setup environment

```bash
cd pdf_to_csv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (for scanned PDFs)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 4. Install Java (for Tabula table extraction)

Tabula requires Java 8+. Download from https://adoptium.net/ if not installed.

## 📖 Usage

### CLI Interface

```bash
# Process single PDF
python main.py --input invoice.pdf --output output.csv

# Process directory of PDFs
python main.py --input ./pdfs/ --output ./output/data.csv

# With custom configuration
python main.py -i ./pdfs/ -o output.csv -c ./config/fields.yaml

# Disable OCR for faster processing
python main.py -i ./pdfs/ -o output.csv --no-ocr

# Verbose mode with log file
python main.py -i ./pdfs/ -o output.csv -v --log-file extraction.log

# Generate JSON report
python main.py -i ./pdfs/ -o output.csv --json-report report.json
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input PDF file or directory (required) |
| `--output` | `-o` | Output CSV file path (required) |
| `--config` | `-c` | Path to fields.yaml configuration |
| `--ocr/--no-ocr` | | Enable/disable OCR (default: enabled) |
| `--ocr-language` | | Tesseract language code (default: eng) |
| `--verbose` | `-v` | Enable verbose logging |
| `--log-file` | | Write logs to file |
| `--json-report` | | Generate detailed JSON report |

### Programmatic API

```python
from pathlib import Path
from main import PDFExtractor, ExtractionConfig

# Configure extraction
config = ExtractionConfig(
    config_path=Path("config/fields.yaml"),
    enable_ocr=True,
    ocr_language="eng",
    confidence_threshold=0.6
)

# Initialize extractor
extractor = PDFExtractor(config=config)

# Process single PDF
report = extractor.process_pdf(Path("invoice.pdf"))

if report.success:
    print("Extracted fields:")
    for name, value in report.fields_extracted.items():
        confidence = report.confidence_scores.get(name, 0)
        print(f"  {name}: {value} ({confidence:.0%})")
else:
    print("Extraction failed:")
    for error in report.errors:
        print(f"  {error}")

# Process directory
reports = extractor.process_directory(
    Path("./pdfs/"),
    Path("./output/data.csv")
)
```

## ⚙️ Configuration

The `config/fields.yaml` file defines what fields to extract and how to find them.

### Field Definition Structure

```yaml
fields:
  - name: invoice_number          # Internal name (used in CSV)
    display_name: "Invoice Number" # Human-readable name
    type: string                   # string, number, currency, date, text_block
    required: true                 # Flag extraction failure if missing
    patterns:                      # Regex patterns to try (in order)
      - '(?i)invoice\s*#?[:\s]*([A-Z0-9-]+)'
      - '(?i)inv[:\s]*([A-Z0-9-]+)'
    keywords:                      # Context keywords for fallback extraction
      - "invoice"
      - "inv"
    multiline: false              # Can span multiple lines
    validation:                   # Validation rules
      min_length: 3
      max_length: 30
```

### Adding New Fields

1. Add field definition to `config/fields.yaml`:

```yaml
fields:
  - name: vendor_email
    display_name: "Vendor Email"
    type: string
    required: false
    patterns:
      - '(?i)email[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    keywords:
      - "email"
      - "e-mail"
    validation:
      pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
```

2. No code changes needed - the field will be automatically extracted.

### Settings

```yaml
settings:
  confidence_threshold: 0.6    # Minimum extraction confidence
  auto_ocr_fallback: true      # Auto-trigger OCR for scanned pages
  ocr_text_threshold: 50       # Chars per page to trigger OCR
  
  date_formats:                # Date formats to recognize
    - "%Y-%m-%d"
    - "%d/%m/%Y"
    - "%m/%d/%Y"
    
  currency_symbols:            # Currency symbols to recognize
    - "$"
    - "€"
    - "£"
```

### Noise Patterns

Remove repeated headers, footers, and watermarks:

```yaml
noise_patterns:
  - '(?i)page\s*\d+\s*(?:of\s*\d+)?'
  - '(?i)confidential'
  - '(?i)www\.[a-z]+\.[a-z]+'
```

## 🔧 How It Works

### Extraction Pipeline

```
PDF File
    │
    ▼
┌──────────────────────────────────────┐
│           EXTRACTION LAYER           │
│                                      │
│  1. Text Layer (pdfplumber/PyMuPDF) │
│     - Fast, high accuracy            │
│     - Works for digital PDFs         │
│                                      │
│  2. Page Analysis                    │
│     - Detect scanned pages           │
│     - Identify tables                │
│                                      │
│  3. OCR (Tesseract) - if needed     │
│     - Image preprocessing            │
│     - Deskew, denoise, binarize     │
│                                      │
│  4. Table Extraction (Camelot)      │
│     - Bordered tables                │
│     - Borderless tables (stream)     │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│            PARSING LAYER             │
│                                      │
│  1. Noise Removal                   │
│     - Headers, footers, watermarks   │
│                                      │
│  2. Field Mapping                   │
│     - Regex pattern matching         │
│     - Keyword proximity fallback     │
│                                      │
│  3. Validation                      │
│     - Type checking                  │
│     - Format validation              │
│     - Cross-field validation         │
│                                      │
│  4. Normalization                   │
│     - Date → ISO format              │
│     - Currency → decimals            │
│     - Text → cleaned/trimmed         │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│            OUTPUT LAYER              │
│                                      │
│  - Clean snake_case columns          │
│  - Consistent data types             │
│  - UTF-8 with BOM (Excel friendly)  │
│  - Proper NULL handling              │
└──────────────────────────────────────┘
    │
    ▼
CSV File
```

### When OCR is Triggered

OCR is automatically triggered when:
- Page has less than 50 characters of extractable text
- Page has images covering >50% of the area
- Page appears to be a scan (large image, little text)

OCR can be disabled with `--no-ocr` for faster processing of digital PDFs.

### Confidence Scoring

Each extracted field gets a confidence score (0.0 - 1.0):

| Score | Meaning |
|-------|---------|
| 0.8-1.0 | High confidence - text layer extraction with strong pattern match |
| 0.6-0.8 | Medium confidence - may need review |
| 0.4-0.6 | Low confidence - OCR or weak pattern match |
| <0.4 | Very low - likely incorrect |

Confidence factors:
- Extraction method (text > table > OCR)
- Pattern match quality
- Character/word validity
- Type-specific validation

## ⚠️ Known Limitations

### PDF Types
- **Best for**: Invoices, receipts, forms with consistent layouts
- **Challenging**: Multi-column documents, complex nested tables
- **Not designed for**: Free-form documents, contracts with flowing text

### OCR Limitations
- Accuracy depends on scan quality (recommend 300+ DPI)
- Handwritten text is not supported
- Very small fonts (<8pt) may not extract well
- Skewed scans (>10°) may have issues

### Table Extraction
- Merged cells often break extraction
- Tables spanning multiple pages are not automatically joined
- Very wide tables may lose columns

### Performance
- Processing speed: ~1-5 seconds per page (text layer)
- OCR: ~5-30 seconds per page (depending on DPI)
- Memory: ~50-200MB per PDF (varies with page count)

## 📊 Sample Output

### Input PDF
```
INVOICE

Invoice #: INV-2024-001234
Date: January 15, 2024

Bill To:
Acme Corporation
123 Business St
New York, NY 10001

Description          Qty    Price    Amount
Widget A             10     $25.00   $250.00
Widget B             5      $50.00   $250.00
Service Fee          1      $100.00  $100.00

Subtotal:                           $600.00
Tax (8%):                           $48.00
Total Due:                          $648.00
```

### Output CSV
```csv
invoice_number,invoice_date,customer_name,customer_address,subtotal,tax_amount,grand_total,_source_file
INV-2024-001234,2024-01-15,Acme Corporation,"123 Business St, New York, NY 10001",600.00,48.00,648.00,invoice.pdf
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=pdf_to_csv --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built with these excellent libraries:
- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF text extraction
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - Fast PDF processing
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [Camelot](https://github.com/camelot-dev/camelot) - Table extraction
- [pandas](https://pandas.pydata.org/) - Data manipulation
- [Pydantic](https://pydantic.dev/) - Data validation
- [Rich](https://github.com/Textualize/rich) - Terminal formatting

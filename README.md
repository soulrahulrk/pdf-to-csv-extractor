# PDF to CSV Extractor

A Python tool for extracting structured data from PDF documents and exporting to CSV format. Features text extraction, OCR for scanned documents, table detection, and configurable field mapping.

## ✨ Features

- **Text Extraction**: Uses pdfplumber for reliable text extraction from PDFs
- **OCR Support**: Tesseract OCR for scanned PDFs with automatic detection
- **Table Extraction**: Camelot and Tabula support for extracting tabular data
- **Smart Field Mapping**: Regex patterns with keyword proximity fallback
- **Configurable Fields**: YAML-based configuration - no code changes needed
- **Validation Pipeline**: Type validation, format checking, custom rules
- **Web UI**: Streamlit-based web interface for easy PDF processing
- **CLI Support**: Command-line interface for batch processing

## 📁 Project Structure

```
pdf_to_csv/
├── app.py                    # Streamlit web UI
├── main.py                   # CLI entry point
├── config/
│   └── fields.yaml           # Field extraction configuration
├── extractor/                # PDF extraction modules
│   ├── pdf_text.py           # Text extraction from PDFs
│   ├── ocr.py                # OCR processing for scanned docs
│   ├── tables.py             # Table extraction
│   └── utils.py              # Utility functions
├── parser/                   # Data parsing modules
│   ├── field_mapper.py       # Field mapping logic
│   ├── validators.py         # Data validation
│   └── normalizers.py        # Data normalization
├── output/                   # Output generation
│   └── csv_writer.py         # CSV file writing
├── tests/                    # Test suite
│   └── test_extraction.py    # Extraction tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 📋 Requirements

- Python 3.9+
- Tesseract OCR (optional, for scanned PDFs)
- Ghostscript (optional, for Camelot table extraction)
- Java (optional, for Tabula table extraction)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/rahul-singh011/pdftodocs.git
cd pdftodocs/pdf_to_csv

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: Install Tesseract OCR (for scanned PDFs)

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

## 🖥️ Web UI (Recommended)

The easiest way to use this tool is through the Streamlit web interface:

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser:

1. Upload a PDF file using the file uploader
2. View extracted text and fields in the interface
3. Download the extracted data as CSV

## ⌨️ Command Line Usage

```bash
# Process single PDF
python main.py -i invoice.pdf -o output.csv

# Process directory of PDFs  
python main.py -i ./pdfs/ -o output.csv

# With OCR enabled
python main.py -i scanned.pdf -o output.csv --ocr

# Custom configuration
python main.py -i invoice.pdf -o output.csv -c custom_fields.yaml

# Verbose logging
python main.py -i invoice.pdf -o output.csv -v
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

## ⚙️ Configuration

The `config/fields.yaml` file defines what fields to extract and how to find them.

### Field Definition Example

```yaml
fields:
  - name: invoice_number
    display_name: "Invoice Number"
    type: string
    required: true
    patterns:
      - '(?i)invoice\s*#?[:\s]*([A-Z0-9-]+)'
      - '(?i)inv[:\s]*([A-Z0-9-]+)'
    keywords:
      - "invoice"
      - "inv"
    validation:
      min_length: 3
      max_length: 30
```

### Field Types

- `string`: General text values
- `number`: Numeric values
- `currency`: Monetary amounts with symbols
- `date`: Date values (auto-normalized to ISO format)
- `text_block`: Multi-line text blocks

### Adding New Fields

1. Edit `config/fields.yaml`
2. Add a new field definition with name, patterns, and validation rules
3. No code changes needed - the field will be automatically extracted

## 🔧 How It Works

```
PDF File
    │
    ▼
┌───────────────────────────────┐
│      EXTRACTION LAYER         │
│                               │
│  • Text Layer (pdfplumber)    │
│  • OCR (Tesseract) if needed  │
│  • Table Extraction (Camelot) │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│       PARSING LAYER           │
│                               │
│  • Regex pattern matching     │
│  • Keyword proximity fallback │
│  • Type validation            │
│  • Value normalization        │
└───────────────────────────────┘
    │
    ▼
┌───────────────────────────────┐
│       OUTPUT LAYER            │
│                               │
│  • Clean CSV export           │
│  • UTF-8 encoding             │
│  • Web UI display             │
└───────────────────────────────┘
    │
    ▼
CSV File / Web Display
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html
```

## 📦 Dependencies

### Core
- `pdfplumber` - PDF text extraction
- `pytesseract` - OCR wrapper for Tesseract
- `Pillow` - Image processing
- `pandas` - Data manipulation

### Table Extraction (Optional)
- `camelot-py` - Table extraction from PDFs
- `tabula-py` - Alternative table extraction

### Web & CLI
- `streamlit` - Web UI framework
- `click` - CLI framework
- `loguru` - Logging
- `rich` - Terminal formatting

### Configuration & Validation
- `PyYAML` - YAML parsing
- `python-dateutil` - Date parsing

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

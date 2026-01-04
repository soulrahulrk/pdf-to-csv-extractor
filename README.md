# PDF to CSV - Generic Content Extractor

A **content-only** PDF extraction tool. No business logic, no field assumptions - just pure text extraction.

## 🎯 What This Is

This tool extracts **all readable content** from any PDF and outputs it to a neutral CSV format. 

**This is NOT:**
- An invoice extractor
- A form parser
- A document intelligence system

**This IS:**
- A content dumper
- A text extraction tool
- A PDF-to-structured-data converter

## 📊 Output Format

| Column | Description |
|--------|-------------|
| `source_file` | PDF filename |
| `page_number` | 1-based page number |
| `block_type` | `paragraph` \| `line` \| `table` \| `ocr_text` \| `empty` |
| `block_index` | 0-based index within the page |
| `content` | The actual text content |

### Example Output

```csv
source_file,page_number,block_type,block_index,content
resume.pdf,1,paragraph,0,"John Doe
Software Engineer"
resume.pdf,1,line,1,Contact: john@email.com
resume.pdf,1,table,2,"Skill | Years
Python | 5
JavaScript | 3"
```

## 📁 Project Structure

```
pdf_to_csv/
├── extract/                  # Content extraction
│   ├── text_blocks.py        # Text block grouping
│   ├── tables.py             # Table extraction
│   └── ocr.py                # OCR for scanned pages
├── output/
│   └── generic_csv_writer.py # CSV output
├── app.py                    # Streamlit Web UI
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/rahul-singh011/pdftodocs.git
cd pdftodocs/pdf_to_csv

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

## 🖥️ Web UI (Recommended)

The easiest way to use this tool:

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser:

1. **Upload** any PDF file
2. **View** extracted content with stats and preview
3. **Download** as CSV

Features:
- Drag & drop PDF upload
- Toggle OCR on/off
- Block type breakdown chart
- Content preview table
- One-click CSV download

## ⌨️ CLI Usage

```bash
# Single PDF
python main.py -i document.pdf -o content.csv

# Directory of PDFs
python main.py -i ./pdfs/ -o all_content.csv

# With verbose output
python main.py -i document.pdf -o content.csv -v

# Disable OCR (faster, text-only PDFs)
python main.py -i document.pdf -o content.csv --no-ocr

# Custom OCR threshold
python main.py -i scan.pdf -o content.csv --ocr-threshold 100
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Input PDF file or directory (required) |
| `-o, --output` | Output CSV file path (required) |
| `--ocr-threshold` | Char count below which OCR triggers (default: 50) |
| `--no-ocr` | Disable OCR completely |
| `-v, --verbose` | Print progress messages |

## 🔧 How It Works

```
PDF File
    │
    ▼
┌─────────────────────────────┐
│     CHARACTER CHECK         │
│                             │
│  chars < threshold?         │
│  YES → OCR path             │
│  NO  → Text extraction      │
└─────────────────────────────┘
    │
    ├── OCR Path ──────────────┐
    │   • Render page to image │
    │   • Tesseract OCR        │
    │   • block_type: ocr_text │
    │                          │
    └── Text Path ─────────────┤
        │                      │
        ├─ Tables              │
        │  • pdfplumber.find_tables()
        │  • Flatten to pipe-separated
        │  • block_type: table │
        │                      │
        └─ Text Blocks         │
           • Extract words     │
           • Group into lines  │
           • Group into blocks │
           • block_type: paragraph/line
                               │
    ┌──────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│        CSV OUTPUT           │
│                             │
│  source_file                │
│  page_number                │
│  block_type                 │
│  block_index                │
│  content                    │
└─────────────────────────────┘
```

## 📝 Block Types

| Type | Description |
|------|-------------|
| `paragraph` | Multi-line text block |
| `line` | Single line of text |
| `table` | Table data (pipe-separated columns) |
| `ocr_text` | Text extracted via OCR |
| `empty` | Empty page or extraction failed |

## ⚠️ Design Principles

### ✅ What This Tool Does

- Extracts **all** readable text
- Preserves **reading order** (top → bottom)
- Groups words into **logical blocks**
- Flattens tables into **pipe-separated** format
- Falls back to **OCR** for scanned pages
- **Never fails** - always outputs CSV

### ❌ What This Tool Does NOT Do

- Look for invoice fields
- Parse dates or currency
- Validate content
- Apply regex for business keywords
- Assume any meaning from content

## 📦 Dependencies

```
pdfplumber>=0.10.0     # Text & table extraction
pytesseract>=0.3.10    # OCR (optional)
Pillow>=10.0.0         # Image processing for OCR
```

### Optional: Tesseract OCR

For scanned PDFs, install Tesseract:

**Windows:** https://github.com/UB-Mannheim/tesseract/wiki

**macOS:** `brew install tesseract`

**Linux:** `sudo apt install tesseract-ocr`

## 🧪 Failure Policy

This tool is designed to **never fail completely**:

| Scenario | Behavior |
|----------|----------|
| Empty page | Outputs row with `block_type=empty` |
| Page extraction error | Outputs row with error message in content |
| PDF open failure | Outputs row with failure message |
| OCR unavailable | Outputs row noting OCR unavailable |

**Goal: Always output a CSV, even if content extraction fails.**

## 📄 License

MIT License

## 🎯 Philosophy

> **Accuracy > Intelligence**
> **Completeness > Correctness**

This is a **content extraction tool**, not an intelligence system.

"""
Streamlit UI for PDF to CSV Extractor

A user-friendly web interface for extracting structured data from PDF documents.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import json
import io
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extractor import (
    PDFTextExtractor,
    OCRExtractor,
    TableExtractor,
    remove_noise,
    merge_extraction_results,
    get_pdf_info,
    check_tesseract_installed,
)
from parser import (
    FieldMapper,
    ConfigLoader,
    FieldValidator,
    InvoiceFieldNormalizer,
)
from output import CSVWriter, CSVConfig


# Page configuration
st.set_page_config(
    page_title="PDF to CSV Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load field configuration."""
    config_path = Path(__file__).parent / "config" / "fields.yaml"
    loader = ConfigLoader()
    if config_path.exists():
        loader.load(config_path)
    return loader


@st.cache_resource
def init_extractors(enable_ocr: bool, ocr_language: str):
    """Initialize extraction components."""
    text_extractor = PDFTextExtractor()
    
    ocr_extractor = None
    if enable_ocr and check_tesseract_installed():
        try:
            ocr_extractor = OCRExtractor(language=ocr_language)
        except Exception:
            pass
    
    table_extractor = None
    try:
        table_extractor = TableExtractor()
    except Exception:
        pass
    
    return text_extractor, ocr_extractor, table_extractor


def process_pdf(
    pdf_file,
    config_loader: ConfigLoader,
    text_extractor: PDFTextExtractor,
    ocr_extractor,
    table_extractor,
    enable_ocr: bool
) -> dict:
    """Process a single PDF file and extract fields."""
    
    result = {
        'success': False,
        'fields': {},
        'confidence': {},
        'line_items': [],
        'warnings': [],
        'errors': [],
        'raw_text': '',
        'method': 'unknown'
    }
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = Path(tmp.name)
    
    try:
        # Validate PDF
        pdf_info = get_pdf_info(tmp_path)
        if not pdf_info.get('readable', False):
            result['errors'].append("Cannot read PDF file. It may be corrupted or password-protected.")
            return result
        
        # Extract text
        text_result = text_extractor.extract_text(tmp_path)
        result['method'] = text_result.method.value
        result['warnings'].extend(text_result.warnings)
        
        # Check if OCR is needed
        if enable_ocr and ocr_extractor:
            page_analyses = text_extractor.analyze_pages(tmp_path)
            ocr_pages = [a.page_number for a in page_analyses if a.needs_ocr(50)]
            
            if ocr_pages:
                result['warnings'].append(f"OCR applied to {len(ocr_pages)} page(s)")
                ocr_result = ocr_extractor.extract_from_pdf(tmp_path, pages=ocr_pages)
                text_result = merge_extraction_results([text_result, ocr_result])
                result['method'] = 'hybrid'
        
        if not text_result.text.strip():
            result['errors'].append("No text could be extracted from the PDF")
            return result
        
        # Store raw text
        result['raw_text'] = text_result.text[:5000]  # Limit for display
        
        # Remove noise
        cleaned_text = remove_noise(text_result.text, config_loader.noise_patterns)
        
        # Extract fields
        field_mapper = FieldMapper(config=config_loader.config)
        extracted_fields = field_mapper.extract_fields(cleaned_text)
        
        # Validate and normalize
        validator = FieldValidator(date_formats=config_loader.get_date_formats())
        normalizer = InvoiceFieldNormalizer(date_formats=config_loader.get_date_formats())
        
        for field_result in extracted_fields:
            if not field_result.is_valid:
                continue
            
            field_def = config_loader.get_field(field_result.name)
            field_type = field_def.field_type if field_def else 'string'
            
            validation = validator.validate_field(
                field_result.name,
                field_result.value,
                field_type
            )
            
            if validation.is_valid:
                normalized = normalizer.normalize_field(
                    validation.validated_value,
                    field_type,
                    field_result.name
                )
                result['fields'][field_result.name] = normalized
                result['confidence'][field_result.name] = round(
                    field_result.confidence + validation.confidence_adjustment, 3
                )
            else:
                result['warnings'].extend(
                    [f"{field_result.name}: {e}" for e in validation.errors]
                )
        
        # Extract line items
        if table_extractor:
            try:
                line_items_config = config_loader.line_items_config
                if line_items_config.get('enabled', False):
                    column_mappings = line_items_config.get('column_mappings', {})
                    if column_mappings:
                        items = table_extractor.extract_line_items(tmp_path, column_mappings)
                        result['line_items'] = items
            except Exception as e:
                result['warnings'].append(f"Table extraction failed: {str(e)}")
        
        # Cross-validation
        cross_warnings = validator.cross_validate(result['fields'])
        result['warnings'].extend(cross_warnings)
        
        result['success'] = len(result['fields']) > 0
        
    except Exception as e:
        result['errors'].append(f"Processing error: {str(e)}")
    
    finally:
        # Cleanup temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass
    
    return result


def results_to_csv(results: list[dict], config_loader: ConfigLoader) -> str:
    """Convert extraction results to CSV string."""
    records = []
    columns = [f.name for f in config_loader.fields]
    field_types = {f.name: f.field_type for f in config_loader.fields}
    
    for res in results:
        if res['success']:
            record = res['fields'].copy()
            record['_source_file'] = res.get('filename', 'unknown')
            record['_extraction_confidence'] = sum(
                res['confidence'].values()
            ) / max(len(res['confidence']), 1)
            records.append(record)
    
    if not records:
        return ""
    
    writer = CSVWriter(
        columns=columns + ['_source_file', '_extraction_confidence'],
        field_types=field_types
    )
    
    return writer.to_string(records)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<p class="main-header">📄 PDF to CSV Extractor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Extract structured data from invoices, receipts, and forms</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        enable_ocr = st.checkbox(
            "Enable OCR",
            value=True,
            help="Enable OCR for scanned documents (requires Tesseract)"
        )
        
        ocr_language = st.selectbox(
            "OCR Language",
            options=['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'nld'],
            index=0,
            help="Select language for OCR processing"
        )
        
        show_raw_text = st.checkbox(
            "Show Raw Text",
            value=False,
            help="Display extracted raw text"
        )
        
        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True,
            help="Display extraction confidence for each field"
        )
        
        st.divider()
        
        # Status indicators
        st.subheader("📊 System Status")
        
        # Check Tesseract
        tesseract_ok = check_tesseract_installed()
        if tesseract_ok:
            st.success("✅ Tesseract OCR installed")
        else:
            st.warning("⚠️ Tesseract not found - OCR disabled")
        
        # Check config
        config_loader = load_config()
        if config_loader.fields:
            st.success(f"✅ {len(config_loader.fields)} fields configured")
        else:
            st.error("❌ No field configuration found")
        
        st.divider()
        
        st.markdown("""
        ### 📖 How to Use
        1. Upload one or more PDF files
        2. Wait for extraction to complete
        3. Review extracted data
        4. Download as CSV
        
        ### 💡 Tips
        - Works best with invoices & forms
        - Enable OCR for scanned documents
        - Check confidence scores for accuracy
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload PDFs")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to extract data from"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            
            for f in uploaded_files:
                st.text(f"  • {f.name} ({f.size / 1024:.1f} KB)")
    
    with col2:
        st.subheader("🎯 Configured Fields")
        
        if config_loader.fields:
            field_df = pd.DataFrame([
                {
                    'Field': f.display_name,
                    'Type': f.field_type,
                    'Required': '✓' if f.required else ''
                }
                for f in config_loader.fields
            ])
            st.dataframe(field_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No fields configured. Add fields to config/fields.yaml")
    
    # Process button
    if uploaded_files:
        st.divider()
        
        if st.button("🚀 Extract Data", type="primary", use_container_width=True):
            
            # Initialize extractors
            text_extractor, ocr_extractor, table_extractor = init_extractors(
                enable_ocr, ocr_language
            )
            
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, pdf_file in enumerate(uploaded_files):
                status_text.text(f"Processing {pdf_file.name}...")
                
                result = process_pdf(
                    pdf_file,
                    config_loader,
                    text_extractor,
                    ocr_extractor,
                    table_extractor,
                    enable_ocr
                )
                result['filename'] = pdf_file.name
                results.append(result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            
            # Store results in session state
            st.session_state['results'] = results
            st.session_state['config_loader'] = config_loader
    
    # Display results
    if 'results' in st.session_state:
        results = st.session_state['results']
        config_loader = st.session_state['config_loader']
        
        st.divider()
        st.subheader("📊 Extraction Results")
        
        # Summary metrics
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", total)
        with col2:
            st.metric("Successful", successful, delta=None)
        with col3:
            st.metric("Failed", total - successful, delta=None)
        with col4:
            avg_conf = 0
            conf_count = 0
            for r in results:
                if r['confidence']:
                    avg_conf += sum(r['confidence'].values())
                    conf_count += len(r['confidence'])
            if conf_count > 0:
                avg_conf = avg_conf / conf_count
            st.metric("Avg Confidence", f"{avg_conf:.0%}")
        
        # Tabs for each result
        tabs = st.tabs([f"📄 {r['filename']}" for r in results])
        
        for tab, result in zip(tabs, results):
            with tab:
                if result['success']:
                    st.success(f"✅ Extraction successful ({result['method']})")
                    
                    # Extracted fields
                    st.markdown("#### 📋 Extracted Fields")
                    
                    field_data = []
                    for name, value in result['fields'].items():
                        field_def = config_loader.get_field(name)
                        display_name = field_def.display_name if field_def else name
                        confidence = result['confidence'].get(name, 0)
                        
                        row = {
                            'Field': display_name,
                            'Value': str(value) if value else '',
                        }
                        
                        if show_confidence:
                            row['Confidence'] = f"{confidence:.0%}"
                        
                        field_data.append(row)
                    
                    if field_data:
                        st.dataframe(
                            pd.DataFrame(field_data),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    # Line items
                    if result['line_items']:
                        st.markdown("#### 📦 Line Items")
                        st.dataframe(
                            pd.DataFrame(result['line_items']),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    # Warnings
                    if result['warnings']:
                        with st.expander(f"⚠️ Warnings ({len(result['warnings'])})"):
                            for w in result['warnings']:
                                st.warning(w)
                    
                    # Raw text
                    if show_raw_text and result['raw_text']:
                        with st.expander("📝 Raw Extracted Text"):
                            st.text(result['raw_text'])
                
                else:
                    st.error("❌ Extraction failed")
                    for error in result['errors']:
                        st.error(error)
        
        # Download section
        st.divider()
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = results_to_csv(results, config_loader)
            if csv_data:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data to download")
        
        with col2:
            # JSON report download
            report = {
                'timestamp': datetime.now().isoformat(),
                'statistics': {
                    'total_documents': total,
                    'successful': successful,
                    'failed': total - successful,
                },
                'documents': [
                    {
                        'filename': r['filename'],
                        'success': r['success'],
                        'fields': r['fields'],
                        'confidence': r['confidence'],
                        'line_items': r['line_items'],
                        'warnings': r['warnings'],
                        'errors': r['errors'],
                        'method': r['method']
                    }
                    for r in results
                ]
            }
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(report, indent=2, default=str),
                file_name=f"extraction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Clear results button
        if st.button("🗑️ Clear Results"):
            del st.session_state['results']
            del st.session_state['config_loader']
            st.rerun()


if __name__ == "__main__":
    main()

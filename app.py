"""
Streamlit Web UI for PDF Content Extractor
Simple interface to upload PDFs and download extracted CSV.
"""

import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd

from extract.text_blocks import extract_text_blocks, get_page_char_count
from extract.tables import extract_tables
from extract.ocr import extract_ocr_text, should_ocr, is_ocr_available
from output.generic_csv_writer import ContentRow

# Page config
st.set_page_config(
    page_title="PDF to CSV Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF to CSV - Content Extractor")
st.markdown("Upload any PDF and extract all content to CSV. **No business logic, pure extraction.**")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
ocr_enabled = st.sidebar.checkbox("Enable OCR", value=True, help="Use OCR for scanned pages")
ocr_threshold = st.sidebar.slider("OCR Threshold", 10, 200, 50, help="Trigger OCR if page has fewer characters")

if ocr_enabled and not is_ocr_available():
    st.sidebar.warning("⚠️ pytesseract not installed. OCR disabled.")
    ocr_enabled = False

# File uploader
uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    
    try:
        import pdfplumber
        
        with st.spinner("Extracting content..."):
            rows = []
            source_file = uploaded_file.name
            
            with pdfplumber.open(tmp_path) as pdf:
                total_pages = len(pdf.pages)
                progress = st.progress(0)
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    block_index = 0
                    
                    # Check if OCR needed
                    char_count = get_page_char_count(page)
                    use_ocr = ocr_enabled and should_ocr(char_count, ocr_threshold)
                    
                    if use_ocr:
                        ocr_blocks = extract_ocr_text(page, page_num)
                        for block in ocr_blocks:
                            rows.append({
                                'source_file': source_file,
                                'page_number': page_num,
                                'block_type': block.block_type,
                                'block_index': block_index,
                                'content': block.content
                            })
                            block_index += 1
                    else:
                        # Extract tables
                        tables = extract_tables(page, page_num)
                        for table in tables:
                            rows.append({
                                'source_file': source_file,
                                'page_number': page_num,
                                'block_type': table.block_type,
                                'block_index': block_index,
                                'content': table.content
                            })
                            block_index += 1
                        
                        # Extract text blocks
                        text_blocks = extract_text_blocks(page, page_num)
                        for block in text_blocks:
                            if block.block_type == 'empty' and (tables or any(b.block_type != 'empty' for b in text_blocks)):
                                continue
                            rows.append({
                                'source_file': source_file,
                                'page_number': page_num,
                                'block_type': block.block_type,
                                'block_index': block_index,
                                'content': block.content
                            })
                            block_index += 1
                    
                    # Empty page fallback
                    if block_index == 0:
                        rows.append({
                            'source_file': source_file,
                            'page_number': page_num,
                            'block_type': 'empty',
                            'block_index': 0,
                            'content': ''
                        })
                    
                    progress.progress((i + 1) / total_pages)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Display results
        st.success(f"✅ Extracted {len(rows)} blocks from {total_pages} pages")
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Blocks", len(rows))
        with col2:
            st.metric("Pages", total_pages)
        with col3:
            block_types = df['block_type'].value_counts().to_dict()
            st.metric("Block Types", len(block_types))
        
        # Block type breakdown
        st.subheader("📊 Block Types")
        st.bar_chart(df['block_type'].value_counts())
        
        # Data preview
        st.subheader("📋 Extracted Content")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=f"{Path(uploaded_file.name).stem}_content.csv",
            mime="text/csv"
        )
        
        # Raw content preview
        with st.expander("🔍 View Raw Content"):
            for idx, row in df.iterrows():
                st.markdown(f"**Page {row['page_number']} | {row['block_type']} | Block {row['block_index']}**")
                st.text(row['content'][:500] + "..." if len(row['content']) > 500 else row['content'])
                st.divider()
                
    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
    
    finally:
        # Cleanup temp file
        tmp_path.unlink(missing_ok=True)

else:
    # Instructions when no file uploaded
    st.info("👆 Upload a PDF file to extract content")
    
    st.markdown("""
    ### Output Format
    
    | Column | Description |
    |--------|-------------|
    | `source_file` | PDF filename |
    | `page_number` | Page number (1-based) |
    | `block_type` | paragraph, line, table, ocr_text, empty |
    | `block_index` | Block index within page |
    | `content` | Extracted text |
    
    ### Block Types
    - **paragraph**: Multi-line text block
    - **line**: Single line of text  
    - **table**: Table data (pipe-separated)
    - **ocr_text**: Text from OCR (scanned pages)
    - **empty**: Empty page
    """)

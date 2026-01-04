"""
HTML Preview Generator

Generates interactive HTML previews for document review.
Includes PDF rendering, field highlighting, and correction forms.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from .review_data import ReviewItem, ReviewField, ReviewStatus

logger = logging.getLogger(__name__)


@dataclass
class PreviewConfig:
    """Configuration for HTML preview generation."""
    
    # Display settings
    page_width: int = 800
    highlight_color: str = '#ffeb3b'  # Yellow
    verified_color: str = '#4caf50'   # Green
    warning_color: str = '#ff9800'    # Orange
    error_color: str = '#f44336'      # Red
    
    # Features
    show_confidence: bool = True
    show_suggestions: bool = True
    show_bounding_boxes: bool = True
    enable_corrections: bool = True
    enable_keyboard_nav: bool = True
    
    # Rendering
    include_thumbnails: bool = True
    thumbnail_width: int = 200
    embed_pdf: bool = False  # Whether to embed PDF in HTML
    
    # Output
    output_dir: str = './review_output'
    template_dir: Optional[str] = None


class HTMLPreviewGenerator:
    """
    Generates interactive HTML previews for document review.
    
    Features:
    - PDF page rendering with highlighted regions
    - Field list with confidence indicators
    - Inline correction forms
    - Keyboard navigation
    - Export corrected data
    
    Usage:
        generator = HTMLPreviewGenerator()
        
        # Generate preview for single item
        html_path = generator.generate(review_item)
        
        # Generate batch preview
        index_path = generator.generate_batch(items)
    """
    
    def __init__(self, config: Optional[PreviewConfig] = None):
        """
        Initialize generator.
        
        Args:
            config: Preview configuration
        """
        self.config = config or PreviewConfig()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def generate(
        self,
        item: ReviewItem,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate HTML preview for a review item.
        
        Args:
            item: ReviewItem to generate preview for
            output_path: Optional output path (default: auto-generated)
            
        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"review_{item.item_id}.html"
            )
        
        html_content = self._generate_html(item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated review preview: {output_path}")
        return output_path
    
    def generate_batch(
        self,
        items: List[ReviewItem],
        index_name: str = 'index.html',
    ) -> str:
        """
        Generate previews for multiple items with an index page.
        
        Args:
            items: List of ReviewItems
            index_name: Name of index file
            
        Returns:
            Path to index HTML file
        """
        # Generate individual previews
        item_paths = []
        for item in items:
            path = self.generate(item)
            item_paths.append((item, path))
        
        # Generate index page
        index_path = os.path.join(self.config.output_dir, index_name)
        index_html = self._generate_index(item_paths)
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_html)
        
        logger.info(f"Generated batch review index: {index_path}")
        return index_path
    
    def _generate_html(self, item: ReviewItem) -> str:
        """Generate HTML content for a review item."""
        # Build field rows
        field_rows = []
        for field in item.fields.values():
            field_rows.append(self._generate_field_row(field))
        
        # Build bounding box data for JavaScript
        bbox_data = {}
        for name, field in item.fields.items():
            if field.bbox:
                bbox_data[name] = field.bbox.to_dict()
        
        # Status badge
        status_badge = self._get_status_badge(item.overall_status)
        
        # Progress bar
        progress_pct = int(item.review_progress * 100)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review: {html.escape(item.document_name)}</title>
    <style>
        {self._get_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Document Review</h1>
            <div class="doc-info">
                <span class="doc-name">{html.escape(item.document_name)}</span>
                {status_badge}
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress_pct}%"></div>
                <span class="progress-text">{progress_pct}% Complete</span>
            </div>
        </header>
        
        <main class="main">
            <div class="pdf-panel">
                <div class="pdf-header">
                    <h2>Document Preview</h2>
                    <div class="page-nav">
                        <button id="prevPage" class="btn">◀ Prev</button>
                        <span id="pageNum">Page 1 of {item.page_count}</span>
                        <button id="nextPage" class="btn">Next ▶</button>
                    </div>
                </div>
                <div id="pdfContainer" class="pdf-container">
                    <div class="pdf-placeholder">
                        <p>PDF Preview</p>
                        <p class="small">Document: {html.escape(item.document_path)}</p>
                    </div>
                    <svg id="highlightLayer" class="highlight-layer"></svg>
                </div>
            </div>
            
            <div class="fields-panel">
                <h2>Extracted Fields</h2>
                <div class="field-filters">
                    <button class="filter-btn active" data-filter="all">All</button>
                    <button class="filter-btn" data-filter="pending">Pending</button>
                    <button class="filter-btn" data-filter="corrected">Corrected</button>
                </div>
                <div class="fields-list">
                    {''.join(field_rows)}
                </div>
            </div>
        </main>
        
        <footer class="footer">
            <div class="actions">
                <button id="saveBtn" class="btn btn-primary">Save Changes</button>
                <button id="exportBtn" class="btn">Export JSON</button>
                <button id="completeBtn" class="btn btn-success">Mark Complete</button>
            </div>
            <div class="stats">
                <span>Fields: {len(item.fields)}</span>
                <span>Corrected: {len(item.corrected_fields)}</span>
                <span>Pending: {len(item.pending_fields)}</span>
            </div>
        </footer>
    </div>
    
    <script>
        const itemData = {json.dumps(item.to_dict())};
        const bboxData = {json.dumps(bbox_data)};
        
        {self._get_javascript()}
    </script>
</body>
</html>'''
        
        return html
    
    def _generate_field_row(self, field: ReviewField) -> str:
        """Generate HTML for a single field row."""
        confidence_class = self._get_confidence_class(field.confidence)
        status_class = field.status.name.lower()
        
        # Confidence indicator
        confidence_html = ''
        if self.config.show_confidence:
            confidence_pct = int(field.confidence * 100)
            confidence_html = f'''
                <div class="confidence {confidence_class}">
                    <div class="confidence-bar" style="width: {confidence_pct}%"></div>
                    <span>{confidence_pct}%</span>
                </div>
            '''
        
        # Suggestions
        suggestions_html = ''
        if self.config.show_suggestions and field.suggestions:
            suggestions = ''.join(f'<li>{html.escape(s)}</li>' for s in field.suggestions)
            suggestions_html = f'<ul class="suggestions">{suggestions}</ul>'
        
        # Correction form
        correction_html = ''
        if self.config.enable_corrections:
            current_value = html.escape(str(field.final_value or ''))
            correction_html = f'''
                <div class="correction-form">
                    <input type="text" 
                           class="correction-input" 
                           data-field="{html.escape(field.field_name)}"
                           value="{current_value}"
                           placeholder="Enter corrected value">
                    <div class="correction-actions">
                        <button class="btn-sm btn-approve" data-field="{html.escape(field.field_name)}">✓</button>
                        <button class="btn-sm btn-reject" data-field="{html.escape(field.field_name)}">✗</button>
                    </div>
                </div>
            '''
        
        return f'''
        <div class="field-row {status_class}" data-field="{html.escape(field.field_name)}">
            <div class="field-header">
                <span class="field-name">{html.escape(field.field_name)}</span>
                <span class="field-status status-{status_class}">{field.status.name}</span>
            </div>
            <div class="field-value">
                <span class="extracted">{html.escape(str(field.extracted_value or ''))}</span>
                {f'<span class="corrected">→ {html.escape(str(field.corrected_value))}</span>' if field.corrected_value else ''}
            </div>
            {confidence_html}
            {suggestions_html}
            {correction_html}
            <div class="field-notes">
                <textarea class="notes-input" 
                          data-field="{html.escape(field.field_name)}"
                          placeholder="Add notes...">{html.escape(field.reviewer_notes)}</textarea>
            </div>
        </div>
        '''
    
    def _generate_index(
        self,
        item_paths: List[tuple],
    ) -> str:
        """Generate index HTML for batch review."""
        rows = []
        
        for item, path in item_paths:
            rel_path = os.path.basename(path)
            status_badge = self._get_status_badge(item.overall_status)
            progress = int(item.review_progress * 100)
            
            rows.append(f'''
            <tr>
                <td><a href="{html.escape(rel_path)}">{html.escape(item.document_name)}</a></td>
                <td>{status_badge}</td>
                <td>
                    <div class="progress-bar-small">
                        <div class="progress-fill" style="width: {progress}%"></div>
                    </div>
                    {progress}%
                </td>
                <td>{len(item.fields)}</td>
                <td>{len(item.corrected_fields)}</td>
                <td>{item.created_at.strftime('%Y-%m-%d %H:%M')}</td>
            </tr>
            ''')
        
        total = len(item_paths)
        completed = sum(1 for item, _ in item_paths if item.is_complete)
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review Queue</title>
    <style>
        {self._get_styles()}
        
        .index-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .index-table th, .index-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .index-table th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        
        .index-table tr:hover {{
            background: #f9f9f9;
        }}
        
        .progress-bar-small {{
            width: 100px;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        
        .summary-card .value {{
            font-size: 32px;
            font-weight: 600;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Review Queue</h1>
        </header>
        
        <div class="summary-cards">
            <div class="summary-card">
                <h3>Total Documents</h3>
                <div class="value">{total}</div>
            </div>
            <div class="summary-card">
                <h3>Completed</h3>
                <div class="value">{completed}</div>
            </div>
            <div class="summary-card">
                <h3>Pending</h3>
                <div class="value">{total - completed}</div>
            </div>
            <div class="summary-card">
                <h3>Completion Rate</h3>
                <div class="value">{int(completed/total*100) if total else 0}%</div>
            </div>
        </div>
        
        <table class="index-table">
            <thead>
                <tr>
                    <th>Document</th>
                    <th>Status</th>
                    <th>Progress</th>
                    <th>Fields</th>
                    <th>Corrections</th>
                    <th>Created</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
</body>
</html>'''
    
    def _get_status_badge(self, status: ReviewStatus) -> str:
        """Get HTML badge for status."""
        colors = {
            ReviewStatus.PENDING: '#ff9800',
            ReviewStatus.APPROVED: '#4caf50',
            ReviewStatus.CORRECTED: '#2196f3',
            ReviewStatus.REJECTED: '#f44336',
            ReviewStatus.SKIPPED: '#9e9e9e',
        }
        color = colors.get(status, '#9e9e9e')
        return f'<span class="status-badge" style="background: {color}">{status.name}</span>'
    
    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class for confidence level."""
        if confidence >= 0.9:
            return 'confidence-high'
        elif confidence >= 0.7:
            return 'confidence-medium'
        elif confidence >= 0.5:
            return 'confidence-low'
        return 'confidence-very-low'
    
    def _get_styles(self) -> str:
        """Get CSS styles."""
        return '''
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin-bottom: 10px;
        }
        
        .doc-info {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .doc-name {
            font-weight: 500;
            color: #666;
        }
        
        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            color: white;
            font-size: 12px;
            font-weight: 500;
        }
        
        .progress-bar {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            position: relative;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: #4caf50;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            position: absolute;
            right: 0;
            top: -20px;
            font-size: 12px;
            color: #666;
        }
        
        .main {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
        }
        
        .pdf-panel, .fields-panel {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }
        
        .pdf-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .pdf-container {
            position: relative;
            min-height: 600px;
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }
        
        .pdf-placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 600px;
            color: #999;
        }
        
        .highlight-layer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .page-nav {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .btn {
            padding: 8px 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        
        .btn:hover {
            background: #f5f5f5;
        }
        
        .btn-primary {
            background: #2196f3;
            color: white;
            border-color: #2196f3;
        }
        
        .btn-primary:hover {
            background: #1976d2;
        }
        
        .btn-success {
            background: #4caf50;
            color: white;
            border-color: #4caf50;
        }
        
        .btn-success:hover {
            background: #388e3c;
        }
        
        .fields-panel h2 {
            margin-bottom: 15px;
        }
        
        .field-filters {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .filter-btn {
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 20px;
            background: white;
            cursor: pointer;
            font-size: 13px;
        }
        
        .filter-btn.active {
            background: #2196f3;
            color: white;
            border-color: #2196f3;
        }
        
        .fields-list {
            max-height: 600px;
            overflow-y: auto;
        }
        
        .field-row {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.2s;
        }
        
        .field-row:hover {
            border-color: #2196f3;
        }
        
        .field-row.approved {
            border-left: 4px solid #4caf50;
        }
        
        .field-row.corrected {
            border-left: 4px solid #2196f3;
        }
        
        .field-row.rejected {
            border-left: 4px solid #f44336;
        }
        
        .field-row.pending {
            border-left: 4px solid #ff9800;
        }
        
        .field-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        
        .field-name {
            font-weight: 600;
        }
        
        .field-status {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }
        
        .status-pending { background: #fff3e0; color: #e65100; }
        .status-approved { background: #e8f5e9; color: #2e7d32; }
        .status-corrected { background: #e3f2fd; color: #1565c0; }
        .status-rejected { background: #ffebee; color: #c62828; }
        
        .field-value {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            margin-bottom: 8px;
        }
        
        .field-value .extracted {
            color: #333;
        }
        
        .field-value .corrected {
            color: #2196f3;
            margin-left: 10px;
        }
        
        .confidence {
            height: 6px;
            background: #e0e0e0;
            border-radius: 3px;
            margin-bottom: 8px;
            position: relative;
        }
        
        .confidence-bar {
            height: 100%;
            border-radius: 3px;
        }
        
        .confidence-high .confidence-bar { background: #4caf50; }
        .confidence-medium .confidence-bar { background: #ff9800; }
        .confidence-low .confidence-bar { background: #f44336; }
        .confidence-very-low .confidence-bar { background: #9e9e9e; }
        
        .confidence span {
            position: absolute;
            right: 0;
            top: -16px;
            font-size: 11px;
            color: #666;
        }
        
        .suggestions {
            font-size: 12px;
            color: #666;
            margin: 8px 0;
            padding-left: 20px;
        }
        
        .suggestions li {
            margin-bottom: 4px;
        }
        
        .correction-form {
            margin-top: 10px;
            display: flex;
            gap: 8px;
        }
        
        .correction-input {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .correction-input:focus {
            outline: none;
            border-color: #2196f3;
        }
        
        .correction-actions {
            display: flex;
            gap: 4px;
        }
        
        .btn-sm {
            padding: 8px 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .btn-approve {
            background: #e8f5e9;
            color: #2e7d32;
        }
        
        .btn-approve:hover {
            background: #c8e6c9;
        }
        
        .btn-reject {
            background: #ffebee;
            color: #c62828;
        }
        
        .btn-reject:hover {
            background: #ffcdd2;
        }
        
        .field-notes {
            margin-top: 10px;
        }
        
        .notes-input {
            width: 100%;
            padding: 8px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            font-size: 13px;
            resize: vertical;
            min-height: 60px;
        }
        
        .footer {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .actions {
            display: flex;
            gap: 10px;
        }
        
        .stats {
            display: flex;
            gap: 20px;
            color: #666;
            font-size: 14px;
        }
        
        @media (max-width: 1200px) {
            .main {
                grid-template-columns: 1fr;
            }
        }
        '''
    
    def _get_javascript(self) -> str:
        """Get JavaScript code."""
        return '''
        // State
        let currentPage = 1;
        let totalPages = itemData.page_count || 1;
        let changes = {};
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            updatePageDisplay();
            setupEventListeners();
            highlightFields();
        });
        
        function setupEventListeners() {
            // Page navigation
            document.getElementById('prevPage').addEventListener('click', () => {
                if (currentPage > 1) {
                    currentPage--;
                    updatePageDisplay();
                    highlightFields();
                }
            });
            
            document.getElementById('nextPage').addEventListener('click', () => {
                if (currentPage < totalPages) {
                    currentPage++;
                    updatePageDisplay();
                    highlightFields();
                }
            });
            
            // Filter buttons
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    filterFields(btn.dataset.filter);
                });
            });
            
            // Correction inputs
            document.querySelectorAll('.correction-input').forEach(input => {
                input.addEventListener('change', (e) => {
                    const fieldName = e.target.dataset.field;
                    changes[fieldName] = {
                        corrected_value: e.target.value,
                        status: 'CORRECTED'
                    };
                    updateFieldStatus(fieldName, 'corrected');
                });
            });
            
            // Approve buttons
            document.querySelectorAll('.btn-approve').forEach(btn => {
                btn.addEventListener('click', () => {
                    const fieldName = btn.dataset.field;
                    changes[fieldName] = { status: 'APPROVED' };
                    updateFieldStatus(fieldName, 'approved');
                });
            });
            
            // Reject buttons
            document.querySelectorAll('.btn-reject').forEach(btn => {
                btn.addEventListener('click', () => {
                    const fieldName = btn.dataset.field;
                    changes[fieldName] = { status: 'REJECTED' };
                    updateFieldStatus(fieldName, 'rejected');
                });
            });
            
            // Notes inputs
            document.querySelectorAll('.notes-input').forEach(input => {
                input.addEventListener('change', (e) => {
                    const fieldName = e.target.dataset.field;
                    if (!changes[fieldName]) changes[fieldName] = {};
                    changes[fieldName].notes = e.target.value;
                });
            });
            
            // Save button
            document.getElementById('saveBtn').addEventListener('click', saveChanges);
            
            // Export button
            document.getElementById('exportBtn').addEventListener('click', exportJSON);
            
            // Complete button
            document.getElementById('completeBtn').addEventListener('click', markComplete);
            
            // Keyboard navigation
            document.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft') {
                    document.getElementById('prevPage').click();
                } else if (e.key === 'ArrowRight') {
                    document.getElementById('nextPage').click();
                }
            });
            
            // Field row click to highlight
            document.querySelectorAll('.field-row').forEach(row => {
                row.addEventListener('click', () => {
                    const fieldName = row.dataset.field;
                    scrollToBBox(fieldName);
                });
            });
        }
        
        function updatePageDisplay() {
            document.getElementById('pageNum').textContent = `Page ${currentPage} of ${totalPages}`;
        }
        
        function highlightFields() {
            const svg = document.getElementById('highlightLayer');
            svg.innerHTML = '';
            
            // Get container dimensions
            const container = document.getElementById('pdfContainer');
            const width = container.clientWidth;
            const height = container.clientHeight;
            
            // Draw bounding boxes for current page
            Object.entries(bboxData).forEach(([fieldName, bbox]) => {
                if (bbox.page === currentPage - 1) {
                    // Scale coordinates (assuming 612x792 PDF page)
                    const scaleX = width / 612;
                    const scaleY = height / 792;
                    
                    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    rect.setAttribute('x', bbox.x0 * scaleX);
                    rect.setAttribute('y', bbox.y0 * scaleY);
                    rect.setAttribute('width', (bbox.x1 - bbox.x0) * scaleX);
                    rect.setAttribute('height', (bbox.y1 - bbox.y0) * scaleY);
                    rect.setAttribute('fill', 'rgba(255, 235, 59, 0.3)');
                    rect.setAttribute('stroke', '#ffc107');
                    rect.setAttribute('stroke-width', '2');
                    rect.setAttribute('data-field', fieldName);
                    rect.style.cursor = 'pointer';
                    
                    rect.addEventListener('click', () => {
                        scrollToField(fieldName);
                    });
                    
                    svg.appendChild(rect);
                }
            });
        }
        
        function filterFields(filter) {
            document.querySelectorAll('.field-row').forEach(row => {
                if (filter === 'all') {
                    row.style.display = 'block';
                } else if (filter === 'pending' && row.classList.contains('pending')) {
                    row.style.display = 'block';
                } else if (filter === 'corrected' && row.classList.contains('corrected')) {
                    row.style.display = 'block';
                } else {
                    row.style.display = filter === 'all' ? 'block' : 'none';
                }
            });
        }
        
        function updateFieldStatus(fieldName, status) {
            const row = document.querySelector(`.field-row[data-field="${fieldName}"]`);
            if (row) {
                row.classList.remove('pending', 'approved', 'corrected', 'rejected');
                row.classList.add(status);
                
                const statusSpan = row.querySelector('.field-status');
                if (statusSpan) {
                    statusSpan.textContent = status.toUpperCase();
                    statusSpan.className = `field-status status-${status}`;
                }
            }
        }
        
        function scrollToField(fieldName) {
            const row = document.querySelector(`.field-row[data-field="${fieldName}"]`);
            if (row) {
                row.scrollIntoView({ behavior: 'smooth', block: 'center' });
                row.style.animation = 'highlight 1s';
                setTimeout(() => row.style.animation = '', 1000);
            }
        }
        
        function scrollToBBox(fieldName) {
            const bbox = bboxData[fieldName];
            if (bbox && bbox.page !== currentPage - 1) {
                currentPage = bbox.page + 1;
                updatePageDisplay();
                highlightFields();
            }
        }
        
        function saveChanges() {
            console.log('Saving changes:', changes);
            
            // In a real implementation, this would POST to server
            const data = {
                item_id: itemData.item_id,
                changes: changes,
                timestamp: new Date().toISOString()
            };
            
            // Store in localStorage as fallback
            localStorage.setItem(`review_${itemData.item_id}`, JSON.stringify(data));
            
            alert('Changes saved locally');
        }
        
        function exportJSON() {
            const exportData = {
                ...itemData,
                changes: changes,
                exported_at: new Date().toISOString()
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `review_${itemData.item_id}.json`;
            a.click();
            
            URL.revokeObjectURL(url);
        }
        
        function markComplete() {
            if (confirm('Mark this document as completely reviewed?')) {
                changes._complete = true;
                saveChanges();
                alert('Document marked as complete');
            }
        }
        '''

"""
Review Export

Export review data in various formats for integration with other systems.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO, TextIO

from .review_data import ReviewItem, ReviewSession, ReviewStatus

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    
    JSON = auto()           # Full JSON with all metadata
    JSON_MINIMAL = auto()   # JSON with just values
    CSV = auto()            # CSV spreadsheet
    JSONL = auto()          # JSON Lines (one record per line)
    COCO = auto()           # COCO format for ML training
    LABELSTUDIO = auto()    # Label Studio format
    

@dataclass
class ExportConfig:
    """Configuration for export."""
    
    # What to include
    include_metadata: bool = True
    include_bboxes: bool = True
    include_confidence: bool = True
    include_corrections: bool = True
    include_notes: bool = True
    
    # Field handling
    use_final_values: bool = True  # Use corrected values when available
    flatten_structure: bool = False
    
    # Output options
    pretty_print: bool = True
    indent: int = 2
    
    # ML export options
    image_root: Optional[str] = None
    category_mapping: Dict[str, int] = None


class ReviewExporter:
    """
    Export review data in various formats.
    
    Supports multiple export formats for different use cases:
    - JSON: Full data for archival
    - CSV: Spreadsheet for analysis
    - COCO/LabelStudio: ML training data
    
    Usage:
        exporter = ReviewExporter()
        
        # Export single item
        exporter.export_item(item, 'output.json', ExportFormat.JSON)
        
        # Export session
        exporter.export_session(session, 'session.json', ExportFormat.JSON)
        
        # Export for ML training
        exporter.export_for_training(session, 'training/', ExportFormat.COCO)
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
    
    def export_item(
        self,
        item: ReviewItem,
        output_path: str,
        format: ExportFormat = ExportFormat.JSON,
    ) -> str:
        """
        Export a single review item.
        
        Args:
            item: ReviewItem to export
            output_path: Output file path
            format: Export format
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        if format == ExportFormat.JSON:
            return self._export_json(item, output_path)
        elif format == ExportFormat.JSON_MINIMAL:
            return self._export_json_minimal(item, output_path)
        elif format == ExportFormat.CSV:
            return self._export_csv([item], output_path)
        elif format == ExportFormat.JSONL:
            return self._export_jsonl([item], output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_session(
        self,
        session: ReviewSession,
        output_path: str,
        format: ExportFormat = ExportFormat.JSON,
    ) -> str:
        """
        Export a review session.
        
        Args:
            session: ReviewSession to export
            output_path: Output file path
            format: Export format
            
        Returns:
            Path to exported file
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        items = list(session.items.values())
        
        if format == ExportFormat.JSON:
            return self._export_session_json(session, output_path)
        elif format == ExportFormat.CSV:
            return self._export_csv(items, output_path)
        elif format == ExportFormat.JSONL:
            return self._export_jsonl(items, output_path)
        elif format == ExportFormat.COCO:
            return self._export_coco(items, output_path)
        elif format == ExportFormat.LABELSTUDIO:
            return self._export_labelstudio(items, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_for_training(
        self,
        session: ReviewSession,
        output_dir: str,
        format: ExportFormat = ExportFormat.COCO,
        split_ratio: float = 0.8,
    ) -> Dict[str, str]:
        """
        Export data for ML training with train/val split.
        
        Args:
            session: ReviewSession to export
            output_dir: Output directory
            format: Export format (COCO or LABELSTUDIO)
            split_ratio: Train/val split ratio
            
        Returns:
            Paths to exported files
        """
        import random
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get completed items only
        items = [i for i in session.items.values() if i.is_complete]
        
        if not items:
            logger.warning("No completed items to export")
            return {}
        
        # Shuffle and split
        random.shuffle(items)
        split_idx = int(len(items) * split_ratio)
        
        train_items = items[:split_idx]
        val_items = items[split_idx:]
        
        paths = {}
        
        if format == ExportFormat.COCO:
            paths['train'] = self._export_coco(
                train_items,
                os.path.join(output_dir, 'train.json')
            )
            paths['val'] = self._export_coco(
                val_items,
                os.path.join(output_dir, 'val.json')
            )
        elif format == ExportFormat.LABELSTUDIO:
            paths['train'] = self._export_labelstudio(
                train_items,
                os.path.join(output_dir, 'train.json')
            )
            paths['val'] = self._export_labelstudio(
                val_items,
                os.path.join(output_dir, 'val.json')
            )
        
        # Export statistics
        stats = {
            'total_items': len(items),
            'train_items': len(train_items),
            'val_items': len(val_items),
            'split_ratio': split_ratio,
            'exported_at': datetime.now().isoformat(),
        }
        
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        paths['stats'] = stats_path
        
        return paths
    
    def _export_json(self, item: ReviewItem, output_path: str) -> str:
        """Export item as JSON."""
        data = self._prepare_item_data(item)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if self.config.pretty_print:
                json.dump(data, f, indent=self.config.indent, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        return output_path
    
    def _export_json_minimal(self, item: ReviewItem, output_path: str) -> str:
        """Export item as minimal JSON with just field values."""
        data = {
            'document': item.document_name,
            'fields': {},
        }
        
        for name, field in item.fields.items():
            if self.config.use_final_values:
                data['fields'][name] = field.final_value
            else:
                data['fields'][name] = field.extracted_value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=self.config.indent, ensure_ascii=False)
        
        return output_path
    
    def _export_session_json(self, session: ReviewSession, output_path: str) -> str:
        """Export session as JSON."""
        data = {
            'session': session.get_statistics(),
            'items': [
                self._prepare_item_data(item)
                for item in session.items.values()
            ],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=self.config.indent, ensure_ascii=False)
        
        return output_path
    
    def _export_csv(self, items: List[ReviewItem], output_path: str) -> str:
        """Export items as CSV."""
        if not items:
            return output_path
        
        # Collect all field names
        all_fields = set()
        for item in items:
            all_fields.update(item.fields.keys())
        
        field_names = sorted(all_fields)
        
        # Build headers
        headers = ['document_name', 'document_path', 'status']
        for name in field_names:
            headers.append(f'{name}_value')
            if self.config.include_confidence:
                headers.append(f'{name}_confidence')
            if self.config.include_corrections:
                headers.append(f'{name}_corrected')
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for item in items:
                row = [
                    item.document_name,
                    item.document_path,
                    item.overall_status.name,
                ]
                
                for name in field_names:
                    field = item.fields.get(name)
                    if field:
                        if self.config.use_final_values:
                            row.append(field.final_value or '')
                        else:
                            row.append(field.extracted_value or '')
                        
                        if self.config.include_confidence:
                            row.append(round(field.confidence, 3))
                        if self.config.include_corrections:
                            row.append(field.corrected_value or '')
                    else:
                        row.append('')
                        if self.config.include_confidence:
                            row.append('')
                        if self.config.include_corrections:
                            row.append('')
                
                writer.writerow(row)
        
        return output_path
    
    def _export_jsonl(self, items: List[ReviewItem], output_path: str) -> str:
        """Export items as JSON Lines."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                data = self._prepare_item_data(item)
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        return output_path
    
    def _export_coco(self, items: List[ReviewItem], output_path: str) -> str:
        """
        Export items in COCO format for object detection training.
        
        COCO format is widely used for training document understanding models.
        """
        # Build category list from field names
        categories = {}
        category_list = []
        
        for item in items:
            for name in item.fields.keys():
                if name not in categories:
                    cat_id = len(categories) + 1
                    categories[name] = cat_id
                    category_list.append({
                        'id': cat_id,
                        'name': name,
                        'supercategory': 'field',
                    })
        
        # Override with config if provided
        if self.config.category_mapping:
            categories = self.config.category_mapping
        
        # Build images and annotations
        images = []
        annotations = []
        annotation_id = 1
        
        for image_id, item in enumerate(items, 1):
            # Image entry
            images.append({
                'id': image_id,
                'file_name': item.document_name,
                'width': 612,  # Standard PDF width
                'height': 792,  # Standard PDF height
            })
            
            # Annotations for each field with bbox
            for field in item.fields.values():
                if field.bbox:
                    bbox = field.bbox
                    
                    # COCO bbox format: [x, y, width, height]
                    coco_bbox = [
                        bbox.x0,
                        bbox.y0,
                        bbox.width,
                        bbox.height,
                    ]
                    
                    annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': categories.get(field.field_name, 1),
                        'bbox': coco_bbox,
                        'area': bbox.width * bbox.height,
                        'iscrowd': 0,
                        'attributes': {
                            'value': str(field.final_value),
                            'confidence': field.confidence,
                            'status': field.status.name,
                        },
                    }
                    
                    annotations.append(annotation)
                    annotation_id += 1
        
        # COCO structure
        coco_data = {
            'info': {
                'description': 'Document extraction training data',
                'date_created': datetime.now().isoformat(),
                'version': '1.0',
            },
            'licenses': [],
            'categories': category_list,
            'images': images,
            'annotations': annotations,
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _export_labelstudio(self, items: List[ReviewItem], output_path: str) -> str:
        """
        Export items in Label Studio format.
        
        Label Studio is a popular annotation tool that can be used
        for reviewing and correcting extraction results.
        """
        tasks = []
        
        for item in items:
            # Build annotations
            annotations = []
            
            for field in item.fields.values():
                result = {
                    'id': field.field_id,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [field.field_name],
                        'text': str(field.final_value),
                    },
                    'from_name': 'label',
                    'to_name': 'document',
                }
                
                # Add bbox if available
                if field.bbox:
                    result['value'].update({
                        'x': (field.bbox.x0 / 612) * 100,
                        'y': (field.bbox.y0 / 792) * 100,
                        'width': (field.bbox.width / 612) * 100,
                        'height': (field.bbox.height / 792) * 100,
                    })
                
                annotations.append(result)
            
            task = {
                'id': item.item_id,
                'data': {
                    'document': item.document_path,
                    'document_name': item.document_name,
                },
                'annotations': [{
                    'result': annotations,
                    'completed_by': item.assigned_reviewer or 'auto',
                }],
                'predictions': [],
            }
            
            tasks.append(task)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _prepare_item_data(self, item: ReviewItem) -> Dict[str, Any]:
        """Prepare item data for export."""
        data = {
            'item_id': item.item_id,
            'document_name': item.document_name,
            'document_path': item.document_path,
            'status': item.overall_status.name,
            'fields': {},
        }
        
        if self.config.include_metadata:
            data['metadata'] = {
                'created_at': item.created_at.isoformat(),
                'completed_at': item.completed_at.isoformat() if item.completed_at else None,
                'review_time': item.total_review_time,
                'page_count': item.page_count,
                'correction_rate': item.correction_rate,
            }
        
        for name, field in item.fields.items():
            field_data = {
                'value': field.final_value if self.config.use_final_values else field.extracted_value,
            }
            
            if self.config.include_confidence:
                field_data['confidence'] = round(field.confidence, 3)
            
            if self.config.include_corrections and field.was_corrected:
                field_data['original_value'] = field.extracted_value
                field_data['corrected_value'] = field.corrected_value
            
            if self.config.include_bboxes and field.bbox:
                field_data['bbox'] = field.bbox.to_dict()
            
            if self.config.include_notes and field.reviewer_notes:
                field_data['notes'] = field.reviewer_notes
            
            field_data['status'] = field.status.name
            
            data['fields'][name] = field_data
        
        return data


def export_corrections_summary(
    session: ReviewSession,
    output_path: str,
) -> str:
    """
    Export a summary of corrections made during review.
    
    Useful for analyzing extraction accuracy and improving the model.
    
    Args:
        session: Review session
        output_path: Output path
        
    Returns:
        Path to exported file
    """
    corrections = []
    
    for item in session.items.values():
        for field in item.fields.values():
            if field.was_corrected:
                corrections.append({
                    'document': item.document_name,
                    'field': field.field_name,
                    'original': field.extracted_value,
                    'corrected': field.corrected_value,
                    'confidence': round(field.confidence, 3),
                    'notes': field.reviewer_notes,
                })
    
    summary = {
        'session_id': session.session_id,
        'total_fields': session.total_fields,
        'total_corrections': len(corrections),
        'correction_rate': round(len(corrections) / session.total_fields, 3) if session.total_fields else 0,
        'corrections': corrections,
        'exported_at': datetime.now().isoformat(),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return output_path

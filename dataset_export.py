"""
Dataset Export Module

Export extracted data in ML-ready formats for training and evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import csv
import random
from enum import Enum


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    COCO = "coco"
    LABEL_STUDIO = "label_studio"
    HUGGINGFACE = "huggingface"
    YOLO = "yolo"


@dataclass
class BoundingBoxAnnotation:
    """Bounding box annotation for ML training."""
    
    label: str
    x: float  # Normalized 0-1
    y: float
    width: float
    height: float
    
    # Optional attributes
    confidence: float = 1.0
    text: str = ""
    page: int = 0
    
    def to_coco(self, image_id: int, annotation_id: int, category_id: int) -> dict:
        """Convert to COCO format."""
        # COCO uses absolute pixels, assuming 1000x1000 for normalized
        return {
            'id': annotation_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': [
                self.x * 1000,
                self.y * 1000,
                self.width * 1000,
                self.height * 1000,
            ],
            'area': self.width * self.height * 1000000,
            'segmentation': [],
            'iscrowd': 0,
            'attributes': {
                'text': self.text,
                'confidence': self.confidence,
            },
        }
    
    def to_yolo(self, class_id: int) -> str:
        """Convert to YOLO format."""
        # YOLO: class x_center y_center width height (normalized)
        x_center = self.x + self.width / 2
        y_center = self.y + self.height / 2
        return f"{class_id} {x_center:.6f} {y_center:.6f} {self.width:.6f} {self.height:.6f}"
    
    def to_label_studio(self) -> dict:
        """Convert to Label Studio format."""
        return {
            'type': 'rectanglelabels',
            'value': {
                'x': self.x * 100,  # Label Studio uses percentages
                'y': self.y * 100,
                'width': self.width * 100,
                'height': self.height * 100,
                'rectanglelabels': [self.label],
            },
            'score': self.confidence,
        }


@dataclass
class DocumentAnnotation:
    """Full document annotation for ML training."""
    
    document_id: str
    document_path: str
    document_type: str
    
    # Field annotations
    fields: Dict[str, Any]
    field_boxes: List[BoundingBoxAnnotation] = field(default_factory=list)
    
    # Table annotations
    tables: List[Dict[str, Any]] = field(default_factory=list)
    table_boxes: List[BoundingBoxAnnotation] = field(default_factory=list)
    
    # Metadata
    page_count: int = 1
    image_width: int = 0
    image_height: int = 0
    
    # Quality info
    human_verified: bool = False
    extraction_confidence: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'document_id': self.document_id,
            'document_path': self.document_path,
            'document_type': self.document_type,
            'fields': self.fields,
            'field_boxes': [
                {
                    'label': b.label,
                    'x': b.x,
                    'y': b.y,
                    'width': b.width,
                    'height': b.height,
                    'text': b.text,
                    'confidence': b.confidence,
                    'page': b.page,
                }
                for b in self.field_boxes
            ],
            'tables': self.tables,
            'page_count': self.page_count,
            'human_verified': self.human_verified,
            'extraction_confidence': self.extraction_confidence,
        }


@dataclass
class DatasetSplit:
    """Dataset split configuration."""
    
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    stratify_by: Optional[str] = None  # e.g., 'document_type'
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class DatasetExporter:
    """
    Export extracted data as ML-ready datasets.
    
    Supports multiple formats:
    - JSON/JSONL: Simple key-value format
    - CSV: Tabular format
    - COCO: Object detection format
    - Label Studio: Interactive annotation format
    - HuggingFace: Datasets library format
    - YOLO: YOLO object detection format
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(
        self,
        annotations: List[DocumentAnnotation],
        format: ExportFormat,
        split: Optional[DatasetSplit] = None,
        prefix: str = "dataset",
    ) -> Dict[str, str]:
        """
        Export dataset in specified format.
        
        Args:
            annotations: List of document annotations
            format: Export format
            split: Optional dataset split configuration
            prefix: Output file prefix
        
        Returns:
            Dictionary of output file paths
        """
        
        if split:
            splits = self._split_data(annotations, split)
        else:
            splits = {'all': annotations}
        
        output_paths = {}
        
        for split_name, split_data in splits.items():
            if format == ExportFormat.JSON:
                path = self._export_json(split_data, split_name, prefix)
            elif format == ExportFormat.JSONL:
                path = self._export_jsonl(split_data, split_name, prefix)
            elif format == ExportFormat.CSV:
                path = self._export_csv(split_data, split_name, prefix)
            elif format == ExportFormat.COCO:
                path = self._export_coco(split_data, split_name, prefix)
            elif format == ExportFormat.LABEL_STUDIO:
                path = self._export_label_studio(split_data, split_name, prefix)
            elif format == ExportFormat.HUGGINGFACE:
                path = self._export_huggingface(split_data, split_name, prefix)
            elif format == ExportFormat.YOLO:
                path = self._export_yolo(split_data, split_name, prefix)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            output_paths[split_name] = str(path)
        
        return output_paths
    
    def _split_data(
        self,
        annotations: List[DocumentAnnotation],
        split: DatasetSplit,
    ) -> Dict[str, List[DocumentAnnotation]]:
        """Split data into train/val/test sets."""
        
        random.seed(split.seed)
        
        # Optionally stratify
        if split.stratify_by:
            groups: Dict[str, List[DocumentAnnotation]] = {}
            for ann in annotations:
                key = getattr(ann, split.stratify_by, 'unknown')
                if key not in groups:
                    groups[key] = []
                groups[key].append(ann)
            
            train, val, test = [], [], []
            for group_items in groups.values():
                random.shuffle(group_items)
                n = len(group_items)
                n_train = int(n * split.train_ratio)
                n_val = int(n * split.val_ratio)
                
                train.extend(group_items[:n_train])
                val.extend(group_items[n_train:n_train + n_val])
                test.extend(group_items[n_train + n_val:])
        else:
            data = annotations.copy()
            random.shuffle(data)
            
            n = len(data)
            n_train = int(n * split.train_ratio)
            n_val = int(n * split.val_ratio)
            
            train = data[:n_train]
            val = data[n_train:n_train + n_val]
            test = data[n_train + n_val:]
        
        return {
            'train': train,
            'val': val,
            'test': test,
        }
    
    def _export_json(
        self,
        annotations: List[DocumentAnnotation],
        split_name: str,
        prefix: str,
    ) -> Path:
        """Export as JSON."""
        
        path = self.output_dir / f"{prefix}_{split_name}.json"
        
        data = {
            'metadata': {
                'format': 'json',
                'split': split_name,
                'count': len(annotations),
            },
            'documents': [ann.to_dict() for ann in annotations],
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return path
    
    def _export_jsonl(
        self,
        annotations: List[DocumentAnnotation],
        split_name: str,
        prefix: str,
    ) -> Path:
        """Export as JSON Lines."""
        
        path = self.output_dir / f"{prefix}_{split_name}.jsonl"
        
        with open(path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(json.dumps(ann.to_dict(), ensure_ascii=False) + '\n')
        
        return path
    
    def _export_csv(
        self,
        annotations: List[DocumentAnnotation],
        split_name: str,
        prefix: str,
    ) -> Path:
        """Export as CSV."""
        
        path = self.output_dir / f"{prefix}_{split_name}.csv"
        
        if not annotations:
            return path
        
        # Collect all field names
        all_fields = set()
        for ann in annotations:
            all_fields.update(ann.fields.keys())
        
        fieldnames = [
            'document_id',
            'document_path',
            'document_type',
            'page_count',
            'human_verified',
            'extraction_confidence',
        ] + sorted(all_fields)
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for ann in annotations:
                row = {
                    'document_id': ann.document_id,
                    'document_path': ann.document_path,
                    'document_type': ann.document_type,
                    'page_count': ann.page_count,
                    'human_verified': ann.human_verified,
                    'extraction_confidence': ann.extraction_confidence,
                }
                row.update(ann.fields)
                writer.writerow(row)
        
        return path
    
    def _export_coco(
        self,
        annotations: List[DocumentAnnotation],
        split_name: str,
        prefix: str,
    ) -> Path:
        """Export as COCO format for object detection."""
        
        path = self.output_dir / f"{prefix}_{split_name}_coco.json"
        
        # Build category list
        categories = {}
        cat_id = 1
        
        for ann in annotations:
            for box in ann.field_boxes:
                if box.label not in categories:
                    categories[box.label] = cat_id
                    cat_id += 1
        
        # Build COCO structure
        coco = {
            'info': {
                'description': f'Document Intelligence Dataset - {split_name}',
                'version': '1.0',
                'year': 2024,
            },
            'licenses': [],
            'categories': [
                {'id': cat_id, 'name': cat_name, 'supercategory': 'field'}
                for cat_name, cat_id in categories.items()
            ],
            'images': [],
            'annotations': [],
        }
        
        annotation_id = 1
        
        for img_id, ann in enumerate(annotations, 1):
            # Add image
            coco['images'].append({
                'id': img_id,
                'file_name': ann.document_path,
                'width': ann.image_width or 1000,
                'height': ann.image_height or 1000,
            })
            
            # Add annotations
            for box in ann.field_boxes:
                if box.label in categories:
                    coco_ann = box.to_coco(
                        img_id, annotation_id, categories[box.label]
                    )
                    coco['annotations'].append(coco_ann)
                    annotation_id += 1
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(coco, f, indent=2)
        
        return path
    
    def _export_label_studio(
        self,
        annotations: List[DocumentAnnotation],
        split_name: str,
        prefix: str,
    ) -> Path:
        """Export as Label Studio format."""
        
        path = self.output_dir / f"{prefix}_{split_name}_labelstudio.json"
        
        tasks = []
        
        for ann in annotations:
            task = {
                'data': {
                    'image': ann.document_path,
                    'document_id': ann.document_id,
                    'document_type': ann.document_type,
                },
                'predictions': [
                    {
                        'model_version': 'document_intelligence_v1',
                        'score': ann.extraction_confidence,
                        'result': [
                            box.to_label_studio()
                            for box in ann.field_boxes
                        ],
                    }
                ] if ann.field_boxes else [],
            }
            tasks.append(task)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2)
        
        return path
    
    def _export_huggingface(
        self,
        annotations: List[DocumentAnnotation],
        split_name: str,
        prefix: str,
    ) -> Path:
        """Export for HuggingFace Datasets."""
        
        # Export as Arrow/Parquet if pyarrow available, else JSONL
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            path = self.output_dir / f"{prefix}_{split_name}.parquet"
            
            # Build records
            records = []
            for ann in annotations:
                record = {
                    'document_id': ann.document_id,
                    'document_path': ann.document_path,
                    'document_type': ann.document_type,
                    'page_count': ann.page_count,
                    'human_verified': ann.human_verified,
                    'extraction_confidence': ann.extraction_confidence,
                    'fields': json.dumps(ann.fields),
                    'boxes': json.dumps([
                        {
                            'label': b.label,
                            'x': b.x,
                            'y': b.y,
                            'width': b.width,
                            'height': b.height,
                            'text': b.text,
                        }
                        for b in ann.field_boxes
                    ]),
                }
                records.append(record)
            
            table = pa.Table.from_pylist(records)
            pq.write_table(table, path)
            
            return path
            
        except ImportError:
            # Fallback to JSONL
            return self._export_jsonl(annotations, split_name, prefix)
    
    def _export_yolo(
        self,
        annotations: List[DocumentAnnotation],
        split_name: str,
        prefix: str,
    ) -> Path:
        """Export as YOLO format."""
        
        # Create directories
        images_dir = self.output_dir / split_name / 'images'
        labels_dir = self.output_dir / split_name / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Build class mapping
        classes = {}
        class_id = 0
        
        for ann in annotations:
            for box in ann.field_boxes:
                if box.label not in classes:
                    classes[box.label] = class_id
                    class_id += 1
        
        # Write classes file
        classes_path = self.output_dir / 'classes.txt'
        with open(classes_path, 'w') as f:
            for label, _ in sorted(classes.items(), key=lambda x: x[1]):
                f.write(f"{label}\n")
        
        # Write label files
        for ann in annotations:
            label_file = labels_dir / f"{ann.document_id}.txt"
            
            with open(label_file, 'w') as f:
                for box in ann.field_boxes:
                    if box.label in classes:
                        yolo_line = box.to_yolo(classes[box.label])
                        f.write(yolo_line + '\n')
        
        # Write dataset yaml
        yaml_path = self.output_dir / f"{prefix}.yaml"
        yaml_content = f"""
path: {self.output_dir}
train: train/images
val: val/images
test: test/images

names:
"""
        for label, idx in sorted(classes.items(), key=lambda x: x[1]):
            yaml_content += f"  {idx}: {label}\n"
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return labels_dir


def create_dataset_from_results(
    results: List['DocumentResult'],
    include_boxes: bool = True,
) -> List[DocumentAnnotation]:
    """
    Create dataset annotations from pipeline results.
    
    Args:
        results: List of DocumentResult from pipeline
        include_boxes: Whether to include bounding boxes
    
    Returns:
        List of DocumentAnnotation objects
    """
    from .pipeline import DocumentResult
    
    annotations = []
    
    for result in results:
        boxes = []
        
        if include_boxes and result.extraction_result:
            for field_name, extracted in result.extraction_result.fields.items():
                if extracted.bbox:
                    x0, y0, x1, y1 = extracted.bbox
                    # Normalize to 0-1 (assuming page size known)
                    box = BoundingBoxAnnotation(
                        label=field_name,
                        x=x0 / 612.0,  # Standard US Letter width
                        y=y0 / 792.0,  # Standard US Letter height
                        width=(x1 - x0) / 612.0,
                        height=(y1 - y0) / 792.0,
                        confidence=extracted.confidence,
                        text=extracted.value,
                        page=extracted.page,
                    )
                    boxes.append(box)
        
        ann = DocumentAnnotation(
            document_id=Path(result.source_path).stem,
            document_path=result.source_path,
            document_type=result.document_type,
            fields=result.fields,
            field_boxes=boxes,
            tables=result.tables,
            page_count=result.page_count,
            human_verified=False,
            extraction_confidence=result.document_decision.overall_confidence,
        )
        annotations.append(ann)
    
    return annotations

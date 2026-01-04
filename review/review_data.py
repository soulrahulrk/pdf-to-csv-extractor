"""
Review Data Structures

Data classes for representing review sessions, items, and decisions.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of a review item."""
    
    PENDING = auto()      # Not yet reviewed
    APPROVED = auto()     # Approved as-is
    CORRECTED = auto()    # Corrected by reviewer
    REJECTED = auto()     # Rejected as unusable
    SKIPPED = auto()      # Skipped for later
    
    @property
    def is_complete(self) -> bool:
        """Whether review is complete."""
        return self in (ReviewStatus.APPROVED, ReviewStatus.CORRECTED, ReviewStatus.REJECTED)


class ReviewDecision(Enum):
    """Decision made by reviewer."""
    
    ACCEPT = auto()       # Accept extracted value
    CORRECT = auto()      # Provide corrected value
    REJECT = auto()       # Mark as invalid/unusable
    UNSURE = auto()       # Flag for escalation
    SKIP = auto()         # Skip for now


@dataclass
class BoundingBox:
    """Bounding box for a field on a page."""
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'page': self.page,
            'x0': round(self.x0, 2),
            'y0': round(self.y0, 2),
            'x1': round(self.x1, 2),
            'y1': round(self.y1, 2),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        return cls(
            page=data['page'],
            x0=data['x0'],
            y0=data['y0'],
            x1=data['x1'],
            y1=data['y1'],
        )


@dataclass
class ReviewField:
    """
    A field requiring or that received review.
    """
    field_id: str
    field_name: str
    extracted_value: Any
    corrected_value: Optional[Any] = None
    confidence: float = 0.0
    decision_reason: str = ''
    bbox: Optional[BoundingBox] = None
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_notes: str = ''
    review_time_seconds: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    alternatives: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def final_value(self) -> Any:
        """Get final value (corrected if available, else extracted)."""
        if self.corrected_value is not None:
            return self.corrected_value
        return self.extracted_value
    
    @property
    def was_corrected(self) -> bool:
        """Whether the value was corrected."""
        return self.status == ReviewStatus.CORRECTED
    
    @property
    def is_pending(self) -> bool:
        """Whether review is still pending."""
        return self.status == ReviewStatus.PENDING
    
    def approve(self) -> None:
        """Mark field as approved."""
        self.status = ReviewStatus.APPROVED
    
    def correct(self, new_value: Any, notes: str = '') -> None:
        """Correct the extracted value."""
        self.corrected_value = new_value
        self.status = ReviewStatus.CORRECTED
        self.reviewer_notes = notes
    
    def reject(self, reason: str = '') -> None:
        """Reject the extracted value."""
        self.status = ReviewStatus.REJECTED
        self.reviewer_notes = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'field_id': self.field_id,
            'field_name': self.field_name,
            'extracted_value': str(self.extracted_value) if self.extracted_value else None,
            'corrected_value': str(self.corrected_value) if self.corrected_value else None,
            'final_value': str(self.final_value) if self.final_value else None,
            'confidence': round(self.confidence, 3),
            'decision_reason': self.decision_reason,
            'bbox': self.bbox.to_dict() if self.bbox else None,
            'status': self.status.name,
            'was_corrected': self.was_corrected,
            'reviewer_notes': self.reviewer_notes,
            'review_time_seconds': round(self.review_time_seconds, 2),
            'suggestions': self.suggestions,
            'alternatives': [str(a) for a in self.alternatives],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewField':
        """Create from dictionary."""
        bbox = None
        if data.get('bbox'):
            bbox = BoundingBox.from_dict(data['bbox'])
        
        status = ReviewStatus[data.get('status', 'PENDING')]
        
        return cls(
            field_id=data['field_id'],
            field_name=data['field_name'],
            extracted_value=data.get('extracted_value'),
            corrected_value=data.get('corrected_value'),
            confidence=data.get('confidence', 0.0),
            decision_reason=data.get('decision_reason', ''),
            bbox=bbox,
            status=status,
            reviewer_notes=data.get('reviewer_notes', ''),
            review_time_seconds=data.get('review_time_seconds', 0.0),
            suggestions=data.get('suggestions', []),
            alternatives=data.get('alternatives', []),
        )


@dataclass
class ReviewItem:
    """
    A document requiring review.
    """
    item_id: str
    document_path: str
    document_name: str
    fields: Dict[str, ReviewField]
    overall_status: ReviewStatus = ReviewStatus.PENDING
    priority: int = 0  # Higher = more urgent
    assigned_reviewer: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_review_time: float = 0.0
    page_count: int = 0
    thumbnail_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def pending_fields(self) -> List[ReviewField]:
        """Get fields still pending review."""
        return [f for f in self.fields.values() if f.is_pending]
    
    @property
    def completed_fields(self) -> List[ReviewField]:
        """Get fields with complete review."""
        return [f for f in self.fields.values() if f.status.is_complete]
    
    @property
    def corrected_fields(self) -> List[ReviewField]:
        """Get fields that were corrected."""
        return [f for f in self.fields.values() if f.was_corrected]
    
    @property
    def review_progress(self) -> float:
        """Review completion percentage."""
        if not self.fields:
            return 1.0
        return len(self.completed_fields) / len(self.fields)
    
    @property
    def is_complete(self) -> bool:
        """Whether all fields are reviewed."""
        return all(f.status.is_complete for f in self.fields.values())
    
    @property
    def correction_rate(self) -> float:
        """Percentage of fields that were corrected."""
        completed = self.completed_fields
        if not completed:
            return 0.0
        return len(self.corrected_fields) / len(completed)
    
    def get_field(self, field_name: str) -> Optional[ReviewField]:
        """Get field by name."""
        return self.fields.get(field_name)
    
    def add_field(self, field: ReviewField) -> None:
        """Add a field for review."""
        self.fields[field.field_name] = field
    
    def complete_review(self) -> None:
        """Mark review as complete."""
        self.completed_at = datetime.now()
        self.total_review_time = sum(f.review_time_seconds for f in self.fields.values())
        
        # Determine overall status
        if all(f.status == ReviewStatus.APPROVED for f in self.fields.values()):
            self.overall_status = ReviewStatus.APPROVED
        elif any(f.status == ReviewStatus.REJECTED for f in self.fields.values()):
            self.overall_status = ReviewStatus.REJECTED
        elif any(f.status == ReviewStatus.CORRECTED for f in self.fields.values()):
            self.overall_status = ReviewStatus.CORRECTED
        else:
            self.overall_status = ReviewStatus.APPROVED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'item_id': self.item_id,
            'document_path': self.document_path,
            'document_name': self.document_name,
            'fields': {name: f.to_dict() for name, f in self.fields.items()},
            'overall_status': self.overall_status.name,
            'priority': self.priority,
            'assigned_reviewer': self.assigned_reviewer,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_review_time': round(self.total_review_time, 2),
            'page_count': self.page_count,
            'review_progress': round(self.review_progress, 2),
            'is_complete': self.is_complete,
            'correction_rate': round(self.correction_rate, 3),
            'pending_count': len(self.pending_fields),
            'completed_count': len(self.completed_fields),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewItem':
        """Create from dictionary."""
        fields = {
            name: ReviewField.from_dict(f_data)
            for name, f_data in data.get('fields', {}).items()
        }
        
        created_at = datetime.now()
        if data.get('created_at'):
            try:
                created_at = datetime.fromisoformat(data['created_at'])
            except Exception:
                pass
        
        completed_at = None
        if data.get('completed_at'):
            try:
                completed_at = datetime.fromisoformat(data['completed_at'])
            except Exception:
                pass
        
        return cls(
            item_id=data['item_id'],
            document_path=data['document_path'],
            document_name=data['document_name'],
            fields=fields,
            overall_status=ReviewStatus[data.get('overall_status', 'PENDING')],
            priority=data.get('priority', 0),
            assigned_reviewer=data.get('assigned_reviewer'),
            created_at=created_at,
            completed_at=completed_at,
            total_review_time=data.get('total_review_time', 0.0),
            page_count=data.get('page_count', 0),
        )


@dataclass
class ReviewSession:
    """
    A review session containing multiple items.
    """
    session_id: str
    name: str
    items: Dict[str, ReviewItem]
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    reviewer_id: Optional[str] = None
    reviewer_name: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, name: str) -> 'ReviewSession':
        """Create a new review session."""
        return cls(
            session_id=str(uuid.uuid4()),
            name=name,
            items={},
        )
    
    @property
    def total_items(self) -> int:
        """Total items in session."""
        return len(self.items)
    
    @property
    def pending_items(self) -> List[ReviewItem]:
        """Items still pending review."""
        return [i for i in self.items.values() if not i.is_complete]
    
    @property
    def completed_items(self) -> List[ReviewItem]:
        """Items with complete review."""
        return [i for i in self.items.values() if i.is_complete]
    
    @property
    def progress(self) -> float:
        """Session completion percentage."""
        if not self.items:
            return 1.0
        return len(self.completed_items) / len(self.items)
    
    @property
    def is_complete(self) -> bool:
        """Whether all items are reviewed."""
        return all(i.is_complete for i in self.items.values())
    
    @property
    def total_fields(self) -> int:
        """Total fields across all items."""
        return sum(len(i.fields) for i in self.items.values())
    
    @property
    def total_corrections(self) -> int:
        """Total corrections made."""
        return sum(len(i.corrected_fields) for i in self.items.values())
    
    @property
    def overall_correction_rate(self) -> float:
        """Overall correction rate."""
        total = self.total_fields
        if total == 0:
            return 0.0
        return self.total_corrections / total
    
    def add_item(self, item: ReviewItem) -> None:
        """Add an item to the session."""
        self.items[item.item_id] = item
    
    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get item by ID."""
        return self.items.get(item_id)
    
    def get_next_item(self) -> Optional[ReviewItem]:
        """Get next item to review (highest priority pending)."""
        pending = self.pending_items
        if not pending:
            return None
        return max(pending, key=lambda i: i.priority)
    
    def start(self, reviewer_id: str = '', reviewer_name: str = '') -> None:
        """Start the review session."""
        self.started_at = datetime.now()
        self.reviewer_id = reviewer_id
        self.reviewer_name = reviewer_name
    
    def complete(self) -> None:
        """Complete the review session."""
        self.completed_at = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        return {
            'session_id': self.session_id,
            'name': self.name,
            'total_items': self.total_items,
            'completed_items': len(self.completed_items),
            'pending_items': len(self.pending_items),
            'progress': round(self.progress, 2),
            'total_fields': self.total_fields,
            'total_corrections': self.total_corrections,
            'correction_rate': round(self.overall_correction_rate, 3),
            'is_complete': self.is_complete,
            'reviewer_id': self.reviewer_id,
            'reviewer_name': self.reviewer_name,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'name': self.name,
            'items': {k: v.to_dict() for k, v in self.items.items()},
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'reviewer_id': self.reviewer_id,
            'reviewer_name': self.reviewer_name,
            'settings': self.settings,
            'statistics': self.get_statistics(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewSession':
        """Create from dictionary."""
        items = {
            k: ReviewItem.from_dict(v)
            for k, v in data.get('items', {}).items()
        }
        
        session = cls(
            session_id=data['session_id'],
            name=data['name'],
            items=items,
            settings=data.get('settings', {}),
        )
        
        if data.get('created_at'):
            try:
                session.created_at = datetime.fromisoformat(data['created_at'])
            except Exception:
                pass
        
        if data.get('started_at'):
            try:
                session.started_at = datetime.fromisoformat(data['started_at'])
            except Exception:
                pass
        
        if data.get('completed_at'):
            try:
                session.completed_at = datetime.fromisoformat(data['completed_at'])
            except Exception:
                pass
        
        session.reviewer_id = data.get('reviewer_id')
        session.reviewer_name = data.get('reviewer_name')
        
        return session


def create_review_item_from_extraction(
    document_path: str,
    extraction_result: Dict[str, Any],
    field_decisions: Optional[Dict[str, Any]] = None,
) -> ReviewItem:
    """
    Create a ReviewItem from extraction results.
    
    Args:
        document_path: Path to source document
        extraction_result: Extraction result dictionary
        field_decisions: Optional field decision data
        
    Returns:
        ReviewItem ready for review
    """
    import os
    
    item_id = str(uuid.uuid4())
    document_name = os.path.basename(document_path)
    
    fields = {}
    
    # Extract fields from result
    field_data = extraction_result.get('fields', extraction_result)
    confidences = extraction_result.get('confidences', {})
    bboxes = extraction_result.get('bounding_boxes', {})
    
    for field_name, value in field_data.items():
        if field_name in ('fields', 'confidences', 'bounding_boxes', 'metadata'):
            continue
        
        field_id = f"{item_id}_{field_name}"
        confidence = confidences.get(field_name, 0.5)
        
        bbox = None
        if field_name in bboxes:
            bbox_data = bboxes[field_name]
            if isinstance(bbox_data, dict):
                bbox = BoundingBox.from_dict(bbox_data)
        
        # Get decision info if available
        decision_reason = ''
        suggestions = []
        
        if field_decisions and field_name in field_decisions:
            fd = field_decisions[field_name]
            decision_reason = fd.get('decision', '')
            suggestions = fd.get('suggestions', [])
        
        review_field = ReviewField(
            field_id=field_id,
            field_name=field_name,
            extracted_value=value,
            confidence=confidence,
            decision_reason=decision_reason,
            bbox=bbox,
            suggestions=suggestions,
        )
        
        fields[field_name] = review_field
    
    return ReviewItem(
        item_id=item_id,
        document_path=document_path,
        document_name=document_name,
        fields=fields,
        page_count=extraction_result.get('page_count', 1),
    )

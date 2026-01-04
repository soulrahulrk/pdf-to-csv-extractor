"""
Document Type Registry

Central registry for document types with auto-detection.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Type

from .document_type import DocumentType

logger = logging.getLogger(__name__)


class DocumentTypeRegistry:
    """
    Registry for document types.
    
    Provides:
    - Registration of document types
    - Auto-detection of document type from content
    - Loading/saving type definitions
    
    Usage:
        registry = DocumentTypeRegistry()
        
        # Register a type
        registry.register(my_invoice_type)
        
        # Get a type by name
        invoice_type = registry.get('invoice')
        
        # Auto-detect type from document
        doc_type = registry.detect(document_text)
    """
    
    _instance: Optional['DocumentTypeRegistry'] = None
    
    def __init__(self):
        """Initialize empty registry."""
        self._types: Dict[str, DocumentType] = {}
        self._categories: Dict[str, List[str]] = {}
    
    @classmethod
    def get_instance(cls) -> 'DocumentTypeRegistry':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def register(
        self,
        doc_type: DocumentType,
        overwrite: bool = False,
    ) -> None:
        """
        Register a document type.
        
        Args:
            doc_type: Document type to register
            overwrite: Whether to overwrite existing type
            
        Raises:
            ValueError: If type already exists and overwrite=False
        """
        if doc_type.name in self._types and not overwrite:
            raise ValueError(f"Document type '{doc_type.name}' already registered")
        
        self._types[doc_type.name] = doc_type
        
        # Update category index
        category = doc_type.category
        if category not in self._categories:
            self._categories[category] = []
        
        if doc_type.name not in self._categories[category]:
            self._categories[category].append(doc_type.name)
        
        logger.info(f"Registered document type: {doc_type.name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a document type.
        
        Args:
            name: Type name to unregister
            
        Returns:
            True if type was removed
        """
        if name in self._types:
            doc_type = self._types.pop(name)
            
            # Update category index
            category = doc_type.category
            if category in self._categories:
                self._categories[category] = [
                    n for n in self._categories[category] if n != name
                ]
            
            logger.info(f"Unregistered document type: {name}")
            return True
        
        return False
    
    def get(self, name: str) -> Optional[DocumentType]:
        """
        Get document type by name.
        
        Args:
            name: Type name
            
        Returns:
            DocumentType or None
        """
        return self._types.get(name)
    
    def get_all(self) -> List[DocumentType]:
        """Get all registered document types."""
        return list(self._types.values())
    
    def get_by_category(self, category: str) -> List[DocumentType]:
        """Get document types by category."""
        names = self._categories.get(category, [])
        return [self._types[n] for n in names if n in self._types]
    
    def list_names(self) -> List[str]:
        """List all registered type names."""
        return list(self._types.keys())
    
    def list_categories(self) -> List[str]:
        """List all categories."""
        return list(self._categories.keys())
    
    def detect(
        self,
        text: str,
        threshold: float = 0.5,
    ) -> Optional[DocumentType]:
        """
        Auto-detect document type from content.
        
        Args:
            text: Document text content
            threshold: Minimum score threshold
            
        Returns:
            Best matching DocumentType or None
        """
        best_type: Optional[DocumentType] = None
        best_score = threshold
        
        for doc_type in self._types.values():
            score = doc_type.identify_document(text)
            
            if score > best_score:
                best_score = score
                best_type = doc_type
        
        if best_type:
            logger.debug(f"Detected document type: {best_type.name} (score: {best_score:.2f})")
        
        return best_type
    
    def detect_all(
        self,
        text: str,
        threshold: float = 0.3,
    ) -> List[tuple]:
        """
        Get all matching document types with scores.
        
        Args:
            text: Document text content
            threshold: Minimum score threshold
            
        Returns:
            List of (DocumentType, score) tuples, sorted by score
        """
        matches = []
        
        for doc_type in self._types.values():
            score = doc_type.identify_document(text)
            
            if score >= threshold:
                matches.append((doc_type, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def save(self, path: str) -> None:
        """
        Save registry to JSON file.
        
        Args:
            path: Output file path
        """
        data = {
            'types': {
                name: doc_type.to_dict()
                for name, doc_type in self._types.items()
            },
            'categories': self._categories,
        }
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved registry to {path}")
    
    def load(self, path: str, merge: bool = False) -> int:
        """
        Load registry from JSON file.
        
        Args:
            path: Input file path
            merge: Whether to merge with existing types
            
        Returns:
            Number of types loaded
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not merge:
            self._types.clear()
            self._categories.clear()
        
        count = 0
        for name, type_data in data.get('types', {}).items():
            try:
                doc_type = DocumentType.from_dict(type_data)
                self.register(doc_type, overwrite=merge)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load type '{name}': {e}")
        
        logger.info(f"Loaded {count} document types from {path}")
        return count
    
    def load_from_directory(
        self,
        directory: str,
        pattern: str = "*.json",
        merge: bool = True,
    ) -> int:
        """
        Load document types from a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern (glob)
            merge: Whether to merge with existing
            
        Returns:
            Number of types loaded
        """
        path = Path(directory)
        count = 0
        
        for file_path in path.glob(pattern):
            try:
                count += self.load(str(file_path), merge=merge)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return count
    
    def clear(self) -> None:
        """Clear all registered types."""
        self._types.clear()
        self._categories.clear()
        logger.info("Registry cleared")


# Global registry instance
_global_registry = DocumentTypeRegistry()


def register_document_type(
    doc_type: DocumentType,
    overwrite: bool = False,
) -> None:
    """
    Register a document type in the global registry.
    
    Args:
        doc_type: Document type to register
        overwrite: Whether to overwrite existing
    """
    _global_registry.register(doc_type, overwrite)


def get_document_type(name: str) -> Optional[DocumentType]:
    """
    Get document type from global registry.
    
    Args:
        name: Type name
        
    Returns:
        DocumentType or None
    """
    return _global_registry.get(name)


def detect_document_type(text: str) -> Optional[DocumentType]:
    """
    Auto-detect document type from text.
    
    Args:
        text: Document text content
        
    Returns:
        Best matching DocumentType or None
    """
    return _global_registry.detect(text)


def list_document_types() -> List[str]:
    """List all registered document type names."""
    return _global_registry.list_names()


def get_registry() -> DocumentTypeRegistry:
    """Get the global registry instance."""
    return _global_registry

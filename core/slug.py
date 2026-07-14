"""
core/slug.py - Unified slug and normalization utilities.

Provides consistent string normalization across the codebase.
"""

import re


def slugify(text: str, default: str = "unnamed") -> str:
    """
    Normalize text into a URL/file-friendly slug.
    
    Converts to lowercase, replaces non-alphanumeric chars with hyphens,
    and strips leading/trailing hyphens.
    
    Args:
        text: The text to slugify
        default: Default value if result is empty
        
    Returns:
        A normalized slug string
        
    Examples:
        >>> slugify("Hello World")
        'hello-world'
        >>> slugify("File (v2).txt")
        'file-v2-txt'
        >>> slugify("   ")
        'unnamed'
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Replace non-alphanumeric with hyphens
    value = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    # Collapse multiple hyphens
    value = re.sub(r"-+", "-", value).strip("-")
    
    return value or default


def normalize_id(text: str, default: str = "unknown") -> str:
    """
    Normalize an identifier with dots and underscores preserved.
    
    Similar to slugify but preserves dots and underscores for IDs.
    
    Args:
        text: The text to normalize
        default: Default value if result is empty
        
    Returns:
        A normalized identifier string
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Allow dots and underscores in addition to alphanumerics
    value = re.sub(r"[^a-z0-9._-]+", "-", text.strip().lower())
    # Collapse multiple hyphens
    value = re.sub(r"-+", "-", value).strip("-")
    
    return value or default

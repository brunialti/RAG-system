"""
Utilities Module
----------------
This module provides miscellaneous utility functions used across the rag_system.
For example, it includes the compute_md5 function, which computes the MD5 hash of a given text (encoded in UTF-8),
useful for detecting duplicate documents.
Other helper functions may be added here as needed.
"""
import hashlib

'''
def generate_doc_id():
    """Genera un ID univoco per un documento."""
    return str(uuid.uuid4())
'''

def compute_md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()
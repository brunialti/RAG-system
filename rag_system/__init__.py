"""
rag_system Package Initialization
-----------------------------------
This module initializes the rag_system package by importing its main components:
  - VectorStorage and VectorStorageManager for storing and indexing document embeddings.
  - DocumentManager for loading and managing document content and metadata.
  - EmbeddingManager for generating embeddings and performing text chunking.
  - SearchEngine for executing various search strategies (dense, sparse, hybrid, and multi-representation).
  - Chunker for dynamic text chunking.
  - Utils for common utility functions.
  - Config for managing persistent system configuration.
It also loads the configuration upon package initialization.
"""

from .vector_storage import VectorStorage, VectorStorageManager
from .document_manager import DocumentManager, load_document
from .embedding_manager import EmbeddingManager
from .search_engine import SearchEngine
from .chunker import dynamic_chunk_text
from .utils import compute_md5
from .config import Config

Config.load()

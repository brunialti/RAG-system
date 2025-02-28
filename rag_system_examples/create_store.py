#!/usr/bin/env python3
"""
create_store.py

This script creates a new vector store for the RAG system.
It uses the EmbeddingManager to generate a dummy embedding in order to determine the embedding dimension,
and then creates a new store using the VectorStorageManager.
The store is saved in the persistent stores directory as defined in the system configuration.

Usage:
    Run the script and enter a store name when prompted.
"""

import sys
from rag_system.embedding_manager import EmbeddingManager
from rag_system.vector_storage import VectorStorageManager


def main():
    store_name = input("Enter the new store name: ").strip()
    if not store_name:
        print("Store name cannot be empty.")
        sys.exit(1)

    # Create an EmbeddingManager instance using default parameters
    embedder_key = "MiniLM"  # You can change this key as needed
    temp_manager = EmbeddingManager(model_key=embedder_key, chunk_size=50, overlap=10)

    # Use a dummy text to compute the embedding dimension
    dummy_text = "This is a test text to determine embedding dimension."
    chunks, embeddings = temp_manager.embed_text(dummy_text)
    if not embeddings or len(embeddings) == 0:
        print("Error generating dummy embedding.")
        sys.exit(1)
    embedding_dim = len(embeddings[0])

    # Create a new vector store
    storage_manager = VectorStorageManager()
    try:
        storage_manager.create_storage(store_name, embedding_dim, embedder=temp_manager.model_name)
        print(f"Store '{store_name}' created successfully with embedding dimension {embedding_dim}.")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

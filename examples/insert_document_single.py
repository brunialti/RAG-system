#!/usr/bin/env python3
"""
insert_single_document.py

This script inserts a single document into an existing vector store.
It prompts the user for the store name and the document file path.
The document is loaded using the load_document function, processed by the EmbeddingManager,
and its chunks (along with their embeddings) are added to the store.
A unique document ID is generated to track the inserted document.

Usage:
    Run the script and follow the prompts.
"""

import sys
import uuid
from rag_system.embedding_manager import EmbeddingManager
from rag_system.vector_storage import VectorStorageManager
from rag_system.document_manager import load_document


def main():
    store_name = input("Enter the store name to insert the document: ").strip()
    file_path = input("Enter the full path of the document: ").strip()
    if not store_name or not file_path:
        print("Store name and file path are required.")
        sys.exit(1)

    # Create an EmbeddingManager instance (adjust parameters as needed)
    embedder_key = "MiniLM"
    embedding_manager = EmbeddingManager(model_key=embedder_key, chunk_size=50, overlap=10)

    # Load the document content
    try:
        content = load_document(file_path)
    except Exception as e:
        print(f"Error loading document: {e}")
        sys.exit(1)
    if not content:
        print("Document is empty or could not be processed.")
        sys.exit(1)

    # Generate document chunks and embeddings
    chunks, embeddings = embedding_manager.embed_text(content, doc_name=file_path)

    # Generate a unique document ID
    doc_id = str(uuid.uuid4())

    # Load the existing store (modify base_dir as needed)
    storage_manager = VectorStorageManager()
    store = storage_manager.get_storage(store_name)
    if store is None:
        try:
            base_dir = "rag_system/persistent_storages"
            store = storage_manager.load_storage(store_name, base_dir)
        except Exception as e:
            print(f"Error loading store '{store_name}': {e}")
            sys.exit(1)

    # Insert each chunk into the store
    for chunk, emb in zip(chunks, embeddings):
        store.add_vector(emb, doc_id, chunk, source=file_path)

    print(f"Document inserted with doc_id: {doc_id}. {len(chunks)} chunks added.")


if __name__ == "__main__":
    main()

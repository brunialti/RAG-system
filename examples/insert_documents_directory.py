#!/usr/bin/env python3
"""
insert_directory_documents.py

This script inserts all documents from a specified directory into an existing vector store.
It prompts the user for the store name and the directory path.
The script recursively walks through the directory, loads each document using the load_document function,
and inserts it into the store by generating embeddings and document chunks.
Duplicate documents (based on MD5 hash) are skipped.

Usage:
    Run the script and follow the prompts.
"""

import os
import sys
import uuid
from rag_system.embedding_manager import EmbeddingManager
from rag_system.vector_storage import VectorStorageManager
from rag_system.document_manager import load_document
from rag_system.utils import compute_md5


def main():
    store_name = input("Enter the store name to insert documents: ").strip()
    dir_path = input("Enter the directory path containing documents: ").strip()
    if not store_name or not dir_path:
        print("Store name and directory path are required.")
        sys.exit(1)
    if not os.path.isdir(dir_path):
        print("The provided directory path does not exist or is not a directory.")
        sys.exit(1)

    # Create an EmbeddingManager instance (adjust parameters as needed)
    embedder_key = "MiniLM"
    embedding_manager = EmbeddingManager(model_key=embedder_key, chunk_size=50, overlap=10)

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

    inserted_docs = 0
    skipped_docs = 0

    # Walk through the directory and process each file
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                content = load_document(file_path)
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")
                continue
            if not content:
                print(f"Skipping empty document: {file_path}")
                continue

            # Check for duplicate using MD5
            md5 = compute_md5(content)
            if md5 in store.signatures:
                print(f"Skipping duplicate document: {file_path}")
                skipped_docs += 1
                continue

            # Generate document chunks and embeddings
            chunks, embeddings = embedding_manager.embed_text(content, doc_name=file)
            doc_id = str(uuid.uuid4())
            for chunk, emb in zip(chunks, embeddings):
                store.add_vector(emb, doc_id, chunk, source=file)
            store.signatures[md5] = doc_id
            inserted_docs += 1
            print(f"Inserted document {file} with doc_id {doc_id} ({len(chunks)} chunks).")

    print(f"Processing completed. Inserted documents: {inserted_docs}, Skipped duplicates: {skipped_docs}.")


if __name__ == "__main__":
    main()

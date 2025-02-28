#!/usr/bin/env python3
"""
query_store.py

This script queries an existing vector store in the RAG system.
It prompts the user for the store name and a query string, then executes the query using the SearchEngine.
For each result, it displays:
    - Document ID
    - Source (file name)
    - Score (or re-ranked score if a cross-encoder is used)
    - The first 100 characters of the content (followed by "..." if longer)
Usage:
    Run the script and follow the prompts.
"""

import sys
from rag_system.embedding_manager import EmbeddingManager
from rag_system.vector_storage import VectorStorageManager
from rag_system.search_engine import SearchEngine


def main():
    store_name = input("Enter the store name to query: ").strip()
    if not store_name:
        print("Store name is required.")
        sys.exit(1)

    query = input("Enter your query: ").strip()
    if not query:
        print("Query cannot be empty.")
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

    # Create a SearchEngine instance; here we use only the FAISS strategy for simplicity
    search_engine = SearchEngine(store, embedding_manager)

    # Execute the query
    results = search_engine.search_with_strategy(query, strategy="faiss", top_k=10, retrieval_mode="document")

    if not results:
        print("No results found for the query.")
    else:
        print("\nQuery Results:")
        for r in results:
            doc_id = r.get("doc_id")
            score = r.get("ce_score", r.get("score", 0.0))
            source = r.get("source", "Unknown")
            content = r.get("text", r.get("chunk", ""))
            # Truncate content to first 100 characters
            truncated_content = content[:100] + ("..." if len(content) > 100 else "")
            print("--------------------------------------------------")
            print(f"Document ID: {doc_id}")
            print(f"Source (File Name): {source}")
            print(f"Score: {score:.4f}")
            print("Content (first 100 characters):")
            print(truncated_content)
        print("--------------------------------------------------")
        print(f"Total results: {len(results)}")


if __name__ == "__main__":
    main()


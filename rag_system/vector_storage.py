"""
Vector Storage Module
-----------------------
This module provides functionality for storing and indexing document representations for retrieval.
It implements a multi-index approach using the following technologies:

1. FAISS:
   - Uses FAISS (IndexFlatL2) to index dense embeddings produced by SentenceTransformer.
2. BM25:
   - Uses BM25Okapi to build a sparse retrieval index over tokenized, aggregated document texts.
3. Persistent TF-IDF Index:
   - Uses scikit-learn’s TfidfVectorizer to build and persist an index computed over the aggregated texts.
   - This index is saved and loaded from disk, reducing on-the-fly computation.

Additionally, the VectorStorageManager class manages multiple persistent storages.
"""

import os
import pickle
import faiss
import numpy as np
import shutil

import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from rag_system.embedding_manager import EmbeddingManager


class VectorStorage:
    def __init__(self, embedding_dim, embedder="all-MiniLM-L6-v2", indices_present=None):
        if indices_present is None:
            # By default, both dense (faiss) and BM25 indices are enabled.
            indices_present = ["faiss", "bm25"]
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self.indices_present = indices_present

        # FAISS index for dense vector search
        self.index = faiss.IndexFlatL2(embedding_dim)

        # BM25 components for sparse search
        self.bm25_corpus = []
        self.bm25_doc_ids = []
        self.bm25 = None

        # Document-level storage:
        # documents: doc_id -> { 'chunks': [ { "chunk": ..., "source": ... }, ... ] }
        self.documents = {}
        self.signatures = {}  # MD5 hash to detect duplicates

        # Chunk-level storage:
        self.doc_ids = []      # List of doc_ids for each chunk
        self.chunks_data = []  # List of dicts: { "doc_id": ..., "chunk": ..., "source": ... }

        # Persistent TF-IDF index components
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_doc_ids = None

    def add_vector(self, vector, doc_id, chunk_text, source=None):
        # Add the vector to the FAISS index.
        vector = np.array(vector, dtype="float32").reshape(1, -1)
        self.index.add(vector)

        self.doc_ids.append(doc_id)
        self.chunks_data.append({
            "doc_id": doc_id,
            "chunk": chunk_text,
            "source": source
        })

        if doc_id not in self.documents:
            self.documents[doc_id] = {'chunks': []}
        self.documents[doc_id]['chunks'].append({
            "chunk": chunk_text,
            "source": source
        })

        # Invalidate the persistent TF-IDF index because new data has been added.
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_doc_ids = None

    def remove_document(self, doc_id, embedding_manager: EmbeddingManager):
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]

        new_doc_ids = []
        new_chunks_data = []
        for cid, cdata in zip(self.doc_ids, self.chunks_data):
            if cid != doc_id:
                new_doc_ids.append(cid)
                new_chunks_data.append(cdata)
        self.doc_ids = new_doc_ids
        self.chunks_data = new_chunks_data

        # Remove the signature for the document.
        keys_to_remove = [k for k, v in self.signatures.items() if v == doc_id]
        for k in keys_to_remove:
            del self.signatures[k]

        # Rebuild indices.
        self.rebuild_faiss_index(embedding_manager)
        if "bm25" in self.indices_present:
            self.rebuild_bm25_index()
        # Invalidate the persistent TF-IDF index.
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.tfidf_doc_ids = None

        return True

    # ------------------- FAISS Index -------------------
    def rebuild_faiss_index(self, embedding_manager: EmbeddingManager):
        if not self.chunks_data:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            return
        all_chunks = [cd["chunk"] for cd in self.chunks_data]
        embeddings = embedding_manager.model.encode(all_chunks)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(embeddings, dtype="float32"))

    def search(self, query_vector, top_k=5):
        if self.index.ntotal == 0:
            return []
        query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            cdata = self.chunks_data[idx]
            results.append({
                "doc_id": cdata["doc_id"],
                "score": float(dist),  # Distance (will be converted to similarity later)
                "chunk": cdata["chunk"],
                "source": cdata["source"]
            })
        return results

    # ------------------- BM25 Index -------------------
    def rebuild_bm25_index(self):
        if "bm25" not in self.indices_present:
            print("[DEBUG] BM25 index is not enabled in indices_present.")
            return
        doc_ids_list = []
        corpus = []
        for doc_id, doc_info in self.documents.items():
            text = " ".join(ch["chunk"] for ch in doc_info["chunks"])
            doc_ids_list.append(doc_id)
            corpus.append(text)
        print(f"[DEBUG] Rebuilding BM25 index for {len(corpus)} documents.")
        tokenized_corpus = []
        for doc_text in corpus:
            tokens = word_tokenize(doc_text.lower())
            tokenized_corpus.append(tokens)
        self.bm25_corpus = tokenized_corpus
        self.bm25_doc_ids = doc_ids_list
        self.bm25 = BM25Okapi(self.bm25_corpus)
        print("[DEBUG] BM25 index rebuilt. Corpus size:", len(self.bm25_corpus))

    def search_bm25(self, query_text, top_k=5):
        if "bm25" not in self.indices_present or not self.bm25:
            print("[DEBUG] BM25 index is not available.")
            return []
        tokens = word_tokenize(query_text.lower())
        print(f"[DEBUG] BM25 query tokens: {tokens}")
        scores = self.bm25.get_scores(tokens)
        print(f"[DEBUG] BM25 raw scores: {scores}")
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in sorted_idx:
            doc_id = self.bm25_doc_ids[idx]
            score_val = float(scores[idx])
            print(f"[DEBUG] BM25 result - doc_id: {doc_id}, score: {score_val}")
            results.append({
                "doc_id": doc_id,
                "score": score_val,
                "chunk": "document-level",  # Indicator that this is a document-level score
                "source": "bm25"
            })
        return results

    # ------------------- Persistent TF-IDF Index -------------------
    def rebuild_tfidf_index(self):
        """
        Builds a TF-IDF index at the document level by aggregating all chunks of each document.
        The index (vectorizer, matrix, and list of doc_ids) is stored in memory.
        """
        doc_ids = []
        corpus = []
        for doc_id, doc_info in self.documents.items():
            full_text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
            doc_ids.append(doc_id)
            corpus.append(full_text)
        print(f"[DEBUG] Rebuilding TF-IDF index for {len(corpus)} documents.")
        if len(corpus) == 0:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.tfidf_doc_ids = None
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        self.tfidf_doc_ids = doc_ids
        print("[DEBUG] TF-IDF index rebuilt. Vocabulary size:", len(self.tfidf_vectorizer.vocabulary_))

    def get_tfidf_index(self):
        """
        Returns a tuple (tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids).
        If the TF-IDF index has not been built, it is rebuilt.
        """
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None or self.tfidf_doc_ids is None:
            self.rebuild_tfidf_index()
        return self.tfidf_vectorizer, self.tfidf_matrix, self.tfidf_doc_ids

    # ------------------- Saving and Loading -------------------
    def save(self, storage_dir):
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        # Save FAISS index
        faiss_path = os.path.join(storage_dir, "index.faiss")
        faiss.write_index(self.index, faiss_path)
        # Save BM25 index (if present)
        bm25_path = os.path.join(storage_dir, "index_bm25.pkl")
        if "bm25" in self.indices_present and self.bm25 is not None:
            with open(bm25_path, "wb") as f_bm:
                pickle.dump({
                    "bm25_corpus": self.bm25_corpus,
                    "bm25_doc_ids": self.bm25_doc_ids
                }, f_bm)
        else:
            if os.path.exists(bm25_path):
                os.remove(bm25_path)
        # Save metadata
        meta_path = os.path.join(storage_dir, "metadata.pkl")
        meta = {
            "embedding_dim": self.embedding_dim,
            "embedder": self.embedder,
            "doc_ids": self.doc_ids,
            "chunks_data": self.chunks_data,
            "documents": self.documents,
            "signatures": self.signatures,
            "indices_present": self.indices_present
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        # Save persistent TF-IDF index
        tfidf_path = os.path.join(storage_dir, "tfidf.pkl")
        with open(tfidf_path, "wb") as f_tfidf:
            pickle.dump({
                "tfidf_vectorizer": self.tfidf_vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "tfidf_doc_ids": self.tfidf_doc_ids
            }, f_tfidf)
        return faiss_path, bm25_path, meta_path, tfidf_path

    @classmethod
    def load(cls, storage_dir):
        faiss_path = os.path.join(storage_dir, "index.faiss")
        bm25_path = os.path.join(storage_dir, "index_bm25.pkl")
        meta_path = os.path.join(storage_dir, "metadata.pkl")
        tfidf_path = os.path.join(storage_dir, "tfidf.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        embedding_dim = meta["embedding_dim"]
        embedder = meta.get("embedder", "all-MiniLM-L6-v2")
        doc_ids = meta.get("doc_ids", [])
        chunks_data = meta.get("chunks_data", [])
        documents = meta.get("documents", {})
        signatures = meta.get("signatures", {})
        indices_present = meta.get("indices_present", ["faiss"])
        storage = cls(embedding_dim, embedder=embedder, indices_present=indices_present)
        storage.index = faiss.read_index(os.path.join(storage_dir, "index.faiss"))
        storage.doc_ids = doc_ids
        storage.chunks_data = chunks_data
        storage.documents = documents
        storage.signatures = signatures
        if "bm25" in indices_present and os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f_bm:
                bm25_data = pickle.load(f_bm)
            storage.bm25_corpus = bm25_data["bm25_corpus"]
            storage.bm25_doc_ids = bm25_data["bm25_doc_ids"]
            storage.bm25 = BM25Okapi(storage.bm25_corpus)
        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f_tfidf:
                tfidf_data = pickle.load(f_tfidf)
            storage.tfidf_vectorizer = tfidf_data["tfidf_vectorizer"]
            storage.tfidf_matrix = tfidf_data["tfidf_matrix"]
            storage.tfidf_doc_ids = tfidf_data["tfidf_doc_ids"]
        return storage


class VectorStorageManager:
    def __init__(self):
        self.storages = {}

    def create_storage(self, name, embedding_dim, embedder="all-MiniLM-L6-v2"):
        if name in self.storages:
            raise ValueError(f"A storage with the name '{name}' already exists.")
        # By default, both FAISS and BM25 indices are enabled.
        new_storage = VectorStorage(embedding_dim, embedder=embedder, indices_present=["faiss", "bm25"])
        self.storages[name] = new_storage
        return new_storage

    def get_storage(self, name):
        return self.storages.get(name)

    def list_storages(self):
        return list(self.storages.keys())

    def delete_storage(self, name, base_dir):
        if name in self.storages:
            del self.storages[name]
        storage_dir = os.path.join(base_dir, name)
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        return True

    def save_storage(self, name, base_dir):
        storage = self.get_storage(name)
        if not storage:
            raise ValueError(f"Storage '{name}' not found.")
        storage_dir = os.path.join(base_dir, name)
        storage.save(storage_dir)

    def load_storage(self, name, base_dir):
        storage_dir = os.path.join(base_dir, name)
        storage = VectorStorage.load(storage_dir)
        self.storages[name] = storage
        return storage

    def list_persistent_storages(self, base_dir):
        persistent = []
        if not os.path.exists(base_dir):
            return persistent
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                meta_path = os.path.join(folder_path, "metadata.pkl")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "rb") as f:
                            meta = pickle.load(f)
                        doc_ids = meta.get("doc_ids", [])
                        documents = meta.get("documents", {})
                        persistent.append({
                            "name": folder,
                            "embedding_dim": meta.get("embedding_dim", 384),
                            "doc_ids": doc_ids,
                            "document_count": len(documents)
                        })
                    except Exception:
                        pass
        return persistent










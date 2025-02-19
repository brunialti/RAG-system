"""
vector_storage.py
-----------------
Questo modulo gestisce la persistenza degli indici FAISS, BM25 e TF-IDF, consentendo di salvare e caricare
lo stato su disco. Fornisce anche l'interfaccia per aggiungere vettori (chunk + embedding) e rimuovere documenti.

Riferimenti:
- FAISS: https://faiss.ai/
- BM25: implementazione rank_bm25
- TF-IDF: scikit-learn TfidfVectorizer
"""

import os
import pickle
import faiss
import numpy as np

import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from sklearn.feature_extraction.text import TfidfVectorizer
from .embedding_manager import EmbeddingManager

class VectorStorage:
    def __init__(self, embedding_dim, embedder="all-MiniLM-L6-v2", indices_present=None):
        if indices_present is None:
            indices_present = ["faiss", "bm25", "tfidf"]
        self.embedding_dim = embedding_dim
        self.embedder = embedder
        self.indices_present = indices_present

        # Inizializza FAISS
        self.index = faiss.IndexFlatL2(embedding_dim)

        # BM25
        self.bm25_corpus = []
        self.bm25_doc_ids = []
        self.bm25 = None

        # TF-IDF
        self.tfidf_index = None
        self.tfidf_vectorizer = None
        self.tfidf_doc_ids = []

        # Metadati
        self.documents = {}   # doc_id -> { 'chunks': [...] }
        self.signatures = {}  # MD5 -> doc_id
        self.doc_ids = []
        self.chunks_data = []

    def add_vector(self, vector, doc_id, chunk_text, source=None):
        """
        Aggiunge un vettore (embedding) e il relativo chunk nel dataset.
        """
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

    def remove_document(self, doc_id, embedding_manager: EmbeddingManager):
        """
        Rimuove un documento dal dataset e ricostruisce gli indici.
        """
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

        # rimuove la firma
        keys_to_remove = [k for k, v in self.signatures.items() if v == doc_id]
        for k in keys_to_remove:
            del self.signatures[k]

        self.rebuild_all_indices(embedding_manager)
        return True

    def rebuild_all_indices(self, embedding_manager: EmbeddingManager):
        """
        Ricostruisce tutti gli indici (FAISS, BM25, TF-IDF).
        """
        self.rebuild_faiss_index(embedding_manager)
        if "bm25" in self.indices_present:
            self.rebuild_bm25_index()
        if "tfidf" in self.indices_present:
            self.rebuild_tfidf_index()

    # FAISS
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
                "score": float(dist),
                "chunk": cdata["chunk"],
                "source": cdata["source"]
            })
        return results

    # BM25
    def rebuild_bm25_index(self):
        """
        Ricostruisce il corpus per BM25. Se non ci sono documenti,
        self.bm25 rimarrà None per evitare errori di divisione.
        """
        doc_ids_list = []
        corpus = []
        for doc_id, doc_info in self.documents.items():
            text = " ".join(ch["chunk"] for ch in doc_info["chunks"]).strip()
            if text:
                doc_ids_list.append(doc_id)
                corpus.append(text)

        if not corpus:
            # Se non ci sono documenti o testi vuoti, impostiamo bm25 a None
            self.bm25_corpus = []
            self.bm25_doc_ids = []
            self.bm25 = None
            return

        from nltk.tokenize import word_tokenize
        tokenized_corpus = [word_tokenize(doc_text.lower()) for doc_text in corpus]
        self.bm25_corpus = tokenized_corpus
        self.bm25_doc_ids = doc_ids_list

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.bm25_corpus)

    def search_bm25(self, query_text, top_k=5):
        if not self.bm25:
            return []
        tokens = word_tokenize(query_text.lower())
        scores = self.bm25.get_scores(tokens)
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in sorted_idx:
            doc_id = self.bm25_doc_ids[idx]
            results.append({
                "doc_id": doc_id,
                "score": float(scores[idx]),
                "chunk": "document-level",
                "source": "bm25"
            })
        return results

    # TF-IDF
    def rebuild_tfidf_index(self):
        corpus = []
        self.tfidf_doc_ids = []
        for doc_id, doc_info in self.documents.items():
            text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
            corpus.append(text)
            self.tfidf_doc_ids.append(doc_id)
        if corpus:
            vectorizer = TfidfVectorizer()
            self.tfidf_index = vectorizer.fit_transform(corpus)
            self.tfidf_vectorizer = vectorizer
        else:
            self.tfidf_index = None
            self.tfidf_vectorizer = None

    def get_tfidf_index(self):
        return self.tfidf_vectorizer, self.tfidf_index, self.tfidf_doc_ids

    # SALVATAGGIO / CARICAMENTO
    def save(self, storage_dir, embedding_manager: EmbeddingManager):
        self.rebuild_all_indices(embedding_manager)

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        # Salva FAISS
        faiss_path = os.path.join(storage_dir, "index.faiss")
        faiss.write_index(self.index, faiss_path)

        # Salva BM25
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

        # Salva TF-IDF
        tfidf_path = os.path.join(storage_dir, "tfidf.pkl")
        if "tfidf" in self.indices_present and self.tfidf_index is not None:
            with open(tfidf_path, "wb") as f_tfidf:
                pickle.dump({
                    "tfidf_index": self.tfidf_index,
                    "tfidf_vectorizer": self.tfidf_vectorizer,
                    "tfidf_doc_ids": self.tfidf_doc_ids
                }, f_tfidf)
        else:
            if os.path.exists(tfidf_path):
                os.remove(tfidf_path)

        # Salva metadati
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
        return faiss_path, bm25_path, tfidf_path, meta_path

    @classmethod
    def load(cls, storage_dir):
        faiss_path = os.path.join(storage_dir, "index.faiss")
        bm25_path = os.path.join(storage_dir, "index_bm25.pkl")
        tfidf_path = os.path.join(storage_dir, "tfidf.pkl")
        meta_path = os.path.join(storage_dir, "metadata.pkl")

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
        storage.doc_ids = doc_ids
        storage.chunks_data = chunks_data
        storage.documents = documents
        storage.signatures = signatures

        # Carica FAISS
        storage.index = faiss.read_index(faiss_path)

        # Carica BM25
        if "bm25" in indices_present and os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f_bm:
                bm25_data = pickle.load(f_bm)
            storage.bm25_corpus = bm25_data["bm25_corpus"]
            storage.bm25_doc_ids = bm25_data["bm25_doc_ids"]
            storage.bm25 = BM25Okapi(storage.bm25_corpus)

        # Carica TF-IDF
        if "tfidf" in indices_present and os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f_tfidf:
                tfidf_data = pickle.load(f_tfidf)
            storage.tfidf_index = tfidf_data["tfidf_index"]
            storage.tfidf_vectorizer = tfidf_data["tfidf_vectorizer"]
            storage.tfidf_doc_ids = tfidf_data["tfidf_doc_ids"]

        return storage

    def add_document_from_excel(self, file_path, embedding_manager, min_chunks=5):
        df = pd.read_excel(file_path)
        header = df.columns.tolist()

        def row_to_sentence(row):
            values = row.tolist()
            return f"Riga con {', '.join([f'{header[i]}: {values[i]}' for i in range(len(header))])}."

        rows_as_sentences = df.apply(row_to_sentence, axis=1).tolist()
        full_text_representation = "\n".join(rows_as_sentences)

        doc_id = compute_md5(full_text_representation)

        if doc_id in self.signatures.values():
            print(f"[INFO] Documento Excel già presente, saltato: {file_path}")
            return None

        if len(rows_as_sentences) <= min_chunks:
            chunks, embeddings = embedding_manager.embed_text(full_text_representation)
        else:
            chunks = rows_as_sentences
            embeddings = embedding_manager.model.encode(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            self.add_vector(embedding, doc_id, chunk, source=os.path.basename(file_path))

        self.signatures[doc_id] = doc_id

        return doc_id


class VectorStorageManager:
    def __init__(self):
        self.storages = {}

    def create_storage(self, name, embedding_dim, embedder="all-MiniLM-L6-v2"):
        if name in self.storages:
            raise ValueError(f"Esiste già uno storage con nome '{name}'.")
        new_storage = VectorStorage(embedding_dim, embedder=embedder, indices_present=["faiss","bm25","tfidf"])
        self.storages[name] = new_storage
        return new_storage

    def get_storage(self, name):
        return self.storages.get(name)

    def list_storages(self):
        return list(self.storages.keys())

    def delete_storage(self, name, base_dir):
        if name in self.storages:
            del self.storages[name]
        import shutil
        storage_dir = os.path.join(base_dir, name)
        if os.path.exists(storage_dir):
            shutil.rmtree(storage_dir)
        return True

    def save_storage(self, name, base_dir, embedding_manager: EmbeddingManager):
        storage = self.get_storage(name)
        if not storage:
            raise ValueError(f"Storage '{name}' non trovato.")
        storage_dir = os.path.join(base_dir, name)
        storage.save(storage_dir, embedding_manager)

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















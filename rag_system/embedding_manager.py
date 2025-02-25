"""
embedding_manager.py
--------------------
Questo modulo si occupa di creare le embedding dense dei testi sfruttando modelli SentenceTransformer.
In caso di testi lunghi, delega il chunking al modulo chunker.py, così da spezzare i documenti in segmenti
più piccoli (chunk) e quindi creare embedding per ogni chunk.

Riferimenti:
- SentenceTransformers: https://www.sbert.net/
"""
import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

from .chunker import dynamic_chunk_text

class EmbeddingManager:
    EMBEDDER_OPTIONS = {
        "MiniLM": "all-MiniLM-L6-v2",
        "MPNet": "all-mpnet-base-v2",
        "DistilRoBERTa": "all-distilroberta-v1"
    }

    def __init__(self, model_key="MiniLM", chunk_size=100, overlap=20):
        """
        :param model_key: chiave del modello (es. "MiniLM", "MPNet", "DistilRoBERTa") oppure il modello completo
        :param chunk_size: lunghezza massima in parole per chunk
        :param overlap: overlap in parole tra chunk consecutivi
        """
        # Se model_key è già uno dei valori usati per caricare il modello, usalo direttamente
        if model_key in self.EMBEDDER_OPTIONS.values():
            self.model_name = model_key
        else:
            self.model_name = self.EMBEDDER_OPTIONS.get(model_key, "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def embed_text(self, text: str):
        """
        Ritorna (chunks, embeddings).
        Se il testo è breve (<50 parole), viene restituito un solo chunk.
        Per file Excel (identificati dal marker [EXCEL]), ogni riga viene trattata come un chunk.
        """
        if text.startswith("[EXCEL]"):
            # Ogni linea (eccetto il marker) corrisponde a un dizionario in formato testo
            lines = text.split("\n")[1:]
            chunks = [line.strip() for line in lines if line.strip()]
        else:
            words = text.split()
            if len(words) < 50:
                chunks = [text]
            else:
                chunks = dynamic_chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)

        # Calcolo delle embedding
        embeddings = self.model.encode(chunks)

        # Normalizzazione L2 per interpretare il dot product come coseno di similarità
        embeddings = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9  # per evitare divisione per zero
        embeddings = embeddings / norms

        return chunks, embeddings

    def embed_query(self, query: str):
        """
        Ritorna l'embedding (1D) della query.
        """
        vec = self.model.encode([query])[0]
        vec = vec.astype("float32")
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            norm = 1.0
        vec /= norm
        return vec

    def set_chunk_params(self, chunk_size: int, overlap: int):
        """
        Consente di modificare a runtime i parametri di chunking.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        print(f"[DEBUG] EmbeddingManager: chunk_size={chunk_size}, overlap={overlap} aggiornati.")

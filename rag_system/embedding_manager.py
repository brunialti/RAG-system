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
from .chunker import dynamic_chunk_text

class EmbeddingManager:
    EMBEDDER_OPTIONS = {
        "MiniLM": "all-MiniLM-L6-v2",
        "MPNet": "all-mpnet-base-v2",
        "DistilRoBERTa": "all-distilroberta-v1"
    }

    def __init__(self, model_key="MiniLM", chunk_size=100, overlap=20):
        """
        :param model_key: chiave del modello (es. "MiniLM", "MPNet", "DistilRoBERTa")
        :param chunk_size: lunghezza massima in parole per chunk
        :param overlap: overlap in parole tra chunk consecutivi
        """
        model_name = self.EMBEDDER_OPTIONS.get(model_key, "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
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
        embeddings = self.model.encode(chunks)
        return chunks, embeddings

    def embed_query(self, query: str):
        """
        Ritorna l'embedding (1D) della query.
        """
        return self.model.encode([query])[0]


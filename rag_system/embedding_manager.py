"""
Embedding Manager Module
--------------------------
This module is responsible for generating text embeddings and splitting long texts into manageable chunks.
It uses SentenceTransformer models (e.g., all-MiniLM-L6-v2, all-mpnet-base-v2, all-distilroberta-v1) to produce
dense vector representations of text.
The module implements a chunking strategy that:
  - Splits text into sentences using NLTK’s sent_tokenize.
  - Groups sentences into chunks of approximately a specified number of words (chunk_size),
    with a defined overlap (overlap) between consecutive chunks.
If the text is short (e.g., fewer than 50 words), it is returned as a single chunk.
These embeddings and chunks are used later by the search engine for retrieval.
"""

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

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
        Se il testo è breve (<50 parole), un solo chunk; altrimenti chunking.
        """
        words = text.split()
        if len(words) < 50:
            chunks = [text]
        else:
            chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)
        embeddings = self.model.encode(chunks)
        return chunks, embeddings

    def embed_query(self, query: str):
        """
        Ritorna l'embedding (1D) della query.
        """
        return self.model.encode([query])[0]

def chunk_text(text, chunk_size=100, overlap=20):
    """
    Spezza il testo in chunk di ~chunk_size parole, con overlap di ~overlap parole.
    Qui usiamo la segmentazione su frasi (sent_tokenize) come base.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        sent_words = sent.split()
        sent_len = len(sent_words)
        if current_length + sent_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # overlap
            if overlap > 0:
                overlap_chunk = []
                total_overlap = 0
                for s in reversed(current_chunk):
                    s_words = s.split()
                    if total_overlap + len(s_words) > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    total_overlap += len(s_words)
            else:
                overlap_chunk = []
            current_chunk = overlap_chunk.copy()
            current_length = sum(len(x.split()) for x in current_chunk)

        current_chunk.append(sent)
        current_length += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks







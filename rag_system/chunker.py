"""
Chunker Module
--------------
This module implements dynamic text chunking functionality.
It first attempts to split the input text into sentences using NLTK’s sent_tokenize, then groups these sentences
into chunks of an approximate target length (chunk_size) with a specified overlap (overlap) between chunks.
If the sentence-based splitting results in too few chunks (less than a defined minimum), a fallback mechanism
splits the text directly based on word counts.
This ensures that long texts are divided into semantically coherent segments for subsequent embedding.
"""

import nltk
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

def dynamic_chunk_text(text, chunk_size=100, overlap=20, min_chunks=3):
    """
    Suddivide il testo in chunk in base al numero di parole,
    cercando di mantenere intatte le frasi. Se il tokenizzatore
    trova pochissime frasi ma il testo è lungo, usa un fallback
    basato sul conteggio delle parole.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    # Prova a suddividere in frasi
    sentences = sent_tokenize(text)
    if not sentences:
        return [text]

    # Chunking basato sulle frasi
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        s_words = sentence.split()
        s_len = len(s_words)
        if current_length + s_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Calcola overlap
            if overlap > 0:
                overlap_chunk = []
                total_overlap = 0
                for s in reversed(current_chunk):
                    s_w = s.split()
                    if total_overlap + len(s_w) > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    total_overlap += len(s_w)
            else:
                overlap_chunk = []
            current_chunk = overlap_chunk.copy()
            current_length = len(" ".join(current_chunk).split())
        current_chunk.append(sentence)
        current_length += s_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Se il numero di chunk è troppo basso rispetto al testo, fallback
    if len(chunks) < min_chunks and len(words) > chunk_size:
        # Fallback: chunking diretto su parole
        return fallback_chunk_by_words(words, chunk_size, overlap)
    return chunks

def fallback_chunk_by_words(words, chunk_size, overlap):
    """
    Fallback: suddivide un array di parole in chunk
    di lunghezza ~chunk_size con overlap.
    """
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(chunk_size - overlap, 1)
    return chunks

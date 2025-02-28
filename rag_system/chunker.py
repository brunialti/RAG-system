"""
chunker.py
----------
Questo modulo si occupa di suddividere (chunking) un testo lungo in porzioni più brevi e coerenti.
Utilizza un approccio basato sulle frasi con overlap. In caso di testo troppo breve (numero di chunk
inferiore a una certa soglia), fa un fallback spezzando il testo in base a un conteggio di parole.

Riferimenti:
- NLTK per sent_tokenize
- Logica di chunking basata su parole e overlap
"""

import nltk
#nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

def dynamic_chunk_text(text, chunk_size=100, overlap=20, min_chunks=3):
    """
    Suddivide il testo in chunk di lunghezza ~chunk_size parole, cercando di mantenere intatte le frasi.
    Se il numero di chunk è minore di min_chunks, fa fallback su un chunking basato su parole.

    :param text: testo completo da suddividere
    :param chunk_size: numero approssimativo di parole per chunk
    :param overlap: numero di parole sovrapposte tra chunk consecutivi
    :param min_chunks: numero minimo di chunk desiderato, se non viene raggiunto si fa fallback
    :return: lista di chunk (stringhe)
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    sentences = sent_tokenize(text)
    if not sentences:
        return [text]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        s_words = sentence.split()
        s_len = len(s_words)
        if current_length + s_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Calcolo overlap
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

    if len(chunks) < min_chunks and len(words) > chunk_size:
        # Fallback su spezzettamento basato solo sulle parole
        return fallback_chunk_by_words(words, chunk_size, overlap)
    return chunks

def fallback_chunk_by_words(words, chunk_size, overlap):
    """
    Fallback: suddivide un array di parole in chunk di lunghezza ~chunk_size con overlap.
    """
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += max(chunk_size - overlap, 1)
    return chunks


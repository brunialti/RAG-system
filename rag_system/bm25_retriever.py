# rag_system/bm25_retriever.py
from rank_bm25 import BM25Okapi
import nltk
#nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize


class BM25Retriever:
    def __init__(self, documents):
        """
        documents: lista di documenti (stringhe)
        """
        # Tokenizza i documenti in minuscolo
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_scores(self, query):
        tokenized_query = word_tokenize(query.lower())
        return self.bm25.get_scores(tokenized_query)

    def get_top_n(self, query, n=5):
        scores = self.get_scores(query)
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return top_n_indices, [scores[i] for i in top_n_indices]

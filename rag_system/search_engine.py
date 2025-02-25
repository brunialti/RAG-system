"""
search_engine.py
----------------
Questo modulo implementa le strategie di ricerca per il sistema RAG. Le strategie supportate sono:
  - faiss: ricerca densa tramite FAISS.
  - bm25: ricerca classica BM25, operante a livello documentale o a livello di chunk.
  - ibrido: fusione lineare dei segnali densi (FAISS) e sparsi (BM25).
  - multi: fusione di FAISS, BM25 e TF-IDF tramite normalizzazione e somma pesata.
  - rrf: Reciprocal Rank Fusion (RRF) che combina i ranking provenienti da FAISS, BM25 e TF-IDF.

I punteggi delle strategie che usano un singolo indice (faiss, bm25) non vengono normalizzati oltre
la conversione da distanza a similarità (FAISS) o il punteggio nativo (BM25).
Nelle strategie di fusione (ibrido, multi, rrf) avviene una normalizzazione intermedia dei punteggi
prima di combinarli.

Inoltre, se un cross-encoder è presente, si può opzionalmente normalizzare (min–max) i punteggi del re-ranking
oppure lasciarli "raw".
Se è attivata la knee detection, dopo l'ordinamento decrescente dei risultati, si applica un cutoff automatico
basato su un'analisi del punto di maggior calo (knee).

Riferimenti:
  - Fox, E.A., & Shaw, J.A. (1994). Combination of multiple searches. TREC.
  - Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and
    individual rank learning methods. SIGIR.
  - FAISS: https://faiss.ai/
  - rank_bm25 per BM25.
  - scikit-learn TfidfVectorizer per TF-IDF.
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def detect_knee(scores: List[float]) -> float:
    """
    Implementa una semplice knee detection:
      - Calcola le differenze tra punteggi consecutivi (assumendo che siano ordinati in ordine decrescente).
      - Restituisce il punteggio immediatamente successivo al punto di maggior calo (max drop).
    Se la lista è troppo corta, restituisce il primo punteggio o 0 se vuota.
    """
    if len(scores) < 2:
        return scores[0] if scores else 0.0
    diffs = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    max_drop_index = diffs.index(max(diffs))
    return scores[max_drop_index + 1]


class SearchEngine:
    def __init__(
            self,
            vector_storage,
            embedding_manager,
            cross_encoder=None,
            weight_dense=0.4,
            weight_bm25=0.3,
            weight_tfidf=0.3
    ):
        """
        :param vector_storage: VectorStorage (gestisce FAISS, BM25 e TF-IDF)
        :param embedding_manager: EmbeddingManager per generare le embedding
        :param cross_encoder: eventuale modello di re-ranking (CrossEncoder)
        :param weight_dense: peso del segnale denso (usato in ibrido e multi)
        :param weight_bm25: peso del segnale BM25 (usato in ibrido e multi)
        :param weight_tfidf: peso del segnale TF-IDF (usato in multi)
        """
        self.vector_storage = vector_storage
        self.embedding_manager = embedding_manager
        self.cross_encoder = cross_encoder
        self.weight_dense = weight_dense
        self.weight_bm25 = weight_bm25
        self.weight_tfidf = weight_tfidf

    def search_with_strategy(
            self,
            query: str,
            strategy="faiss",
            top_k=10,
            retrieval_mode="chunk",
            threshold=0.0,
            use_knee_detection: bool = False
    ) -> List[Dict[str, Any]]:
        strategy = strategy.lower().strip()
        retrieval_mode = retrieval_mode.lower().strip()
        print(f"[DEBUG] search_with_strategy called with strategy={strategy}, top_k={top_k}, retrieval_mode={retrieval_mode}, threshold={threshold}")

        if strategy == "faiss":
            results = self.search_faiss(query, top_k, retrieval_mode)
        elif strategy == "bm25":
            results = self.search_bm25(query, top_k, retrieval_mode)
        elif strategy in ("ibrido", "hybrid"):
            results = self.search_hybrid(query, top_k, retrieval_mode)
        elif strategy == "multi":
            results = self.search_multi_representation(query, top_k, "document")
        elif strategy == "rrf":
            results = self.search_rrf(query, top_k, "document")
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        print(f"[DEBUG] search_with_strategy: obtained {len(results)} results before threshold filtering")
        results.sort(key=lambda r: r.get("ce_score", r.get("score", 0.0)), reverse=True)

        if use_knee_detection and results:
            scores = [r.get("ce_score", r.get("score", 0.0)) for r in results]
            cutoff = detect_knee(scores)
            print(f"[DEBUG] Knee detection cutoff: {cutoff}")
            results = [r for r in results if r.get("ce_score", r.get("score", 0.0)) > cutoff]

        filtered = [r for r in results if r.get("ce_score", r.get("score", 0.0)) >= threshold]
        return filtered

    # ------------------- FAISS -------------------
    def search_faiss(self, query: str, top_k=10, retrieval_mode="chunk") -> List[Dict[str, Any]]:
        if "faiss" not in self.vector_storage.indices_present:
            return []
        q_vec = self.embedding_manager.embed_query(query)

        if retrieval_mode == "document":
            effective_top_k = top_k * 3
            candidate_chunks = self.vector_storage.search(q_vec, top_k=effective_top_k)
            for r in candidate_chunks:
                r["score"] = float(r["score"])
            doc_results = {}
            for r in candidate_chunks:
                doc_id = r["doc_id"]
                if doc_id not in doc_results:
                    doc_results[doc_id] = {"doc_id": doc_id, "score": r["score"], "chunks": [r["chunk"]]}
                else:
                    doc_results[doc_id]["score"] = max(doc_results[doc_id]["score"], r["score"])
                    doc_results[doc_id]["chunks"].append(r["chunk"])
            aggregated_results = []
            for doc_id, info in doc_results.items():
                full_text = " ".join(info["chunks"])
                aggregated_results.append({"doc_id": doc_id, "score": info["score"], "text": full_text})
            if self.cross_encoder:
                aggregated_results = self.rerank_with_cross_encoder(query, aggregated_results, normalize_ce=True)
            aggregated_results.sort(key=lambda x: x["score"], reverse=True)
            return aggregated_results[:top_k]
        else:
            results = self.vector_storage.search(q_vec, top_k=top_k)
            for r in results:
                r["score"] = float(r["score"])
            if self.cross_encoder:
                results = self.rerank_with_cross_encoder(query, results, normalize_ce=True)
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    # ------------------- BM25 -------------------
    def search_bm25(self, query: str, top_k=10, retrieval_mode="document") -> List[Dict[str, Any]]:
        if "bm25" not in self.vector_storage.indices_present:
            return []
        results = self.vector_storage.search_bm25(query, top_k=top_k)
        if retrieval_mode == "document":
            aggregated = []
            for r in results:
                doc_id = r["doc_id"]
                doc_info = self.vector_storage.documents.get(doc_id, {})
                full_text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
                aggregated.append({"doc_id": doc_id, "score": r["score"], "text": full_text})
            if self.cross_encoder:
                aggregated = self.rerank_with_cross_encoder(query, aggregated, normalize_ce=False)
            return aggregated
        else:
            # retrieval_mode == "chunk": restituisce direttamente i risultati a livello di chunk
            if self.cross_encoder:
                results = self.rerank_with_cross_encoder(query, results, normalize_ce=True)
            return results

    # ------------------- HYBRID (faiss + bm25) -------------------
    def search_hybrid(self, query: str, top_k=10, retrieval_mode="chunk") -> List[Dict[str, Any]]:
        print(f"[DEBUG] search_hybrid: retrieval_mode={retrieval_mode}")

        dense_res = []
        if "faiss" in self.vector_storage.indices_present:
            q_vec = self.embedding_manager.embed_query(query)
            effective_top_k = top_k * 3 if retrieval_mode == "document" else top_k
            dense_res = self.vector_storage.search(q_vec, top_k=effective_top_k)
            for r in dense_res:
                r["score"] = float(r["score"])

        bm25_res = []
        if "bm25" in self.vector_storage.indices_present:
            bm25_res = self.search_bm25(query, top_k=top_k, retrieval_mode=retrieval_mode)

        if retrieval_mode == "document":
            dense_group = {}
            for r in dense_res:
                doc_id = r["doc_id"]
                if doc_id not in dense_group:
                    dense_group[doc_id] = {"score": r["score"], "chunks": [r["chunk"]]}
                else:
                    dense_group[doc_id]["score"] = max(dense_group[doc_id]["score"], r["score"])
                    dense_group[doc_id]["chunks"].append(r["chunk"])
            hybrid_scores = {}
            for doc_id, info in dense_group.items():
                hybrid_scores[doc_id] = self.weight_dense * info["score"]
            for r in bm25_res:
                doc_id = r["doc_id"]
                hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0) + self.weight_bm25 * r["score"]
            aggregated_results = []
            for doc_id, sc in hybrid_scores.items():
                if doc_id in dense_group:
                    text = " ".join(dense_group[doc_id]["chunks"])
                else:
                    doc_info = self.vector_storage.documents.get(doc_id, {})
                    text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
                aggregated_results.append({
                    "doc_id": doc_id,
                    "score": sc,
                    "text": text,
                    "source": "hybrid"
                })
            aggregated_results.sort(key=lambda x: x["score"], reverse=True)
            aggregated_results = aggregated_results[:top_k]
            if self.cross_encoder:
                aggregated_results = self.rerank_with_cross_encoder(query, aggregated_results, normalize_ce=True)
            return aggregated_results
        else:
            best_dense_chunk = {}
            for r in dense_res:
                doc_id = r["doc_id"]
                if doc_id not in best_dense_chunk or r["score"] > best_dense_chunk[doc_id][1]:
                    best_dense_chunk[doc_id] = (r["chunk"], r["score"])
            hybrid_scores = {}
            for doc_id, (chunk_text, chunk_score) in best_dense_chunk.items():
                hybrid_scores[doc_id] = self.weight_dense * chunk_score
            for r in bm25_res:
                doc_id = r["doc_id"]
                if doc_id in best_dense_chunk:
                    hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0) + self.weight_bm25 * r["score"]
            final_results = []
            for doc_id, sc in hybrid_scores.items():
                text = best_dense_chunk[doc_id][0] if doc_id in best_dense_chunk else ""
                final_results.append({
                    "doc_id": doc_id,
                    "score": sc,
                    "chunk": text,
                    "source": "hybrid"
                })
            final_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = final_results[:top_k]
            if self.cross_encoder:
                final_results = self.rerank_with_cross_encoder(query, final_results, normalize_ce=True)
            return final_results

    # ------------------- MULTI (faiss + bm25 + TF-IDF) -------------------
    def search_multi_representation(self, query: str, top_k=10, retrieval_mode="document") -> List[Dict[str, Any]]:
        dense_candidates = self.search_faiss(query, top_k=top_k * 3, retrieval_mode="document")
        dense_dict = {d["doc_id"]: d["score"] for d in dense_candidates}
        bm25_candidates = self.search_bm25(query, top_k=top_k * 3, retrieval_mode="document")
        bm25_dict = {d["doc_id"]: d["score"] for d in bm25_candidates}

        try:
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = self.vector_storage.get_tfidf_index()
        except Exception as e:
            print(f"[DEBUG] Errore in get_tfidf_index: {e}")
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = None, None, []
        if tfidf_vectorizer is not None and tfidf_matrix is not None:
            query_vec = tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            tfidf_dict = {tfidf_doc_ids[i]: float(tfidf_scores[i]) for i in range(len(tfidf_doc_ids))}
        else:
            tfidf_dict = {}

        def normalize(score_dict):
            if not score_dict:
                return {}
            mx = max(score_dict.values())
            if mx > 0:
                return {k: v / mx for k, v in score_dict.items()}
            return score_dict

        dense_norm = normalize(dense_dict)
        bm25_norm = normalize(bm25_dict)
        tfidf_norm = normalize(tfidf_dict)
        all_doc_ids = set(dense_norm.keys()) | set(bm25_norm.keys()) | set(tfidf_norm.keys())

        combined_results = []
        for doc_id in all_doc_ids:
            sd = dense_norm.get(doc_id, 0)
            sb = bm25_norm.get(doc_id, 0)
            st = tfidf_norm.get(doc_id, 0)
            fused = self.weight_dense * sd + self.weight_bm25 * sb + self.weight_tfidf * st
            doc_info = self.vector_storage.documents.get(doc_id, {})
            full_text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
            combined_results.append({"doc_id": doc_id, "score": fused, "text": full_text})

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        combined_results = combined_results[:top_k]
        if self.cross_encoder:
            combined_results = self.rerank_with_cross_encoder(query, combined_results, normalize_ce=True)
        return combined_results

    # ------------------- RRF (Reciprocal Rank Fusion) -------------------
    def search_rrf(self, query: str, top_k=10, retrieval_mode="document") -> List[Dict[str, Any]]:
        faiss_ranking = self.search_faiss(query, top_k=top_k * 3, retrieval_mode="document")
        bm25_ranking = self.search_bm25(query, top_k=top_k * 3, retrieval_mode="document")

        try:
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = self.vector_storage.get_tfidf_index()
        except Exception as e:
            print(f"[DEBUG] Errore in get_tfidf_index: {e}")
            tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = None, None, []
        if tfidf_vectorizer is not None and tfidf_matrix is not None:
            query_vec = tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            tfidf_ranking = sorted(
                [{"doc_id": tfidf_doc_ids[i], "score": float(tfidf_scores[i])} for i in range(len(tfidf_doc_ids))],
                key=lambda x: x["score"],
                reverse=True
            )
        else:
            tfidf_ranking = []

        rrf_scores = self.reciprocal_rank_fusion(faiss_ranking, bm25_ranking, tfidf_ranking, k=60)
        combined = []
        for doc_id, score in rrf_scores.items():
            doc_info = self.vector_storage.documents.get(doc_id, {})
            full_text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
            combined.append({"doc_id": doc_id, "score": score, "text": full_text})
        combined.sort(key=lambda x: x["score"], reverse=True)
        combined = combined[:top_k]
        if self.cross_encoder:
            combined = self.rerank_with_cross_encoder(query, combined, normalize_ce=True)
        return combined

    def reciprocal_rank_fusion(self, *rankings, k=60) -> Dict[str, float]:
        rrf_scores = {}
        for ranking in rankings:
            sorted_docs = sorted(ranking, key=lambda x: x["score"], reverse=True)
            for rank, doc in enumerate(sorted_docs):
                doc_id = doc["doc_id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)
        return rrf_scores

    # ------------------- CROSS-ENCODER RERANK -------------------
    def rerank_with_cross_encoder(
            self,
            query: str,
            results: List[Dict[str, Any]],
            normalize_ce: bool = True
    ) -> List[Dict[str, Any]]:
        if not results:
            return results
        passages = [r.get("text", r.get("chunk", "")) for r in results]
        pairs = [(query, p) for p in passages]
        ce_scores = self.cross_encoder.predict(pairs)
        print(f"[DEBUG] Raw cross-encoder scores: {ce_scores}")
        ce_scores = np.array(ce_scores)
        if normalize_ce:
            if ce_scores.size > 0:
                min_score = float(ce_scores.min())
                max_score = float(ce_scores.max())
            else:
                min_score = 0.0
                max_score = 0.0
            if max_score - min_score > 1e-6:
                normalized = [(s - min_score) / (max_score - min_score) for s in ce_scores]
            else:
                normalized = [0.5] * len(ce_scores)
            for r, raw, norm in zip(results, ce_scores, normalized):
                r["ce_score"] = norm
                print(f"[DEBUG] Doc_id {r.get('doc_id')} - raw CE score: {raw}, normalized: {norm}")
        else:
            for r, raw in zip(results, ce_scores):
                r["ce_score"] = float(raw)
                print(f"[DEBUG] Doc_id {r.get('doc_id')} - raw CE score: {raw}, no normalization")
        results.sort(key=lambda x: x["ce_score"], reverse=True)
        return results





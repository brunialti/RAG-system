from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:
    def __init__(self, vector_storage, embedding_manager, cross_encoder=None,
                 weight_dense=0.4, weight_bm25=0.3, weight_tfidf=0.3):
        """
        :param vector_storage: VectorStorage (supports FAISS, BM25, and persistent TF-IDF)
        :param embedding_manager: EmbeddingManager used to generate dense embeddings
        :param cross_encoder: Optional re-ranking model (e.g., CrossEncoder)
        :param weight_dense: Weight for the dense (FAISS) signal
        :param weight_bm25: Weight for the BM25 signal
        :param weight_tfidf: Weight for the TF-IDF signal
        """
        self.vector_storage = vector_storage
        self.embedding_manager = embedding_manager
        self.cross_encoder = cross_encoder
        self.weight_dense = weight_dense
        self.weight_bm25 = weight_bm25
        self.weight_tfidf = weight_tfidf

    def search_with_strategy(self, query: str, strategy="faiss", top_k=10, retrieval_mode="chunk", threshold=0.0):
        """
        Executes the search using the specified strategy and then filters out results whose
        score (or normalized ce_score if available) is below the threshold.
        Supported strategies:
          - "faiss": Dense search using FAISS
          - "bm25": Sparse search using BM25
          - "ibrido"/"hybrid": Combination of FAISS and BM25 signals
          - "multi": Multi-representation search combining FAISS, BM25, and persistent TF-IDF

        :param query: Query string.
        :param strategy: Retrieval strategy.
        :param top_k: Number of top results to return.
        :param retrieval_mode: "chunk" or "document".
        :param threshold: Minimum score required after cross-encoder normalization.
        :return: Filtered list of results.
        """
        strategy = strategy.lower()
        print(
            f"[DEBUG] search_with_strategy called with strategy={strategy}, top_k={top_k}, retrieval_mode={retrieval_mode}, threshold={threshold}")
        if strategy == "faiss":
            results = self.search_dense(query, top_k, retrieval_mode)
        elif strategy == "bm25":
            results = self.search_bm25(query, top_k, retrieval_mode)
        elif strategy in ("ibrido", "hybrid"):
            results = self.search_hybrid(query, top_k, retrieval_mode)
        elif strategy == "multi":
            results = self.search_multi_representation(query, top_k, retrieval_mode)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        print(f"[DEBUG] search_with_strategy: obtained {len(results)} results before threshold filtering")
        filtered = [r for r in results if r.get("ce_score", r.get("score", 0)) >= threshold]
        print(f"[DEBUG] search_with_strategy: {len(filtered)} results remain after applying threshold {threshold}")
        return filtered

    # ------------------- FAISS -------------------
    def search_dense(self, query: str, top_k=10, retrieval_mode="chunk"):
        if "faiss" not in self.vector_storage.indices_present:
            return []
        q_vec = self.embedding_manager.embed_query(query)
        if retrieval_mode == "document":
            effective_top_k = top_k * 3
            candidate_chunks = self.vector_storage.search(q_vec, top_k=effective_top_k)
            for r in candidate_chunks:
                dist = r["score"]
                similarity = 1.0 / (dist + 1e-6)
                r["score"] = similarity
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
                aggregated_results = self.rerank_with_cross_encoder(query, aggregated_results)
            aggregated_results.sort(key=lambda x: x["score"], reverse=True)
            return aggregated_results[:top_k]
        else:
            results = self.vector_storage.search(q_vec, top_k=top_k)
            for r in results:
                dist = r["score"]
                similarity = 1.0 / (dist + 1e-6)
                r["score"] = similarity
            if self.cross_encoder:
                results = self.rerank_with_cross_encoder(query, results)
            return results

    # ------------------- BM25 -------------------
    def search_bm25(self, query: str, top_k=10, retrieval_mode="chunk"):
        if "bm25" not in self.vector_storage.indices_present:
            return []
        results = self.vector_storage.search_bm25(query, top_k=top_k)
        # In BM25, we now aggregate actual document text regardless of mode, as BM25 is inherently document-level.
        aggregated = []
        for r in results:
            doc_id = r["doc_id"]
            doc_info = self.vector_storage.documents.get(doc_id, {})
            full_text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
            aggregated.append({
                "doc_id": doc_id,
                "score": r["score"],
                "text": full_text,
                "source": r.get("source", "bm25")
            })
        results = aggregated
        if self.cross_encoder:
            results = self.rerank_with_cross_encoder(query, results)
        return results

    # ------------------- IBRIDO (Hybrid) -------------------
    def search_hybrid(self, query: str, top_k=10, retrieval_mode="chunk"):
        dense_res = []
        if "faiss" in self.vector_storage.indices_present:
            q_vec = self.embedding_manager.embed_query(query)
            effective_top_k = top_k * 3 if retrieval_mode == "document" else top_k
            dense_res = self.vector_storage.search(q_vec, top_k=effective_top_k)
            for r in dense_res:
                dist = r["score"]
                sim = 1.0 / (dist + 1e-6)
                r["score"] = sim
        bm25_res = []
        if "bm25" in self.vector_storage.indices_present:
            bm25_res = self.vector_storage.search_bm25(query, top_k=top_k)
        if retrieval_mode == "document":
            # Document mode: aggregate all chunks per doc.
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
                aggregated_results = self.rerank_with_cross_encoder(query, aggregated_results)
            return aggregated_results
        else:
            # Chunk mode: return only the best dense chunk per document.
            best_dense_chunk = {}
            for r in dense_res:
                doc_id = r["doc_id"]
                if doc_id not in best_dense_chunk or r["score"] > best_dense_chunk[doc_id][1]:
                    best_dense_chunk[doc_id] = (r["chunk"], r["score"])
            # Combine BM25 scores only for docs that have a dense chunk.
            hybrid_scores = {}
            for doc_id, (chunk_text, chunk_score) in best_dense_chunk.items():
                hybrid_scores[doc_id] = self.weight_dense * chunk_score
            for r in bm25_res:
                doc_id = r["doc_id"]
                if doc_id in best_dense_chunk:
                    hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0) + self.weight_bm25 * r["score"]
            final_results = []
            for doc_id, sc in hybrid_scores.items():
                text = best_dense_chunk[doc_id][0]  # Only the best chunk
                final_results.append({
                    "doc_id": doc_id,
                    "score": sc,
                    "chunk": text,
                    "source": "hybrid"
                })
            final_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = final_results[:top_k]
            if self.cross_encoder:
                final_results = self.rerank_with_cross_encoder(query, final_results)
            return final_results

    # ------------------- MULTI-REPRESENTATION INDEXING -------------------
    def search_multi_representation(self, query: str, top_k=10, retrieval_mode="document"):
        """
        Combines three signals:
          - Dense signal from FAISS
          - BM25 signal
          - Persistent TF-IDF signal
        The scores from each signal are normalized and fused using a weighted sum.
        Operates at the document level.
        """
        dense_candidates = self.search_dense(query, top_k=top_k * 3, retrieval_mode="document")
        dense_dict = {d["doc_id"]: d["score"] for d in dense_candidates}
        bm25_candidates = self.search_bm25(query, top_k=top_k * 3, retrieval_mode="document")
        bm25_dict = {d["doc_id"]: d["score"] for d in bm25_candidates}
        tfidf_vectorizer, tfidf_matrix, tfidf_doc_ids = self.vector_storage.get_tfidf_index()
        if tfidf_vectorizer is not None:
            query_vec = tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
            tfidf_dict = {tfidf_doc_ids[i]: float(tfidf_scores[i]) for i in range(len(tfidf_doc_ids))}
        else:
            tfidf_dict = {}

        def normalize(score_dict):
            if not score_dict:
                return {}
            max_score = max(score_dict.values())
            if max_score > 0:
                return {k: v / max_score for k, v in score_dict.items()}
            else:
                return score_dict

        dense_norm = normalize(dense_dict)
        bm25_norm = normalize(bm25_dict)
        tfidf_norm = normalize(tfidf_dict)
        all_doc_ids = set(dense_norm.keys()) | set(bm25_norm.keys()) | set(tfidf_norm.keys())
        combined_results = []
        for doc_id in all_doc_ids:
            score_dense = dense_norm.get(doc_id, 0)
            score_bm25 = bm25_norm.get(doc_id, 0)
            score_tfidf = tfidf_norm.get(doc_id, 0)
            final_score = (self.weight_dense * score_dense +
                           self.weight_bm25 * score_bm25 +
                           self.weight_tfidf * score_tfidf)
            doc_info = self.vector_storage.documents.get(doc_id, {})
            full_text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
            combined_results.append({
                "doc_id": doc_id,
                "score": final_score,
                "text": full_text
            })
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        if self.cross_encoder:
            combined_results = self.rerank_with_cross_encoder(query, combined_results)
        return combined_results[:top_k]

    # ------------------- CROSS-ENCODER RERANK -------------------
    def rerank_with_cross_encoder(self, query: str, results: List[Dict[str, Any]]):
        if not results:
            return results
        # Prepare passages for each result. Use 'text' if available; otherwise, use 'chunk'.
        passages = []
        for r in results:
            text = r.get("text", r.get("chunk", ""))
            # If the passage is a placeholder, try to get the real aggregated text.
            if text.strip().lower() == "document-level":
                doc_id = r.get("doc_id")
                doc_info = self.vector_storage.documents.get(doc_id, {})
                aggregated_text = " ".join(ch["chunk"] for ch in doc_info.get("chunks", []))
                text = aggregated_text
            passages.append(text)
        pairs = [(query, p) for p in passages]
        print("[DEBUG] Cross-encoder input pairs:")
        for i, (q, p) in enumerate(pairs):
            print(f" Pair {i}: Query: {q[:100]}... Passage: {p[:100]}...")
        ce_scores = self.cross_encoder.predict(pairs)
        print(f"[DEBUG] Raw cross-encoder scores: {ce_scores}")
        if ce_scores.size > 0:
            min_score = float(ce_scores.min())
            max_score = float(ce_scores.max())
            mean_score = float(ce_scores.mean())
            std_score = float(ce_scores.std())
            print(f"[DEBUG] Cross-encoder stats: min={min_score}, max={max_score}, mean={mean_score}, std={std_score}")
        else:
            print("[DEBUG] Cross-encoder returned no scores.")
        if len(set(ce_scores.tolist())) == 1:
            print("[WARN] Cross-encoder returned a constant score for all inputs. Check model and inputs!")
        if ce_scores.size > 0:
            if max_score - min_score > 1e-6:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in ce_scores]
            else:
                normalized_scores = [0.5] * len(ce_scores)
        else:
            normalized_scores = []
        for r, raw, norm in zip(results, ce_scores, normalized_scores):
            r["ce_score"] = norm
            print(f"[DEBUG] Doc_id {r.get('doc_id')}: raw score: {raw}, normalized: {norm}")
        results.sort(key=lambda x: x["ce_score"], reverse=True)
        return results


















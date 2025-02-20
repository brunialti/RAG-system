<b>RAGWIZ  - Rag system wizard<br>
Author: Roberto Brunialti<br>
Copyright: (c) 2025</b>

WARNING: This code is a Beta release. It may or may not fully meet your needs.
Use it "as is" at your own risk.
<b>Consider RAGWIZ system as a workbench to better understand RAG systems</b> and to practice with.
No extensive stress or scalability tests have been performed; nevertheless, it should work 
effectively with stores up to 500,000 chunks at least. The document ingestion module manages 
documents of the following types: .txt, .doc (with some limitations), .docx, .pdf.
It can also add  excel files but with unsatisfactory search results and with some dimensions and
format constraints.
Other document types can be added by modifying document_manager.py, provided that your tools 
can extract plain text from the desired format.

---
1) GENERAL DESCRIPTION

This program implements a Retrieval-Augmented Generation (RAG) system that indexes a document
collection using multiple retrieval methods: a dense approach (FAISS), a sparse approach (BM25),
and a persistent TF-IDF index for multi-representation retrieval. The system is designed to 
efficiently handle very heterogeneous documents—from very short to very long—and return 
relevant results for queries, optionally enriched by a cross-encoder re-ranking phase.
The system returns raw text chunks that form a “context” for a user-defined LLM 
(integration with an LLM is out of scope).

In practice:
- Long documents are split into coherent chunks using strategies based on sentence boundaries 
  and word counts. The splitting is delegated to the dedicated module chunker.py, while the 
  embedding module (embedding_manager.py) computes dense representations on these chunks.
- A dense index is built using FAISS (saved as "index.faiss"), and a BM25 index is built 
  (stored in "metadata.pkl") that aggregates full document text.
- A persistent TF-IDF index is maintained and saved (in "tfidf.pkl") to support multi-
  representation retrieval.
- When a query is made, the user can select:
    • The search strategy ("faiss", "bm25", "ibrido", "multi", or "rrf")
    • The retrieval mode ("chunk" or "document"). Note that for some strategies (e.g. BM25) 
      only document-level retrieval is supported.
- Optionally, a cross-encoder (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2") re-ranks the top-
  k candidates, applying min–max normalization to ensure consistency across signals.
- An optional knee detection algorithm is available to automatically set a cutoff threshold 
  based on the distribution of scores when multiple signals are fused (applicable only to multi-
  signal fusion strategies).
- excel files are allowed by the system, provided:
    • the table has an header
    • the file is not too big (well, I know it is somehow vague, but I did not tested the system limits...)
---

2) DESIGN AND TECHNICAL CHOICES

Dual/Multi Index:
  - FAISS: Captures semantic similarity via dense embeddings (using SentenceTransformer).
  - BM25: A classical sparse retrieval method using a bag-of-words approach (after tokenization 
    with NLTK).
  - Persistent TF-IDF: Built using scikit-learn’s TfidfVectorizer over aggregated document text; 
    stored to disk to avoid recalculation.

Retrieval Strategies:
  - faiss: Performs dense search via FAISS. Returns individual chunks or aggregated documents 
    based on the retrieval mode.
  - bm25: Retrieves documents using BM25. (Only document-level retrieval is available.)
  - ibrido (hybrid): Combines FAISS and BM25 scores via a linear weighted sum.
      In "chunk" mode, returns the best chunk per document; in "document" mode, aggregates all 
      chunks.
  - multi: Extends hybrid by fusing FAISS, BM25, and TF-IDF signals (with normalization) for 
    robust ranking.
  - rrf: Applies Reciprocal Rank Fusion (RRF) to combine rankings from FAISS, BM25, and TF-IDF.
    RRF assigns each document a score of 1/(k+rank) and sums these scores across rankings.
    (See Fox & Shaw, 1994; Cormack et al., 2009.)

Retrieval Mode & Knee Detection:
  - Depending on the selected strategy, the retrieval mode can be "chunk" or "document". For 
    example, BM25 forces document-level retrieval.
  - An optional knee detection algorithm is provided to automatically determine a cutoff 
    threshold when fusing signals.

Re-ranking:
  - An optional cross-encoder (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2") is applied to re-rank 
    the top results, with detailed debugging and min–max normalization.

---


3) MODULE STRUCTURE
<tt>
your_app_root_directory/
├── rag_system/
│   ├── persistent_stores/         # Vector store directory (created at first store creation)
│   │   └── TEST/                      # Example Vector Store (one directory for each store)
│   │       ├── index.faiss            # FAISS index
│   │       ├── index_bm25.pkl         # BM25 index
│   │       ├── index.faiss            # FAISS index
│   │       ├── tfidf.pkl              # TF-IDF index
│   │       └── metadata.pkl           # Metadata index
│   ├── models/                    # Local models (installed by setup_models.py)
│   │   ├── all-distilroberta-v1
│   │   ├── all-MiniLM-L6-v2
│   │   ├── all-mpnet-base-v2
│   │   ├── cross-encoder
│   │   └── cross-encoder_ms-marco-MiniLM-L-6-v2
│   ├── __init__.py                # Initializes the rag_system package
│   ├── bm25_retriever.py          # BM25 retrieval logic
│   ├── chunker.py                 # Dynamic text chunking functions
│   ├── README.md                  # This file
│   ├── config.py                  # Global configuration (config.json)
│   ├── document_manager.py        # Document loading and duplicate detection
│   ├── embedding_manager.py       # Splits documents and computes dense embeddings 
│   │                              # (delegates chunking to chunker.py)
│   ├── search_engine.py           # Search strategies: faiss, bm25, ibrido, multi, rrf; includes 
│   │                              # cross-encoder re-ranking and optional knee detection
│   ├── utils.py                   # Helper functions (e.g., MD5 computation)
│   └── vector_storage.py          # Manages indices (FAISS, BM25, persistent TF-IDF) and metadata
├── UI_manager.py                  # Graphical UI for managing the system
├── UI_manager.json                # UI configuration file
├── setup_models.py                # Downloads local models from HuggingFace
├── requiremnts.txt                # Required libraries
└── config.json                    # Configuration file for UI_manager.py
</tt>


Module Details:
  - __init__.py: Initializes the rag_system package.
  - bm25_retriever.py: Contains BM25 retrieval logic.
  - chunker.py: Provides functions for dynamic text chunking.
  - config.py: Manages global configuration parameters.
  - document_manager.py: Loads documents and avoids duplicates.
  - embedding_manager.py: Computes dense embeddings using SentenceTransformer;
      uses chunker.py for splitting long texts.
  - search_engine.py: Implements retrieval strategies and handles re-ranking 
      and knee detection.
  - utils.py: Contains utility functions (e.g., MD5, UUID generation).
  - vector_storage.py: Core module for storing document chunks and building indices.
  - UI_manager.py: Provides the Tkinter UI for creating/loading vector stores, 
      adding documents, and querying the system.
  - setup_models.py: Utility script to download local models for offline use.

---
4) INSTALLATION

4.1. Drop the rag_system directory "as-is" where you want to use it.
4.2. Install required libraries:
      pip install -r requirements.txt
4.3. Create the models directory (if it exists, delete it and run setup_models.py). Ensure
     setup_models.py is one level above rag_system. Then run:
      python setup_models.py

Note: The persistent_stores directory will be created automatically when you create your
first store.

---
5) HOW TO START

Once installed, the system is not fully functional until a vector store is created.
To start:
  5.1 Launch the UI (python UI_manager.py)
  5.2 Create a vector store
  5.3 Insert some test documents

![image](https://github.com/user-attachments/assets/0a151b3d-3278-4a54-9b53-97ca623aff06)

These operations can also be done via a Python script; see provided samples if available.

---
6) REFERENCES

Introductory papers:
1. https://medium.com/@alexrodriguesj/retrieval-augmented-generation-rag-with-langchain-and-faiss-a3997f95b551
2. https://div.beehiiv.com/p/advanced-rag-series-indexing

Additional references:
  - Fox, E.A., & Shaw, J.A. (1994). Combination of multiple searches. TREC.
  - Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). Reciprocal Rank Fusion.

-------------------------------------------------



</head>
<body>
  <h1>RAG SYSTEM - README</h1>
  <p><strong>Author:</strong> Roberto Brunialti</p>
  <p><strong>Copyright:</strong> (c) 2025</p>

  <hr>

  <p><strong>WARNING:</strong> This code is a Beta release. It may or may not fully meet your needs.
  Use it "as is" at your own risk. Consider <code>rag_system</code> as a workbench to better understand RAG systems and to practice with.
  No extensive stress or scalability tests have been performed; nevertheless, it should work effectively with stores up to 500,000 chunks at least.
  The document ingestion module manages documents of the following types: <code>.txt</code>, <code>.doc</code> (with some limitations), <code>.docx</code>, <code>.pdf</code>.
  It can also add Excel files but with unsatisfactory search results and with some dimensional and format constraints.
  Other document types can be added by modifying <code>document_manager.py</code>, provided that your tools can extract plain text from the desired format.</p>

  <hr>

  <h2>1) GENERAL DESCRIPTION</h2>
  <p>This program implements a Retrieval-Augmented Generation (RAG) system that indexes a document
  collection using multiple retrieval methods: a dense approach (FAISS), a sparse approach (BM25),
  and a persistent TF-IDF index for multi-representation retrieval. The system is designed to 
  efficiently handle very heterogeneous documents—from very short to very long—and return 
  relevant results for queries, optionally enriched by a cross-encoder re-ranking phase.
  The system returns raw text chunks that form a “context” for a user-defined LLM 
  (integration with an LLM is out of scope).</p>

  <p><strong>In practice:</strong></p>
  <ul>
    <li>Long documents are split into coherent chunks using strategies based on sentence boundaries 
      and word counts. The splitting is delegated to the dedicated module <code>chunker.py</code>, while the 
      embedding module (<code>embedding_manager.py</code>) computes dense representations on these chunks.</li>
    <li>A dense index is built using FAISS (saved as <code>index.faiss</code>), and a BM25 index is built 
      (stored in <code>metadata.pkl</code>) that aggregates full document text.</li>
    <li>A persistent TF-IDF index is maintained and saved (in <code>tfidf.pkl</code>) to support multi-representation retrieval.</li>
    <li>When a query is made, the user can select:
      <ul>
        <li>The search strategy ("faiss", "bm25", "ibrido", "multi", or "rrf")</li>
        <li>The retrieval mode ("chunk" or "document"). Note that for some strategies (e.g. BM25) 
          only document-level retrieval is supported.</li>
      </ul>
    </li>
    <li>Optionally, a cross-encoder re-ranks the top candidates, applying min–max normalization 
      to ensure consistency across signals.</li>
  </ul>

  <p><strong>New Functionality:</strong></p>
  <ul>
    <li><strong>New Cross Encoder Options:</strong> The system now supports multiple cross-encoder models for re-ranking:
      <ul>
        <li><strong>cross-encoder/ms-marco-MiniLM-L-6-v2:</strong> The standard cross encoder fine-tuned on the 
          MS MARCO passage ranking dataset, optimized for English re-ranking.</li>
        <li><strong>jinaai/jina-reranker-v2-base-multilingual:</strong> A state-of-the-art multilingual cross encoder 
          fine-tuned on diverse query–document pairs. This model is optimized for cross-lingual re-ranking 
          and supports multiple languages beyond English. <em>Note: This model requires setting <code>trust_remote_code=True</code></em>.</li>
        <li><strong>osiria/minilm-l6-h384-italian-cross-encoder:</strong> An Italian-specific cross encoder based on the 
          MiniLM-L6 architecture with a hidden size of 384. Fully trained on Italian text, it is designed 
          to optimize ranking performance on Italian documents.</li>
      </ul>
    </li>
    <li><strong>Enhanced Re-ranking Options:</strong> Users can now select, tramite l'interfaccia grafica, il cross encoder 
      che meglio si adatta al proprio caso d'uso (ad esempio, scegliendo il modello multilingue per query in lingue diverse 
      o il modello italiano per documenti in italiano).</li>
    <li><strong>Optional Knee Detection &amp; MMR Re-ranking:</strong> As before, the system includes a knee detection algorithm 
      to automatically determine a cutoff threshold when fusing multiple retrieval signals and an MMR-based re-ranking 
      mechanism to balance relevance with diversity.</li>
  </ul>

  <hr>

  <h2>2) DESIGN AND TECHNICAL CHOICES</h2>
  <h3>Dual/Multi Index:</h3>
  <ul>
    <li><strong>FAISS:</strong> Captures semantic similarity via dense embeddings (using SentenceTransformer).</li>
    <li><strong>BM25:</strong> A classical sparse retrieval method using a bag-of-words approach (after tokenization 
      with NLTK).</li>
    <li><strong>Persistent TF-IDF:</strong> Built using scikit-learn’s TfidfVectorizer over aggregated document text; 
      stored to disk to avoid recalculation.</li>
  </ul>

  <h3>Retrieval Strategies:</h3>
  <ul>
    <li><strong>faiss:</strong> Performs dense search via FAISS. Returns individual chunks or aggregated documents 
      based on the retrieval mode.</li>
    <li><strong>bm25:</strong> Retrieves documents using BM25. (Only document-level retrieval is available.)</li>
    <li><strong>ibrido (hybrid):</strong> Combines FAISS and BM25 scores via a linear weighted sum.
      In "chunk" mode, returns the best chunk per document; in "document" mode, aggregates all 
      chunks.</li>
    <li><strong>multi:</strong> Extends hybrid by fusing FAISS, BM25, and TF-IDF signals (with normalization) for 
      robust ranking.</li>
    <li><strong>rrf:</strong> Applies Reciprocal Rank Fusion (RRF) to combine rankings from FAISS, BM25, and TF-IDF.
      RRF assigns each document a score of 1/(k+rank) and sums these scores across rankings.
      (See Fox &amp; Shaw, 1994; Cormack et al., 2009.)</li>
  </ul>

  <h3>Retrieval Mode &amp; Knee Detection:</h3>
  <ul>
    <li>Depending on the selected strategy, the retrieval mode can be "chunk" or "document". For 
      example, BM25 forces document-level retrieval.</li>
    <li>An optional knee detection algorithm is provided to automatically determine a cutoff 
      threshold when fusing signals.</li>
  </ul>

  <h3>Re-ranking:</h3>
  <ul>
    <li>An optional cross-encoder re-ranking phase is applied to the top candidates.</li>
    <li>With the new cross encoder options available, users have more flexibility in choosing 
      the re-ranking model best suited for their language or domain requirements.</li>
    <li>Detailed debugging information and min–max normalization are applied to ensure consistency 
      across re-ranking scores.</li>
  </ul>

  <hr>

  <h2>3) MODULE STRUCTURE</h2>
  <pre>
your_app_root_directory/
├── rag_system/
│   ├── persistent_stores/         # Vector store directory (created automatically)
│   │   └── TEST/                  # Example Vector Store
│   │       ├── index.faiss        # FAISS index
│   │       ├── index_bm25.pkl     # BM25 index
│   │       ├── tfidf.pkl          # TF-IDF index
│   │       └── metadata.pkl       # Metadata index
│   ├── models/                    # Local models (installed by setup_models.py)
│   │   ├── all-distilroberta-v1
│   │   ├── all-MiniLM-L6-v2
│   │   ├── all-mpnet-base-v2
│   │   ├── cross-encoder
│   │   │   └── ms-marco-MiniLM-L-6-v2
│   │   ├── osiria
│   │   │   └── minilm-l6-h384-italian-cross-encoder
│   │   └── jinaai
│   │       └── jina-reranker-v2-base-multilingual
│   ├── __init__.py                # Initializes the rag_system package
│   ├── bm25_retriever.py          # BM25 retrieval logic
│   ├── chunker.py                 # Dynamic text chunking functions
│   ├── README.md                  # This file
│   ├── config.py                  # Global configuration (rag_system.json)
│   ├── document_manager.py        # Document loading and duplicate detection
│   ├── embedding_manager.py       # Computes dense embeddings (delegates chunking to chunker.py)
│   ├── search_engine.py           # Search strategies: faiss, bm25, ibrido, multi, rrf; includes 
│   │                              # cross-encoder re-ranking and optional knee detection
│   ├── utils.py                   # Helper functions (e.g., MD5 computation)
│   └── vector_storage.py          # Manages indices (FAISS, BM25, persistent TF-IDF) and metadata
├── UI_manager.py                  # Graphical UI for managing the system
├── UI_manager.json                # UI configuration file
├── setup_models.py                # Downloads local models from HuggingFace and configures the system
├── requirements.txt               # Required libraries
└── config.json                    # Configuration file for UI_manager.py
  </pre>

  <hr>

  <h2>4) INSTALLATION</h2>
  <h3>4.1.</h3>
  <p>Drop the <code>rag_system</code> directory "as-is" where you want to use it.</p>
  <h3>4.2.</h3>
  <p>Install required libraries:</p>
  <pre>pip install -r requirements.txt</pre>
  <h3>4.3.</h3>
  <p>Create the models directory (if it exists, delete it and run <code>setup_models.py</code>). Ensure
     <code>setup_models.py</code> is one level above <code>rag_system</code>. Then run:</p>
  <pre>python setup_models.py</pre>
  <p><em>Note: The persistent_stores directory will be created automatically when you create your first store.</em></p>

  <hr>

  <h2>5) HOW TO START</h2>
  <p>Once installed, the system is not fully functional until a vector store is created.
  To start:</p>
  <ol>
    <li>Launch the UI (<code>python UI_manager.py</code>)</li>
    <li>Create a vector store</li>
    <li>Insert some test documents</li>
  </ol>
  <p>These operations can also be done via a Python script; see provided samples if available.</p>

  ![image](https://github.com/user-attachments/assets/cab0c746-2394-4a5d-9979-52ff8ec2d5f8)

  
  ![image](https://github.com/user-attachments/assets/f5d49e73-eea5-4261-8ad1-6e5041801b24)

  <hr>

  <h2>6) REFERENCES</h2>
  <p><strong>Introductory papers:</strong></p>
  <ol>
    <li><a href="https://medium.com/@alexrodriguesj/retrieval-augmented-generation-rag-with-langchain-and-faiss-a3997f95b551">https://medium.com/@alexrodriguesj/retrieval-augmented-generation-rag-with-langchain-and-faiss-a3997f95b551</a></li>
    <li><a href="https://div.beehiiv.com/p/advanced-rag-series-indexing">https://div.beehiiv.com/p/advanced-rag-series-indexing</a></li>
  </ol>
  <p><strong>Additional references:</strong></p>
  <ul>
    <li>Fox, E.A., &amp; Shaw, J.A. (1994). Combination of multiple searches. TREC.</li>
    <li>Cormack, G.V., Clarke, C.L.A., &amp; Buettcher, S. (2009). Reciprocal Rank Fusion.</li>
    <li>FAISS: <a href="https://faiss.ai/">https://faiss.ai/</a></li>
    <li>Rank-BM25 for BM25.</li>
    <li>scikit-learn’s TfidfVectorizer for TF-IDF.</li>
  </ul>

  <hr>

  <h2>7) FILE DESCRIPTIONS</h2>
  <ul>
    <li><strong>rag_system.json:</strong> Configuration file containing global parameters for the RAG system, including paths, embedding model key, chunk size, overlap, and other settings.</li>
    <li><strong>search_engine.py:</strong> Implements various search strategies (FAISS, BM25, hybrid, multi-representation, and RRF) and includes optional cross-encoder re-ranking and knee detection functionalities.</li>
    <li><strong>utils.py:</strong> Provides utility functions such as computing MD5 checksums for strings and files, used for duplicate detection and file integrity.</li>
    <li><strong>vector_storage.py:</strong> Manages persistent storage and retrieval of vectors and metadata for FAISS, BM25, and TF-IDF indices. It handles adding, removing, and rebuilding indices.</li>
    <li><strong>config.py:</strong> Handles loading and saving of configuration settings from the <code>rag_system.json</code> file, and defines default values for various system parameters.</li>
    <li><strong>document_manager.py:</strong> Manages document ingestion, text extraction, normalization, and duplicate detection from various file formats (e.g., txt, pdf, docx, Excel).</li>
    <li><strong>embedding_manager.py:</strong> Generates dense and sparse embeddings using SentenceTransformer models, handles text chunking, and normalizes embeddings for similarity calculations.</li>
    <li><strong>UI_manager.py:</strong> Provides a graphical user interface built with Tkinter, allowing users to create and manage vector stores, add documents, perform queries, and modify system parameters.</li>
  </ul>

  <hr>
</body>
</html>


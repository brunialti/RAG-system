"""
UI_manager.py
-------------
Graphical UI (Tkinter) for the RAG system.
Features:
  - Create a store (using FAISS + BM25 + TF-IDF)
  - Load/Delete existing stores
  - Add documents (single file or directory)
  - Query using various retrieval strategies (faiss, bm25, ibrido, multi, rrf)

Retrieval strategies are shown in the combo as:
  <strategy> - <description>

When a strategy is selected, only the strategy code (before the dash) is used.
An optional knee detection cutoff can be enabled.
Additionally, the Query Store tab now features:
  - a long query input field with a search button to the right.
  - a container labeled "Retrieval Strategy" that contains:
       • the strategy combobox in column 0, row 0,
       • the mode radiobuttons ("Chunk" and "Document") side by side in column 0, row 1,
       • a label in column 1, row 0 (spanning 2 rows) that displays a detailed description of the selected strategy,
       • an info icon in column 2 (row 0 for "Enable Knee Detection" and row 1 for "Auto select strategy"),
       • and in column 3, the checkboxes for these functions aligned to the left.

In the Add Documents tab, the LabelFrame "Select doc(s) to store" contains (from left to right):
  - an input field for the path,
  - a "Browse" button,
  - and (to the right, arranged vertically) two radiobuttons ("Single Doc" and "All Docs in Folder").
"""

import rag_system
import os
import re
import json
import threading
import time
import uuid
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import traceback

from sentence_transformers import CrossEncoder
from sympy import expand

from rag_system.embedding_manager import EmbeddingManager
from rag_system.vector_storage import VectorStorageManager, align_embedding_manager
from rag_system.search_engine import SearchEngine
from rag_system.document_manager import DocumentManager, load_document
from rag_system.utils import compute_md5

STRATEGY_DESCRIPTIONS = {
    "faiss": "faiss - Dense search via FAISS",
    "bm25": "bm25 - Sparse search via BM25",
    "ibrido": "ibrido - Hybrid (dense + sparse)",
    "multi": "multi - Fusion of FAISS, BM25, TF-IDF",
    "rrf": "rrf - Reciprocal Rank Fusion"
}

# Updated detailed descriptions for each strategy.
STRATEGY_DETAILS = {
    "faiss": ("Dense search using neural embeddings. It leverages advanced semantic representations "
              "to match queries with contextually similar documents. It is ideal for complex queries that include rich contextual "
              "information, allowing the system to capture subtle nuances in language."),
    "bm25": (
        "Sparse search based on term frequency and inverse document frequency. This method is best suited for short, "
        "keyword-centric queries where precision in matching exact terms is essential. It performs well when the query consists "
        "of one or two critical words."),
    "ibrido": (
        "Hybrid search that combines the strengths of both dense and sparse retrieval methods. It merges semantic matching "
        "with exact term matching, making it suitable for queries that require a balanced approach, where both context and keywords "
        "are important."),
    "multi": (
        "Multi-representation fusion that combines scores from dense (FAISS), sparse (BM25), and TF-IDF methods. This approach leverages "
        "multiple signals to provide comprehensive search results and is ideal for complex queries where no single method is sufficient."),
    "rrf": (
        "Reciprocal Rank Fusion integrates rankings from various retrieval techniques by combining their reciprocal ranks. This robust method "
        "reduces biases from individual models and is effective across a wide range of query types, ensuring balanced results.")
}

# Descriptions for the checkboxes.
CHECKBOX_INFO_KNEE = (
    "Enable Knee Detection:\n\n"
    "When enabled, the system applies a knee detection algorithm to automatically determine a cutoff in the ranking scores, "
    "filtering out lower-quality results."
)
CHECKBOX_INFO_AUTO = (
    "Auto Select Strategy:\n\n"
    "When enabled, the system automatically selects the most appropriate retrieval strategy based on the query token number (parameter UI_QUERY_AUTO_MIN_TOKENS)"
)


class UIConfig:
    DEFAULTS = {
        "BASE_DIR": "rag_system/persistent_storages",
        "EMBEDDING_MODEL_KEY": "MiniLM",
        "CHUNK_SIZE": 100,
        "OVERLAP": 20,
        "CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "DEFAULT_TOP_K": 5,
        "DEFAULT_THRESHOLD": 0.0,
        "MIN_CHUNKS": 5,
        "UI_LAST_STORE": "",
        "UI_QUERY_AUTO_MIN_TOKENS": 3
    }
    FILENAME = "rag_system.json"  # The UI uses the same file rag_system.json

    def __init__(self):
        self.params = dict(self.DEFAULTS)
        self.load()

    def load(self):
        if os.path.exists(self.FILENAME):
            try:
                with open(self.FILENAME, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    self.params[k] = v
            except Exception as e:
                print(f"[DEBUG] Error loading config: {e}")
        else:
            self.save()

    def save(self):
        try:
            with open(self.FILENAME, "w", encoding="utf-8") as f:
                json.dump(self.params, f, indent=2)
        except Exception as e:
            print(f"[DEBUG] Error saving config: {e}")

    def __getattr__(self, item):
        if item in self.params:
            return self.params[item]
        raise AttributeError(f"UIConfig has no attribute {item}")

    def __setattr__(self, key, value):
        if key in ["params", "FILENAME", "DEFAULTS"]:
            super().__setattr__(key, value)
        else:
            self.params[key] = value


class VectorStoreManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.UIConfig = UIConfig()

        self.title("Rag System - Control Center")
        self.geometry("1000x700")
        self.current_store = None
        self.query_results_data = []

        self.label_current_store = ttk.Label(self, text="Current Store: None", foreground="blue")
        self.label_current_store.pack(pady=5)

        # Container for status bar and notebook
        self.top_container = ttk.Frame(self)
        self.top_container.pack(side=tk.TOP, fill=tk.X)

        # Status bar
        self.status_frame = ttk.Frame(self.top_container, style="Status.TFrame")
        self.status_frame.pack(side=tk.TOP, fill=tk.X)
        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor="w", background="#c0c0c0")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)
        style = ttk.Style()
        style.configure("Status.TFrame", background="#c0c0c0")

        # Notebook
        self.notebook = ttk.Notebook(self.top_container)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.embedding_manager = EmbeddingManager(
            model_key=self.UIConfig.params["EMBEDDING_MODEL_KEY"],
            chunk_size=self.UIConfig.params["CHUNK_SIZE"],
            overlap=self.UIConfig.params["OVERLAP"]
        )
        self.vector_storage_manager = VectorStorageManager()
        self.doc_manager = DocumentManager()
        self.cross_encoder = CrossEncoder(self.UIConfig.params["CROSS_ENCODER_MODEL"])

        self.add_mode_var = tk.StringVar(value="single")
        self.use_knee = tk.BooleanVar(value=False)
        self.retrieval_mode_var = tk.StringVar(value="chunk")
        self.auto_strategy = tk.BooleanVar(value=False)

        self.create_widgets()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.combo_strategy.bind("<<ComboboxSelected>>", self.on_strategy_change)
        self.on_strategy_change(None)

        last_store = self.UIConfig.params["UI_LAST_STORE"].strip()
        if last_store:
            try:
                self.load_store_by_name(last_store)
                self.notebook.select(self.tab_query)
            except Exception as e:
                print(f"[DEBUG] Unable to load '{last_store}': {e}")
                self.notebook.select(self.tab_store_mngt)
        else:
            self.notebook.select(self.tab_store_mngt)

    def set_status(self, message, color=None):
        if color:
            self.status_label.config(text=message, foreground=color)
        else:
            self.status_label.config(text=message, foreground="black")

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[10, 10])
        spacer_height = 10

        # TAB: Store Management
        self.tab_store_mngt = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_store_mngt, text="Store Mngt")
        self.store_container = ttk.Frame(self.tab_store_mngt)
        self.store_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=spacer_height)
        self.store_container.rowconfigure(0, weight=3)
        self.store_container.rowconfigure(1, weight=5)
        self.store_container.rowconfigure(2, weight=2)
        self.store_container.columnconfigure(0, weight=1)

        # LabelFrame "List Store"
        self.lf_list_store = ttk.LabelFrame(self.store_container, text="List Store", padding=(10, 5))
        self.lf_list_store.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.lf_list_store.rowconfigure(0, weight=1)
        self.lf_list_store.columnconfigure(0, weight=1)
        self.lf_list_store.columnconfigure(1, weight=0)
        self.lf_list_store.columnconfigure(2, weight=0)

        self.store_listbox = tk.Listbox(self.lf_list_store, height=10)
        self.store_listbox.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.store_listbox.bind("<<ListboxSelect>>", self.on_store_info_select)
        self.scroll_list = ttk.Scrollbar(self.lf_list_store, orient="vertical", command=self.store_listbox.yview)
        self.scroll_list.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=5)
        self.store_listbox.configure(yscrollcommand=self.scroll_list.set)

        self.frame_list_buttons = ttk.Frame(self.lf_list_store)
        self.frame_list_buttons.grid(row=0, column=2, sticky="ns", padx=5, pady=5)
        self.btn_load_store = ttk.Button(self.frame_list_buttons, text="Load Selected Store",
                                         command=self.load_selected_store)
        self.btn_load_store.pack(fill=tk.X, pady=(0, 5))
        self.btn_delete_store = ttk.Button(self.frame_list_buttons, text="Delete Selected Store",
                                           command=self.delete_store)
        self.btn_delete_store.pack(fill=tk.X)

        # LabelFrame "View store"
        self.lf_view_store = ttk.LabelFrame(self.store_container, text="View store", padding=(10, 5))
        self.lf_view_store.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.lf_view_store.rowconfigure(0, weight=1)
        self.lf_view_store.columnconfigure(0, weight=1)
        self.lf_view_store.columnconfigure(1, weight=0)
        self.lf_view_store.columnconfigure(2, weight=0)

        self.store_info_list = tk.Listbox(self.lf_view_store, height=10)
        self.store_info_list.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.store_info_list.bind("<<ListboxSelect>>", self.on_store_info_select)
        self.scroll_view = ttk.Scrollbar(self.lf_view_store, orient="vertical", command=self.store_info_list.yview)
        self.scroll_view.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=5)
        self.store_info_list.configure(yscrollcommand=self.scroll_view.set)

        self.frame_view_buttons = ttk.Frame(self.lf_view_store)
        self.frame_view_buttons.grid(row=0, column=2, sticky="ns", padx=5, pady=5)
        self.btn_delete_doc = ttk.Button(self.frame_view_buttons, text="Drop Selected doc(s)",
                                         command=self.delete_document)
        self.btn_delete_doc.pack(fill=tk.X, pady=(0, 5))
        self.btn_delete_doc["state"] = tk.DISABLED

        # LabelFrame "Create store"
        self.lf_create_store = ttk.LabelFrame(self.store_container, text="Create store", padding=(10, 5))
        self.lf_create_store.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.lf_create_store.rowconfigure(0, weight=1)
        self.lf_create_store.columnconfigure(0, weight=1)
        self.lf_create_store.columnconfigure(1, weight=0)
        self.lf_create_store.columnconfigure(2, weight=0)

        self.frame_create_store = ttk.Frame(self.lf_create_store)
        self.frame_create_store.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame_create_store.columnconfigure(1, weight=1)
        ttk.Label(self.frame_create_store, text="Store Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_store_name = ttk.Entry(self.frame_create_store)
        self.entry_store_name.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(self.frame_create_store, text="Select Embedder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.embedder_options = ["MiniLM", "MPNet", "DistilRoBERTa","multilingual-MiniLM-L12-v2"]
        self.combo_embedder = ttk.Combobox(self.frame_create_store, values=self.embedder_options, state="readonly")
        self.combo_embedder.current(0)
        self.combo_embedder.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.frame_create_buttons = ttk.Frame(self.frame_create_store)
        self.frame_create_buttons.grid(row=0, column=2, sticky="ns", padx=5, pady=5)
        self.btn_create_store = ttk.Button(self.frame_create_buttons, text=" Create New Store(s)",
                                           command=self.create_store)
        self.btn_create_store.pack(fill=tk.X)

        # TAB: Add Documents
        self.tab_add = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_add, text="Add Documents")
        add_container = ttk.Frame(self.tab_add)
        add_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=spacer_height)
        self.add_doc_frame = ttk.LabelFrame(add_container, text="Select doc(s) to store", padding=(10, 5))
        self.add_doc_frame.pack(fill=tk.X, padx=10, pady=5)
        self.add_path_entry = ttk.Entry(self.add_doc_frame)
        self.add_path_entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.add_browse_btn = ttk.Button(self.add_doc_frame, text="Browse", command=self.browse_directory_or_file)
        self.add_browse_btn.grid(row=0, column=1, padx=5, pady=5)
        self.add_doc_frame.columnconfigure(0, weight=1)
        self.add_mode_frame = ttk.Frame(self.add_doc_frame)
        self.add_mode_frame.grid(row=0, column=2, padx=10, pady=5, sticky="n")
        self.add_mode_var = tk.StringVar(value="single")
        self.rb_single_doc = ttk.Radiobutton(self.add_mode_frame, text="Single Doc", variable=self.add_mode_var,
                                             value="single")
        self.rb_single_doc.pack(anchor="w", pady=2)
        self.rb_all_docs = ttk.Radiobutton(self.add_mode_frame, text="All Docs in Folder", variable=self.add_mode_var,
                                           value="directory")
        self.rb_all_docs.pack(anchor="w", pady=2)
        self.btn_add_docs = ttk.Button(add_container, text="Add Documents", command=self.add_documents_async)
        self.btn_add_docs.pack(anchor="e", padx=10, pady=5)
        self.text_progress = tk.Text(add_container, wrap=tk.WORD)
        self.text_progress.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # TAB: Query Store
        self.tab_query = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_query, text="Query Store")
        query_container = ttk.Frame(self.tab_query)
        query_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=spacer_height)
        self.query_frame = ttk.Frame(query_container)
        self.query_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(self.query_frame, text="Enter Query:").grid(row=0, column=0, sticky="w")
        self.entry_query = ttk.Entry(self.query_frame)
        self.entry_query.grid(row=0, column=1, sticky="ew", padx=5)
        self.btn_query = ttk.Button(self.query_frame, text="Search", command=self.query_store)
        self.btn_query.grid(row=0, column=2, sticky="e", padx=5)
        self.query_frame.columnconfigure(1, weight=1)
        self.strategy_container = ttk.Frame(query_container)
        self.strategy_container.pack(fill=tk.X, padx=10, pady=5)
        self.rf_frame = ttk.LabelFrame(self.strategy_container, text="Retrieval Strategy", padding=(10, 5))
        self.rf_frame.pack(fill=tk.X, padx=10, pady=5,expand=True)
        # Set grid with four columns:
        # Column 0: combobox (row0) and radiobuttons (row1, horizontally arranged)
        # Column 1: description label (row0, spanning 2 rows and filling horizontally)
        # Column 2: info icons (one per row)
        # Column 3: checkboxes (rows 0 and 1)
        self.rf_frame.columnconfigure(0, weight=0)
        self.rf_frame.columnconfigure(1, weight=1)
        self.rf_frame.columnconfigure(2, weight=0)
        self.rf_frame.columnconfigure(3, weight=0)
        # Combobox in column 0, row 0:
        self.combo_strategy = ttk.Combobox(self.rf_frame, values=list(STRATEGY_DESCRIPTIONS.values()), state="readonly",
                                           width=50)
        self.combo_strategy.current(0)
        self.combo_strategy.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        # Radiobuttons in column 0, row 1 (horizontally arranged):
        self.mode_frame = ttk.Frame(self.rf_frame)
        self.mode_frame.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.rb_chunk = ttk.Radiobutton(self.mode_frame, text="Chunk", variable=self.retrieval_mode_var, value="chunk")
        self.rb_chunk.pack(side=tk.LEFT, padx=(0, 10))
        self.rb_document = ttk.Radiobutton(self.mode_frame, text="Document", variable=self.retrieval_mode_var,
                                           value="document")
        self.rb_document.pack(side=tk.LEFT)
        # Description label in column 1, row 0 (spanning 2 rows, fills horizontally):
        self.lbl_strategy_info = ttk.Label(self.rf_frame, text="", wraplength=1000, anchor="nw", justify="left")
        self.lbl_strategy_info.config(text=STRATEGY_DETAILS['faiss'])
        self.lbl_strategy_info.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        self.lbl_strategy_info.bind("<Configure>", self.on_resize_strategy_info)
        # Info icon and checkbox for "Enable Knee Detection" in row 0:
        self.info_icon_knee = ttk.Label(self.rf_frame, text=" ⓘ ", cursor="hand2", foreground="blue")
        self.info_icon_knee.grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.info_icon_knee.bind("<Button-1>",
                                 lambda event: messagebox.showinfo("Enable Knee Detection", CHECKBOX_INFO_KNEE))
        self.chk_knee = ttk.Checkbutton(self.rf_frame, text="Enable Knee Detection", variable=self.use_knee)
        self.chk_knee.grid(row=0, column=3, sticky="w", padx=5, pady=5)
        # Info icon and checkbox for "Auto Select Strategy" in row 1:
        self.info_icon_auto = ttk.Label(self.rf_frame, text=" ⓘ ", cursor="hand2", foreground="blue")
        self.info_icon_auto.grid(row=1, column=2, sticky="e", padx=5, pady=5)
        self.info_icon_auto.bind("<Button-1>",
                                 lambda event: messagebox.showinfo("Auto Select Strategy", CHECKBOX_INFO_AUTO))
        self.chk_auto_strategy = ttk.Checkbutton(self.rf_frame, text="Auto select strategy",
                                                 variable=self.auto_strategy)
        self.chk_auto_strategy.grid(row=1, column=3, sticky="w", padx=5, pady=5)

        self.tree_query_results = ttk.Treeview(query_container, columns=("doc_id", "doc_name", "score", "length"),
                                               show="headings")
        self.tree_query_results.heading("doc_id", text="DocID")
        self.tree_query_results.heading("doc_name", text="DocName")
        self.tree_query_results.heading("score", text="Score")
        self.tree_query_results.heading("length", text="Length (bytes)")
        self.tree_query_results.column("doc_id", width=100)
        self.tree_query_results.column("doc_name", width=150)
        self.tree_query_results.column("score", width=80)
        self.tree_query_results.column("length", width=100)
        self.tree_query_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.tree_query_results.bind("<Double-1>", self.show_result_detail)

        # TAB: Parameters
        self.tab_params = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_params, text="Parameters")
        spacer = tk.Frame(self.tab_params, height=spacer_height)
        spacer.pack()
        ttk.Label(self.tab_params, text="Parameters (double-click on value to edit):").pack(pady=5)
        self.tree_params = ttk.Treeview(self.tab_params, columns=("key", "value"), show="headings")
        self.tree_params.heading("key", text="Key")
        self.tree_params.heading("value", text="Value")
        self.tree_params.column("key", width=150, anchor="w")
        self.tree_params.column("value", width=150, anchor="w")
        self.tree_params.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        frame_params_btn = ttk.Frame(self.tab_params)
        frame_params_btn.pack(pady=5)
        self.btn_new_param = ttk.Button(frame_params_btn, text="New", command=self.new_parameter)
        self.btn_new_param.pack(side=tk.LEFT, padx=5)
        self.btn_delete_param = ttk.Button(frame_params_btn, text="Delete", command=self.delete_parameter)
        self.btn_delete_param.pack(side=tk.LEFT, padx=5)
        self.btn_save_params = ttk.Button(frame_params_btn, text="Save", command=self.save_parameters)
        self.btn_save_params.pack(side=tk.LEFT, padx=5)
        self.tree_params.bind("<Double-Button-1>", self.edit_parameter)
        self.load_parameters()

    def on_resize_strategy_info(self, event):
        """
        Called whenever the label is resized.
        We dynamically set wraplength to the new width.
        """
        new_width = event.width
        # Update the label's wraplength so it matches the label's actual width
        self.lbl_strategy_info.config(wraplength=new_width)

    def on_tab_changed(self, event):
        self.set_status("")
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == "Store Mngt":
            self.set_status("Select, view or create a store.", "green")
            self.refresh_store_list()
        elif tab_text == "Add Documents":
            self.set_status("Add documents to the store.", "green")
        elif tab_text == "Query Store":
            self.set_status("Retrieve relevant documents from the current store.", "green")
        elif tab_text == "Parameters":
            self.set_status("View and modify system params. Deletions might damage the system.", "green")

    def on_strategy_change(self, event):
        selected_strat = self.combo_strategy.get().split(" - ")[0].lower()
        if selected_strat == "bm25":
            self.retrieval_mode_var.set("chunk")
            self.rb_chunk.config(state="normal")
            self.rb_document.config(state="disabled")
        elif selected_strat in ["multi", "ibrido", "tfidf", "rrf"]:
            self.retrieval_mode_var.set("document")
            self.rb_chunk.config(state="disabled")
            self.rb_document.config(state="normal")
        else:
            self.rb_chunk.config(state="normal")
            self.rb_document.config(state="normal")
            if self.retrieval_mode_var.get() not in ["chunk", "document"]:
                self.retrieval_mode_var.set("chunk")
        self.update_strategy_info()

    def update_strategy_info(self):
        selected_strat = self.combo_strategy.get().split(" - ")[0].lower()
        desc = STRATEGY_DETAILS.get(selected_strat, "")
        self.lbl_strategy_info.config(text=desc)

    def load_store_by_name(self, store_name):
        try:
            storage = self.vector_storage_manager.get_storage(store_name)
            if storage is None:
                storage = self.vector_storage_manager.load_storage(store_name, self.UIConfig.params["BASE_DIR"])
            self.current_store = store_name
            self.label_current_store.config(text=f"Current Store: {store_name}")
            self.UIConfig.params["UI_LAST_STORE"] = store_name
            self.UIConfig.save()

            self.store_info_list.delete(0, tk.END)
            self.store_info_list.insert(tk.END, f"Store Name: {store_name}")
            self.store_info_list.insert(tk.END, f"Embedding Dimension: {storage.embedding_dim}")
            self.store_info_list.insert(tk.END, f"Embedder: {storage.embedder}")
            self.store_info_list.insert(tk.END, f"Chunk Size: {storage.chunk_size}")
            self.store_info_list.insert(tk.END, f"Overlap: {storage.overlap}")
            self.store_info_list.insert(tk.END, f"Total Chunks: {len(storage.doc_ids)}")
            self.store_info_list.insert(tk.END, f"Unique Documents: {len(storage.documents)}")
            self.store_info_list.insert(tk.END, "Documents and Files:")

            for doc_id, doc_info in storage.documents.items():
                sources = set(ch.get("source", "Unknown") for ch in doc_info["chunks"])
                sources_str = ", ".join(sources) if sources else "Unknown"
                line = f"- {doc_id}: {sources_str}"
                self.store_info_list.insert(tk.END, line)

            indices = getattr(storage, "indices_present", ["faiss"])
            allowed = []
            if "faiss" in indices:
                allowed.append("faiss")
            if "bm25" in indices:
                allowed.append("bm25")
            if all(x in indices for x in ["faiss", "bm25"]):
                allowed.append("ibrido")
            if all(x in indices for x in ["faiss", "bm25", "tfidf"]):
                allowed.append("multi")
                allowed.append("rrf")
            if not allowed:
                allowed = ["__None__"]
            allowed_list = [STRATEGY_DESCRIPTIONS[code] for code in allowed if code in STRATEGY_DESCRIPTIONS]
            self.combo_strategy.config(values=allowed_list)
            self.combo_strategy.current(0)
            self.on_strategy_change(None)

            self.verify_store_compatibility(storage)

            self.set_status("Store loaded.")
        except Exception as e:
            self.set_status(f"Error loading store:{str(e)}", "red")

    def verify_store_compatibility(self, storage):
        dummy_text = "Just to check embedding dimension."
        chunks, dummy_emb = self.embedding_manager.embed_text(dummy_text)
        if len(dummy_emb.shape) == 1:
            local_dim = dummy_emb.shape[0]
        else:
            local_dim = dummy_emb.shape[1] if dummy_emb.shape[0] > 0 else 0

        mismatch_msgs = []
        if storage.embedding_dim != local_dim:
            mismatch_msgs.append(f"Embedding dimension mismatch (Store={storage.embedding_dim}, Local={local_dim})")
        local_embedder = self.embedding_manager.model_name
        if storage.embedder != local_embedder:
            mismatch_msgs.append(f"Embedder mismatch (Store='{storage.embedder}', Local='{local_embedder}')")
        if storage.chunk_size != self.embedding_manager.chunk_size:
            mismatch_msgs.append(
                f"Chunk Size mismatch (Store={storage.chunk_size}, Local={self.embedding_manager.chunk_size})")
        if storage.overlap != self.embedding_manager.overlap:
            mismatch_msgs.append(f"Overlap mismatch (Store={storage.overlap}, Local={self.embedding_manager.overlap})")

        if mismatch_msgs:
            msg = "Detected the following mismatches:\n\n" + "\n".join(mismatch_msgs) + "\n\n"
            msg += "Do you want to align the local embedding manager parameters to the store's?\n"
            msg += "This does NOT reindex existing docs, but ensures future additions are consistent."
            answer = messagebox.askyesno("Store Mismatch", msg)
            if answer:
                self.embedding_manager.set_chunk_params(storage.chunk_size, storage.overlap)
                if storage.embedder != local_embedder or storage.embedding_dim != local_dim:
                    messagebox.showwarning(
                        "Model Mismatch",
                        "The embedding model of the current store is different from that of the Control Center.\n"
                        "For consistency, the EmbeddingManager should be reloaded with the same model as the store.\n"
                        "Currently, this automatic replacement is not supported. Please check it manually."
                    )
                else:
                    self.set_status("Local chunk parameters aligned to store.")
            else:
                self.set_status(
                    "Warning: current store and local parameters differ.\nThis could yield biased results. Use with caution.",
                    "red")

    def refresh_store_list(self):
        self.store_listbox.delete(0, tk.END)
        stores = self.vector_storage_manager.list_persistent_storages(self.UIConfig.params["BASE_DIR"])
        for store in stores:
            display_text = (
                f"{store['name']} - Embedding Dim: {store['embedding_dim']} - "
                f"Total Vectors: {len(store['doc_ids'])} - Unique Documents: {store['document_count']}"
            )
            self.store_listbox.insert(tk.END, display_text)

    def load_selected_store(self):
        selection = self.store_listbox.curselection()
        if not selection:
            self.set_status("Select a Store first!", "red")
            return
        line = self.store_listbox.get(selection[0])
        store_name = line.split(" - ")[0]
        self.load_store_by_name(store_name)

    def delete_store(self):
        selection = self.store_listbox.curselection()
        if not selection:
            self.set_status("Please select a store to delete.", "red")
            return
        line = self.store_listbox.get(selection[0])
        store_name = line.split(" - ")[0]
        confirm = messagebox.askyesno("Confirm", f"Delete store '{store_name}'?")
        if confirm:
            try:
                self.vector_storage_manager.delete_storage(store_name, self.UIConfig.params["BASE_DIR"])
                self.set_status(f"Store {store_name} has been successfully deleted.")
                if self.current_store == store_name:
                    self.current_store = None
                    self.label_current_store.config(text="Current Store: None")
                    self.UIConfig.params["UI_LAST_STORE"] = ""
                    self.UIConfig.save()
                    self.store_info_list.delete(0, tk.END)
                self.refresh_store_list()
            except Exception as e:
                self.set_status(f"Error during store deletion: {str(e)}", "red")

    def create_store(self):
        self.set_status("Creating store...")
        store_name = self.entry_store_name.get().strip()
        if not store_name:
            self.set_status("Store name cannot be empty.", "red")
            return
        embedder_key = self.combo_embedder.get()
        temp_manager = EmbeddingManager(model_key=embedder_key, chunk_size=50, overlap=10)
        dummy_text = ("Questo è un testo di prova per verificare dimensione embedding. " * 3).strip()
        chunks, dummy_embeddings = temp_manager.embed_text(dummy_text)
        if dummy_embeddings is None or dummy_embeddings.shape[0] == 0:
            self.set_status("Error generating dummy embedding.", "red")
            return
        embedding_dim = len(dummy_embeddings[0])
        if embedding_dim == 0:
            self.set_status("Error: embedding dimension is 0.", "red")
            return
        try:
            self.vector_storage_manager.create_storage(store_name, embedding_dim, embedder=temp_manager.model_name)
            self.set_status("Index saving...")
            self.vector_storage_manager.save_storage(store_name, self.UIConfig.params["BASE_DIR"],
                                                     self.embedding_manager)
            self.refresh_store_list()
            self.load_store_by_name(store_name)
            self.set_status(f"Store {store_name} created using embedder {embedder_key}.")
        except Exception as e:
            traceback.print_exc()
            self.set_status("Error during store creation.", "red")
        self.set_status(f"New store {store_name} created.")

    def on_store_info_select(self, event):
        selection = self.store_info_list.curselection()
        if not selection:
            self.btn_delete_doc["state"] = tk.DISABLED
            return
        line = self.store_info_list.get(selection[0]).strip()
        pattern = r"^- ([0-9a-fA-F-]+): (.+)$"
        if re.match(pattern, line):
            self.btn_delete_doc["state"] = tk.NORMAL
        else:
            self.btn_delete_doc["state"] = tk.DISABLED

    def delete_document(self):
        selection = self.store_info_list.curselection()
        if not selection:
            self.set_status("No document selected for deletion.", "red")
            return
        line = self.store_info_list.get(selection[0]).strip()
        pattern = r"^- ([0-9a-fA-F-]+): (.+)$"
        match = re.match(pattern, line)
        if not match:
            self.set_status("Invalid document selection.", "red")
            return
        doc_id = match.group(1)
        confirm = messagebox.askyesno("Confirm", f"Are you sure you want to delete document '{doc_id}'?")
        if not confirm:
            self.set_status("Document deletion cancelled.")
            return
        try:
            storage = self.vector_storage_manager.get_storage(self.current_store)
            if storage is None:
                storage = self.vector_storage_manager.load_storage(self.current_store, self.UIConfig.params["BASE_DIR"])
            success = storage.remove_document(doc_id, self.embedding_manager)
            if success:
                self.vector_storage_manager.save_storage(self.current_store, self.UIConfig.params["BASE_DIR"],
                                                         self.embedding_manager)
                self.set_status("Document deletion completed.")
                self.load_store_by_name(self.current_store)
            else:
                self.set_status("Document not found.", "red")
        except Exception as e:
            traceback.print_exc()
            self.set_status("Error during document deletion.", "red")

    def browse_directory_or_file(self):
        if self.add_mode_var.get() == "single":
            path = filedialog.askopenfilename()
        else:
            path = filedialog.askdirectory()
        if path:
            self.add_path_entry.delete(0, tk.END)
            self.add_path_entry.insert(0, path)

    def add_documents_async(self):
        if not self.current_store:
            self.set_status("Load a Store first!", "red")
            return
        path_input = self.add_path_entry.get().strip()
        if not path_input or not os.path.exists(path_input):
            self.set_status("Invalid file or directory selected.", "red")
            return
        self.text_progress.delete("1.0", tk.END)
        t = threading.Thread(target=self._add_documents_worker, args=(path_input,))
        t.start()

    def _add_documents_worker(self, path_input):
        start_time = time.time()
        storage = self.vector_storage_manager.get_storage(self.current_store)
        if storage is None:
            storage = self.vector_storage_manager.load_storage(self.current_store, self.UIConfig.params["BASE_DIR"])

        if (storage.embedder != self.embedding_manager.model_name or
                storage.chunk_size != self.embedding_manager.chunk_size or
                storage.overlap != self.embedding_manager.overlap):
            print("WARNING: Parameter mismatch detected. New documents will be processed using the store's parameters.")
            embedding_manager_to_use = EmbeddingManager(model_key=storage.embedder,
                                                        chunk_size=storage.chunk_size,
                                                        overlap=storage.overlap)
        else:
            embedding_manager_to_use = self.embedding_manager

        files_to_process = []
        if self.add_mode_var.get() == "single":
            if os.path.isfile(path_input):
                files_to_process = [path_input]
            else:
                self._append_progress("ERROR: Selected path is not a file.\n")
                self.set_status("Invalid file selected.", "red")
                return
        else:
            if os.path.isdir(path_input):
                for root, dirs, files in os.walk(path_input):
                    for f in files:
                        files_to_process.append(os.path.join(root, f))
                if len(files_to_process) > 1:
                    ans = messagebox.askyesno("Confirm",
                                              f"Are you sure you want to load {len(files_to_process)} files?")
                    if not ans:
                        self._append_progress("Operation cancelled by user.\n")
                        self.set_status("Document loading cancelled.")
                        return
            else:
                self._append_progress("ERROR: Selected path is not a directory.\n")
                self.set_status("Invalid directory selected.", "red")
                return

        total_files = len(files_to_process)
        total_docs_added = 0
        total_docs_skipped = 0
        total_chunks = 0
        total_duplicates = 0
        total_processed_kb = 0.0

        for file_index, file_path in enumerate(files_to_process, start=1):
            try:
                file_size_kb = os.path.getsize(file_path) / 1024.0
                self._append_progress(f"Processing: {file_path} ({file_size_kb:.1f} KB)\n")
                try:
                    content = load_document(file_path)
                except Exception as e:
                    self._append_progress(f"  ⚠️ Skipped. Cannot read document: {e}\n")
                    continue
                if not content:
                    self._append_progress("  ⚠️ Skipped. Empty or unsupported.\n")
                    continue

                total_processed_kb += file_size_kb

                signature = compute_md5(content)
                if signature in storage.signatures:
                    total_docs_skipped += 1
                    total_duplicates += 1
                    self._append_progress("  ⚠️ Skipped. Document duplicated (same MD5).\n")
                    continue
                doc_id = str(uuid.uuid4())
                source_file = os.path.basename(file_path)
                chunks, embeddings = embedding_manager_to_use.embed_text(content)
                chunk_count = len(chunks)
                for chunk, emb in zip(chunks, embeddings):
                    storage.add_vector(emb, doc_id, chunk, source=source_file)
                storage.signatures[signature] = doc_id
                total_docs_added += 1
                total_chunks += chunk_count
                self._append_progress(f"  Inserted doc_id={doc_id}, {chunk_count} chunks.\n")
            except Exception as e:
                traceback.print_exc()
                self._append_progress(f"  ERROR: {e}\n")
            self.after(0,
                       lambda i=file_index, total=total_files: self.set_status(f"Document {i} of {total} processed."))

        if "bm25" in storage.indices_present:
            storage.rebuild_bm25_index()
        self.set_status("Index saving...")
        self.vector_storage_manager.save_storage(self.current_store, self.UIConfig.params["BASE_DIR"],
                                                 self.embedding_manager)
        self.set_status("Index saved. Documents added, indexes built and saved.")
        elapsed = time.time() - start_time
        dup_percentage = (total_duplicates / (total_chunks + total_duplicates)) * 100 if (
                                                                                                     total_chunks + total_duplicates) > 0 else 0.0
        summary = (
            "\n==== Summary ====\n"
            f"Inserted docs: {total_docs_added}\n"
            f"Skipped docs (MD5 duplicates): {total_docs_skipped}\n"
            f"Total chunks generated: {total_chunks}\n"
            f"Duplicate chunks: {total_duplicates} ({dup_percentage:.2f}% of chunk attempts)\n"
            f"Total processed Kb: {total_processed_kb:.2f} KB\n"
            f"Elapsed time: {elapsed:.2f} s\n"
            "=================\n\nBuilding indexes..."
        )
        self._append_progress(summary)
        self.after(0, lambda: self.load_store_by_name(self.current_store))
        self.after(0, lambda: self.set_status("Documents added, indexes built and saved."))

    def _append_progress(self, text):
        def append():
            self.text_progress.insert(tk.END, text)
            self.text_progress.see(tk.END)

        self.after(0, append)

    def query_store(self):
        if not self.current_store:
            self.set_status("Load a Store first!", "red")
            return
        query_text = self.entry_query.get().strip()
        if not query_text:
            self.set_status("Enter a query.")
            return

        self.set_status("Query executing...")
        selected_mode = self.retrieval_mode_var.get()
        selected_strategy = self.combo_strategy.get().split(" - ")[0]
        t = threading.Thread(target=self.run_query, args=(query_text, selected_mode, selected_strategy))
        t.start()

    def run_query(self, query_text, mode, strategy):
        if self.auto_strategy.get():
            tokens = query_text.split()
            if len(tokens) < self.UIConfig.params["UI_QUERY_AUTO_MIN_TOKENS"]:
                strategy = "bm25"
            else:
                strategy = "faiss"
            self.combo_strategy.set(STRATEGY_DESCRIPTIONS[strategy])
            self.update_strategy_info()
        try:
            storage = self.vector_storage_manager.get_storage(self.current_store)
            if storage is None:
                storage = self.vector_storage_manager.load_storage(self.current_store, self.UIConfig.params["BASE_DIR"])
            search_engine = SearchEngine(storage, self.embedding_manager, cross_encoder=self.cross_encoder)
            results = search_engine.search_with_strategy(
                query=query_text,
                strategy=strategy,
                top_k=self.UIConfig.params["DEFAULT_TOP_K"],
                retrieval_mode=mode,
                threshold=self.UIConfig.params["DEFAULT_THRESHOLD"],
                use_knee_detection=self.use_knee.get()
            )
            self.query_results_data = results

            def update_results():
                for item in self.tree_query_results.get_children():
                    self.tree_query_results.delete(item)
                if not self.query_results_data:
                    self.tree_query_results.insert("", tk.END, values=("No results found.", "", "", ""))
                    return
                for i, r in enumerate(results):
                    doc_id = r["doc_id"]
                    doc_info = storage.documents.get(doc_id, {})
                    sources = {ch.get("source", "Unknown") for ch in doc_info.get("chunks", [])}
                    doc_name = ", ".join(sources) if sources else "Unknown"
                    result_text = r.get("text", r.get("chunk", ""))
                    length_bytes = len(result_text.encode("utf-8"))
                    final_score = r.get("ce_score", r.get("score", 0.0))
                    self.tree_query_results.insert("", tk.END, iid=str(i), values=(
                        doc_id,
                        doc_name,
                        f"{final_score:.2f}",
                        f"{length_bytes}"
                    ))

            self.after(0, update_results)
            self.set_status("Query executed.")
        except Exception as e:
            traceback.print_exc()
            self.set_status("Error during query execution.", "red")

    def show_result_detail(self, event):
        selected = self.tree_query_results.selection()
        if not selected:
            return
        idx = int(selected[0])
        if idx >= len(self.query_results_data):
            return
        result = self.query_results_data[idx]
        detail_win = tk.Toplevel(self)
        detail_win.title("Result Detail")
        detail_win.geometry("600x400")
        text_widget = tk.Text(detail_win, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)
        if "text" in result:
            detail_text = (
                f"Doc ID: {result['doc_id']}\n"
                f"Score: {result.get('ce_score', result.get('score', 0.0)):.2f}\n\n"
                f"Document Content:\n{result['text']}\n"
            )
        else:
            detail_text = (
                f"Doc ID: {result.get('doc_id', '?')}\n"
                f"Score: {result.get('ce_score', result.get('score', 0.0)):.2f}\n"
                f"Source: {result.get('source', '?')}\n\n"
                f"Chunk Content:\n{result.get('chunk', '')}\n"
            )
        text_widget.insert(tk.END, detail_text)
        text_widget.config(state=tk.DISABLED)

    def load_parameters(self):
        for item in self.tree_params.get_children():
            self.tree_params.delete(item)
        for key, value in self.UIConfig.params.items():
            self.tree_params.insert("", tk.END, iid=key, values=(key, value))

    def new_parameter(self):
        key = simpledialog.askstring("New Parameter", "Enter new parameter key:")
        if not key:
            return
        if key in self.UIConfig.params:
            self.set_status(f"Error: Key '{key}' already exists.", "red")
            return
        value = simpledialog.askstring("New Parameter", f"Enter value for {key}:")
        if value is None:
            return
        try:
            new_val = int(value)
        except ValueError:
            try:
                new_val = float(value)
            except ValueError:
                new_val = value
        self.UIConfig.params[key] = new_val
        self.load_parameters()

    def delete_parameter(self):
        selected = self.tree_params.selection()
        if not selected:
            self.set_status("Error: No parameter selected.", "red")
            return
        key = selected[0]
        confirm = messagebox.askyesno("Delete", f"Are you sure you want to delete the parameter '{key}'?")
        if confirm:
            if key in self.UIConfig.params:
                del self.UIConfig.params[key]
            self.load_parameters()

    def edit_parameter(self, event):
        region = self.tree_params.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.tree_params.identify_column(event.x)
        if col != "#2":
            return
        item = self.tree_params.identify_row(event.y)
        if not item:
            return
        current_val = self.tree_params.set(item, "value")
        new_val = simpledialog.askstring("Edit Parameter", f"Enter new value for {item}:", initialvalue=current_val)
        if new_val is None:
            return
        try:
            new_val_converted = int(new_val)
        except ValueError:
            try:
                new_val_converted = float(new_val)
            except ValueError:
                new_val_converted = new_val
        self.UIConfig.params[item] = new_val_converted
        self.load_parameters()

    def save_parameters(self):
        self.UIConfig.save()
        self.embedding_manager.chunk_size = self.UIConfig.params["CHUNK_SIZE"]
        self.embedding_manager.overlap = self.UIConfig.params["OVERLAP"]
        expected_model_name = EmbeddingManager.EMBEDDER_OPTIONS.get(self.UIConfig.params["EMBEDDING_MODEL_KEY"])
        if self.embedding_manager.model_name != expected_model_name:
            print(
                "WARNING: EMBEDDING_MODEL_KEY has been changed in configuration, but the current EmbeddingManager is not reloaded automatically.")
        self.load_parameters()
        self.set_status("Parameters saved to rag_system.json.")


if __name__ == "__main__":
    base_dir = UIConfig().params["BASE_DIR"]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    app = VectorStoreManagerApp()
    app.mainloop()

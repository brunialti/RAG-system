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
  - A long query input field with a search button to the right.
  - A container labeled "Retrieval Strategy" that contains:
       • the strategy combobox,
       • a vertical stack of two radiobuttons ("Chunk" and "Document"),
       • and, in a new row, the "Enable Knee Detection" checkbox.

In the Add Documents tab, the LabelFrame "Select doc(s) to store" contains (from left to right):
  - an input field for the path,
  - a "Browse" button,
  - and (to the right, arranged vertically) two radiobuttons ("Single Doc" and "All Docs in Folder").
"""
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import traceback

from sentence_transformers import CrossEncoder
from sympy import expand

from rag_system.embedding_manager import EmbeddingManager
from rag_system.vector_storage import VectorStorageManager
from rag_system.search_engine import SearchEngine
from rag_system.document_manager import DocumentManager, load_document
from rag_system.utils import compute_md5

# Mapping of strategy codes to descriptions
STRATEGY_DESCRIPTIONS = {
    "faiss": "faiss - Dense search via FAISS",
    "bm25": "bm25 - Sparse search via BM25",
    "ibrido": "ibrido - Hybrid (dense + sparse)",
    "multi": "multi - Fusion of FAISS, BM25, TF-IDF",
    "rrf": "rrf - Reciprocal Rank Fusion"
}


class UIConfig:
    DEFAULTS = {
        "BASE_DIR": "rag_system/persistent_stores",
        "EMBEDDING_MODEL_KEY": "MiniLM",
        "CHUNK_SIZE": 100,
        "OVERLAP": 20,
        "CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "DEFAULT_TOP_K": 5,
        "DEFAULT_THRESHOLD": 0.0,
        "LAST_STORE": "",
        "MIN_CHUNKS": 5
    }
    FILENAME = "UI_manager.json"

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

        self.title("Vector Store Manager - UI")
        self.geometry("1000x800")
        self.current_store = None
        self.query_results_data = []

        self.label_current_store = ttk.Label(self, text="Current Store: None", foreground="blue")
        self.label_current_store.pack(pady=5)

        # Crea un container per status bar e notebook: la status bar viene posizionata subito sotto il titolo.
        self.top_container = ttk.Frame(self)
        self.top_container.pack(side=tk.TOP, fill=tk.X)

        # Status bar (con sfondo grigio leggermente più scuro)
        self.status_frame = ttk.Frame(self.top_container, style="Status.TFrame")
        self.status_frame.pack(side=tk.TOP, fill=tk.X)
        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor="w", background="#c0c0c0")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.X, expand=True)
        style = ttk.Style()
        style.configure("Status.TFrame", background="#c0c0c0")

        # Notebook sotto la status bar
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

        self.create_widgets()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.combo_strategy.bind("<<ComboboxSelected>>", self.on_strategy_change)
        self.on_strategy_change(None)

        last_store = self.UIConfig.params["LAST_STORE"].strip()
        if last_store:
            try:
                self.load_store_by_name(last_store)
                self.notebook.select(self.tab_query)
            except Exception as e:
                print(f"[DEBUG] Unable to load '{last_store}': {e}")
                self.notebook.select(self.tab_store_mngt)
        else:
            self.notebook.select(self.tab_store_mngt)

    def set_status(self, message):
        self.status_label.config(text=message)

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[10, 10])
        spacer_height = 10

        # --- Merged Tab: Store Mngt ---
        self.tab_store_mngt = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_store_mngt, text="Store Mngt")
        self.store_container = ttk.Frame(self.tab_store_mngt)
        self.store_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=spacer_height)
        self.store_container.rowconfigure(0, weight=3)  # 30%
        self.store_container.rowconfigure(1, weight=5)  # 50%
        self.store_container.rowconfigure(2, weight=2)  # 20%
        self.store_container.columnconfigure(0, weight=1)

        # LabelFrame "List Store" (row 0)
        self.lf_list_store = ttk.LabelFrame(self.store_container, text="List Store", padding=(10, 5))
        self.lf_list_store.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.lf_list_store.columnconfigure(0, weight=1)
        self.lf_list_store.columnconfigure(1, weight=0)
        self.frame_list_store = ttk.Frame(self.lf_list_store)
        self.frame_list_store.grid(row=0, column=0, sticky="nsew")
        self.frame_list_store.rowconfigure(0, weight=1)
        self.frame_list_store.columnconfigure(0, weight=1)
        self.frame_list_store.columnconfigure(1, weight=0)
        self.store_listbox = tk.Listbox(self.frame_list_store, height=10)
        self.store_listbox.grid(row=0, column=0, sticky="nsew", padx=(0,5), pady=5)
        self.store_listbox.bind("<<ListboxSelect>>", self.on_store_info_select)
        self.scroll_list = ttk.Scrollbar(self.frame_list_store, orient="vertical", command=self.store_listbox.yview)
        self.scroll_list.grid(row=0, column=0, sticky="nse")
        self.store_listbox.configure(yscrollcommand=self.scroll_list.set)
        self.frame_list_buttons = ttk.Frame(self.frame_list_store)
        self.frame_list_buttons.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.btn_load_store = ttk.Button(self.frame_list_buttons, text="Load Selected Store", command=self.load_selected_store)
        self.btn_load_store.pack(side=tk.TOP, fill=tk.X, pady=(0,5))
        self.btn_delete_store = ttk.Button(self.frame_list_buttons, text="Delete Selected Store", command=self.delete_store)
        self.btn_delete_store.pack(side=tk.TOP, fill=tk.X)

        # LabelFrame "View store" (row 1)
        self.lf_view_store = ttk.LabelFrame(self.store_container, text="View store", padding=(10, 5))
        self.lf_view_store.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.lf_view_store.rowconfigure(1, weight=1)
        self.lf_view_store.columnconfigure(0, weight=1)
        self.frame_view_buttons = ttk.Frame(self.lf_view_store)
        self.frame_view_buttons.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.btn_delete_doc = ttk.Button(self.frame_view_buttons, text="Delete Document", command=self.delete_document)
        self.btn_delete_doc.pack(side=tk.RIGHT)
        self.btn_delete_doc["state"] = tk.DISABLED
        self.store_info_list = tk.Listbox(self.lf_view_store)
        self.store_info_list.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.store_info_list.bind("<<ListboxSelect>>", self.on_store_info_select)
        self.scroll_view = ttk.Scrollbar(self.lf_view_store, orient="vertical", command=self.store_info_list.yview)
        self.scroll_view.grid(row=1, column=0, sticky="nse")
        self.store_info_list.configure(yscrollcommand=self.scroll_view.set)

        # LabelFrame "Create store" (row 2)
        self.lf_create_store = ttk.LabelFrame(self.store_container, text="Create store", padding=(10, 5))
        self.lf_create_store.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.lf_create_store.columnconfigure(0, weight=1)
        self.frame_create_store = ttk.Frame(self.lf_create_store)
        self.frame_create_store.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame_create_store.columnconfigure(1, weight=1)
        ttk.Label(self.frame_create_store, text="Store Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_store_name = ttk.Entry(self.frame_create_store)
        self.entry_store_name.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Label(self.frame_create_store, text="Select Embedder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.embedder_options = ["MiniLM", "MPNet", "DistilRoBERTa"]
        self.combo_embedder = ttk.Combobox(self.frame_create_store, values=self.embedder_options, state="readonly")
        self.combo_embedder.current(0)
        self.combo_embedder.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.btn_create_store = ttk.Button(self.frame_create_store, text="Create Store", command=self.create_store)
        self.btn_create_store.grid(row=2, column=0, columnspan=2, pady=10)

        # --- Tab: Add Documents ---
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

        # --- Tab: Query Store ---
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
        self.rf_frame.pack(fill=tk.X, padx=10, pady=5)
        self.rf_frame.columnconfigure(1, weight=1)
        self.combo_strategy = ttk.Combobox(self.rf_frame, values=list(STRATEGY_DESCRIPTIONS.values()), state="readonly", width=50)
        self.combo_strategy.current(0)
        self.combo_strategy.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="w")
        self.mode_frame = ttk.Frame(self.rf_frame)
        self.mode_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="w")
        self.rb_chunk = ttk.Radiobutton(self.mode_frame, text="Chunk", variable=self.retrieval_mode_var, value="chunk")
        self.rb_chunk.pack(anchor="w", pady=2)
        self.rb_document = ttk.Radiobutton(self.mode_frame, text="Document", variable=self.retrieval_mode_var,
                                           value="document")
        self.rb_document.pack(anchor="w", pady=2)
        self.chk_knee = ttk.Checkbutton(self.rf_frame, text="Enable Knee Detection", variable=self.use_knee)
        self.chk_knee.grid(row=0, column=2, padx=5, pady=5, sticky="e")
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

        # --- Tab: Parameters ---
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

    def on_tab_changed(self, event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == "Store Mngt":
            self.refresh_store_list()

    def on_strategy_change(self, event):
        selected_strat = self.combo_strategy.get().split(" - ")[0].lower()
        if selected_strat in ["bm25", "multi", "ibrido", "tfidf", "rrf"]:
            self.retrieval_mode_var.set("document")
            self.rb_chunk.config(state="disabled")
            self.rb_document.config(state="normal")
        else:
            self.rb_chunk.config(state="normal")
            self.rb_document.config(state="normal")
            if self.retrieval_mode_var.get() not in ["chunk", "document"]:
                self.retrieval_mode_var.set("chunk")

    def load_store_by_name(self, store_name):
        try:
            storage = self.vector_storage_manager.get_storage(store_name)
            if storage is None:
                storage = self.vector_storage_manager.load_storage(store_name, self.UIConfig.params["BASE_DIR"])
            self.current_store = store_name
            self.label_current_store.config(text=f"Current Store: {store_name}")
            self.UIConfig.params["LAST_STORE"] = store_name
            self.UIConfig.save()

            self.store_info_list.delete(0, tk.END)
            self.store_info_list.insert(tk.END, f"Store Name: {store_name}")
            self.store_info_list.insert(tk.END, f"Embedding Dimension: {storage.embedding_dim}")
            self.store_info_list.insert(tk.END, f"Embedder: {storage.embedder}")
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
            self.set_status("Store loaded.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error loading store.")

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
            self.set_status("Load a Store first!")
            return
        line = self.store_listbox.get(selection[0])
        store_name = line.split(" - ")[0]
        self.load_store_by_name(store_name)

    def delete_store(self):
        selection = self.store_listbox.curselection()
        if not selection:
            self.set_status("Please select a store to delete.")
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
                    self.UIConfig.params["LAST_STORE"] = ""
                    self.UIConfig.save()
                self.refresh_store_list()
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.set_status("Error during store deletion.")

    def create_store(self):
        store_name = self.entry_store_name.get().strip()
        if not store_name:
            self.set_status("Store name cannot be empty.")
            return
        embedder_key = self.combo_embedder.get()
        temp_embedding_manager = EmbeddingManager(model_key=embedder_key, chunk_size=50, overlap=10)
        dummy_text = ("Questo è un testo di prova per verificare le dimensioni dell'embedding. " * 20).strip()
        print(f"[DEBUG] Dummy text length: {len(dummy_text.split())} words")
        chunks, dummy_embeddings = temp_embedding_manager.embed_text(dummy_text)
        if dummy_embeddings is None or dummy_embeddings.shape[0] == 0:
            self.set_status("Error generating dummy embedding.")
            return
        embedding_dim = len(dummy_embeddings[0])
        print(f"[DEBUG] embedding_dim = {embedding_dim}")
        if embedding_dim == 0:
            self.set_status("Error: embedding dimension is 0.")
            return
        try:
            self.vector_storage_manager.create_storage(store_name, embedding_dim,
                                                       embedder=temp_embedding_manager.model_name)
            self.set_status("Index saving...")
            self.vector_storage_manager.save_storage(store_name, self.UIConfig.params["BASE_DIR"],
                                                     self.embedding_manager)
            self.refresh_store_list()
            # Carica il nuovo store come corrente
            self.load_store_by_name(store_name)
            self.set_status(f"Index saved. Store {store_name} successfully created using {embedder_key} embedder.")
        except Exception as e:
            traceback.print_exc()
            self.set_status("Error during store creation.")

    def on_store_info_select(self, event):
        print("DEBUG: on_store_info_select triggered!")  # Per verificare la chiamata
        selection = self.store_info_list.curselection()
        print("DEBUG: selection:", selection)
        if not selection:
            self.btn_delete_doc["state"] = tk.DISABLED
            return

        line = self.store_info_list.get(selection[0]).strip()
        print("DEBUG: selected line:", line)  # Usa line, non lin

        import re
        pattern = r"^- ([0-9a-fA-F-]+): (.+)$"
        if re.match(pattern, line):
            self.btn_delete_doc["state"] = tk.NORMAL
        else:
            self.btn_delete_doc["state"] = tk.DISABLED

    def delete_document(self):
        selection = self.store_info_list.curselection()
        if not selection:
            self.set_status("No document selected for deletion.")
            return
        line = self.store_info_list.get(selection[0]).strip()
        import re
        pattern = r"^- ([0-9a-f-]+): (.+)$"
        match = re.match(pattern, line)
        if not match:
            self.set_status("Invalid document selection.")
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
                self.set_status("Document not found.")
        except Exception as e:
            traceback.print_exc()
            self.set_status("Error during document deletion.")

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
            self.set_status("Load a Store first!")
            return
        path_input = self.add_path_entry.get().strip()
        if not path_input or not os.path.exists(path_input):
            self.set_status("Invalid file or directory selected.")
            return
        self.text_progress.delete("1.0", tk.END)
        import threading
        t = threading.Thread(target=self._add_documents_worker, args=(path_input,))
        t.start()

    def _add_documents_worker(self, path_input):
        import time
        start_time = time.time()
        storage = self.vector_storage_manager.get_storage(self.current_store)
        if storage is None:
            storage = self.vector_storage_manager.load_storage(self.current_store, self.UIConfig.params["BASE_DIR"])
        if self.embedding_manager.model_name != storage.embedder:
            possible_keys = [k for k, v in EmbeddingManager.EMBEDDER_OPTIONS.items() if v == storage.embedder]
            if possible_keys:
                self.embedding_manager = EmbeddingManager(
                    model_key=possible_keys[0],
                    chunk_size=self.embedding_manager.chunk_size,
                    overlap=self.embedding_manager.overlap
                )
            else:
                self.embedding_manager = EmbeddingManager(
                    model_key="MiniLM",
                    chunk_size=self.embedding_manager.chunk_size,
                    overlap=self.embedding_manager.overlap
                )
        import os
        files_to_process = []
        if self.add_mode_var.get() == "single":
            if os.path.isfile(path_input):
                files_to_process = [path_input]
            else:
                self._append_progress("ERROR: Selected path is not a file.\n")
                self.set_status("Invalid file selected.")
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
                self.set_status("Invalid directory selected.")
                return

        total_files = len(files_to_process)
        total_docs_added = 0
        total_docs_skipped = 0
        total_chunks = 0
        total_duplicates = 0

        for file_index, file_path in enumerate(files_to_process, start=1):
            try:
                file_size_kb = os.path.getsize(file_path) / 1024.0
                self._append_progress(f"Processing: {file_path} ({file_size_kb:.1f} KB)\n")
                try:
                    content = load_document(file_path)
                except Exception as e:
                    self._append_progress(f"  Skipped. Cannot read document: {e}\n")
                    continue
                if not content:
                    self._append_progress("  Skipped. Empty or unsupported.\n")
                    continue
                signature = compute_md5(content)
                if signature in storage.signatures:
                    total_docs_skipped += 1
                    total_duplicates += 1
                    self._append_progress("  Skipped. Document duplicated (same MD5).\n")
                    continue
                import uuid
                doc_id = str(uuid.uuid4())
                source_file = os.path.basename(file_path)
                chunks, embeddings = self.embedding_manager.embed_text(content)
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
            # Aggiorna la status bar con il progresso
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
            self.set_status("Load a Store first!")
            return
        query_text = self.entry_query.get().strip()
        if not query_text:
            self.set_status("Enter a query.")
            return

        self.set_status("Query executing...")
        selected_mode = self.retrieval_mode_var.get()
        selected_strategy = self.combo_strategy.get().split(" - ")[0]

        import threading
        t = threading.Thread(target=self.run_query, args=(query_text, selected_mode, selected_strategy))
        t.start()

    def run_query(self, query_text, mode, strategy):
        try:
            storage = self.vector_storage_manager.get_storage(self.current_store)
            if storage is None:
                storage = self.vector_storage_manager.load_storage(self.current_store, self.UIConfig.params["BASE_DIR"])
            if self.embedding_manager.model_name != storage.embedder:
                possible_keys = [k for k, v in EmbeddingManager.EMBEDDER_OPTIONS.items() if v == storage.embedder]
                if possible_keys:
                    self.embedding_manager = EmbeddingManager(
                        model_key=possible_keys[0],
                        chunk_size=self.embedding_manager.chunk_size,
                        overlap=self.embedding_manager.overlap
                    )
                else:
                    self.embedding_manager = EmbeddingManager(
                        model_key="MiniLM",
                        chunk_size=self.embedding_manager.chunk_size,
                        overlap=self.embedding_manager.overlap
                    )
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
            self.set_status("Error during query execution.")

    def show_result_detail(self, event):
        selected = self.tree_query_results.selection()
        if not selected:
            return
        idx = int(selected[0])
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
            messagebox.showerror("Error", f"Key '{key}' already exists.")
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
            messagebox.showerror("Error", "No parameter selected.")
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
        messagebox.showinfo("Saved", "Parameters saved to UI_manager.json.")
        self.load_parameters()


if __name__ == "__main__":
    base_dir = UIConfig().params["BASE_DIR"]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    app = VectorStoreManagerApp()
    app.mainloop()

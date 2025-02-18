
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import traceback
import threading

from numpy.ma.core import indices
from sentence_transformers import CrossEncoder

from rag_system.embedding_manager import EmbeddingManager
from rag_system.vector_storage import VectorStorageManager
from rag_system.search_engine import SearchEngine
from rag_system.document_manager import DocumentManager, load_document
from rag_system.utils import compute_md5


class UIConfig:
    """
    Gestisce la configurazione locale della UI, salvata in UI_manager.json.
    """
    DEFAULTS = {
        "BASE_DIR": "persistent_stores",
        "EMBEDDING_MODEL_KEY": "MiniLM",
        "CHUNK_SIZE": 100,
        "OVERLAP": 20,
        "CROSS_ENCODER_MODEL": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "DEFAULT_TOP_K": 5,
        "DEFAULT_THRESHOLD": 0.0,
        "LAST_STORE": ""
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
                print(f"[DEBUG] Errore nel caricamento della configurazione: {e}")
        else:
            self.save()

    def save(self):
        try:
            with open(self.FILENAME, "w", encoding="utf-8") as f:
                json.dump(self.params, f, indent=2)
        except Exception as e:
            print(f"[DEBUG] Errore nel salvataggio della configurazione: {e}")

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
        self.query_results_data = []  # Lista dei risultati della query

        self.label_current_store = ttk.Label(self, text="Current Store: None", foreground="blue")
        self.label_current_store.pack(pady=5)

        # Inizializzo i manager principali
        self.embedding_manager = EmbeddingManager(
            model_key=self.UIConfig.params["EMBEDDING_MODEL_KEY"],
            chunk_size=self.UIConfig.params["CHUNK_SIZE"],
            overlap=self.UIConfig.params["OVERLAP"]
        )
        self.vector_storage_manager = VectorStorageManager()
        self.doc_manager = DocumentManager()
        self.cross_encoder = CrossEncoder(self.UIConfig.params["CROSS_ENCODER_MODEL"])

        # Variabile per la modalità di add (single file vs directory)
        self.add_mode_var = tk.StringVar(value="single")  # default "single"

        self.create_widgets()

        # Bind per aggiornare automaticamente la lista degli store se viene selezionata la tab "List Stores"
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Logica di avvio:
        # - Se config ha LAST_STORE e può essere caricato, seleziona tab "View Store"
        # - Se esiste LAST_STORE ma non è possibile caricarlo, seleziona tab "List Stores"
        # - Se non esiste LAST_STORE, seleziona tab "Create Store"
        last_store = self.UIConfig.params["LAST_STORE"].strip()
        if last_store:
            try:
                self.load_store_by_name(last_store)
                self.notebook.select(self.tab_view)
            except Exception as e:
                print(f"[DEBUG] Impossibile caricare '{last_store}': {e}")
                self.notebook.select(self.tab_list)
        else:
            self.notebook.select(self.tab_create)

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # ----------------- Tab: List Stores -----------------
        self.tab_list = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_list, text="List Stores")
        # Rimuovo il pulsante "Refresh Store List" (poiché la lista verrà aggiornata automaticamente)
        # self.btn_refresh = ttk.Button(self.tab_list, text="Refresh Store List", command=self.refresh_store_list)
        # self.btn_refresh.pack(pady=5)

        self.store_listbox = tk.Listbox(self.tab_list, width=100, height=10)
        self.store_listbox.pack(padx=10, pady=5)

        self.btn_load_store = ttk.Button(self.tab_list, text="Load Selected Store", command=self.load_selected_store)
        self.btn_load_store.pack(pady=5)

        self.btn_delete_store = ttk.Button(self.tab_list, text="Delete Selected Store", command=self.delete_store)
        self.btn_delete_store.pack(pady=5)

        # ----------------- Tab: Create Store -----------------
        self.tab_create = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_create, text="Create Store")

        ttk.Label(self.tab_create, text="Store Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.entry_store_name = ttk.Entry(self.tab_create, width=40)
        self.entry_store_name.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self.tab_create, text="Select Embedder:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.embedder_options = ["MiniLM", "MPNet", "DistilRoBERTa"]
        self.combo_embedder = ttk.Combobox(self.tab_create, values=self.embedder_options, state="readonly", width=20)
        self.combo_embedder.current(0)
        self.combo_embedder.grid(row=1, column=1, padx=5, pady=5)

        self.btn_create_store = ttk.Button(self.tab_create, text="Create Store", command=self.create_store)
        self.btn_create_store.grid(row=2, column=0, columnspan=2, pady=10)

        # ----------------- Tab: View Store -----------------
        self.tab_view = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_view, text="View Store")

        self.store_info_list = tk.Listbox(self.tab_view, width=100, height=20)
        self.store_info_list.pack(padx=10, pady=5)
        self.store_info_list.bind("<<ListboxSelect>>", self.on_store_info_select)

        self.btn_delete_doc = ttk.Button(self.tab_view, text="Delete Document", command=self.delete_document)
        self.btn_delete_doc.pack(pady=5)
        self.btn_delete_doc["state"] = tk.DISABLED

        # ----------------- Tab: Add Documents -----------------
        self.tab_add = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_add, text="Add Documents")

        # Layout per la selezione del percorso e del tipo di add
        frame_add = ttk.Frame(self.tab_add)
        frame_add.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frame_add, text="Select Add Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        # Radio buttons per selezionare tra single file e directory
        self.radio_single = ttk.Radiobutton(frame_add, text="Single File", variable=self.add_mode_var, value="single")
        self.radio_single.grid(row=0, column=1, padx=5, pady=5)
        self.radio_directory = ttk.Radiobutton(frame_add, text="Directory", variable=self.add_mode_var, value="directory")
        self.radio_directory.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(frame_add, text="Path:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_directory = ttk.Entry(frame_add, width=60)
        self.entry_directory.grid(row=1, column=1, columnspan=2, sticky="we", padx=5, pady=5)

        self.btn_browse = ttk.Button(frame_add, text="Browse", command=self.browse_directory_or_file)
        self.btn_browse.grid(row=1, column=3, padx=5, pady=5)

        self.btn_add_docs = ttk.Button(frame_add, text="Add Documents", command=self.add_documents_async)
        self.btn_add_docs.grid(row=2, column=0, columnspan=4, pady=10)

        self.text_progress = tk.Text(self.tab_add, width=80, height=15, wrap=tk.WORD)
        self.text_progress.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # ----------------- Tab: Query Store -----------------
        self.tab_query = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_query, text="Query Store")

        ttk.Label(self.tab_query, text="Enter Query:").pack(pady=5)
        self.entry_query = ttk.Entry(self.tab_query, width=60)
        self.entry_query.pack(pady=5)

        ttk.Label(self.tab_query, text="Retrieval Strategy:").pack(pady=5)
        # Aggiorna il combobox per includere anche "ibrido" e "multi"
        self.combo_strategy = ttk.Combobox(self.tab_query, values=["faiss", "bm25", "ibrido", "multi"], state="readonly", width=20)
        self.combo_strategy.current(0)
        self.combo_strategy.pack(pady=5)

        ttk.Label(self.tab_query, text="Retrieval Mode:").pack(pady=5)
        self.combo_mode = ttk.Combobox(self.tab_query, values=["chunk", "document"], state="readonly", width=20)
        self.combo_mode.set("chunk")
        self.combo_mode.pack(pady=5)

        self.btn_query = ttk.Button(self.tab_query, text="Search", command=self.query_store)
        self.btn_query.pack(pady=5)

        # Treeview per i risultati della query, con header e layout dinamico
        columns = ("doc_id", "doc_name", "score", "length")
        self.tree_query_results = ttk.Treeview(self.tab_query, columns=columns, show="headings")
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

        # ----------------- Tab: Parameters -----------------
        self.tab_params = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_params, text="Parameters")

        ttk.Label(self.tab_params, text="Parameters (double-click on value to edit):").pack(pady=5)
        self.tree_params = ttk.Treeview(self.tab_params, columns=("key", "value"), show="headings")
        self.tree_params.heading("key", text="Key")
        self.tree_params.heading("value", text="Value")
        self.tree_params.column("key", width=150, anchor="w")
        self.tree_params.column("value", width=150, anchor="w")
        self.tree_params.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

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
        if tab_text == "List Stores":
            self.refresh_store_list()

    # ----------------- Funzioni di caricamento store -----------------
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
            print("indices:",indices)
            allowed_strategies = []
            if "faiss" in indices:
                allowed_strategies.append("faiss")
            if "bm25" in indices:
                allowed_strategies.append("bm25")
            if "faiss" in indices and "bm25" in indices:
                allowed_strategies.append("ibrido")
            if "tfidf" in indices:
                allowed_strategies.append('multi')
            if not allowed_strategies:
                allowed_strategies = ["faiss"]
            self.combo_strategy.config(values=allowed_strategies)
            self.combo_strategy.current(0)
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
            messagebox.showerror("Error", "Please select a store from the list.")
            return
        line = self.store_listbox.get(selection[0])
        store_name = line.split(" - ")[0]
        self.load_store_by_name(store_name)

    def delete_store(self):
        selection = self.store_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a store to delete.")
            return
        line = self.store_listbox.get(selection[0])
        store_name = line.split(" - ")[0]
        confirm = messagebox.askyesno("Confirm", f"Delete store '{store_name}'?")
        if confirm:
            try:
                self.vector_storage_manager.delete_storage(store_name, self.UIConfig.params["BASE_DIR"])
                messagebox.showinfo("Deleted", f"Store '{store_name}' has been deleted.")
                if self.current_store == store_name:
                    self.current_store = None
                    self.label_current_store.config(text="Current Store: None")
                    self.UIConfig.params["LAST_STORE"] = ""
                    self.UIConfig.save()
                self.refresh_store_list()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # ----------------- Creazione Store -----------------

    def create_store(self):
        store_name = self.entry_store_name.get().strip()
        if not store_name:
            messagebox.showerror("Error", "Store name cannot be empty")
            return
        embedder_key = self.combo_embedder.get()
        temp_embedding_manager = EmbeddingManager(model_key=embedder_key, chunk_size=50, overlap=10)
        # Dummy text esteso per garantire embedding validi
        dummy_text = ("Questo è un testo di prova per verificare le dimensioni dell'embedding. " * 5).strip()

        chunks, dummy_embeddings = temp_embedding_manager.embed_text(dummy_text)
        # Controllo esplicito per evitare ambiguità con array numpy
        if dummy_embeddings is None or (hasattr(dummy_embeddings, "size") and dummy_embeddings.size == 0) or (
                isinstance(dummy_embeddings, list) and len(dummy_embeddings) == 0):
            messagebox.showerror("Error", "Embedding generato vuoto. Verifica il modello.")
            return
        embedding_dim = len(dummy_embeddings[0])
        if embedding_dim == 0:
            messagebox.showerror("Error", "L'embedding generato è vuoto. Controlla il modello o il dummy text.")
            return
        try:
            self.vector_storage_manager.create_storage(store_name, embedding_dim,
                                                       embedder=temp_embedding_manager.model_name)
            # Salva subito lo store passando l'istanza di embedding_manager
            self.vector_storage_manager.save_storage(store_name, self.UIConfig.params["BASE_DIR"],self.embedding_manager)
            messagebox.showinfo("Success", f"Store '{store_name}' created successfully using {embedder_key}!")
            self.refresh_store_list()
        except Exception as e:
            print(f"[DEBUG] Errore nella creazione dello store: {e}")
            messagebox.showerror("Error", str(e))

    # ----------------- View Store -----------------
    def on_store_info_select(self, event):
        selection = self.store_info_list.curselection()
        if not selection:
            self.btn_delete_doc["state"] = tk.DISABLED
            return
        line = self.store_info_list.get(selection[0]).strip()
        import re
        pattern = r"^- ([0-9a-f-]+): (.+)$"
        if re.match(pattern, line):
            self.btn_delete_doc["state"] = tk.NORMAL
        else:
            self.btn_delete_doc["state"] = tk.DISABLED

    def delete_document(self):
        selection = self.store_info_list.curselection()
        if not selection:
            messagebox.showerror("Error", "No item selected.")
            return
        line = self.store_info_list.get(selection[0]).strip()
        import re
        pattern = r"^- ([0-9a-f-]+): (.+)$"
        match = re.match(pattern, line)
        if not match:
            messagebox.showerror("Error", "Selected line is not a valid document.")
            return
        doc_id = match.group(1)
        confirm = messagebox.askyesno("Confirm", f"Are you sure you want to delete document '{doc_id}'?")
        if not confirm:
            return
        try:
            storage = self.vector_storage_manager.get_storage(self.current_store)
            if storage is None:
                storage = self.vector_storage_manager.load_storage(self.current_store, self.UIConfig.params["BASE_DIR"])
            success = storage.remove_document(doc_id, self.embedding_manager)
            if success:
                self.vector_storage_manager.save_storage(self.current_store, self.UIConfig.params["BASE_DIR"],self.embedding_manager)
                messagebox.showinfo("Deleted", f"Document '{doc_id}' deleted successfully.")
                self.load_store_by_name(self.current_store)
            else:
                messagebox.showerror("Error", "Document not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ----------------- Add Documents -----------------
    def browse_directory_or_file(self):
        if self.add_mode_var.get() == "single":
            path = filedialog.askopenfilename()
        else:
            path = filedialog.askdirectory()
        if path:
            self.entry_directory.delete(0, tk.END)
            self.entry_directory.insert(0, path)

    def add_documents_async(self):
        if not self.current_store:
            messagebox.showerror("Error", "Load a store first.")
            return
        path_input = self.entry_directory.get().strip()
        if not path_input or not os.path.exists(path_input):
            messagebox.showerror("Error", "Please select a valid file or directory.")
            return
        self.text_progress.delete("1.0", tk.END)
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
                self.embedding_manager = EmbeddingManager(model_key=possible_keys[0],
                                                           chunk_size=self.embedding_manager.chunk_size,
                                                           overlap=self.embedding_manager.overlap)
            else:
                self.embedding_manager = EmbeddingManager(model_key="MiniLM",
                                                           chunk_size=self.embedding_manager.chunk_size,
                                                           overlap=self.embedding_manager.overlap)
        import os
        files_to_process = []
        if self.add_mode_var.get() == "single":
            if os.path.isfile(path_input):
                files_to_process = [path_input]
            else:
                self._append_progress("ERROR: Selected path is not a file.\n")
                return
        else:
            if os.path.isdir(path_input):
                for root, dirs, files in os.walk(path_input):
                    for f in files:
                        files_to_process.append(os.path.join(root, f))
                if len(files_to_process) > 1:
                    ans = messagebox.askyesno("Confirm", f"Are you sure you want to load {len(files_to_process)} files?")
                    if not ans:
                        self._append_progress("Operation cancelled by user.\n")
                        return
            else:
                self._append_progress("ERROR: Selected path is not a directory.\n")
                return

        total_docs_added = 0
        total_docs_skipped = 0
        total_chunks = 0
        total_duplicates = 0

        for file_path in files_to_process:
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
                self._append_progress(f"  ERROR: {e}\n")

        if "bm25" in storage.indices_present:
            storage.rebuild_bm25_index()

        self.vector_storage_manager.save_storage(self.current_store, self.UIConfig.params["BASE_DIR"],self.embedding_manager)
        elapsed = time.time() - start_time
        dup_percentage = (total_duplicates / (total_chunks + total_duplicates)) * 100 if (total_chunks + total_duplicates) > 0 else 0.0
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

    def _append_progress(self, text):
        def append():
            self.text_progress.insert(tk.END, text)
            self.text_progress.see(tk.END)
        self.after(0, append)

    # ----------------- Query Store -----------------
    def query_store(self):
        if not self.current_store:
            messagebox.showerror("Error", "Load a store first.")
            return
        query_text = self.entry_query.get().strip()
        if not query_text:
            messagebox.showerror("Error", "Please enter a query.")
            return
        selected_strategy = self.combo_strategy.get()
        selected_mode = self.combo_mode.get()       # "chunk" o "document"
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
                    self.embedding_manager = EmbeddingManager(model_key=possible_keys[0],
                                                               chunk_size=self.embedding_manager.chunk_size,
                                                               overlap=self.embedding_manager.overlap)
                else:
                    self.embedding_manager = EmbeddingManager(model_key="MiniLM",
                                                               chunk_size=self.embedding_manager.chunk_size,
                                                               overlap=self.embedding_manager.overlap)
            search_engine = SearchEngine(storage, self.embedding_manager, cross_encoder=self.cross_encoder)
            results = search_engine.search_with_strategy(
                query=query_text,
                strategy=strategy,
                top_k=self.UIConfig.params["DEFAULT_TOP_K"],
                retrieval_mode=mode,
                threshold=self.UIConfig.params["DEFAULT_THRESHOLD"]
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
        except Exception as e:
            traceback.print_exc()
            self.after(0, lambda e=e: messagebox.showerror("Error", str(e)))

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
                f"Doc ID: {result.get('doc_id','?')}\n"
                f"Score: {result.get('ce_score', result.get('score', 0.0)):.2f}\n"
                f"Source: {result.get('source','?')}\n\n"
                f"Chunk Content:\n{result.get('chunk','')}\n"
            )
        text_widget.insert(tk.END, detail_text)
        text_widget.config(state=tk.DISABLED)

    # ----------------- Parameters Tab Functions -----------------
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

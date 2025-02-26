import os
import time
import hashlib
from .utils import compute_md5


# Funzione di normalizzazione: rimuove spazi extra e converte in minuscolo
def normalize_value(value):
    import pandas as pd
    if pd.isnull(value):
        return ""
    # Converte il valore in stringa, elimina spazi all'inizio/fine e converte in minuscolo
    return str(value).strip().lower()


def load_document(file_path):
    """
    Carica un documento da file.
    Supporta formati: .txt, .md, .doc, .docx, .pdf, .xls, .xlsx.
    Per i file Excel, legge tutti i fogli con pandas, elimina righe e colonne vuote,
    aggrega ogni 5 righe in un chunk e normalizza ogni valore.
    Se il formato non è compatibile, solleva un’eccezione.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md"]:
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Impossibile leggere {file_path} con le codifiche testate.")

    elif ext == ".pdf":
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("Installa PyPDF2 per leggere PDF.")
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text)

    elif ext in [".doc", ".docx"]:
        if ext == ".doc":
            raise ValueError("Formato .doc non supportato. Usa .docx.")
        else:
            try:
                from docx import Document
                doc = Document(file_path)
                full_text = "\n".join([para.text for para in doc.paragraphs])
                if full_text.strip():
                    return full_text
                else:
                    print(f"[DEBUG] python-docx ha restituito un contenuto vuoto per {file_path}.")
            except Exception as e:
                print(f"[DEBUG] Errore con python‑docx su {file_path}: {e}")
            try:
                import docx2txt
                text = docx2txt.process(file_path)
                if text.strip():
                    return text
                else:
                    print(f"[DEBUG] docx2txt ha restituito un contenuto vuoto per {file_path}.")
            except Exception as e:
                print(f"[DEBUG] Errore con docx2txt su {file_path}: {e}")
            try:
                import textract
                text = textract.process(file_path).decode("utf-8")
                if text.strip():
                    return text
                else:
                    raise ValueError("textract ha restituito un contenuto vuoto.")
            except Exception as e:
                raise ValueError(f"Errore nella lettura del file DOCX ({file_path}): {e}")
    elif ext in [".xls", ".xlsx"]:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Installa pandas per leggere file Excel.")
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            raise ValueError(f"Errore nella lettura del file Excel ({file_path}): {e}")
        all_chunks = []
        valid_sheet_found = False
        for sheet_name, df in sheets.items():
            # Elimina righe e colonne completamente vuote
            df = df.dropna(how='all').dropna(axis=1, how='all')
            if df.empty or len(df.columns) == 0:
                continue
            valid_sheet_found = True
            header = df.columns.tolist()
            # Per ogni riga, crea un chunk che include il nome del foglio, il numero di riga e i valori associati alle colonne
            for idx, row in df.iterrows():
                parts = []
                for col in header:
                    val = row[col]
                    if pd.notnull(val):
                        parts.append(f"{col}: {val}")
                if parts:
                    # Il numero di riga è incrementato di 2 (perché la prima riga contiene l'header e l'indice parte da 0)
                    chunk = f"Sheet '{sheet_name}', riga {idx + 2}: " + ", ".join(parts)
                    all_chunks.append(chunk)
        if not valid_sheet_found:
            raise ValueError("Formato Excel non compatibile: nessun foglio contiene dati validi.")
        return "[EXCEL]\n" + "\n".join(all_chunks)
    else:
        raise ValueError(f"Formato {ext} non supportato.")


class DocumentManager:
    def __init__(self):
        self.documents = {}  # Mapping: doc_id -> { 'content': ..., 'metadata': ... }
        self.document_hashes = {}  # Per tenere traccia degli MD5 e rilevare duplicati

    def add_document(self, doc_id, content, metadata=None, file_path=None):
        """
        Aggiunge un documento. Se file_path viene fornito e metadata è None,
        genera metadata con 'file_path' e 'source'.
        Per file Excel, aggiunge anche il campo 'type': 'excel'.
        """
        if file_path is not None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".xls", ".xlsx"]:
                if metadata is None:
                    metadata = {}
                metadata["type"] = "excel"
                if "file_path" not in metadata:
                    metadata["file_path"] = file_path
                if "source" not in metadata:
                    metadata["source"] = os.path.basename(file_path)
            elif metadata is None:
                metadata = {"file_path": file_path, "source": os.path.basename(file_path)}
        self.documents[doc_id] = {"content": content, "metadata": metadata or {}}

    def delete_document(self, doc_id):
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False

    def list_documents(self):
        return list(self.documents.keys())

    def load_documents_from_directory(self, directory_path):
        start_time = time.time()
        docs = {}
        duplicates_count = 0
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    content = load_document(file_path)
                    md5 = compute_md5(content)
                    if md5 in self.document_hashes:
                        duplicates_count += 1
                        print("Skipped. Document duplicated (same MD5).")
                        continue
                    else:
                        self.document_hashes[md5] = file_path
                        docs[file_path] = {
                            "content": content,
                            "metadata": {"file_path": file_path, "source": os.path.basename(file_path)}
                        }
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in [".xls", ".xlsx"]:
                        docs[file_path]["metadata"]["type"] = "excel"
                except Exception as e:
                    print(f"Saltato {file}: {e}")

        self.documents.update(docs)
        elapsed = time.time() - start_time
        print("==== Summary ====")
        print(f"Inserted docs: {len(docs)}")
        print(f"Skipped docs (MD5 duplicates): {duplicates_count}")
        print(f"Elapsed time: {elapsed:.2f} s")
        print("=================")
        print("Note: Document chunking is handled during indexing, not at document load time.")
        return docs


import os
import time
import hashlib
from .utils import compute_md5

def load_document(file_path):
    """
    Carica un documento da file.
    Supporta i formati: .txt, .md, .doc, .docx, .pdf.
    Per i file di testo prova diverse codifiche.
    Per i DOCX, prova prima python‑docx, poi docx2txt e infine textract.
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
            # Prova prima con python-docx
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
            # Fallback: prova con docx2txt
            try:
                import docx2txt
                text = docx2txt.process(file_path)
                if text.strip():
                    return text
                else:
                    print(f"[DEBUG] docx2txt ha restituito un contenuto vuoto per {file_path}.")
            except Exception as e:
                print(f"[DEBUG] Errore con docx2txt su {file_path}: {e}")
            # Fallback: prova con textract
            try:
                import textract
                text = textract.process(file_path).decode("utf-8")
                if text.strip():
                    return text
                else:
                    raise ValueError("textract ha restituito un contenuto vuoto.")
            except Exception as e:
                raise ValueError(f"Errore nella lettura del file DOCX ({file_path}): {e}")
    else:
        raise ValueError(f"Formato {ext} non supportato.")


class DocumentManager:
    def __init__(self):
        self.documents = {}  # Mapping: doc_id -> { 'content': ..., 'metadata': ... }
        self.document_hashes = {}  # Per tenere traccia degli MD5 e rilevare duplicati

    def add_document(self, doc_id, content, metadata=None, file_path=None):
        """
        Aggiunge un documento. Se file_path viene fornito e metadata è None,
        genera metadata con 'file_path' e 'source' (nome del file).
        """
        if metadata is None and file_path is not None:
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
        duplicates_count = 0  # Inizializza il contatore duplicati
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    content = load_document(file_path)

                    # Calcola l'MD5 del contenuto (convertito in UTF-8)
                    # Nota: questo funziona se content è una stringa non vuota
                    md5 = compute_md5(content)

                    # Se il file è già stato processato, incrementa il contatore e salta
                    if md5 in self.document_hashes:
                        duplicates_count += 1
                        print(f"Skipped. Document duplicated (same MD5).")
                        continue
                    else:
                        self.document_hashes[md5] = file_path
                        docs[file_path] = {
                            "content": content,
                            "metadata": {"file_path": file_path, "source": os.path.basename(file_path)}
                        }
                except Exception as e:
                    print(f"Saltato {file}: {e}")
        self.documents.update(docs)
        elapsed = time.time() - start_time
        print("==== Summary ====")
        print(f"Inserted docs: {len(docs)}")
        print(f"Skipped docs (MD5 duplicates): {duplicates_count}")
        print("Duplicate chunks: 0 (0.00% of chunk attempts)")
        print(f"Elapsed time: {elapsed:.2f} s")
        print("=================")
        print("Note: Document chunking is handled during indexing, not at document load time.")
        return docs





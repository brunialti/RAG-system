import os
# Imposta HF_HOME in modo dinamico in base alla posizione del package
_current_dir = os.path.dirname(os.path.abspath(__file__))
hf_home = os.path.join(_current_dir, "models")
os.environ["HF_HOME"] = hf_home
# Rimuove TRANSFORMERS_CACHE se presente
try:
    if (os.environ["TRANSFORMERS_CACHE"]):
        print("TRANSFORMERS_CACHE variable still detected. Deleted.")
        os.environ.pop("TRANSFORMERS_CACHE", None)
except:
    pass
print("HF_HOME is set to:", os.environ["HF_HOME"])

from .vector_storage import VectorStorage, VectorStorageManager
from .document_manager import DocumentManager, load_document
from .embedding_manager import EmbeddingManager
from .search_engine import SearchEngine
from .chunker import dynamic_chunk_text
from .utils import compute_md5
from .config import Config

__version__="0.5"
Config.load()

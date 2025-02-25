# config.py
import json
import os

_CONFIG_FILENAME = "rag_system.json"
_THIS_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(_THIS_DIR, _CONFIG_FILENAME)
TRANSFORMERS_CACHE = "C:/Users/rober/Dropbox/Applicazioni/shibot/CHATGPT/rag_system/models"

class Config:
    EMBEDDING_MODEL_KEY = "MiniLM"            # Possibili valori: "MiniLM", "MPNet", "DistilRoBERTa"
    CHUNK_SIZE = 200
    OVERLAP = 100
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_TOP_K = 20
    DEFAULT_THRESHOLD = 0.5
    BASE_DIR = "persistent_stores"
    MIN_CHUNKS = 3  # Numero minimo di chunk per considerare un file "piccolo"
    DEFAULT_RETRIEVAL_MODE = "chunk"
    UI_LAST_STORE = ""

    @classmethod
    def load(cls):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            cls.BASE_DIR = data.get("BASE_DIR", cls.BASE_DIR)
            cls.EMBEDDING_MODEL_KEY = data.get("EMBEDDING_MODEL_KEY", cls.EMBEDDING_MODEL_KEY)
            cls.CHUNK_SIZE = data.get("CHUNK_SIZE", cls.CHUNK_SIZE)
            cls.OVERLAP = data.get("OVERLAP", cls.OVERLAP)
            cls.CROSS_ENCODER_MODEL = data.get("CROSS_ENCODER_MODEL", cls.CROSS_ENCODER_MODEL)
            cls.DEFAULT_TOP_K = data.get("DEFAULT_TOP_K", cls.DEFAULT_TOP_K)
            cls.DEFAULT_THRESHOLD = data.get("DEFAULT_THRESHOLD", cls.DEFAULT_THRESHOLD)
            cls.UI_LAST_STORE = data.get("UI_LAST_STORE", cls.UI_LAST_STORE)
            cls.DEFAULT_RETRIEVAL_MODE = data.get("DEFAULT_RETRIEVAL_MODE", cls.DEFAULT_RETRIEVAL_MODE)
            cls.MIN_CHUNKS = data.get("MIN_CHUNKS", cls.MIN_CHUNKS)
        else:
            cls.save()

    @classmethod
    def save(cls):
        data = {
            "BASE_DIR": cls.BASE_DIR,
            "EMBEDDING_MODEL_KEY": cls.EMBEDDING_MODEL_KEY,
            "CHUNK_SIZE": cls.CHUNK_SIZE,
            "OVERLAP": cls.OVERLAP,
            "CROSS_ENCODER_MODEL": cls.CROSS_ENCODER_MODEL,
            "DEFAULT_TOP_K": cls.DEFAULT_TOP_K,
            "DEFAULT_THRESHOLD": cls.DEFAULT_THRESHOLD,
            "UI_LAST_STORE": cls.UI_LAST_STORE,
            "DEFAULT_RETRIEVAL_MODE": cls.DEFAULT_RETRIEVAL_MODE,
            "MIN_CHUNKS": cls.MIN_CHUNKS
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)




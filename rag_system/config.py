"""
Configuration Module
----------------------
This module manages the configuration parameters for the rag_system.
It defines default values (such as embedding model key, chunk size, overlap, cross-encoder model, etc.)
and provides methods to load and save these parameters from/to a JSON file (rag_system.json).
This centralized configuration management allows the system to be easily tuned and persistently configured.
"""

import json
import os

_CONFIG_FILENAME = "rag_system.json"
_THIS_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(_THIS_DIR, _CONFIG_FILENAME)

class Config:
    EMBEDDING_MODEL_KEY = "MiniLM"            # embedding Sentence Transformers. Valori possibili: "MiniLM", "MPNet", "DistilRoBERTa"
    CHUNK_SIZE = 200
    OVERLAP = 100
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DEFAULT_TOP_K = 20
    DEFAULT_THRESHOLD = 0.5
    BASE_DIR = "persistent_stores"
    MIN_CHUNKS=3 #il numero minimo di chunks per cui considerare "piccolo" un file, nella implementazione di una strategia ibrida
    DEFAULT_RETRIEVAL_MODE = "chunk"
    LAST_STORE = ""

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
            cls.LAST_STORE = data.get("LAST_STORE", cls.LAST_STORE)
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
            "LAST_STORE": cls.LAST_STORE,
            "DEFAULT_RETRIEVAL_MODE": cls.DEFAULT_RETRIEVAL_MODE,
            "MIN_CHUNKS" : cls.MIN_CHUNKS
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
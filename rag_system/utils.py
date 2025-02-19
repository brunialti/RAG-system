import uuid
import hashlib

'''
def generate_doc_id():
    """Genera un ID univoco per un documento."""
    return str(uuid.uuid4())
'''

def compute_md5(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def compute_file_md5(file_path: str) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

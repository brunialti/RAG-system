import os
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
from rag_system.config import Config

def download_models(models_dir):
    """
    Scarica e salva i modelli embedding e i cross encoder nella directory models_dir.
    I modelli embedding vengono salvati in:
      models_dir/<model_name>
    I modelli cross encoder vengono salvati in:
      models_dir/<nome_gruppo>/<nome_modello>
    """
    # Scarica i modelli embedding
    embedding_models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "all-distilroberta-v1",
        "paraphrase-multilingual-MiniLM-L12-v2"
    ]
    for model_name in embedding_models:
        print(f"Downloading embedding model: {model_name} ...")
        model = SentenceTransformer(model_name)
        save_path = os.path.join(models_dir, model_name)
        model.save(save_path)
        print(f"Saved embedding model {model_name} to {save_path}")

    # Scarica i modelli cross encoder
    # Primo cross encoder: MS MARCO
    existing_ce_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Downloading cross encoder model: {existing_ce_model} ...")
    existing_ce = CrossEncoder(existing_ce_model)
    existing_ce_save_path = os.path.join(models_dir, *existing_ce_model.split("/"))
    os.makedirs(existing_ce_save_path, exist_ok=True)
    existing_ce.save(existing_ce_save_path)
    print(f"Saved cross encoder model {existing_ce_model} to {existing_ce_save_path}")

    # Secondo cross encoder: Italian Cross-Encoder
    italian_ce_model = "osiria/minilm-l6-h384-italian-cross-encoder"
    print(f"Downloading cross encoder model: {italian_ce_model} ...")
    italian_ce = CrossEncoder(italian_ce_model)
    italian_ce_save_path = os.path.join(models_dir, *italian_ce_model.split("/"))
    os.makedirs(italian_ce_save_path, exist_ok=True)
    italian_ce.save(italian_ce_save_path)
    print(f"Saved cross encoder model {italian_ce_model} to {italian_ce_save_path}")

    # Terzo cross encoder: Jina Reranker
    jina_ce_model = "jinaai/jina-reranker-v2-base-multilingual"
    print(f"Downloading cross encoder model: {jina_ce_model} ...")
    jina_ce = CrossEncoder(jina_ce_model, trust_remote_code=True)
    jina_ce_save_path = os.path.join(models_dir, *jina_ce_model.split("/"))
    os.makedirs(jina_ce_save_path, exist_ok=True)
    jina_ce.to('cpu')  # Usa 'cuda' se disponibile una GPU
    jina_ce.save(jina_ce_save_path)
    print(f"Saved cross encoder model {jina_ce_model} to {jina_ce_save_path}")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    print("base_dir:", base_dir)
    # La directory rag_system deve essere una sottocartella di base_dir
    rag_system_dir = os.path.join(base_dir, "rag_system")
    if not os.path.exists(rag_system_dir):
        print(f"Error: rag_system directory not found in {base_dir}.")
        sys.exit(1)

    # Crea la directory models all'interno di rag_system se non esiste già
    models_dir = os.path.join(rag_system_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Created models directory:", models_dir)
    else:
        print("Models directory already exists:", models_dir)

    # Scarica i modelli e salvali localmente
    download_models(models_dir)

    # Istanzia la classe di configurazione; questo creerà rag_system.json se non esiste
    Config.load()
    # Aggiorna il valore di CROSS_ENCODER_MODEL con il valore del primo cross encoder
    Config.CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    Config.save()
    print("Configuration updated: CROSS_ENCODER_MODEL set to", Config.CROSS_ENCODER_MODEL)

if __name__ == "__main__":
    main()




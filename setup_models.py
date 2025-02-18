import os
import sys
import re
from sentence_transformers import SentenceTransformer, CrossEncoder


def find_app_root(start_dir):
    """
    Walk upward from start_dir until a .env file is found.
    Returns the directory path where .env was located.
    If not found, exits with an error.
    """
    current_dir = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(current_dir, ".env")):
            return current_dir
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            print("Error: Could not find a .env file in the directory hierarchy.")
            sys.exit(1)
        current_dir = parent


def update_env_file(env_path, key, value):
    """
    Updates the .env file by appending the key=value if the key is not present.
    If the key is already present with a different value, emits a warning without modifying it.
    """
    with open(env_path, "r") as f:
        lines = f.readlines()
    env_dict = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env_dict[k.strip()] = v.strip()
    if key in env_dict:
        if env_dict[key] != value:
            print(f"Warning: {key} already defined in .env with value '{env_dict[key]}'. Not modifying it.")
        else:
            print(f"{key} is already correctly defined in .env.")
    else:
        with open(env_path, "a") as f:
            f.write(f"\n{key}={value}\n")
        print(f"Added {key}={value} to {env_path}")


def update_config_file(config_path, key, value):
    """
    Updates the config.py file in system_rag.
    If the key already exists with a different value, emits a warning and leaves it unchanged.
    Otherwise, appends the key=value at the end of the file.
    """
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found; skipping config update.")
        return
    with open(config_path, "r") as f:
        lines = f.readlines()
    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(key):
            found = True
            parts = line.split("=")
            if len(parts) >= 2:
                current_value = parts[1].strip().strip('"\'')

                if current_value != value:
                    print(f"Warning: In {config_path}, {key} is set to '{current_value}'. Not overwriting it.")
                    new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f'\n{key} = "{value}"\n')
        print(f"Added {key} = \"{value}\" to {config_path}")
    with open(config_path, "w") as f:
        f.writelines(new_lines)


def download_models(models_dir):
    """
    Downloads the required models (embedding models and cross encoder)
    and saves them in models_dir.
    """
    embedding_models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "all-distilroberta-v1"
    ]
    for model_name in embedding_models:
        print(f"Downloading embedding model: {model_name} ...")
        model = SentenceTransformer(model_name)
        save_path = os.path.join(models_dir, model_name)
        model.save(save_path)
        print(f"Saved embedding model {model_name} to {save_path}")

    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Downloading cross encoder model: {cross_encoder_model} ...")
    cross_encoder = CrossEncoder(cross_encoder_model)
    ce_folder = cross_encoder_model.replace("/", "_")
    ce_save_path = os.path.join(models_dir, ce_folder)
    cross_encoder.save(ce_save_path)
    print(f"Saved cross encoder model to {ce_save_path}")


def main():
    # Determine the directory where setup_models.py resides
    base_dir = os.path.abspath(os.path.dirname(__file__))
    # rag_system is assumed to be a subdirectory of the directory containing setup_models.py
    rag_system_dir = os.path.join(base_dir, "rag_system")
    if not os.path.exists(rag_system_dir):
        print(f"Error: rag_system directory not found in {base_dir}.")
        sys.exit(1)

    # Create the models directory inside rag_system
    models_dir = os.path.join(rag_system_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Created models directory:", models_dir)
    else:
        print("Models directory already exists:", models_dir)

    # Set TRANSFORMERS_CACHE to point to models_dir
    os.environ["TRANSFORMERS_CACHE"] = models_dir
    print("Set TRANSFORMERS_CACHE to:", models_dir)

    # Download models and save them locally
    download_models(models_dir)

    # Find the app root by looking for .env, starting from base_dir
    app_root = find_app_root(base_dir)
    print("App root found at:", app_root)

    # Update the .env file in the app root
    env_path = os.path.join(app_root, ".env")
    update_env_file(env_path, "TRANSFORMERS_CACHE", models_dir)

    # Update the config.py file in rag_system (if it exists)
    config_path = os.path.join(rag_system_dir, "config.py")
    update_config_file(config_path, "TRANSFORMERS_CACHE", models_dir)

    print("Setup completed successfully.")


if __name__ == "__main__":
    main()




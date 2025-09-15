#!/usr/bin/env python3
import argparse
import os
import json
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

def publish(model_dir: str, model_id: str, token: str, readme_path: str = None, index_path: str = None):
    api = HfApi()

    # 1) ensure repo exists
    try:
        create_repo(repo_id=model_id, token=token, exist_ok=True)
        print(f"Repo ensured: {model_id}")
    except Exception as e:
        print("Warning while creating repo (may already exist):", e)

    # 2) upload model files (folder)
    print("Uploading model files from:", model_dir)
    upload_folder(
        folder_path=model_dir,
        path_in_repo="",
        repo_id=model_id,
        token=token,
        repo_type="model",
        ignore_patterns=["*.pyc", "__pycache__", ".git/*"]
    )
    print("Model folder upload complete.")

    # 3) upload model card (README.md) if provided
    if readme_path and os.path.exists(readme_path):
        print("Uploading model card:", readme_path)
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=model_id,
            token=token,
            repo_type="model"
        )

    # 4) upload machine metadata (model-index.json) if provided
    if index_path and os.path.exists(index_path):
        print("Uploading model-index.json:", index_path)
        upload_file(
            path_or_fileobj=index_path,
            path_in_repo="model-index.json",
            repo_id=model_id,
            token=token,
            repo_type="model"
        )

    print("Publish complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--index", default="model-index.json")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in environment. Please set the secret in GitHub.")

    publish(args.model_dir, args.model_id, token, readme_path=args.readme, index_path=args.index)

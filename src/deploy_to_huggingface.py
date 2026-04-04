from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

MODEL_PATH = Path("models/best_model.joblib")


def deploy_to_hugging_face(repo_id: str, token: str, model_path: Path = MODEL_PATH) -> str:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="best_model.joblib",
        repo_id=repo_id,
        repo_type="model",
    )
    return repo_id


def main() -> None:
    repo_id = os.environ["HF_REPO_ID"]
    token = os.environ["HF_TOKEN"]
    deployed_repo = deploy_to_hugging_face(repo_id=repo_id, token=token)
    print(f"Deployed model to Hugging Face repo: {deployed_repo}")


if __name__ == "__main__":
    main()

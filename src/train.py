from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from huggingface_hub import HfApi

from pipeline import MODELS_DIR, register_best_model, run_training_pipeline


def upload_model_to_huggingface(model_path: Path) -> str | None:
    """Upload the trained model artifact to a Hugging Face model repository.

    Required environment variables:
    - HF_TOKEN: access token with write access to the target repo.
    - HF_REPO_ID: repo id in the format "username/repo-name".
    """
    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")

    if not hf_token or not hf_repo_id:
        return None

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=hf_repo_id,
        repo_type="model",
    )
    return f"https://huggingface.co/{hf_repo_id}"


def main() -> None:
    results, best_result = run_training_pipeline()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_model.joblib"
    with best_model_path.open("wb") as model_file:
        pickle.dump(best_result.model, model_file)

    model_version = register_best_model(best_result)
    summary = {
        "best_model": best_result.name,
        "accuracy": round(best_result.accuracy, 4),
        "precision": round(best_result.precision, 4),
        "run_id": best_result.run_id,
        "saved_model": str(best_model_path),
        "registered_version": model_version,
    }

    hf_repo_url = upload_model_to_huggingface(best_model_path)
    if hf_repo_url:
        summary["huggingface_repo"] = hf_repo_url

    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

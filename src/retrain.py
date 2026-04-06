from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import mlflow

from pipeline import (
    MODELS_DIR,
    configure_mlflow,
    register_best_model,
    run_training_pipeline,
)

SUMMARY_PATH = MODELS_DIR / "training_summary.json"
DECISION_PATH = MODELS_DIR / "retraining_decision.json"
MODEL_PATH = MODELS_DIR / "best_model.joblib"


def load_existing_summary(summary_path: Path = SUMMARY_PATH) -> dict | None:
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def get_existing_accuracy(summary: dict | None) -> float | None:
    if not summary:
        return None
    accuracy = summary.get("accuracy")
    if accuracy is None:
        return None
    return float(accuracy)


def should_promote_model(
    new_accuracy: float,
    current_accuracy: float | None,
    min_improvement: float = 0.0,
) -> bool:
    if current_accuracy is None:
        return True
    return new_accuracy > (current_accuracy + min_improvement)


def save_model(model, model_path: Path = MODEL_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as model_file:
        pickle.dump(model, model_file)


def run_retraining(trigger_type: str = "schedule") -> dict:
    configure_mlflow()
    mlflow.set_experiment(os.getenv("MLFLOW_RETRAIN_EXPERIMENT", "Breast_Cancer_Retraining"))

    min_improvement = float(os.getenv("MIN_ACCURACY_IMPROVEMENT", "0.0"))
    previous_summary = load_existing_summary()
    previous_accuracy = get_existing_accuracy(previous_summary)
    previous_model_name = previous_summary.get("best_model") if previous_summary else None

    _, best_result = run_training_pipeline()

    promoted = should_promote_model(
        new_accuracy=best_result.accuracy,
        current_accuracy=previous_accuracy,
        min_improvement=min_improvement,
    )
    accuracy_delta = (
        best_result.accuracy - previous_accuracy
        if previous_accuracy is not None
        else best_result.accuracy
    )

    with mlflow.start_run(run_name="retraining_decision"):
        mlflow.set_tag("run_type", "retraining_decision")
        mlflow.log_param("trigger_type", trigger_type)
        mlflow.log_param("candidate_model", best_result.name)
        mlflow.log_param("previous_model", previous_model_name or "none")
        mlflow.log_param("promoted", str(promoted).lower())
        mlflow.log_metric("new_accuracy", best_result.accuracy)
        mlflow.log_metric("new_precision", best_result.precision)
        mlflow.log_metric("accuracy_improvement", accuracy_delta)
        if previous_accuracy is not None:
            mlflow.log_metric("previous_accuracy", previous_accuracy)

    decision = {
        "trigger": trigger_type,
        "previous_model": previous_model_name,
        "previous_accuracy": previous_accuracy,
        "candidate_model": best_result.name,
        "candidate_accuracy": round(best_result.accuracy, 4),
        "candidate_precision": round(best_result.precision, 4),
        "accuracy_improvement": round(accuracy_delta, 4),
        "min_required_improvement": min_improvement,
        "promoted": promoted,
        "run_id": best_result.run_id,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if promoted:
        save_model(best_result.model)
        decision["registered_version"] = register_best_model(best_result)
        SUMMARY_PATH.write_text(json.dumps({
            "best_model": best_result.name,
            "accuracy": round(best_result.accuracy, 4),
            "precision": round(best_result.precision, 4),
            "run_id": best_result.run_id,
            "saved_model": str(MODEL_PATH),
            "registered_version": decision["registered_version"],
            "updated_by": "auto_retraining",
            "trigger": trigger_type,
        }, indent=2), encoding="utf-8")

    DECISION_PATH.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    trigger_type = os.getenv("RETRAIN_TRIGGER", "schedule")
    decision = run_retraining(trigger_type=trigger_type)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

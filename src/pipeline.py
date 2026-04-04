from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", DEFAULT_TRACKING_URI)
ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "my_model")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "Breast_Cancer_Best_Model")
PROMOTE_STAGE = os.getenv("MLFLOW_PROMOTE_STAGE", "Production")


@dataclass(frozen=True)
class RunResult:
    name: str
    accuracy: float
    precision: float
    run_id: str
    model: Pipeline


def configure_mlflow() -> tuple[str, str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", DEFAULT_REGISTRY_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    return tracking_uri, registry_uri


def load_and_split_data(test_size: float = 0.2, random_state: int = 42):
    data = load_breast_cancer()
    return train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target,
    )


def build_models() -> dict[str, object]:
    return {
        "Random_Forest_v1": RandomForestClassifier(n_estimators=50, random_state=42),
        "Logistic_Regression_v1": LogisticRegression(max_iter=1000),
        "Random_Forest_v2_Tuned": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=42,
        ),
    }


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def evaluate_predictions(y_true, y_pred) -> tuple[float, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    return accuracy, precision


def train_model(
    name: str,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    artifact_path: str = ARTIFACT_PATH,
) -> RunResult:
    pipeline = build_pipeline(estimator)

    with mlflow.start_run(run_name=name) as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy, precision = evaluate_predictions(y_test, y_pred)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.sklearn.log_model(pipeline, artifact_path)

        print(f"{name} completed with accuracy: {accuracy:.4f}")

        return RunResult(
            name=name,
            accuracy=accuracy,
            precision=precision,
            run_id=run.info.run_id,
            model=pipeline,
        )


def select_best_result(results: Iterable[RunResult]) -> RunResult:
    return max(results, key=lambda item: item.accuracy)


def register_best_model(best_run: RunResult) -> str | None:
    tracking_uri = mlflow.get_tracking_uri()
    if not tracking_uri.startswith(("http://", "https://")):
        return None

    client = MlflowClient()
    registered_model = mlflow.register_model(
        f"runs:/{best_run.run_id}/{ARTIFACT_PATH}",
        REGISTERED_MODEL_NAME,
    )
    client.transition_model_version_stage(
        name=registered_model.name,
        version=registered_model.version,
        stage=PROMOTE_STAGE,
    )
    return registered_model.version


def run_training_pipeline() -> tuple[list[RunResult], RunResult]:
    configure_mlflow()
    X_train, X_test, y_train, y_test = load_and_split_data()

    results = [
        train_model(name, estimator, X_train, X_test, y_train, y_test)
        for name, estimator in build_models().items()
    ]
    best_result = select_best_result(results)
    return results, best_result

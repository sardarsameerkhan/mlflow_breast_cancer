from __future__ import annotations

import json
from pathlib import Path

import joblib

from pipeline import MODELS_DIR, register_best_model, run_training_pipeline


def main() -> None:
    results, best_result = run_training_pipeline()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_result.model, best_model_path)

    model_version = register_best_model(best_result)
    summary = {
        "best_model": best_result.name,
        "accuracy": round(best_result.accuracy, 4),
        "precision": round(best_result.precision, 4),
        "run_id": best_result.run_id,
        "saved_model": str(best_model_path),
        "registered_version": model_version,
    }
    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

# 1. Dataset Selection (Part 1.1)
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Preprocessing & Split (Part 1.2)
# Scaling is important for models like Logistic Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

def train_model(name, model):
    # 3. MLflow Tracking (Part 1.3)
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        
        # Log params, metrics, and the model [cite: 9]
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.sklearn.log_model(model, "my_model")
        
        print(f"{name} completed with accuracy: {acc:.4f}")
        return acc, mlflow.active_run().info.run_id

# Requirement: Train at least 2 models & At least 3 runs [cite: 6, 10]
    results = []
    results.append(("Random_Forest_v1",) + train_model("Random_Forest_v1", RandomForestClassifier(n_estimators=50)))
    results.append(("Logistic_Regression_v1",) + train_model("Logistic_Regression_v1", LogisticRegression()))
    results.append(("Random_Forest_v2_Tuned",) + train_model("Random_Forest_v2_Tuned", RandomForestClassifier(n_estimators=150, max_depth=5)))

    best_name, best_accuracy, best_run_id = max(results, key=lambda item: item[1])
    client = MlflowClient()
    registered_model = client.register_model(f"runs:/{best_run_id}/my_model", "Breast_Cancer_Best_Model")
    client.transition_model_version_stage(
        name=registered_model.name,
        version=registered_model.version,
        stage="Production",
    )

    print(f"Best model: {best_name} with accuracy {best_accuracy:.4f}")
    print("Registered as Breast_Cancer_Best_Model and promoted to Production")

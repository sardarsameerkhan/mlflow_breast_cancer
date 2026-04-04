# MLflow Breast Cancer CI/CD Project

This project demonstrates a complete MLflow workflow and a GitHub CI/CD pipeline for the breast cancer dataset.

## Project Structure

```text
project/
├── data/
├── src/
├── models/
├── requirements.txt
└── README.md
```

## Part 1: MLflow Workflow

- Dataset: `sklearn.datasets.load_breast_cancer`
- Models: `RandomForestClassifier` and `LogisticRegression`
- Metrics: accuracy and precision
- MLflow tracking: 3 runs are logged automatically
- Model registry: best run can be registered as `Breast_Cancer_Best_Model`
- Production stage: the best version can be promoted to `Production`

## Local Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start MLflow UI if you want to view runs locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

4. Run training:

```bash
python src/train.py
```

If you want to use the local MLflow UI, keep `MLFLOW_TRACKING_URI` pointed at `http://127.0.0.1:5000`.

## Part 2: GitHub CI/CD Workflow

The workflow file is located at `.github/workflows/ci-cd.yml`.

What it does on push:

1. Installs dependencies.
2. Runs `pytest`.
3. Executes the training script.
4. Saves the best model to `models/best_model.joblib`.
5. Uploads the model artifact.
6. Deploys to Hugging Face when secrets are configured.

## Hugging Face Deployment

To enable deployment, add these repository settings in GitHub:

- Secret: `HF_TOKEN`
- Variable: `HF_REPO_ID` such as `your-username/breast-cancer-model`

The deployment script uploads `models/best_model.joblib` to the Hugging Face Model Hub.

## Testing

Run tests locally with:

```bash
pytest -q
```

## Notes

- The training script logs the best model and saves the local artifact.
- The MLflow registry step runs only when the tracking URI is an HTTP endpoint.
- The Hugging Face deploy step runs only when the required GitHub secret and variable are present.

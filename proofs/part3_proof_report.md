# Part 3: Auto Retraining Module - Proof Report

## 1) Evidence of retraining runs in MLflow

- MLflow tracking URI used for proof generation: file:./mlruns
- Retraining experiment name: Breast_Cancer_Retraining
- Retraining decision runs exported: 4

Evidence files:
- MLflow decision runs export: proofs/mlflow_retraining_decision_runs.json
- Full MLflow recent runs export: proofs/mlflow_retraining_runs.json

## 2) Logs showing multiple retraining cycles

Three cycles were executed and logged:
- Cycle 1 trigger: schedule
- Cycle 2 trigger: schedule
- Cycle 3 trigger: performance-check

Evidence file:
- Console log of all cycles: proofs/retraining_cycles.log

Key observed outcomes from logs:
- Candidate model selected: Logistic_Regression_v1
- Candidate accuracy: 0.9825
- Previous production accuracy: 0.9825
- Promotion decision: false when no improvement threshold was met

## 3) Explanation of retraining logic (for report section)

The auto-retraining module is implemented in src/retrain.py and follows this flow:

1. Trigger received (scheduled cron or manual run).
2. Pipeline retrains model candidates and logs training runs through MLflow.
3. Best new candidate is compared against the existing production summary (models/training_summary.json).
4. Improvement is computed as:

   accuracy_improvement = new_accuracy - previous_accuracy

5. Promotion rule:

   promote if new_accuracy > previous_accuracy + MIN_ACCURACY_IMPROVEMENT

6. Retraining decision run is logged in MLflow with parameters and metrics:
   - trigger_type
   - candidate_model
   - previous_model
   - promoted
   - new_accuracy
   - previous_accuracy
   - accuracy_improvement

7. If promoted, the new model artifact replaces the local production artifact and training summary is updated.

## 4) CI/CD automation proof

Scheduled workflow file:
- .github/workflows/retrain.yml

Configured triggers:
- schedule (weekly cron)
- workflow_dispatch (manual trigger)

Execution command used by workflow:
- python src/retrain.py

## 5) Model selection logic proof

Selection logic is tested in:
- tests/test_retraining.py

Verified cases:
- Promote when no existing production model exists.
- Promote when new model outperforms old model.
- Do not promote when performance is equal or lower.
- Respect minimum improvement threshold.

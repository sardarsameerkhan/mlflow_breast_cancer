from __future__ import annotations

import numpy as np

from pipeline import RunResult, build_models, build_pipeline, evaluate_predictions, load_and_split_data, select_best_result


def test_load_and_split_data_returns_matching_feature_dimensions():
    X_train, X_test, y_train, y_test = load_and_split_data()

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_build_models_returns_three_models():
    models = build_models()

    assert set(models) == {
        "Random_Forest_v1",
        "Logistic_Regression_v1",
        "Random_Forest_v2_Tuned",
    }


def test_select_best_result_chooses_highest_accuracy():
    results = [
        RunResult("model_a", 0.9, 0.8, "run-a", build_pipeline(build_models()["Random_Forest_v1"])),
        RunResult("model_b", 0.95, 0.9, "run-b", build_pipeline(build_models()["Logistic_Regression_v1"])),
        RunResult("model_c", 0.91, 0.85, "run-c", build_pipeline(build_models()["Random_Forest_v2_Tuned"])),
    ]

    best = select_best_result(results)

    assert best.name == "model_b"
    assert best.accuracy == 0.95


def test_evaluate_predictions_returns_expected_metrics():
    accuracy, precision = evaluate_predictions(np.array([1, 0, 1]), np.array([1, 0, 1]))

    assert accuracy == 1.0
    assert precision == 1.0

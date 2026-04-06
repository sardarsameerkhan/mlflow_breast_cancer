from __future__ import annotations

from sklearn.datasets import load_breast_cancer


def test_breast_cancer_dataset_loads_with_expected_feature_count():
    data = load_breast_cancer()

    assert data.data is not None
    assert data.target is not None
    assert data.data.shape[0] > 0
    assert data.data.shape[1] == 30

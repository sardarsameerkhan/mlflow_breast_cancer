from __future__ import annotations

from retrain import should_promote_model


def test_promotes_when_no_existing_model():
    assert should_promote_model(new_accuracy=0.9, current_accuracy=None)


def test_promotes_when_new_model_is_better():
    assert should_promote_model(new_accuracy=0.95, current_accuracy=0.94)


def test_does_not_promote_when_new_model_is_not_better():
    assert not should_promote_model(new_accuracy=0.94, current_accuracy=0.94)


def test_respects_minimum_improvement_threshold():
    assert not should_promote_model(
        new_accuracy=0.95,
        current_accuracy=0.94,
        min_improvement=0.02,
    )

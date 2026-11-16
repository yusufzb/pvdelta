import pytest

from pvdelta.evaluation import evaluate_keyed_detections


def test_evaluation_reports_counts_rates_and_excludes_unsupported_cells():
    result = evaluate_keyed_detections(
        {"P01", "P03"}, {"P01", "P02"}, evaluated_keys={"P01", "P02", "P03"}
    )
    assert result.true_positive == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.false_alarms_per_100 == pytest.approx(100 / 3)
    assert result.coverage == pytest.approx(1.0)


def test_evaluation_rejects_predictions_outside_population():
    with pytest.raises(ValueError, match="outside evaluated population"):
        evaluate_keyed_detections({"P99"}, {"P01"}, evaluated_keys={"P01"})


def test_empty_reference_keeps_precision_undefined_and_recall_undefined():
    result = evaluate_keyed_detections(set(), set(), evaluated_keys={"P01"})
    assert result.precision is None
    assert result.recall is None
    assert result.false_alarms_per_100 == 0.0


def test_coverage_reports_reference_labels_without_native_support():
    result = evaluate_keyed_detections(
        {"P01"}, {"P01", "P02"}, evaluated_keys={"P01"}
    )
    assert result.true_positive == 1
    assert result.recall == 1.0
    assert result.coverage == pytest.approx(0.5)

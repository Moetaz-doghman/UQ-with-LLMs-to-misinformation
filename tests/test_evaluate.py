import unittest

from src.evaluate import compute_threshold_metrics, macro_f1_score, roc_auc_score_binary


class EvaluateTests(unittest.TestCase):
    def test_macro_f1_score_binary(self) -> None:
        y_true = ["fake", "real", "fake", "real"]
        y_pred = ["fake", "real", "real", "real"]
        score = macro_f1_score(y_true, y_pred, labels=["fake", "real"])
        self.assertEqual(round(score, 4), 0.7333)

    def test_roc_auc_score_binary(self) -> None:
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        score = roc_auc_score_binary(y_true, y_score)
        self.assertEqual(score, 0.75)

    def test_threshold_metrics(self) -> None:
        metrics = compute_threshold_metrics(
            confidences=[0.95, 0.8, 0.4, 0.6],
            correct_flags=[1, 0, 1, 1],
            thresholds=[0.5],
        )
        first = metrics[0]
        self.assertEqual(first.kept_count, 3)
        self.assertEqual(first.coverage, 0.75)
        self.assertEqual(round(first.selective_accuracy or 0.0, 4), 0.6667)
        self.assertEqual(round(first.missed_correct_rate, 4), 0.3333)


if __name__ == "__main__":
    unittest.main()

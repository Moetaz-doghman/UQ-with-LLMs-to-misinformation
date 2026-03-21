import unittest

from src.parser import parse_prediction


class ParsePredictionTests(unittest.TestCase):
    def test_parse_prediction_valid_json(self) -> None:
        parsed = parse_prediction(
            '{"label":"real","confidence":0.82,"justification":"The article cites a mainstream wire report."}'
        )
        self.assertTrue(parsed.parse_ok)
        self.assertEqual(parsed.pred_label, "real")
        self.assertEqual(parsed.confidence, 0.82)

    def test_parse_prediction_rejects_invalid_label(self) -> None:
        parsed = parse_prediction('{"label":"unclear","confidence":0.5,"justification":"x"}')
        self.assertFalse(parsed.parse_ok)
        self.assertIn("Invalid label", parsed.parse_error)


if __name__ == "__main__":
    unittest.main()

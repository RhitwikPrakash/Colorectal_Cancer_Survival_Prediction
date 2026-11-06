import unittest

from predictor import FEATURE_NAMES, TREATMENT_CODES, load_artifacts, predict_survival


class TestSavedModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.scaler = load_artifacts()

    def test_training_feature_names_and_order(self):
        self.assertEqual(tuple(self.scaler.feature_names_in_), FEATURE_NAMES)
        self.assertEqual(self.model.n_features_in_, 5)
        self.assertEqual(set(self.model.classes_), {"No", "Yes"})

    def test_all_treatment_types_have_valid_codes(self):
        self.assertEqual(
            TREATMENT_CODES,
            {"Chemotherapy": 0, "Combination": 1, "Radiotherapy": 2, "Surgery": 3},
        )
        for code in TREATMENT_CODES.values():
            with self.subTest(treatment_code=code):
                prediction = predict_survival(
                    self.model,
                    self.scaler,
                    healthcare_costs=72000.0,
                    tumor_size=42.0,
                    treatment_type=code,
                    diabetes=0,
                    mortality_rate=17.0,
                )
                self.assertIn(prediction, {"No", "Yes"})


if __name__ == "__main__":
    unittest.main()

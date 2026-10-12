"""Comprehensive verification test suite for MindBloom AI Stress Prediction."""
import unittest
import numpy as np
import pandas as pd
from preprocessing import load_dataset, transform_features, STRESS_LABELS, FEATURE_COLUMNS
from prediction import load_model_and_features, predict_stress_details, predict_stress, MixedNaiveBayes
from recommendation_engine import generate_recommendations


class TestMindBloomSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata = load_model_and_features()
        cls.X, cls.y, cls.df_raw, cls.dataset_file = load_dataset()

    def test_01_dataset_integrity(self):
        self.assertGreater(len(self.X), 100)
        self.assertEqual(len(self.X), len(self.y))
        self.assertEqual(set(self.y.unique()), {1, 2, 3, 4, 5})
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.X.columns)

    def test_02_model_artifacts_loaded(self):
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.metadata)
        self.assertIn("selected_model", self.metadata)
        self.assertEqual(self.metadata["stress_levels"], [1, 2, 3, 4, 5])
        self.assertIn("priors", self.metadata)

    def test_03_prior_distribution_validity(self):
        priors = self.metadata["priors"]
        self.assertEqual(len(priors), 5)
        prior_sum = sum(priors.values())
        self.assertAlmostEqual(prior_sum, 1.0, places=5)
        for cls, p in priors.items():
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)

    def test_04_gaussian_statistics_present(self):
        g_stats = self.metadata["gaussian_stats"]
        self.assertIn("Age", g_stats)
        self.assertIn("How many hours do you sleep on average?", g_stats)
        self.assertIn("Year of Study", g_stats)
        for col, stats in g_stats.items():
            for lvl in [1, 2, 3, 4, 5]:
                self.assertIn(lvl, stats)
                self.assertGreater(stats[lvl]["variance"], 0.0)

    def test_05_prediction_posterior_sum_and_classes(self):
        sample = {
            "Age": "18–20",
            "Year of Study": "2nd Year",
            "How often do you feel stressed?": "Sometimes",
            "What is the main cause of your stress?": "Academic pressure",
            "How many hours do you sleep on average?": "6–8",
            "How often do you exercise or do physical activity?": "Weekly",
            "Have you experienced any of the following due to stress?": "Headache",
            "How do you usually cope with stress?": "Music",
        }
        res = predict_stress_details(sample)
        self.assertIn(res["predicted_level"], [1, 2, 3, 4, 5])
        self.assertEqual(res["label"], STRESS_LABELS[res["predicted_level"]])

        probs = res["probabilities"]
        self.assertEqual(len(probs), 5)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        for lvl in ["1", "2", "3", "4", "5"]:
            self.assertGreaterEqual(probs[lvl], 0.0)
            self.assertLessEqual(probs[lvl], 1.0)

    def test_06_unseen_category_laplace_smoothing(self):
        unseen_sample = {
            "Age": "18–20",
            "Year of Study": "2nd Year",
            "How often do you feel stressed?": "ExtremelyUnseenFrequencyXYZ",
            "What is the main cause of your stress?": "RareNovelStressorABC",
            "How many hours do you sleep on average?": "6–8",
            "How often do you exercise or do physical activity?": "NeverBeforeEncounteredActivity",
            "Have you experienced any of the following due to stress?": "UniqueSymptom123",
            "How do you usually cope with stress?": "ExperimentalCopingTechnique999",
        }
        res = predict_stress_details(unseen_sample)
        self.assertIn(res["predicted_level"], [1, 2, 3, 4, 5])
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        for p in res["probabilities"].values():
            self.assertGreater(p, 0.0)
            self.assertFalse(np.isnan(p))

    def test_07_recommendations_generation(self):
        sample = {
            "Age": "18–20",
            "Year of Study": "3rd Year",
            "How often do you feel stressed?": "Often",
            "What is the main cause of your stress?": "Academic pressure",
            "How many hours do you sleep on average?": "Less than 5",
            "How often do you exercise or do physical activity?": "Never",
            "Have you experienced any of the following due to stress?": "Headache, Anxiety",
            "How do you usually cope with stress?": "Nothing",
        }
        recs = generate_recommendations(sample, predicted_stress=4)
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 5)
        self.assertTrue(any("sleep" in r.lower() for r in recs))


if __name__ == "__main__":
    unittest.main()

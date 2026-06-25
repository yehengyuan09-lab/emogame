import unittest

from models.sales_calibration import (
    CalibrationSample,
    RbfSalesCalibrator,
    binomial_lower_tail,
    fit_until_gap_probability,
    minimum_samples_for_zero_failures,
)


def sample(index: int, target: int) -> CalibrationSample:
    return CalibrationSample(
        source_key=f"s-{index}",
        hero_name=f"hero-{index}",
        skin_name=f"skin-{index}",
        features={
            "base_score": 25.0 + index,
            "official_prior_score": 25.0 + index,
            "confidence": 40.0,
            "evidence_coverage": 20.0,
            "official_tier": float(index % 5),
            "quality_score": float((index % 5) * 20),
            "hero_skin_count": float(index % 12),
            "skin_age_years": float(index) / 10,
        },
        base_score=25 + index,
        target_sales_score=target,
        sales_basis="sales_rank",
        confidence=0.6,
    )


class SalesCalibrationTest(unittest.TestCase):
    def test_minimum_samples_for_zero_failures(self):
        self.assertEqual(minimum_samples_for_zero_failures(0.10, 0.10), 22)

    def test_binomial_lower_tail_zero_failures(self):
        self.assertGreater(binomial_lower_tail(0, 12, 0.10), 0.10)
        self.assertLess(binomial_lower_tail(0, 32, 0.10), 0.10)

    def test_fit_until_gap_probability_passes_on_training_samples(self):
        samples = [sample(index, 60 + (index % 35)) for index in range(24)]

        result = fit_until_gap_probability(samples, gap_threshold=10, target_probability=0.10, alpha=0.10)

        self.assertTrue(result["calibrated"]["passed"])
        self.assertEqual(result["calibrated"]["exceedances"], 0)
        model = result["model"]
        loaded = RbfSalesCalibrator.from_dict(model.to_dict())
        self.assertIsInstance(loaded.predict(samples[0].features), int)


if __name__ == "__main__":
    unittest.main()

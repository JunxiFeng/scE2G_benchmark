from __future__ import annotations

import unittest

from sce2g_benchmark.resources import build_method_coverage_df, build_missing_requirements_df


class ResourceValidationTest(unittest.TestCase):
    def test_method_coverage_flags_missing_eqtl_paper_methods(self) -> None:
        coverage = build_method_coverage_df()
        subset = coverage[
            (coverage["analysis"] == "eQTL")
            & (coverage["method"].isin(["STARE", "SCENT", "distanceToTSS"]))
        ]
        self.assertFalse(subset["present_in_local_tables"].all())

    def test_missing_requirements_contains_eqtl_parity_gaps(self) -> None:
        missing = build_missing_requirements_df(analyses=("eQTL",))
        identifiers = set(missing["identifier"].tolist())
        self.assertIn("STARE", identifiers)
        self.assertIn("SCENT", identifiers)
        self.assertIn("distanceToTSS", identifiers)


if __name__ == "__main__":
    unittest.main()

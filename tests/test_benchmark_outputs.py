from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sce2g_benchmark.eqtl import build_eqtl_outputs
from sce2g_benchmark.gwas import build_gwas_outputs


class BenchmarkOutputMaterializationTest(unittest.TestCase):
    def test_local_subset_eqtl_outputs_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = build_eqtl_outputs(output_dir=Path(tmp_dir) / "eqtl", mode="local_subset_debug")
            self.assertTrue(outputs["curve_table"].exists())
            self.assertTrue(outputs["fixed_recall_table"].exists())
            self.assertTrue(outputs["panel_traceability"].exists())

    def test_local_subset_gwas_outputs_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = build_gwas_outputs(output_dir=Path(tmp_dir) / "gwas", mode="local_subset_debug")
            self.assertTrue(outputs["curve_table"].exists())
            self.assertTrue(outputs["thresholded_overlap"].exists())
            self.assertTrue(outputs["thresholded_gene_linking"].exists())


if __name__ == "__main__":
    unittest.main()

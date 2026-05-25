from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sce2g_benchmark.normalize import normalize_prediction_table


class NormalizePredictionTableTest(unittest.TestCase):
    def test_normalize_prediction_table_applies_aliases_and_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "raw.tsv"
            input_path.write_text(
                "chr\tstart\tend\tTargetGene\tCellType\tE2G.Score\textra\n"
                "chr1\t10\t20\tGENE1\t10_cDC\t0.5\tfoo\n",
                encoding="utf-8",
            )
            aliases = tmp_path / "aliases.tsv"
            aliases.write_text("alias\tcanonical_biosample\n10_cDC\tPBMC_10_cDC\n", encoding="utf-8")

            output_path = tmp_path / "normalized.tsv"
            normalized_path, summary = normalize_prediction_table(
                input_path=input_path,
                output_path=output_path,
                column_map={
                    "chr": "chrom",
                    "TargetGene": "target_gene",
                    "CellType": "biosample",
                    "E2G.Score": "score",
                },
                method_name="test_method",
                biosample_aliases_path=aliases,
            )

            self.assertEqual(normalized_path, output_path)
            df = pd.read_csv(output_path, sep="\t")
            self.assertEqual(
                list(df.columns[:7]),
                ["chrom", "start", "end", "target_gene", "biosample", "score", "method"],
            )
            self.assertEqual(df.loc[0, "biosample"], "PBMC_10_cDC")
            self.assertEqual(df.loc[0, "method"], "test_method")
            self.assertEqual(summary["methods"], ["test_method"])


if __name__ == "__main__":
    unittest.main()

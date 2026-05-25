from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, read_tsv, write_df
from .paths import DEFAULT_OUTPUT_ROOT, EQTL_FINE_ROOT, FIGURE2_REPRO_ROOT, GWAS_FINE_ROOT, PAPER_PDF


PAPER_METHODS = [
    {"analysis": "eQTL", "panel": "2d,2e", "method": "scE2G_multiome", "paper_number": 1, "display_name": "scE2G (Multiome)", "required_for_paper_parity": True},
    {"analysis": "eQTL", "panel": "2d,2e", "method": "scE2G_ATAC", "paper_number": 2, "display_name": "scE2G (scATAC)", "required_for_paper_parity": True},
    {"analysis": "eQTL", "panel": "2d,2e", "method": "scABC", "paper_number": 5, "display_name": "ABC (A=scATAC, C=power law)", "required_for_paper_parity": True},
    {"analysis": "eQTL", "panel": "2d,2e", "method": "STARE", "paper_number": 6, "display_name": "STARE", "required_for_paper_parity": True},
    {"analysis": "eQTL", "panel": "2d,2e", "method": "SCENT", "paper_number": 11, "display_name": "SCENT", "required_for_paper_parity": True},
    {"analysis": "eQTL", "panel": "2d,2e", "method": "distanceToTSS", "paper_number": 15, "display_name": "Distance to TSS", "required_for_paper_parity": True},
    {"analysis": "eQTL", "panel": "2d,2e", "method": "Kendall", "paper_number": 4, "display_name": "Kendall correlation", "required_for_paper_parity": False},
    {"analysis": "eQTL", "panel": "2d,2e", "method": "ABC_distanceToTSS", "paper_number": 19, "display_name": "In element (ABC) & distance to TSS", "required_for_paper_parity": False},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "scE2G_multiome", "paper_number": 1, "display_name": "scE2G (Multiome)", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "scE2G_ATAC", "paper_number": 2, "display_name": "scE2G (scATAC)", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "Kendall", "paper_number": 4, "display_name": "Kendall correlation", "required_for_paper_parity": False},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "scABC", "paper_number": 5, "display_name": "ABC (A=scATAC, C=power law)", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "SnapATAC", "paper_number": 7, "display_name": "SnapATAC", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "Signac", "paper_number": 8, "display_name": "Signac", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "Cicero", "paper_number": 9, "display_name": "Cicero", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "FigR", "paper_number": 10, "display_name": "FigR", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "ScenicPlus", "paper_number": 12, "display_name": "SCENIC+", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "DIRECTNET", "paper_number": 13, "display_name": "DIRECT-NET", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "ArchR", "paper_number": 14, "display_name": "ArchR", "required_for_paper_parity": True},
    {"analysis": "GWAS", "panel": "2f,2g,2h", "method": "ABC_distanceToTSS", "paper_number": 19, "display_name": "In element (ABC) & distance to TSS", "required_for_paper_parity": True},
]


def local_eqtl_methods() -> set[str]:
    methods = set()
    curve = read_tsv(EQTL_FINE_ROOT / "enrichmentRecall" / "enrichmentRecall.GTExTissueAllMatches.tsv")
    fixed = read_tsv(EQTL_FINE_ROOT / "enrichmentAtRecall" / "enrichments.Recall0.05.GTExTissueAllMatches.tsv")
    methods.update(curve["method"].dropna().unique().tolist())
    methods.update(fixed["method"].dropna().unique().tolist())
    return methods


def local_gwas_methods() -> set[str]:
    methods = set()
    curve = pd.read_csv(GWAS_FINE_ROOT / "enrichmentRecallCurves" / "Blood_matched.values.tsv.gz", sep="\t")
    link = read_tsv(GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.geneLinking.tsv")
    overlap = read_tsv(GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.variantOverlap.tsv")
    methods.update(curve["method"].dropna().unique().tolist())
    methods.update(link["method"].dropna().unique().tolist())
    methods.update(overlap["method"].dropna().unique().tolist())
    return methods


def build_resource_manifest_df() -> pd.DataFrame:
    resources = [
        {
            "analysis": "common",
            "resource_id": "paper_pdf",
            "required_for_paper_parity": True,
            "path": PAPER_PDF,
            "notes": "Manuscript PDF used as the panel source of truth.",
        },
        {
            "analysis": "eQTL",
            "resource_id": "eqtl_curve_all_matches",
            "required_for_paper_parity": True,
            "path": EQTL_FINE_ROOT / "enrichmentRecall" / "enrichmentRecall.GTExTissueAllMatches.tsv",
            "notes": "All-matches eQTL enrichment-recall curves.",
        },
        {
            "analysis": "eQTL",
            "resource_id": "eqtl_fixed_recall_0_05",
            "required_for_paper_parity": True,
            "path": EQTL_FINE_ROOT / "enrichmentAtRecall" / "enrichments.Recall0.05.GTExTissueAllMatches.tsv",
            "notes": "All-matches eQTL enrichment at 5% recall.",
        },
        {
            "analysis": "eQTL",
            "resource_id": "eqtl_fixed_recall_pairwise_0_05",
            "required_for_paper_parity": True,
            "path": EQTL_FINE_ROOT / "enrichmentAtRecall" / "pairwiseComparisons.Recall0.05.GTExTissueAllMatches.tsv",
            "notes": "Pairwise significance for 5% recall panel.",
        },
        {
            "analysis": "GWAS",
            "resource_id": "gwas_curve_all_matches",
            "required_for_paper_parity": True,
            "path": GWAS_FINE_ROOT / "enrichmentRecallCurves" / "Blood_matched.values.tsv.gz",
            "notes": "All-matches GWAS enrichment-recall curves.",
        },
        {
            "analysis": "GWAS",
            "resource_id": "gwas_thresholded_overlap",
            "required_for_paper_parity": True,
            "path": GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.variantOverlap.tsv",
            "notes": "Thresholded GWAS variant-overlap summary.",
        },
        {
            "analysis": "GWAS",
            "resource_id": "gwas_thresholded_gene_linking",
            "required_for_paper_parity": True,
            "path": GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.geneLinking.tsv",
            "notes": "Thresholded GWAS gene-linking summary.",
        },
        {
            "analysis": "common",
            "resource_id": "local_prediction_availability",
            "required_for_paper_parity": False,
            "path": FIGURE2_REPRO_ROOT / "outputs" / "local_prediction_availability.tsv",
            "notes": "Audited local raw-prediction availability.",
        },
    ]
    rows = []
    for row in resources:
        path = Path(row["path"])
        rows.append(
            {
                "analysis": row["analysis"],
                "resource_id": row["resource_id"],
                "required_for_paper_parity": row["required_for_paper_parity"],
                "path": str(path),
                "exists_locally": path.exists(),
                "notes": row["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_method_coverage_df() -> pd.DataFrame:
    eqtl_methods = local_eqtl_methods()
    gwas_methods = local_gwas_methods()
    rows = []
    for row in PAPER_METHODS:
        local_methods = eqtl_methods if row["analysis"] == "eQTL" else gwas_methods
        present = row["method"] in local_methods
        notes = ""
        if row["analysis"] == "eQTL" and row["method"] == "STARE" and not present:
            notes = "Missing from checked-in fine-grained eQTL summary tables."
        elif row["analysis"] == "eQTL" and row["method"] == "SCENT" and not present:
            notes = "Missing from checked-in fine-grained eQTL summary tables."
        elif row["analysis"] == "eQTL" and row["method"] == "distanceToTSS" and not present:
            notes = "No standalone fine-grained eQTL distance-to-TSS method is exposed locally."
        rows.append(
            {
                **row,
                "present_in_local_tables": present,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis", "paper_number", "method"])


def build_missing_requirements_df(analyses: tuple[str, ...] = ("eQTL", "GWAS")) -> pd.DataFrame:
    resource_df = build_resource_manifest_df()
    method_df = build_method_coverage_df()
    missing_resources = resource_df[
        resource_df["required_for_paper_parity"] & ~resource_df["exists_locally"] & resource_df["analysis"].isin(set(analyses) | {"common"})
    ].copy()
    if not missing_resources.empty:
        missing_resources = missing_resources.assign(requirement_type="resource", identifier=missing_resources["resource_id"])
    missing_methods = method_df[
        method_df["required_for_paper_parity"] & ~method_df["present_in_local_tables"] & method_df["analysis"].isin(analyses)
    ].copy()
    if not missing_methods.empty:
        missing_methods = missing_methods.assign(
            requirement_type="method",
            identifier=missing_methods["method"],
            path="",
            exists_locally=False,
        )
    rows = []
    if not missing_resources.empty:
        rows.append(
            missing_resources[["requirement_type", "analysis", "identifier", "path", "exists_locally", "notes"]]
        )
    if not missing_methods.empty:
        rows.append(
            missing_methods[["requirement_type", "analysis", "identifier", "path", "exists_locally", "notes"]]
        )
    if not rows:
        return pd.DataFrame(columns=["requirement_type", "analysis", "identifier", "path", "exists_locally", "notes"])
    return pd.concat(rows, ignore_index=True)


def write_validation_bundle(output_dir: Path, analyses: tuple[str, ...] = ("eQTL", "GWAS")) -> dict[str, Path]:
    ensure_dir(output_dir)
    resource_path = output_dir / "resource_manifest.tsv"
    coverage_path = output_dir / "method_coverage.tsv"
    missing_path = output_dir / "missing_requirements.tsv"
    write_df(resource_path, build_resource_manifest_df())
    write_df(coverage_path, build_method_coverage_df())
    write_df(missing_path, build_missing_requirements_df(analyses=analyses))
    return {
        "resource_manifest": resource_path,
        "method_coverage": coverage_path,
        "missing_requirements": missing_path,
    }


def enforce_mode(mode: str, analyses: tuple[str, ...] = ("eQTL", "GWAS"), output_dir: Path | None = None) -> dict[str, Path]:
    bundle = write_validation_bundle(output_dir or DEFAULT_OUTPUT_ROOT / "paper_parity_validation", analyses=analyses)
    if mode == "paper_parity":
        missing_df = read_tsv(bundle["missing_requirements"])
        if not missing_df.empty:
            raise SystemExit(
                "Exact paper parity is not available in this checkout. "
                f"See {bundle['missing_requirements']} for the missing methods and resources."
            )
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate whether local resources satisfy exact Figure 2 paper parity.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "paper_parity_validation",
        help="Directory for resource and method validation reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    bundle = write_validation_bundle(args.output_dir)
    missing = read_tsv(bundle["missing_requirements"])
    print(f"Wrote resource manifest to {bundle['resource_manifest']}")
    print(f"Wrote method coverage to {bundle['method_coverage']}")
    print(f"Wrote missing requirements to {bundle['missing_requirements']}")
    if not missing.empty:
        raise SystemExit(
            "Exact paper parity is not available in this checkout. "
            f"Inspect {bundle['missing_requirements']}."
        )


if __name__ == "__main__":
    main()

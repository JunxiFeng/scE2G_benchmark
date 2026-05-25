from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, read_tsv, write_df, write_json
from .normalize import normalize_prediction_table
from .paths import DEFAULT_OUTPUT_ROOT, EQTL_FINE_ROOT, FIGURE2_REPRO_ROOT
from .resources import write_validation_bundle
from .scoring import MethodStyle, build_method_styles, load_normalized_predictions, resolve_bg_variants_path, score_eqtl_predictions


def nearest_threshold_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("method", sort=False):
        idx = (sub["threshold"] - sub["score_threshold"]).abs().idxmin()
        rows.append(df.loc[idx])
    return pd.DataFrame(rows)


def stage_normalized_predictions(
    normalized_predictions: Path | None,
    output_dir: Path,
    method_name: str | None,
) -> Path | None:
    if normalized_predictions is None:
        return None
    staged_dir = ensure_dir(output_dir / "normalized_predictions")
    staged_path = staged_dir / normalized_predictions.name
    if method_name:
        _, _summary = normalize_prediction_table(
            input_path=normalized_predictions,
            output_path=staged_path,
            method_name=method_name,
        )
    else:
        shutil.copy2(normalized_predictions, staged_path)
    return staged_path


def build_eqtl_outputs(
    *,
    output_dir: Path,
    mode: str,
    normalized_predictions: Path | None = None,
    method_name: str | None = None,
    bg_variants: Path | None = None,
    score_threshold: float | None = None,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    manifests_dir = ensure_dir(output_dir / "manifests")
    metrics_dir = ensure_dir(output_dir / "metrics")

    validation = write_validation_bundle(manifests_dir, analyses=("eQTL",))
    staged_predictions = stage_normalized_predictions(normalized_predictions, output_dir, method_name)

    traceability = read_tsv(FIGURE2_REPRO_ROOT / "outputs" / "panel_source_traceability.tsv")
    traceability = traceability[traceability["panel"].isin(["2d", "2e"])].copy()
    write_df(manifests_dir / "panel_traceability.tsv", traceability)

    curve_df = read_tsv(EQTL_FINE_ROOT / "enrichmentRecall" / "enrichmentRecall.GTExTissueAllMatches.tsv")
    fixed_df = read_tsv(EQTL_FINE_ROOT / "enrichmentAtRecall" / "enrichments.Recall0.05.GTExTissueAllMatches.tsv")
    pairwise_df = read_tsv(EQTL_FINE_ROOT / "enrichmentAtRecall" / "pairwiseComparisons.Recall0.05.GTExTissueAllMatches.tsv")

    if normalized_predictions is not None:
        bg_variants_path = resolve_bg_variants_path(bg_variants)
        normalized_df = load_normalized_predictions(staged_predictions or normalized_predictions, method_name=method_name)
        styles = build_method_styles([curve_df, fixed_df])
        scored = score_eqtl_predictions(
            normalized_df,
            bg_variants_path=bg_variants_path,
            reference_styles=styles,
            score_threshold=score_threshold,
        )
        if not scored["curve_table"].empty:
            curve_df = pd.concat([curve_df, scored["curve_table"]], ignore_index=True, sort=False)
        if not scored["fixed_recall_table"].empty:
            fixed_df = pd.concat([fixed_df, scored["fixed_recall_table"]], ignore_index=True, sort=False)
        if not scored["pairwise_significance"].empty:
            pairwise_df = pd.concat([pairwise_df, scored["pairwise_significance"]], ignore_index=True, sort=False)

    threshold_df = nearest_threshold_rows(curve_df)

    curve_df = curve_df.sort_values(["method", "threshold"]).reset_index(drop=True)
    fixed_df = fixed_df.sort_values(["enrichment"], ascending=False).reset_index(drop=True)
    pairwise_df = pairwise_df.sort_values(["group1", "group2"]).reset_index(drop=True)
    threshold_df = threshold_df.sort_values(["method"]).reset_index(drop=True)

    curve_path = metrics_dir / "figure2d_curve_table.tsv"
    threshold_path = metrics_dir / "figure2d_threshold_points.tsv"
    fixed_path = metrics_dir / "figure2e_fixed_recall_table.tsv"
    pairwise_path = metrics_dir / "figure2e_pairwise_significance.tsv"
    write_df(curve_path, curve_df)
    write_df(threshold_path, threshold_df)
    write_df(fixed_path, fixed_df)
    write_df(pairwise_path, pairwise_df)

    run_manifest = {
        "mode": mode,
        "analysis": "eQTL",
        "normalized_predictions": str(staged_predictions) if staged_predictions else "",
        "bg_variants": str(bg_variants) if bg_variants else "",
        "score_threshold": score_threshold if score_threshold is not None else "",
        "source_tables": [
            str(EQTL_FINE_ROOT / "enrichmentRecall" / "enrichmentRecall.GTExTissueAllMatches.tsv"),
            str(EQTL_FINE_ROOT / "enrichmentAtRecall" / "enrichments.Recall0.05.GTExTissueAllMatches.tsv"),
            str(EQTL_FINE_ROOT / "enrichmentAtRecall" / "pairwiseComparisons.Recall0.05.GTExTissueAllMatches.tsv"),
        ],
        "outputs": {
            "curve_table": str(curve_path),
            "threshold_points": str(threshold_path),
            "fixed_recall_table": str(fixed_path),
            "pairwise_significance": str(pairwise_path),
            "resource_manifest": str(validation["resource_manifest"]),
            "method_coverage": str(validation["method_coverage"]),
            "missing_requirements": str(validation["missing_requirements"]),
            "panel_traceability": str(manifests_dir / "panel_traceability.tsv"),
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    if mode == "paper_parity":
        missing_df = read_tsv(validation["missing_requirements"])
        if not missing_df.empty:
            raise SystemExit(
                "eQTL paper parity requirements are not satisfied. "
                f"See {validation['missing_requirements']}."
            )

    return {
        "curve_table": curve_path,
        "threshold_points": threshold_path,
        "fixed_recall_table": fixed_path,
        "pairwise_significance": pairwise_path,
        "resource_manifest": validation["resource_manifest"],
        "method_coverage": validation["method_coverage"],
        "missing_requirements": validation["missing_requirements"],
        "panel_traceability": manifests_dir / "panel_traceability.tsv",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the Figure 2 eQTL benchmark artifacts.")
    parser.add_argument("--mode", choices=["paper_parity", "local_subset_debug"], default="local_subset_debug")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "reference_local_subset_debug" / "eqtl",
    )
    parser.add_argument("--normalized-predictions", type=Path, default=None, help="Optional normalized prediction TSV to stage.")
    parser.add_argument("--method-name", default=None, help="Optional method name override when staging predictions.")
    parser.add_argument("--bg-variants", type=Path, default=None, help="Distal-noncoding background SNP BED used for real enrichment scoring.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional selected score threshold for the new model.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = build_eqtl_outputs(
        output_dir=args.output_dir,
        mode=args.mode,
        normalized_predictions=args.normalized_predictions,
        method_name=args.method_name,
        bg_variants=args.bg_variants,
        score_threshold=args.score_threshold,
    )
    for key, path in outputs.items():
        print(f"{key}\t{path}")


if __name__ == "__main__":
    main()

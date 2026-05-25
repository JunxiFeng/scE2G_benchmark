from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .io_utils import ensure_dir, read_tsv, write_df, write_json
from .normalize import normalize_prediction_table
from .paths import DEFAULT_OUTPUT_ROOT, FIGURE2_REPRO_ROOT, GWAS_FINE_ROOT
from .resources import write_validation_bundle
from .scoring import build_method_styles, load_normalized_predictions, resolve_bg_variants_path, score_gwas_predictions


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


def build_gwas_outputs(
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

    validation = write_validation_bundle(manifests_dir, analyses=("GWAS",))
    staged_predictions = stage_normalized_predictions(normalized_predictions, output_dir, method_name)

    traceability = read_tsv(FIGURE2_REPRO_ROOT / "outputs" / "panel_source_traceability.tsv")
    traceability = traceability[traceability["panel"].isin(["2f", "2g", "2h", "2f-2h"])].copy()
    write_df(manifests_dir / "panel_traceability.tsv", traceability)

    curve_df = pd.read_csv(GWAS_FINE_ROOT / "enrichmentRecallCurves" / "Blood_matched.values.tsv.gz", sep="\t")
    curve_df = curve_df[(curve_df["biosample"] == "all_matched") & (curve_df["trait"] == "all_matched")].copy()
    curve_df = curve_df.sort_values(["method", "threshold"]).reset_index(drop=True)
    overlap_df = read_tsv(GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.variantOverlap.tsv")
    overlap_df = overlap_df.sort_values(["recall", "enrichment"], ascending=[False, False]).reset_index(drop=True)
    link_df = read_tsv(GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.geneLinking.tsv")
    link_df = link_df.sort_values(["intersectPoPS", "recall", "precision"], ascending=[True, False, False]).reset_index(drop=True)

    if normalized_predictions is not None:
        bg_variants_path = resolve_bg_variants_path(bg_variants)
        normalized_df = load_normalized_predictions(staged_predictions or normalized_predictions, method_name=method_name)
        styles = build_method_styles([curve_df, overlap_df, link_df])
        scored = score_gwas_predictions(
            normalized_df,
            bg_variants_path=bg_variants_path,
            reference_styles=styles,
            score_threshold=score_threshold,
        )
        if not scored["curve_table"].empty:
            curve_df = pd.concat([curve_df, scored["curve_table"]], ignore_index=True, sort=False)
        if not scored["thresholded_overlap"].empty:
            overlap_df = pd.concat([overlap_df, scored["thresholded_overlap"]], ignore_index=True, sort=False)
        augmented_links = []
        if not scored["thresholded_gene_linking"].empty:
            augmented_links.append(scored["thresholded_gene_linking"])
        if not scored["thresholded_gene_linking_with_pops"].empty:
            augmented_links.append(scored["thresholded_gene_linking_with_pops"])
        if augmented_links:
            link_df = pd.concat([link_df, *augmented_links], ignore_index=True, sort=False)

    link_eqtl_df = link_df[link_df["intersectPoPS"] == False].reset_index(drop=True)  # noqa: E712
    link_pops_df = link_df[link_df["intersectPoPS"] == True].reset_index(drop=True)  # noqa: E712

    curve_path = metrics_dir / "figure2f_curve_table.tsv"
    overlap_path = metrics_dir / "figure2f_thresholded_variant_overlap.tsv"
    link_path = metrics_dir / "figure2g_thresholded_gene_linking.tsv"
    pops_path = metrics_dir / "figure2h_thresholded_gene_linking_with_pops.tsv"
    write_df(curve_path, curve_df)
    write_df(overlap_path, overlap_df)
    write_df(link_path, link_eqtl_df)
    write_df(pops_path, link_pops_df)

    run_manifest = {
        "mode": mode,
        "analysis": "GWAS",
        "normalized_predictions": str(staged_predictions) if staged_predictions else "",
        "bg_variants": str(bg_variants) if bg_variants else "",
        "score_threshold": score_threshold if score_threshold is not None else "",
        "source_tables": [
            str(GWAS_FINE_ROOT / "enrichmentRecallCurves" / "Blood_matched.values.tsv.gz"),
            str(GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.variantOverlap.tsv"),
            str(GWAS_FINE_ROOT / "thresholdedPerformanceComparison" / "Blood_matched.geneLinking.tsv"),
        ],
        "outputs": {
            "curve_table": str(curve_path),
            "thresholded_overlap": str(overlap_path),
            "thresholded_gene_linking": str(link_path),
            "thresholded_gene_linking_with_pops": str(pops_path),
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
                "GWAS paper parity requirements are not satisfied. "
                f"See {validation['missing_requirements']}."
            )

    return {
        "curve_table": curve_path,
        "thresholded_overlap": overlap_path,
        "thresholded_gene_linking": link_path,
        "thresholded_gene_linking_with_pops": pops_path,
        "resource_manifest": validation["resource_manifest"],
        "method_coverage": validation["method_coverage"],
        "missing_requirements": validation["missing_requirements"],
        "panel_traceability": manifests_dir / "panel_traceability.tsv",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the Figure 2 GWAS benchmark artifacts.")
    parser.add_argument("--mode", choices=["paper_parity", "local_subset_debug"], default="local_subset_debug")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "reference_local_subset_debug" / "gwas",
    )
    parser.add_argument("--normalized-predictions", type=Path, default=None, help="Optional normalized prediction TSV to stage.")
    parser.add_argument("--method-name", default=None, help="Optional method name override when staging predictions.")
    parser.add_argument("--bg-variants", type=Path, default=None, help="Distal-noncoding background SNP BED used for real enrichment scoring.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional selected score threshold for the new model.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = build_gwas_outputs(
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

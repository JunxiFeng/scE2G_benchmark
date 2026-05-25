from __future__ import annotations

import gzip
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from scipy.stats import hypergeom, norm

from .io_utils import ensure_dir, read_tsv
from .normalize import REQUIRED_COLUMNS
from .paths import (
    EQTL_CHR_SIZES,
    EQTL_PARTITION,
    EQTL_TSS,
    EQTL_VARIANTS_FINE,
    GWAS_CHR_SIZES,
    GWAS_GENE_PRIORITIZATION,
    GWAS_TSS,
    GWAS_VARIANT_KEY,
    GWAS_VARIANT_ROOT,
    SUPP_TABLE_S4,
    SUPP_TABLE_S5,
)


DISTAL_NONCODING_LABELS = {"ABC", "AllPeaks", "Other", "OtherIntron"}
DEFAULT_N_THRESHOLD_STEPS = 25
DEFAULT_EQTL_TARGET_RECALL = 0.05
DEFAULT_GWAS_TOP_POPS_GENES = 2
DEFAULT_GWAS_TOP_PRED_GENES = 2
DEFAULT_P_THRESHOLD = 0.05
BG_VARIANTS_ENV = "SCE2G_BG_VARIANTS"
EQTL_BG_COUNT_FALLBACK = 9_746_610


class ScoringError(RuntimeError):
    """Raised when real benchmark scoring cannot be completed safely."""


@dataclass(frozen=True)
class MethodStyle:
    display_name: str
    color: str


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "t", "yes"})


def _read_tsv_maybe_gzip(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer", **kwargs)


def _validate_normalized_predictions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ScoringError(f"Normalized predictions are missing required columns: {', '.join(missing)}")
    normalized = df.copy()
    normalized["chrom"] = normalized["chrom"].astype(str)
    normalized["start"] = pd.to_numeric(normalized["start"], errors="raise").astype(int)
    normalized["end"] = pd.to_numeric(normalized["end"], errors="raise").astype(int)
    normalized["target_gene"] = normalized["target_gene"].astype(str)
    normalized["biosample"] = normalized["biosample"].astype(str)
    normalized["score"] = pd.to_numeric(normalized["score"], errors="raise")
    normalized["method"] = normalized["method"].astype(str)
    normalized = normalized[normalized["end"] > normalized["start"]].reset_index(drop=True)
    return normalized


def load_normalized_predictions(path: Path, method_name: str | None = None) -> pd.DataFrame:
    df = read_tsv(path)
    if method_name:
        df["method"] = method_name
    return _validate_normalized_predictions(df)


def _read_gene_names_from_tss(path: Path) -> set[str]:
    df = _read_tsv_maybe_gzip(path, header=None)
    return set(df.iloc[:, 3].astype(str))


def _read_chrom_order(path: Path) -> dict[str, int]:
    df = _read_tsv_maybe_gzip(path, header=None)
    return {chrom: index for index, chrom in enumerate(df.iloc[:, 0].astype(str).tolist())}


def _sort_bed_like(df: pd.DataFrame, chrom_order: dict[str, int], chrom_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    sortable = df.copy()
    sortable["_chrom_order"] = sortable[chrom_col].map(lambda chrom: chrom_order.get(str(chrom), 10**9))
    sortable = sortable.sort_values(["_chrom_order", start_col, end_col]).drop(columns="_chrom_order")
    return sortable.reset_index(drop=True)


def _build_interval_trees(df: pd.DataFrame, *, chrom_col: str, start_col: str, end_col: str, data_cols: Iterable[str]) -> dict[str, IntervalTree]:
    trees: dict[str, IntervalTree] = {}
    payload_columns = list(data_cols)
    for row in df[[chrom_col, start_col, end_col, *payload_columns]].itertuples(index=False, name=None):
        chrom = str(row[0])
        start = int(row[1])
        end = int(row[2])
        payload = row[3:] if payload_columns else ()
        trees.setdefault(chrom, IntervalTree()).addi(start, end, payload)
    return trees


def _build_partition_trees(path: Path) -> dict[str, IntervalTree]:
    partition = _read_tsv_maybe_gzip(path, header=None)
    partition = partition[partition.iloc[:, 3].astype(str).isin(DISTAL_NONCODING_LABELS)].copy()
    return _build_interval_trees(partition, chrom_col=0, start_col=1, end_col=2, data_cols=[])


def _overlaps_any(trees: dict[str, IntervalTree], chrom: str, start: int, end: int) -> bool:
    tree = trees.get(str(chrom))
    if tree is None:
        return False
    return bool(tree.overlap(int(start), int(end)))


def _filter_to_partition(df: pd.DataFrame, partition_trees: dict[str, IntervalTree], chrom_col: str, start_col: str, end_col: str) -> pd.DataFrame:
    keep = [
        _overlaps_any(partition_trees, row[0], int(row[1]), int(row[2]))
        for row in df[[chrom_col, start_col, end_col]].itertuples(index=False, name=None)
    ]
    return df.loc[keep].reset_index(drop=True)


def _default_style(method: str) -> MethodStyle:
    digest = hashlib.md5(method.encode("utf-8")).hexdigest()
    color = f"#{digest[:6]}"
    return MethodStyle(display_name=method, color=color)


def build_method_styles(reference_tables: Iterable[pd.DataFrame]) -> dict[str, MethodStyle]:
    styles: dict[str, MethodStyle] = {}
    for table in reference_tables:
        needed = {"method", "pred_name_long", "hex"}
        if not needed.issubset(table.columns):
            continue
        for row in table[["method", "pred_name_long", "hex"]].drop_duplicates().itertuples(index=False):
            method = str(row[0])
            styles[method] = MethodStyle(display_name=str(row[1]), color=str(row[2]))
    return styles


def _get_method_style(method: str, styles: dict[str, MethodStyle]) -> MethodStyle:
    return styles.get(method, _default_style(method))


def _quantile_thresholds(scores: np.ndarray, explicit_threshold: float | None, n_steps: int = DEFAULT_N_THRESHOLD_STEPS) -> np.ndarray:
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        raise ScoringError("No finite overlap scores were found for the provided predictions.")
    distinct_scores = np.unique(finite_scores)
    if distinct_scores.size == 1:
        thresholds = distinct_scores
    else:
        probabilities = np.linspace(0.0, 1.0, num=n_steps)
        thresholds = np.quantile(finite_scores, probabilities)
        thresholds = np.unique(np.concatenate([thresholds, np.linspace(finite_scores.min(), finite_scores.max(), num=n_steps)]))
    if explicit_threshold is not None:
        thresholds = np.unique(np.concatenate([thresholds, np.array([explicit_threshold], dtype=float)]))
    return np.sort(thresholds)


def _counts_at_threshold(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        return np.zeros(len(thresholds), dtype=int)
    sorted_scores = np.sort(finite_scores)
    positions = np.searchsorted(sorted_scores, thresholds, side="left")
    return (sorted_scores.size - positions).astype(int)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _add_enrichment_stats(df: pd.DataFrame, p_threshold: float = DEFAULT_P_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    z_score = norm.ppf(1 - p_threshold / 2)
    se_values = []
    ci_lows = []
    ci_highs = []
    p_values = []
    for row in out.itertuples(index=False):
        n1 = float(getattr(row, "nVariantsTotal"))
        x1 = float(getattr(row, "nVariantsOverlappingEnhancers"))
        n2 = float(getattr(row, "nCommonVariantsTotal"))
        x2 = float(getattr(row, "nCommonVariantsOverlappingEnhancers"))
        enrichment = float(getattr(row, "enrichment"))
        if x1 <= 0 or x2 <= 0 or n1 <= 0 or n2 <= 0 or enrichment <= 0:
            se = math.inf
            ci_low = 0.0
            ci_high = math.nan
            p_value = 1.0 if x1 <= 0 else 0.0
        else:
            se = math.sqrt((((n1 - x1) / x1) / n1) + (((n2 - x2) / x2) / n2))
            ci_low = math.exp(math.log(enrichment) - z_score * se)
            ci_high = math.exp(math.log(enrichment) + z_score * se)
            p_value = float(hypergeom.sf(int(x1), int(n1 + n2), int(n1), int(x1 + x2)))
        se_values.append(se)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        p_values.append(p_value)
    out["SE_log_enr"] = se_values
    out["CI_enr_low"] = ci_lows
    out["CI_enr_high"] = ci_highs
    out["p_enr"] = p_values
    out["p_adjust_enr"] = np.minimum(1.0, out["p_enr"] * max(len(out), 1))
    return out


def _add_recall_overlap_stats(df: pd.DataFrame, p_threshold: float = DEFAULT_P_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    z_score = norm.ppf(1 - p_threshold / 2)
    out["recall_adjust"] = (out["nVariantsOverlappingEnhancers"] + 2) / (out["nVariantsTotal"] + 4)
    out["SE_recall"] = np.sqrt(out["recall_adjust"] * (1 - out["recall_adjust"]) / (out["nVariantsTotal"] + 4))
    out["CI_recall_low"] = out["recall"] - z_score * out["SE_recall"]
    out["CI_recall_high"] = out["recall"] + z_score * out["SE_recall"]
    return out


def _add_recall_linking_stats(df: pd.DataFrame, numerator_col: str, denominator_col: str, p_threshold: float = DEFAULT_P_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    z_score = norm.ppf(1 - p_threshold / 2)
    out["recall_adjust"] = (out[numerator_col] + 2) / (out[denominator_col] + 4)
    out["SE_recall"] = np.sqrt(out["recall_adjust"] * (1 - out["recall_adjust"]) / (out[denominator_col] + 4))
    out["CI_recall_low"] = out["recall"] - z_score * out["SE_recall"]
    out["CI_recall_high"] = out["recall"] + z_score * out["SE_recall"]
    return out


def _add_precision_linking_stats(df: pd.DataFrame, numerator_col: str, denominator_col: str, p_threshold: float = DEFAULT_P_THRESHOLD) -> pd.DataFrame:
    out = df.copy()
    z_score = norm.ppf(1 - p_threshold / 2)
    out["precision_adjust"] = (out[numerator_col] + 2) / (out[denominator_col] + 4)
    out["SE_precision"] = np.sqrt(out["precision_adjust"] * (1 - out["precision_adjust"]) / (out[denominator_col] + 4))
    out["CI_precision_low"] = out["precision"] - z_score * out["SE_precision"]
    out["CI_precision_high"] = out["precision"] + z_score * out["SE_precision"]
    return out


def _load_bg_variant_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _cat_command(path: Path) -> str:
    return f"zcat {path}" if path.suffix == ".gz" else f"cat {path}"


def _bg_with_bedtools_available() -> bool:
    return shutil.which("bedtools") is not None


def _write_prediction_bed(df: pd.DataFrame, path: Path, chrom_order: dict[str, int]) -> None:
    ordered = _sort_bed_like(df, chrom_order, "chrom", "start", "end")
    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        ordered[["chrom", "start", "end", "biosample", "target_gene", "score"]].to_csv(handle, sep="\t", header=False, index=False)


def _max_bg_scores_with_bedtools(pred_df: pd.DataFrame, bg_variants_path: Path, chrom_order: dict[str, int], chr_sizes: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        pred_path = tmp_root / "predictions.bed.gz"
        out_path = tmp_root / "bg_intersections.tsv"
        _write_prediction_bed(pred_df, pred_path, chrom_order)
        command = (
            f"set -euo pipefail; "
            f"{_cat_command(bg_variants_path)} | "
            f"bedtools intersect -wa -wb -sorted -a stdin -b <({_cat_command(pred_path)}) -g {chr_sizes} > {out_path}"
        )
        subprocess.run(["bash", "-lc", command], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not out_path.exists() or out_path.stat().st_size == 0:
            return np.array([], dtype=float)
        df = pd.read_csv(out_path, sep="\t", header=None)
        if df.empty:
            return np.array([], dtype=float)
        bg_key = df.iloc[:, :4].astype(str).agg("|".join, axis=1)
        scores = pd.to_numeric(df.iloc[:, -1], errors="coerce")
        grouped = pd.DataFrame({"bg_key": bg_key, "score": scores}).groupby("bg_key", sort=False)["score"].max()
        return grouped.to_numpy(dtype=float)


def _max_bg_scores_with_intervaltree(pred_df: pd.DataFrame, bg_variants_df: pd.DataFrame) -> np.ndarray:
    trees = _build_interval_trees(pred_df, chrom_col="chrom", start_col="start", end_col="end", data_cols=["target_gene", "score"])
    scores: list[float] = []
    for row in bg_variants_df[["chrom", "start", "end"]].itertuples(index=False):
        hits = trees.get(str(row.chrom), IntervalTree()).overlap(int(row.start), int(row.end))
        if hits:
            scores.append(max(float(hit.data[1]) for hit in hits))
    return np.asarray(scores, dtype=float)


def _compute_bg_max_scores(pred_df: pd.DataFrame, bg_variants_path: Path, chrom_order: dict[str, int], chr_sizes: Path) -> tuple[np.ndarray, int]:
    bg_total = _load_bg_variant_count(bg_variants_path)
    if _bg_with_bedtools_available():
        try:
            return _max_bg_scores_with_bedtools(pred_df, bg_variants_path, chrom_order, chr_sizes), bg_total
        except Exception as exc:  # pragma: no cover - best-effort acceleration path
            raise ScoringError(
                "bedtools-based background scoring failed. "
                "Make sure bedtools is installed and the provided background SNP BED is sorted."
            ) from exc
    if bg_total > 250_000:
        raise ScoringError(
            "Background SNP scoring without bedtools is only intended for small test fixtures. "
            f"Install bedtools or provide a much smaller background BED; current line count is {bg_total}."
        )
    bg_df = _read_tsv_maybe_gzip(bg_variants_path, header=None)
    bg_df = bg_df.iloc[:, :3].copy()
    bg_df.columns = ["chrom", "start", "end"]
    return _max_bg_scores_with_intervaltree(pred_df, bg_df), bg_total


def _load_eqtl_mapping() -> list[dict[str, object]]:
    mapping = pd.read_csv(SUPP_TABLE_S4, sep="\t", comment="#")
    mapping = mapping[mapping["analysis_type"] == "fine_grained"].copy()
    rows: list[dict[str, object]] = []
    for row in mapping[["cell_type", "variant_biosample"]].itertuples(index=False):
        tissues = [piece.strip() for piece in str(row.variant_biosample).split(",") if piece.strip()]
        rows.append({"biosample": str(row.cell_type), "tissues": tissues})
    return rows


def _prepare_eqtl_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = _read_gene_names_from_tss(EQTL_TSS)
    partition_trees = _build_partition_trees(EQTL_PARTITION)
    eqtl = _read_tsv_maybe_gzip(EQTL_VARIANTS_FINE)
    eqtl = eqtl.rename(columns={"gene_hgnc": "target_gene", "tissue": "eqtl_tissue", "chr": "chrom"})
    eqtl = eqtl[eqtl["target_gene"].isin(genes)].copy()
    eqtl["start"] = pd.to_numeric(eqtl["start"], errors="raise").astype(int)
    eqtl["end"] = pd.to_numeric(eqtl["end"], errors="raise").astype(int)
    eqtl = _filter_to_partition(eqtl, partition_trees, "chrom", "start", "end")
    vargene = eqtl[["chrom", "start", "end", "target_gene", "eqtl_tissue"]].drop_duplicates().reset_index(drop=True)
    variants = eqtl[["chrom", "start", "end", "eqtl_tissue"]].drop_duplicates().reset_index(drop=True)
    return variants, vargene


def _query_variant_any_scores(pred_df: pd.DataFrame, variants_df: pd.DataFrame) -> pd.DataFrame:
    trees = _build_interval_trees(pred_df, chrom_col="chrom", start_col="start", end_col="end", data_cols=["target_gene", "score"])
    rows = []
    for row in variants_df.itertuples(index=False):
        hits = trees.get(str(row.chrom), IntervalTree()).overlap(int(row.start), int(row.end))
        if hits:
            rows.append(
                {
                    "chrom": str(row.chrom),
                    "start": int(row.start),
                    "end": int(row.end),
                    "eqtl_tissue": str(row.eqtl_tissue),
                    "score_any": max(float(hit.data[1]) for hit in hits),
                }
            )
    return pd.DataFrame(rows)


def _query_vargene_scores(pred_df: pd.DataFrame, vargene_df: pd.DataFrame) -> pd.DataFrame:
    trees = _build_interval_trees(pred_df, chrom_col="chrom", start_col="start", end_col="end", data_cols=["target_gene", "score"])
    rows = []
    for row in vargene_df.itertuples(index=False):
        hits = trees.get(str(row.chrom), IntervalTree()).overlap(int(row.start), int(row.end))
        if not hits:
            continue
        any_score = max(float(hit.data[1]) for hit in hits)
        link_scores = [float(hit.data[1]) for hit in hits if str(hit.data[0]) == str(row.target_gene)]
        link_score = max(link_scores) if link_scores else math.nan
        rows.append(
            {
                "chrom": str(row.chrom),
                "start": int(row.start),
                "end": int(row.end),
                "target_gene": str(row.target_gene),
                "eqtl_tissue": str(row.eqtl_tissue),
                "score_any": any_score,
                "score_link": link_score,
            }
        )
    return pd.DataFrame(rows)


def score_eqtl_predictions(
    predictions: pd.DataFrame,
    *,
    bg_variants_path: Path,
    reference_styles: dict[str, MethodStyle],
    score_threshold: float | None = None,
    n_steps: int = DEFAULT_N_THRESHOLD_STEPS,
) -> dict[str, pd.DataFrame]:
    variants_df, vargene_df = _prepare_eqtl_truth()
    mapping_rows = _load_eqtl_mapping()
    chrom_order = _read_chrom_order(EQTL_CHR_SIZES)
    curve_rows: list[dict[str, object]] = []
    fixed_rows: list[dict[str, object]] = []
    for method, method_df in predictions.groupby("method", sort=False):
        style = _get_method_style(method, reference_styles)
        method_pairs = [row for row in mapping_rows if row["biosample"] in set(method_df["biosample"])]
        if not method_pairs:
            continue
        variant_pair_scores: dict[tuple[str, str], np.ndarray] = {}
        vargene_pair_any_scores: dict[tuple[str, str], np.ndarray] = {}
        vargene_pair_link_scores: dict[tuple[str, str], np.ndarray] = {}
        bg_scores_by_biosample: dict[str, np.ndarray] = {}
        total_variants_by_tissue = variants_df.groupby("eqtl_tissue").size().to_dict()
        total_vargene_by_tissue = vargene_df.groupby("eqtl_tissue").size().to_dict()

        for biosample, pred_bio in method_df.groupby("biosample", sort=False):
            pred_bio = pred_bio[["chrom", "start", "end", "biosample", "target_gene", "score"]].drop_duplicates().reset_index(drop=True)
            tissues = next((row["tissues"] for row in method_pairs if row["biosample"] == biosample), [])
            if not tissues:
                continue
            var_hits = _query_variant_any_scores(pred_bio, variants_df[variants_df["eqtl_tissue"].isin(tissues)])
            if not var_hits.empty:
                for tissue, sub in var_hits.groupby("eqtl_tissue", sort=False):
                    variant_pair_scores[(biosample, tissue)] = sub["score_any"].to_numpy(dtype=float)
            vg_hits = _query_vargene_scores(pred_bio, vargene_df[vargene_df["eqtl_tissue"].isin(tissues)])
            if not vg_hits.empty:
                for tissue, sub in vg_hits.groupby("eqtl_tissue", sort=False):
                    vargene_pair_any_scores[(biosample, tissue)] = sub["score_any"].to_numpy(dtype=float)
                    vargene_pair_link_scores[(biosample, tissue)] = sub["score_link"].to_numpy(dtype=float)
            bg_scores, bg_total = _compute_bg_max_scores(pred_bio, bg_variants_path, chrom_order, EQTL_CHR_SIZES)
            bg_scores_by_biosample[biosample] = bg_scores

        all_variant_scores = np.concatenate([values for values in variant_pair_scores.values() if values.size > 0], dtype=float) if variant_pair_scores else np.array([], dtype=float)
        if all_variant_scores.size == 0:
            continue
        thresholds = _quantile_thresholds(all_variant_scores, score_threshold, n_steps=n_steps)
        overlap_variant_counts = {key: _counts_at_threshold(values, thresholds) for key, values in variant_pair_scores.items()}
        overlap_vargene_counts = {key: _counts_at_threshold(values, thresholds) for key, values in vargene_pair_any_scores.items()}
        link_vargene_counts = {key: _counts_at_threshold(values, thresholds) for key, values in vargene_pair_link_scores.items()}
        bg_overlap_counts = {biosample: _counts_at_threshold(values, thresholds) for biosample, values in bg_scores_by_biosample.items()}

        pair_keys = [
            (str(row["biosample"]), tissue)
            for row in method_pairs
            for tissue in row["tissues"]
            if (str(row["biosample"]), tissue) in overlap_variant_counts
        ]
        if not pair_keys:
            continue
        aggregated_rows = []
        total_pairs = len(pair_keys)
        for index, threshold in enumerate(thresholds):
            n_variants_total = sum(total_variants_by_tissue.get(tissue, 0) for _, tissue in pair_keys)
            n_variants_overlap = sum(int(overlap_variant_counts[(biosample, tissue)][index]) for biosample, tissue in pair_keys)
            n_vargene_total = sum(total_vargene_by_tissue.get(tissue, 0) for _, tissue in pair_keys)
            n_vargene_overlap = sum(int(overlap_vargene_counts[(biosample, tissue)][index]) for biosample, tissue in pair_keys)
            n_vargene_link = sum(int(link_vargene_counts[(biosample, tissue)][index]) for biosample, tissue in pair_keys)
            n_common_overlap = sum(int(bg_overlap_counts[biosample][index]) for biosample, _ in pair_keys if biosample in bg_overlap_counts)
            n_common_total = int(bg_total) * total_pairs if pair_keys else EQTL_BG_COUNT_FALLBACK * total_pairs
            enrichment = _safe_ratio(n_variants_overlap, n_variants_total) / max(_safe_ratio(n_common_overlap, n_common_total), np.finfo(float).eps)
            aggregated_rows.append(
                {
                    "Biosample": "all_matches",
                    "threshold": float(threshold),
                    "GTExTissue": "all_matches",
                    "nVariantGenePairsOverlappingEnhancers": n_vargene_overlap,
                    "nVariantsOverlappingEnhancersCorrectGene": n_vargene_link,
                    "total.variants": n_vargene_total,
                    "recall.total": _safe_ratio(n_vargene_overlap, n_vargene_total),
                    "recall.linking": _safe_ratio(n_vargene_link, n_vargene_total),
                    "correctGene.ifOverlap": _safe_ratio(n_vargene_link, n_vargene_overlap),
                    "method": method,
                    "nVariantsTotal": n_variants_total,
                    "nVariantsOverlappingEnhancers": n_variants_overlap,
                    "nCommonVariantsTotal": n_common_total,
                    "nCommonVariantsOverlappingEnhancers": n_common_overlap,
                    "enrichment": enrichment,
                    "pred_name_long": style.display_name,
                    "hex": style.color,
                    "key": f"{style.display_name} (all_matches)",
                    "nPoints": len(thresholds),
                }
            )
        curve = pd.DataFrame(aggregated_rows)
        curve = _add_enrichment_stats(curve)
        curve["CI_enr_low"] = curve["CI_enr_low"].fillna(0.0)
        curve["CI_enr_high"] = curve["CI_enr_high"].fillna(np.nan)
        curve["SE_log_enr"] = curve["SE_log_enr"].replace([np.inf, -np.inf], np.nan)
        selected_threshold = score_threshold
        if selected_threshold is None:
            best_index = (curve["recall.linking"] - DEFAULT_EQTL_TARGET_RECALL).abs().idxmin()
            selected_threshold = float(curve.loc[best_index, "threshold"])
        curve["score_threshold"] = float(selected_threshold)
        curve["p_adjust_enr"] = curve["p_adjust_enr"].fillna(1.0)
        curve_rows.extend(curve[[
            "Biosample",
            "threshold",
            "GTExTissue",
            "nVariantGenePairsOverlappingEnhancers",
            "nVariantsOverlappingEnhancersCorrectGene",
            "total.variants",
            "recall.total",
            "recall.linking",
            "correctGene.ifOverlap",
            "method",
            "enrichment",
            "CI_enr_low",
            "CI_enr_high",
            "SE_log_enr",
            "p_adjust_enr",
            "nPoints",
            "pred_name_long",
            "score_threshold",
            "key",
            "hex",
        ]].to_dict("records"))
        fixed_index = (curve["recall.linking"] - DEFAULT_EQTL_TARGET_RECALL).abs().idxmin()
        fixed = curve.loc[[fixed_index], [
            "Biosample",
            "threshold",
            "GTExTissue",
            "nVariantGenePairsOverlappingEnhancers",
            "nVariantsOverlappingEnhancersCorrectGene",
            "total.variants",
            "recall.total",
            "recall.linking",
            "correctGene.ifOverlap",
            "method",
            "enrichment",
            "CI_enr_low",
            "CI_enr_high",
            "SE_log_enr",
            "p_adjust_enr",
            "pred_name_long",
            "hex",
            "key",
        ]].copy()
        fixed["recall.linking.rounded"] = fixed["recall.linking"].round(3)
        fixed["plotting_label"] = fixed.apply(
            lambda row: f"{row['pred_name_long']} (all_matches) ({row['recall.linking.rounded']})",
            axis=1,
        )
        fixed_rows.extend(fixed.to_dict("records"))

    curve_df = pd.DataFrame(curve_rows)
    fixed_df = pd.DataFrame(fixed_rows)
    pairwise_df = pd.DataFrame(
        columns=[
            "group1",
            "group2",
            "p.value",
            "p.adjust",
            "group1_name",
            "group2_name",
        ]
    )
    return {
        "curve_table": curve_df,
        "fixed_recall_table": fixed_df,
        "pairwise_significance": pairwise_df,
    }


def _load_gwas_pairings(column: str) -> list[dict[str, object]]:
    table = pd.read_csv(SUPP_TABLE_S5, sep="\t", comment="#")
    table = table[table["analysis_id"] == "fine_grained"].copy()
    table = table[_to_bool(table[column])].copy()
    rows: list[dict[str, object]] = []
    for row in table[["cell_type_category", "cell_types", "trait"]].itertuples(index=False):
        biosamples = [piece.strip() for piece in str(row.cell_types).split(",") if piece.strip()]
        traits = [piece.strip() for piece in str(row.trait).split(",") if piece.strip()]
        rows.append({"category": str(row.cell_type_category), "biosamples": biosamples, "traits": traits})
    return rows


def _load_gwas_variant_files(relevant_traits: set[str]) -> pd.DataFrame:
    variant_key = pd.read_csv(GWAS_VARIANT_KEY, sep="\t")
    rows = []
    for record in variant_key.itertuples(index=False):
        trait = str(record.trait)
        if trait not in relevant_traits:
            continue
        path = GWAS_VARIANT_ROOT / str(record.variant_file)
        variant_df = pd.read_csv(path, sep="\t")
        variant_df = variant_df[
            (pd.to_numeric(variant_df["pip"], errors="coerce") > 0.1)
            & (~_to_bool(variant_df["Coding"]))
            & (~_to_bool(variant_df["SpliceSite"]))
            & (~_to_bool(variant_df["Promoter"]))
        ].copy()
        variant_df = variant_df.rename(columns={"chr": "chrom", "Disease": "trait", "TargetGene": "target_gene"})
        variant_df["chrom"] = variant_df["chrom"].astype(str)
        variant_df["start"] = pd.to_numeric(variant_df["start"], errors="raise").astype(int)
        variant_df["end"] = pd.to_numeric(variant_df["end"], errors="raise").astype(int)
        rows.append(variant_df[["chrom", "start", "end", "rsid", "CredibleSet", "trait"]])
    if not rows:
        return pd.DataFrame(columns=["chrom", "start", "end", "rsid", "CredibleSet", "trait"])
    return pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)


def _load_gene_prioritization() -> pd.DataFrame:
    gene_pri = pd.read_csv(GWAS_GENE_PRIORITIZATION, sep="\t")
    gene_pri = gene_pri.rename(columns={"Disease": "trait"})
    genes = _read_gene_names_from_tss(GWAS_TSS)
    gene_pri = gene_pri[gene_pri["TargetGene"].isin(genes)].copy()
    return gene_pri


def _query_gwas_variant_any_scores(pred_df: pd.DataFrame, variants_df: pd.DataFrame) -> pd.DataFrame:
    trees = _build_interval_trees(pred_df, chrom_col="chrom", start_col="start", end_col="end", data_cols=["target_gene", "score"])
    rows = []
    for row in variants_df.itertuples(index=False):
        hits = trees.get(str(row.chrom), IntervalTree()).overlap(int(row.start), int(row.end))
        if hits:
            rows.append(
                {
                    "chrom": str(row.chrom),
                    "start": int(row.start),
                    "end": int(row.end),
                    "trait": str(row.trait),
                    "score_any": max(float(hit.data[1]) for hit in hits),
                }
            )
    return pd.DataFrame(rows)


def _query_gwas_variant_gene_predictions(pred_df: pd.DataFrame, variants_df: pd.DataFrame) -> pd.DataFrame:
    trees = _build_interval_trees(pred_df, chrom_col="chrom", start_col="start", end_col="end", data_cols=["target_gene", "score"])
    rows = []
    for row in variants_df.itertuples(index=False):
        hits = trees.get(str(row.chrom), IntervalTree()).overlap(int(row.start), int(row.end))
        if not hits:
            continue
        for hit in hits:
            rows.append(
                {
                    "CredibleSet": str(row.CredibleSet),
                    "trait": str(row.trait),
                    "TargetGene": str(hit.data[0]),
                    "predScore": float(hit.data[1]),
                }
            )
    return pd.DataFrame(rows)


def score_gwas_predictions(
    predictions: pd.DataFrame,
    *,
    bg_variants_path: Path,
    reference_styles: dict[str, MethodStyle],
    score_threshold: float | None = None,
    n_steps: int = DEFAULT_N_THRESHOLD_STEPS,
) -> dict[str, pd.DataFrame]:
    overlap_pairs = _load_gwas_pairings("variant_overlap")
    linking_pairs = _load_gwas_pairings("gene_linking")
    relevant_traits = {trait for row in overlap_pairs + linking_pairs for trait in row["traits"]}
    variants_df = _load_gwas_variant_files(relevant_traits)
    gene_pri = _load_gene_prioritization()
    chrom_order = _read_chrom_order(GWAS_CHR_SIZES)

    curve_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    linking_rows: list[dict[str, object]] = []

    for method, method_df in predictions.groupby("method", sort=False):
        style = _get_method_style(method, reference_styles)
        available_biosamples = set(method_df["biosample"])
        overlap_groups = [row for row in overlap_pairs if set(row["biosamples"]) & available_biosamples]
        linking_groups = [row for row in linking_pairs if set(row["biosamples"]) & available_biosamples]
        if not overlap_groups and not linking_groups:
            continue

        bg_scores_by_group: dict[str, np.ndarray] = {}
        group_predictions: dict[str, pd.DataFrame] = {}
        for row in overlap_groups + linking_groups:
            category = str(row["category"])
            if category in group_predictions:
                continue
            pred_group = method_df[method_df["biosample"].isin(row["biosamples"])][["chrom", "start", "end", "biosample", "target_gene", "score"]].drop_duplicates().reset_index(drop=True)
            if pred_group.empty:
                continue
            group_predictions[category] = pred_group
            bg_scores_by_group[category], bg_total = _compute_bg_max_scores(pred_group, bg_variants_path, chrom_order, GWAS_CHR_SIZES)

        # Variant-overlap curves and thresholded point
        pair_variant_scores: dict[tuple[str, str], np.ndarray] = {}
        total_variants_per_trait = variants_df.groupby("trait")[["chrom", "start", "end"]].apply(lambda frame: frame.drop_duplicates().shape[0]).to_dict()
        for row in overlap_groups:
            category = str(row["category"])
            pred_group = group_predictions.get(category)
            if pred_group is None:
                continue
            traits = row["traits"]
            hits = _query_gwas_variant_any_scores(pred_group, variants_df[variants_df["trait"].isin(traits)])
            if hits.empty:
                continue
            for trait, sub in hits.groupby("trait", sort=False):
                pair_variant_scores[(category, trait)] = sub["score_any"].to_numpy(dtype=float)

        if pair_variant_scores:
            all_scores = np.concatenate([values for values in pair_variant_scores.values() if values.size > 0], dtype=float)
            thresholds = _quantile_thresholds(all_scores, score_threshold, n_steps=n_steps)
            overlap_counts = {key: _counts_at_threshold(values, thresholds) for key, values in pair_variant_scores.items()}
            bg_overlap_counts = {category: _counts_at_threshold(scores, thresholds) for category, scores in bg_scores_by_group.items()}
            pair_keys = list(pair_variant_scores.keys())
            aggregated = []
            for index, threshold in enumerate(thresholds):
                n_variants_total = sum(total_variants_per_trait.get(trait, 0) for _, trait in pair_keys)
                n_variants_overlap = sum(int(overlap_counts[(category, trait)][index]) for category, trait in pair_keys)
                n_common_overlap = sum(int(bg_overlap_counts[category][index]) for category, _ in pair_keys if category in bg_overlap_counts)
                n_common_total = int(bg_total) * len(pair_keys)
                enrichment = _safe_ratio(n_variants_overlap, n_variants_total) / max(_safe_ratio(n_common_overlap, n_common_total), np.finfo(float).eps)
                aggregated.append(
                    {
                        "trait": "all_matched",
                        "nVariantsOverlappingEnhancers": n_variants_overlap,
                        "nCommonVariantsOverlappingEnhancers": n_common_overlap,
                        "threshold": float(threshold),
                        "nVariantsTotal": n_variants_total,
                        "nCommonVariantsTotal": n_common_total,
                        "recall": _safe_ratio(n_variants_overlap, n_variants_total),
                        "enrichment": enrichment,
                        "biosample": "all_matched",
                        "method": method,
                        "key": "all_matched.all_matched",
                        "pred_name_long": style.display_name,
                        "hex": style.color,
                        "n_points": len(thresholds),
                    }
                )
            curve = pd.DataFrame(aggregated)
            curve = _add_enrichment_stats(curve)
            curve = _add_recall_overlap_stats(curve)
            if score_threshold is None:
                utility = curve["recall"].to_numpy(dtype=float) * curve["enrichment"].to_numpy(dtype=float)
                selected_threshold = float(curve.iloc[int(np.nanargmax(utility))]["threshold"])
            else:
                selected_threshold = float(score_threshold)
            curve["score_threshold"] = selected_threshold
            curve_rows.extend(curve.to_dict("records"))
            overlap_index = (curve["threshold"] - selected_threshold).abs().idxmin()
            overlap_rows.extend(curve.loc[[overlap_index]].to_dict("records"))
        else:
            selected_threshold = float(score_threshold) if score_threshold is not None else 0.0

        # Gene linking at threshold
        if not linking_groups:
            continue
        gene_pri_subset = gene_pri[gene_pri["trait"].isin({trait for row in linking_groups for trait in row["traits"]})].copy()
        cs_per_trait = gene_pri_subset[["CredibleSet", "trait"]].drop_duplicates().groupby("trait").size().to_dict()
        truth_genes = gene_pri_subset[_to_bool(gene_pri_subset["truth"])][["CredibleSet", "trait", "TargetGene"]].rename(columns={"TargetGene": "TruthGene"})
        pops_genes = (
            gene_pri_subset[["CredibleSet", "trait", "TargetGene", "POPS.Score"]]
            .sort_values(["CredibleSet", "trait", "POPS.Score"], ascending=[True, True, False])
            .groupby(["CredibleSet", "trait"])
            .head(DEFAULT_GWAS_TOP_POPS_GENES)
            .rename(columns={"TargetGene": "PoPSGene"})
        )

        link_pair_rows = []
        for row in linking_groups:
            category = str(row["category"])
            pred_group = group_predictions.get(category)
            if pred_group is None:
                continue
            pred_group = pred_group[pred_group["score"] >= selected_threshold].reset_index(drop=True)
            if pred_group.empty:
                for trait in row["traits"]:
                    link_pair_rows.append(
                        {
                            "trait": trait,
                            "biosample": category,
                            "group": True,
                            "intersectPoPS": False,
                            "nCredibleSetsOverlappingEnhancers": 0,
                            "nCredibleSetsOverlappingEnhancersAnyGene": 0,
                            "nCredibleSetsOverlappingEnhancersCorrectGene": 0,
                            "nCredibleSetsTotal": cs_per_trait.get(trait, 0),
                        }
                    )
                    link_pair_rows.append(
                        {
                            "trait": trait,
                            "biosample": category,
                            "group": True,
                            "intersectPoPS": True,
                            "nCredibleSetsOverlappingEnhancers": 0,
                            "nCredibleSetsOverlappingEnhancersAnyGene": 0,
                            "nCredibleSetsOverlappingEnhancersCorrectGene": 0,
                            "nCredibleSetsTotal": cs_per_trait.get(trait, 0),
                        }
                    )
                continue
            trait_variants = variants_df[variants_df["trait"].isin(row["traits"])].copy()
            raw_predictions = _query_gwas_variant_gene_predictions(pred_group, trait_variants)
            for trait in row["traits"]:
                raw_trait = raw_predictions[raw_predictions["trait"] == trait].copy()
                if raw_trait.empty:
                    for pops_flag in (False, True):
                        link_pair_rows.append(
                            {
                                "trait": trait,
                                "biosample": category,
                                "group": True,
                                "intersectPoPS": pops_flag,
                                "nCredibleSetsOverlappingEnhancers": 0,
                                "nCredibleSetsOverlappingEnhancersAnyGene": 0,
                                "nCredibleSetsOverlappingEnhancersCorrectGene": 0,
                                "nCredibleSetsTotal": cs_per_trait.get(trait, 0),
                            }
                        )
                    continue
                pred_only = (
                    raw_trait.groupby(["CredibleSet", "trait", "TargetGene"], sort=False)["predScore"]
                    .max()
                    .reset_index()
                    .sort_values(["CredibleSet", "predScore"], ascending=[True, False])
                )
                pred_only["predRank"] = pred_only.groupby("CredibleSet")["predScore"].rank(method="min", ascending=False)
                pred_only = pred_only[pred_only["predRank"] <= DEFAULT_GWAS_TOP_PRED_GENES].copy()
                truth_this = truth_genes[truth_genes["trait"] == trait]
                pred_truth = pred_only.merge(truth_this, on=["CredibleSet", "trait"], how="left")
                pops_this = pops_genes[pops_genes["trait"] == trait]
                pred_pops = pred_truth.merge(pops_this[["CredibleSet", "trait", "PoPSGene"]], on=["CredibleSet", "trait"], how="left")
                pred_pops = pred_pops[pred_pops["TargetGene"] == pred_pops["PoPSGene"]].copy()
                for subset, pops_flag in ((pred_truth, False), (pred_pops, True)):
                    n_overlap = subset[["CredibleSet", "TruthGene"]].drop_duplicates().shape[0]
                    n_any_gene = subset[["CredibleSet", "TargetGene"]].drop_duplicates().shape[0]
                    n_correct = subset[subset["TargetGene"] == subset["TruthGene"]][["CredibleSet", "TargetGene", "TruthGene"]].drop_duplicates().shape[0]
                    link_pair_rows.append(
                        {
                            "trait": trait,
                            "biosample": category,
                            "group": True,
                            "intersectPoPS": pops_flag,
                            "nCredibleSetsOverlappingEnhancers": n_overlap,
                            "nCredibleSetsOverlappingEnhancersAnyGene": n_any_gene,
                            "nCredibleSetsOverlappingEnhancersCorrectGene": n_correct,
                            "nCredibleSetsTotal": cs_per_trait.get(trait, 0),
                        }
                    )

        link_pair_df = pd.DataFrame(link_pair_rows)
        if link_pair_df.empty:
            continue
        for pops_flag in (False, True):
            subset = link_pair_df[link_pair_df["intersectPoPS"] == pops_flag].copy()
            if subset.empty:
                continue
            n_total = int(subset["nCredibleSetsTotal"].sum())
            n_any = int(subset["nCredibleSetsOverlappingEnhancersAnyGene"].sum())
            n_correct = int(subset["nCredibleSetsOverlappingEnhancersCorrectGene"].sum())
            n_overlap = int(subset["nCredibleSetsOverlappingEnhancers"].sum())
            summary = pd.DataFrame(
                [
                    {
                        "method": method,
                        "pred_name_long": style.display_name,
                        "intersectPoPS": pops_flag,
                        "nCredibleSetsTotal": n_total,
                        "nCredibleSetsOverlappingEnhancers": n_overlap,
                        "nCredibleSetsOverlappingEnhancersAnyGene": n_any,
                        "nCredibleSetsOverlappingEnhancersCorrectGene": n_correct,
                        "recall": _safe_ratio(n_correct, n_total),
                        "precision": _safe_ratio(n_correct, n_any),
                    }
                ]
            )
            summary = _add_recall_linking_stats(summary, "nCredibleSetsOverlappingEnhancersCorrectGene", "nCredibleSetsTotal")
            summary = _add_precision_linking_stats(summary, "nCredibleSetsOverlappingEnhancersCorrectGene", "nCredibleSetsOverlappingEnhancersAnyGene")
            linking_rows.extend(summary[[
                "method",
                "pred_name_long",
                "intersectPoPS",
                "nCredibleSetsTotal",
                "recall",
                "precision",
                "recall_adjust",
                "SE_recall",
                "CI_recall_low",
                "CI_recall_high",
                "precision_adjust",
                "SE_precision",
                "CI_precision_low",
                "CI_precision_high",
            ]].to_dict("records"))

    curve_df = pd.DataFrame(curve_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    linking_df = pd.DataFrame(linking_rows)
    return {
        "curve_table": curve_df,
        "thresholded_overlap": overlap_df,
        "thresholded_gene_linking": linking_df[linking_df["intersectPoPS"] == False].reset_index(drop=True),  # noqa: E712
        "thresholded_gene_linking_with_pops": linking_df[linking_df["intersectPoPS"] == True].reset_index(drop=True),  # noqa: E712
    }


def resolve_bg_variants_path(bg_variants: Path | None) -> Path:
    candidate = bg_variants
    if candidate is None:
        env_value = os.environ.get(BG_VARIANTS_ENV, "").strip()
        candidate = Path(env_value) if env_value else None
    if candidate is None:
        raise ScoringError(
            "Real benchmark scoring requires a distal-noncoding background SNP BED. "
            f"Pass --bg-variants or set {BG_VARIANTS_ENV}."
        )
    if not candidate.exists():
        raise ScoringError(f"Background SNP BED does not exist: {candidate}")
    return candidate

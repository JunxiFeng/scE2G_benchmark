from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eqtl import build_eqtl_outputs
from .gwas import build_gwas_outputs
from .io_utils import ensure_dir, read_tsv, write_df, write_json
from .paths import DEFAULT_OUTPUT_ROOT


METHOD_META = {
    "scE2G_multiome": {"num": 1, "short": "scE2G (Multiome)"},
    "scE2G_ATAC": {"num": 2, "short": "scE2G (scATAC)"},
    "Kendall": {"num": 4, "short": "Kendall correlation"},
    "scABC": {"num": 5, "short": "ABC (A=scATAC, C=power law)"},
    "STARE": {"num": 6, "short": "STARE"},
    "SnapATAC": {"num": 7, "short": "SnapATAC"},
    "Signac": {"num": 8, "short": "Signac"},
    "Cicero": {"num": 9, "short": "Cicero"},
    "FigR": {"num": 10, "short": "FigR"},
    "SCENT": {"num": 11, "short": "SCENT"},
    "ScenicPlus": {"num": 12, "short": "SCENIC+"},
    "DIRECTNET": {"num": 13, "short": "DIRECT-NET"},
    "ArchR": {"num": 14, "short": "ArchR"},
    "distanceToTSS": {"num": 15, "short": "Distance to TSS"},
    "ABC_distanceToTSS": {"num": 19, "short": "In element (ABC) & distance to TSS"},
}

PAPER_STYLE_METHOD_ORDER = [
    "scE2G_multiome",
    "distanceToTSS",
    "scE2G_ATAC",
    "scABC",
    "STARE",
    "Kendall",
    "SnapATAC",
    "Signac",
    "Cicero",
    "FigR",
    "SCENT",
    "ScenicPlus",
    "DIRECTNET",
    "ArchR",
    "ABC_distanceToTSS",
]

PRIMARY_CURVE_METHODS = ["scE2G_multiome", "scE2G_ATAC", "scABC", "STARE", "distanceToTSS", "ABC_distanceToTSS"]

PANEL_LABEL_OFFSETS = {
    "d": {"scE2G_multiome": (0.008, 2.0), "scE2G_ATAC": (0.009, 1.3), "scABC": (0.009, 0.4), "ABC_distanceToTSS": (0.009, -0.3), "Kendall": (0.008, -1.0), "SnapATAC": (0.009, 0.2), "Signac": (0.009, -0.9), "Cicero": (0.009, -1.8), "FigR": (0.009, -0.2), "ScenicPlus": (0.009, 0.8), "DIRECTNET": (0.009, -1.4), "ArchR": (0.009, 1.3), "STARE": (0.009, 0.7), "SCENT": (0.009, 0.6), "distanceToTSS": (0.009, 1.5)},
    "f": {"scE2G_multiome": (0.010, 0.4), "scE2G_ATAC": (0.010, -0.1), "scABC": (0.010, 0.2), "ABC_distanceToTSS": (0.010, -0.3), "Kendall": (0.010, -0.6), "SnapATAC": (0.010, 0.3), "Signac": (0.010, -0.2), "FigR": (0.010, -0.7), "ScenicPlus": (0.010, 0.1), "DIRECTNET": (0.010, -0.5), "ArchR": (0.010, 0.5), "STARE": (0.010, 0.3), "SCENT": (0.010, 0.1), "distanceToTSS": (0.010, -0.4)},
    "g": {"scE2G_multiome": (0.012, -0.025), "scE2G_ATAC": (0.012, 0.015), "scABC": (0.012, -0.05), "ABC_distanceToTSS": (0.012, 0.035), "Kendall": (0.012, -0.005), "SnapATAC": (0.008, 0.02), "Signac": (0.008, 0.01), "Cicero": (0.008, 0.03), "FigR": (0.008, 0.015), "ScenicPlus": (0.008, -0.005), "DIRECTNET": (0.008, 0.02), "ArchR": (0.008, -0.02), "STARE": (0.012, 0.02), "SCENT": (0.012, 0.01), "distanceToTSS": (0.012, -0.03)},
    "h": {"scE2G_multiome": (0.010, -0.035), "scE2G_ATAC": (0.010, 0.03), "scABC": (0.010, 0.0), "ABC_distanceToTSS": (0.010, -0.03), "Kendall": (0.010, -0.06), "SnapATAC": (0.008, 0.015), "Signac": (0.008, -0.015), "Cicero": (0.008, 0.04), "FigR": (0.008, 0.015), "ScenicPlus": (0.008, 0.02), "DIRECTNET": (0.008, -0.01), "ArchR": (0.008, 0.03), "STARE": (0.010, 0.02), "SCENT": (0.010, 0.01), "distanceToTSS": (0.010, -0.04)},
}

LOCAL_SUBSET_NOTE = (
    "Local subset debug mode: missing exact paper eQTL methods 6 (STARE), 11 (SCENT), "
    "and a standalone method 15 distance-to-TSS series."
)


def savefig(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def theme_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def add_note(fig: plt.Figure, note: str | None) -> None:
    if note:
        fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8, color="#555555")


def nearest_threshold_rows(df: pd.DataFrame, threshold_col: str, score_col: str) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("method", sort=False):
        idx = (sub[threshold_col] - sub[score_col]).abs().idxmin()
        rows.append(df.loc[idx])
    return pd.DataFrame(rows)


def build_palette(*dfs: pd.DataFrame) -> dict[str, str]:
    palette: dict[str, str] = {}
    for df in dfs:
        if "hex" not in df.columns:
            continue
        for _, row in df[["method", "hex"]].drop_duplicates().iterrows():
            method = row["method"]
            if pd.notna(row["hex"]):
                palette[method] = row["hex"]
    return palette


def add_method_label(ax: plt.Axes, method: str, x: float, y: float, panel: str, color: str) -> None:
    dx, dy = PANEL_LABEL_OFFSETS.get(panel, {}).get(method, (0.008, 0.0))
    label = str(METHOD_META[method]["num"]) if method in METHOD_META else method
    ax.text(x + dx, y + dy, label, color=color, fontsize=8, va="center", ha="left")


def dynamic_method_order(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].dropna().unique().tolist())
    known = [method for method in PAPER_STYLE_METHOD_ORDER if method in present]
    unknown = sorted(method for method in present if method not in PAPER_STYLE_METHOD_ORDER)
    return known + unknown


def plot_curve_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    threshold_col: str,
    score_col: str,
    palette: dict[str, str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    panel: str,
    title: str,
    xlabel: str,
    ylabel: str,
    n_text: str,
) -> None:
    threshold_rows = nearest_threshold_rows(df, threshold_col, score_col)
    for method in dynamic_method_order(df):
        sub = df[df["method"] == method].sort_values(x_col)
        color = palette.get(method, "#555555")
        is_primary = method in PRIMARY_CURVE_METHODS
        if len(sub) > 2:
            ax.plot(sub[x_col], sub[y_col], color=color, linewidth=2.1 if is_primary else 1.4, alpha=1.0 if is_primary else 0.65)
        row = threshold_rows[threshold_rows["method"] == method].iloc[0]
        ax.scatter(row[x_col], row[y_col], s=80 if is_primary else 58, color=color, edgecolor="white", linewidth=0.6, zorder=4)
        add_method_label(ax, method, float(row[x_col]), float(row[y_col]), panel, color)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(f"{xlabel}\n{n_text}", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    theme_axis(ax)


def plot_bar_panel(ax: plt.Axes, df: pd.DataFrame, palette: dict[str, str]) -> None:
    plot_df = df.copy()
    plot_df["paper_number"] = plot_df["method"].map(lambda method: METHOD_META[method]["num"] if method in METHOD_META else math.inf)
    plot_df["paper_label"] = plot_df["method"].map(lambda method: str(METHOD_META[method]["num"]) if method in METHOD_META else method)
    plot_df["known_method"] = plot_df["method"].map(lambda method: method in METHOD_META)
    plot_df = plot_df.sort_values(["known_method", "paper_number", "method"], ascending=[False, True, True]).reset_index(drop=True)
    x = np.arange(len(plot_df))
    colors = [palette.get(method, "#777777") for method in plot_df["method"]]
    ax.bar(x, plot_df["enrichment"], color=colors, width=0.86)
    yerr = np.vstack(
        [
            (plot_df["enrichment"] - plot_df["CI_enr_low"]).to_numpy(),
            (plot_df["CI_enr_high"] - plot_df["enrichment"]).to_numpy(),
        ]
    )
    ax.errorbar(x, plot_df["enrichment"], yerr=yerr, fmt="none", ecolor="black", elinewidth=1, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["paper_label"].astype(str).tolist(), fontsize=8)
    ax.set_ylabel("Enrichment at 5% recall\n(eQTLs versus common variants)", fontsize=9)
    ax.set_ylim(0, max(plot_df["CI_enr_high"]) * 1.08)
    theme_axis(ax)


def plot_linking_panel(ax: plt.Axes, df: pd.DataFrame, panel: str, title: str, xlim: tuple[float, float], xlabel: str) -> None:
    plot_df = df.copy()
    plot_df["CI_recall_low"] = plot_df["CI_recall_low"].clip(lower=0)
    plot_df["CI_recall_high"] = plot_df["CI_recall_high"].clip(lower=0, upper=1)
    plot_df["CI_precision_low"] = plot_df["CI_precision_low"].clip(lower=0)
    plot_df["CI_precision_high"] = plot_df["CI_precision_high"].clip(lower=0, upper=1)
    for method in dynamic_method_order(plot_df):
        sub = plot_df[plot_df["method"] == method]
        if sub.empty:
            continue
        row = sub.iloc[0]
        color = row.get("hex", "#555555")
        ax.plot([row["CI_recall_low"], row["CI_recall_high"]], [row["precision"], row["precision"]], color=color, linewidth=1.5)
        ax.plot([row["recall"], row["recall"]], [row["CI_precision_low"], row["CI_precision_high"]], color=color, linewidth=1.5)
        ax.scatter(row["recall"], row["precision"], s=76, color=color, alpha=0.78, edgecolor="white", linewidth=0.6, zorder=4)
        add_method_label(ax, method, float(row["recall"]), float(row["precision"]), panel, color)

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Precision\n(Fraction of correct\npredicted credible set-gene links)", fontsize=9)
    theme_axis(ax)


def add_paper_legend(ax: plt.Axes, palette: dict[str, str], note: str | None) -> None:
    ax.axis("off")
    ax.text(0.0, 0.98, "Predictors", fontsize=13, fontweight="bold", va="top")
    ax.text(0.0, 0.92, "Methods present in this run", fontsize=10, fontweight="bold", va="top")
    y = 0.88
    ordered = PAPER_STYLE_METHOD_ORDER + sorted(method for method in palette if method not in PAPER_STYLE_METHOD_ORDER)
    for method in ordered:
        if method not in palette:
            continue
        left = f"({METHOD_META[method]['num']})" if method in METHOD_META else ""
        right = METHOD_META[method]["short"] if method in METHOD_META else method
        ax.text(0.0, y, left, color=palette[method], fontsize=9, va="center")
        ax.text(0.15, y, right, color=palette[method], fontsize=8.5, va="center")
        y -= 0.055
    if note:
        ax.text(0.0, max(y - 0.03, 0.08), note, fontsize=8, color="#555555", va="top", wrap=True)


def build_full_figure() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(14.4, 8.6))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.45, 0.9, 1.2, 1.45], height_ratios=[1.0, 1.0], wspace=0.55, hspace=0.68)
    axes = {
        "d": fig.add_subplot(grid[0, 0]),
        "e": fig.add_subplot(grid[0, 1]),
        "f": fig.add_subplot(grid[1, 0]),
        "g": fig.add_subplot(grid[1, 1]),
        "h": fig.add_subplot(grid[1, 2]),
        "legend": fig.add_subplot(grid[:, 3]),
    }
    return fig, axes


def render_eqtl_and_gwas_panels(
    *,
    eqtl_curve: pd.DataFrame,
    eqtl_bar: pd.DataFrame,
    gwas_curve: pd.DataFrame,
    gwas_overlap: pd.DataFrame,
    gwas_linking: pd.DataFrame,
    output_dir: Path,
    note: str | None,
) -> dict[str, Path]:
    ensure_dir(output_dir)
    palette = build_palette(eqtl_curve, eqtl_bar, gwas_curve, gwas_overlap, gwas_linking)
    outputs: dict[str, Path] = {}

    def render_single_d() -> None:
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        plot_curve_panel(
            ax, eqtl_curve, "recall.linking", "enrichment", "threshold", "score_threshold",
            palette, (0.0, 0.25), (-1.0, 42.0), "d", "d", "Recall\n(Fraction of eVariants overlapping prediction linked to eGene)",
            "Enrichment\n(eQTLs versus common variants)", "26,233-67,287 variant-biosample pairs tested",
        )
        add_note(fig, note)
        path = output_dir / "figure2d.paper_style.pdf"
        savefig(fig, path)
        outputs["figure2d"] = path

    def render_single_e() -> None:
        fig, ax = plt.subplots(figsize=(4.0, 4.5))
        ax.set_title("e", loc="left", fontweight="bold", fontsize=12)
        plot_bar_panel(ax, eqtl_bar, palette)
        add_note(fig, note)
        path = output_dir / "figure2e.paper_style.pdf"
        savefig(fig, path)
        outputs["figure2e"] = path

    def render_single_f() -> None:
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        plot_curve_panel(
            ax, gwas_curve, "recall", "enrichment", "threshold", "score_threshold",
            palette, (0.0, 0.245), (-0.5, 9.5), "f", "f  Variant overlap", "Recall (Fraction of variants in predicted enhancers)",
            "Enrichment\n(GWAS variants versus common variants)", "7,209 variants from 11 traits tested\n12,035 variant-biosample pairs",
        )
        add_note(fig, note)
        path = output_dir / "figure2f.paper_style.pdf"
        savefig(fig, path)
        outputs["figure2f"] = path

    def render_single_g() -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        plot_linking_panel(
            ax, gwas_linking[gwas_linking["intersectPoPS"] == False].copy(),  # noqa: E712
            "g", "g  Linking variants to known genes:\nE2G model only", (-0.015, 0.31),
            "Recall\n(Fraction of credible sets linked to target gene)\n159 credible sets from 9 traits tested\n645 credible set-biosample pairs",
        )
        add_note(fig, note)
        path = output_dir / "figure2g.paper_style.pdf"
        savefig(fig, path)
        outputs["figure2g"] = path

    def render_single_h() -> None:
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        plot_linking_panel(
            ax, gwas_linking[gwas_linking["intersectPoPS"] == True].copy(),  # noqa: E712
            "h", "h  Linking variants to known genes:\nE2G model with PoPS", (-0.015, 0.31),
            "Recall\n(Fraction of credible sets linked to target gene)\n159 credible sets from 9 traits tested\n645 credible set-biosample pairs",
        )
        add_note(fig, note)
        path = output_dir / "figure2h.paper_style.pdf"
        savefig(fig, path)
        outputs["figure2h"] = path

    render_single_d()
    render_single_e()
    render_single_f()
    render_single_g()
    render_single_h()

    fig, axes = build_full_figure()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.09)
    fig.text(0.33, 0.98, "eQTL benchmark", fontsize=18, ha="center", va="top")
    fig.text(0.38, 0.49, "GWAS benchmark", fontsize=18, ha="center", va="top")

    axes["d"].set_title("d", loc="left", fontweight="bold", fontsize=12)
    plot_curve_panel(
        axes["d"], eqtl_curve, "recall.linking", "enrichment", "threshold", "score_threshold",
        palette, (0.0, 0.25), (-1.0, 42.0), "d", "",
        "Recall\n(Fraction of eVariants overlapping prediction linked to eGene)",
        "Enrichment\n(eQTLs versus common variants)", "26,233-67,287 variant-biosample pairs tested",
    )
    axes["e"].set_title("e", loc="left", fontweight="bold", fontsize=12)
    plot_bar_panel(axes["e"], eqtl_bar, palette)
    axes["f"].set_title("f", loc="left", fontweight="bold", fontsize=12)
    plot_curve_panel(
        axes["f"], gwas_curve, "recall", "enrichment", "threshold", "score_threshold",
        palette, (0.0, 0.245), (-0.5, 9.5), "f", "Variant overlap",
        "Recall (Fraction of variants in predicted enhancers)",
        "Enrichment\n(GWAS variants versus common variants)", "7,209 variants from 11 traits tested\n12,035 variant-biosample pairs",
    )
    axes["g"].set_title("g", loc="left", fontweight="bold", fontsize=12)
    plot_linking_panel(
        axes["g"], gwas_linking[gwas_linking["intersectPoPS"] == False].copy(),  # noqa: E712
        "g", "Linking variants to known genes:\nE2G model only", (-0.015, 0.31),
        "Recall\n(Fraction of credible sets linked to target gene)\n159 credible sets from 9 traits tested\n645 credible set-biosample pairs",
    )
    axes["h"].set_title("h", loc="left", fontweight="bold", fontsize=12)
    plot_linking_panel(
        axes["h"], gwas_linking[gwas_linking["intersectPoPS"] == True].copy(),  # noqa: E712
        "h", "Linking variants to known genes:\nE2G model with PoPS", (-0.015, 0.31),
        "Recall\n(Fraction of credible sets linked to target gene)\n159 credible sets from 9 traits tested\n645 credible set-biosample pairs",
    )
    add_paper_legend(axes["legend"], palette, note)
    add_note(fig, note)
    composite_path = output_dir / "figure2d_to_h.paper_style.pdf"
    savefig(fig, composite_path)
    outputs["figure2d_to_h"] = composite_path

    return outputs


def render_all(
    *,
    output_dir: Path,
    mode: str,
    normalized_predictions: Path | None = None,
    method_name: str | None = None,
    bg_variants: Path | None = None,
    score_threshold: float | None = None,
) -> dict[str, Path]:
    eqtl_dir = output_dir / "eqtl"
    gwas_dir = output_dir / "gwas"
    eqtl_outputs = build_eqtl_outputs(
        output_dir=eqtl_dir,
        mode=mode,
        normalized_predictions=normalized_predictions,
        method_name=method_name,
        bg_variants=bg_variants,
        score_threshold=score_threshold,
    )
    gwas_outputs = build_gwas_outputs(
        output_dir=gwas_dir,
        mode=mode,
        normalized_predictions=normalized_predictions,
        method_name=method_name,
        bg_variants=bg_variants,
        score_threshold=score_threshold,
    )

    eqtl_curve = read_tsv(eqtl_outputs["curve_table"])
    eqtl_bar = read_tsv(eqtl_outputs["fixed_recall_table"])
    gwas_curve = read_tsv(gwas_outputs["curve_table"])
    gwas_overlap = read_tsv(gwas_outputs["thresholded_overlap"])
    link_false = read_tsv(gwas_outputs["thresholded_gene_linking"])
    link_true = read_tsv(gwas_outputs["thresholded_gene_linking_with_pops"])
    link_false = link_false.assign(intersectPoPS=False)
    link_true = link_true.assign(intersectPoPS=True)
    gwas_linking = pd.concat([link_false, link_true], ignore_index=True)

    plots_dir = ensure_dir(output_dir / "plots")
    note = None if mode == "paper_parity" else LOCAL_SUBSET_NOTE
    plot_outputs = render_eqtl_and_gwas_panels(
        eqtl_curve=eqtl_curve,
        eqtl_bar=eqtl_bar,
        gwas_curve=gwas_curve,
        gwas_overlap=gwas_overlap,
        gwas_linking=gwas_linking,
        output_dir=plots_dir,
        note=note,
    )

    combined_missing = pd.concat(
        [
            read_tsv(eqtl_outputs["missing_requirements"]),
            read_tsv(gwas_outputs["missing_requirements"]),
        ],
        ignore_index=True,
    ).drop_duplicates()
    write_df(output_dir / "missing_requirements.tsv", combined_missing)

    combined_traceability = pd.concat(
        [
            read_tsv(eqtl_outputs["panel_traceability"]),
            read_tsv(gwas_outputs["panel_traceability"]),
        ],
        ignore_index=True,
    ).drop_duplicates()
    write_df(output_dir / "panel_traceability.tsv", combined_traceability)

    combined_coverage = read_tsv(eqtl_outputs["method_coverage"])
    write_df(output_dir / "method_coverage.tsv", combined_coverage)
    combined_resources = read_tsv(eqtl_outputs["resource_manifest"])
    write_df(output_dir / "resource_manifest.tsv", combined_resources)

    run_manifest = {
        "mode": mode,
        "normalized_predictions": str(normalized_predictions.resolve()) if normalized_predictions else "",
        "method_name": method_name or "",
        "bg_variants": str(bg_variants.resolve()) if bg_variants else "",
        "score_threshold": score_threshold if score_threshold is not None else "",
        "eqtl_output_dir": str(eqtl_dir),
        "gwas_output_dir": str(gwas_dir),
        "plot_output_dir": str(plots_dir),
        "plots": {key: str(path) for key, path in plot_outputs.items()},
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    return plot_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render paper-style Figure 2 benchmark panels.")
    parser.add_argument("--mode", choices=["paper_parity", "local_subset_debug"], default="local_subset_debug")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "reference_local_subset_debug")
    parser.add_argument("--normalized-predictions", type=Path, default=None, help="Optional normalized prediction TSV to stage.")
    parser.add_argument("--method-name", default=None, help="Optional method name override when staging predictions.")
    parser.add_argument("--bg-variants", type=Path, default=None, help="Distal-noncoding background SNP BED used for real enrichment scoring.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional selected score threshold for a new model.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = render_all(
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

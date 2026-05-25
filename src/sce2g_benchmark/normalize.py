from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .io_utils import ensure_dir, write_json
from .paths import CONFIGS_ROOT, DEFAULT_OUTPUT_ROOT


REQUIRED_COLUMNS = ["chrom", "start", "end", "target_gene", "biosample", "score", "method"]


class NormalizationError(ValueError):
    """Raised when a prediction table cannot be normalized safely."""


def load_alias_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t")
    return dict(zip(df["alias"], df["canonical_biosample"]))


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_prediction_table(
    *,
    input_path: Path,
    output_path: Path,
    column_map: Mapping[str, str] | None = None,
    method_name: str | None = None,
    biosample_aliases_path: Path | None = None,
) -> tuple[Path, dict]:
    df = pd.read_csv(input_path, sep="\t")
    if column_map:
        df = df.rename(columns=dict(column_map))

    if "method" not in df.columns:
        if not method_name:
            raise NormalizationError("Missing required column 'method' and no --method override was provided.")
        df["method"] = method_name
    elif method_name:
        df["method"] = method_name

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise NormalizationError(f"Missing required columns after normalization: {', '.join(missing)}")

    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(int)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="raise")
    df["chrom"] = df["chrom"].astype(str)
    df["target_gene"] = df["target_gene"].astype(str)
    df["biosample"] = df["biosample"].astype(str)
    df["method"] = df["method"].astype(str)

    alias_map = load_alias_map(biosample_aliases_path)
    if alias_map:
        df["biosample"] = df["biosample"].map(lambda value: alias_map.get(value, value))

    ordered_columns = REQUIRED_COLUMNS + [column for column in df.columns if column not in REQUIRED_COLUMNS]
    df = df[ordered_columns]

    ensure_dir(output_path.parent)
    df.to_csv(output_path, sep="\t", index=False)

    summary = {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "rows": int(len(df)),
        "methods": sorted(df["method"].dropna().unique().tolist()),
        "biosamples": sorted(df["biosample"].dropna().unique().tolist()),
        "columns": ordered_columns,
    }
    return output_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a prediction TSV into the Figure 2 benchmark schema.")
    parser.add_argument("--input", required=True, type=Path, help="Input prediction TSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "normalized" / "predictions.normalized.tsv",
        help="Output normalized TSV.",
    )
    parser.add_argument("--config-json", type=Path, default=None, help="Optional normalization config JSON.")
    parser.add_argument("--method", default=None, help="Override or fill the method column.")
    parser.add_argument(
        "--biosample-aliases",
        type=Path,
        default=CONFIGS_ROOT / "biosample_aliases.tsv",
        help="Alias TSV with 'alias' and 'canonical_biosample' columns.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config_json)
    output_path, summary = normalize_prediction_table(
        input_path=args.input,
        output_path=args.output,
        column_map=config.get("column_map"),
        method_name=args.method or config.get("method"),
        biosample_aliases_path=args.biosample_aliases,
    )
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

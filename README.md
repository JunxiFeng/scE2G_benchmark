# scE2G Benchmark

`scE2G_benchmark` is a standalone, prediction-table-driven benchmark package
for the scE2G manuscript Figure `2d-h` evaluation framework.

It is the canonical place in `wenkai/` to:

- normalize enhancer-gene prediction TSVs into one benchmark schema
- validate whether local resources are sufficient for exact paper parity
- materialize eQTL and GWAS benchmark tables from the checked-in local assets
- score a new model against the same Figure 2 reference tables when a
  distal-noncoding background SNP BED is provided
- render paper-style panels `2d`, `2e`, `2f`, `2g`, and `2h`
- write reproducible manifests for missing resources, method coverage, and
  panel traceability

## What Figure 2d-h Measure

- `2d`: eQTL enrichment-recall curves with suggested-threshold highlights
- `2e`: eQTL enrichment at fixed 5% recall
- `2f`: GWAS variant-overlap enrichment versus recall
- `2g`: GWAS gene-linking precision versus recall
- `2h`: GWAS gene-linking precision versus recall after intersecting with PoPS

The paper PDF at `wenkai/scE2G_analysis/sce2g.pdf` is treated as the source of
truth for panel meaning.

## Input Schema

Normalized prediction inputs must contain these required columns:

- `chrom`
- `start`
- `end`
- `target_gene`
- `biosample`
- `score`
- `method`

Additional columns are preserved but ignored by the scorer.

## Extra Requirement For Real Scoring

To score a new model rather than just replay the checked-in manuscript subset,
you must provide a distal-noncoding background SNP BED through either:

- `--bg-variants /path/to/distal_noncoding_background_snps.bed`
- `SCE2G_BG_VARIANTS=/path/to/distal_noncoding_background_snps.bed`

The benchmark expects the same background-variant resource used by the original
scE2G eQTL and GWAS benchmarking pipelines:

- Synapse: `https://www.synapse.org/#!Synapse:syn52264319`
- original filename: `all.bg.SNPs.hg38.baseline.v1.1.bed.sorted`

If you already downloaded it locally, you can point the benchmark directly at
it. For example, in this workspace:

- `/data/pinello/PROJECTS/2023_09_JF_SIMBAvariant/wenkai/scE2G_analysis/all.bg.SNPs.hg38.baseline.v1.1.bed.sorted`

The file should already be filtered to the distal-noncoding partition used by
the manuscript. A sorted BED is strongly recommended. `bedtools` is also
strongly recommended for production-size background files.

## Directory Layout

- `configs/`: panel specs, method specs, alias maps, example normalization config
- `examples/`: example normalized prediction TSV
- `references/`: notes and generated parity manifests
- `scripts/`: public CLI entrypoints
- `src/sce2g_benchmark/`: package code
- `outputs/`: default run destination

## Quick Start

Validate whether the current checkout can satisfy exact paper parity:

```bash
python wenkai/scE2G_benchmark/scripts/validate_paper_parity.py \
  --output-dir wenkai/scE2G_benchmark/outputs/paper_parity_validation
```

Normalize a prediction TSV that is already close to the target schema:

```bash
python wenkai/scE2G_benchmark/scripts/normalize_predictions.py \
  --input my_model_predictions.tsv \
  --output wenkai/scE2G_benchmark/outputs/normalized/my_model.normalized.tsv \
  --method my_model
```

Build the local subset eQTL benchmark artifacts:

```bash
python wenkai/scE2G_benchmark/scripts/run_eqtl_benchmark.py \
  --mode local_subset_debug \
  --output-dir wenkai/scE2G_benchmark/outputs/reference_local_subset_debug/eqtl
```

Build the local subset GWAS benchmark artifacts:

```bash
python wenkai/scE2G_benchmark/scripts/run_gwas_benchmark.py \
  --mode local_subset_debug \
  --output-dir wenkai/scE2G_benchmark/outputs/reference_local_subset_debug/gwas
```

Render all final paper-style panels from the available local subset:

```bash
python wenkai/scE2G_benchmark/scripts/render_figure2_panels.py \
  --mode local_subset_debug \
  --output-dir wenkai/scE2G_benchmark/outputs/reference_local_subset_debug
```

Score a new model and append it to the local reference panels:

```bash
python wenkai/scE2G_benchmark/scripts/render_figure2_panels.py \
  --mode local_subset_debug \
  --normalized-predictions wenkai/scE2G_benchmark/outputs/normalized/my_model.normalized.tsv \
  --bg-variants /data/pinello/PROJECTS/2023_09_JF_SIMBAvariant/wenkai/scE2G_analysis/all.bg.SNPs.hg38.baseline.v1.1.bed.sorted \
  --score-threshold 0.15 \
  --output-dir wenkai/scE2G_benchmark/outputs/my_model_vs_reference
```

If `--score-threshold` is omitted:

- eQTL uses the threshold nearest 5% recall for the new model
- GWAS uses the threshold that maximizes `recall * enrichment` on the new
  model's variant-overlap curve

Attempt an exact paper-parity render:

```bash
python wenkai/scE2G_benchmark/scripts/render_figure2_panels.py \
  --mode paper_parity \
  --output-dir wenkai/scE2G_benchmark/outputs/paper_parity_attempt
```

This command is expected to fail in the current checkout because the local
fine-grained eQTL assets are still missing paper method `6` (`STARE`),
paper method `11` (`SCENT`), and a standalone paper-style method `15`
distance-to-TSS series.

## Outputs

Each benchmark command writes:

- normalized prediction copies or manifests
- metric tables
- panel-ready summary tables
- `run_manifest.json`
- `missing_requirements.tsv`
- `method_coverage.tsv`
- `panel_traceability.tsv`
- `resource_manifest.tsv`

The panel renderer also writes:

- standalone `2d`, `2e`, `2f`, `2g`, `2h` PDF and PNG files
- a combined `2d-h` composite PDF and PNG
- when `--normalized-predictions` is supplied, the new model is appended to the
  checked-in local reference tables before plotting

## Modes

- `paper_parity`: requires complete method and resource coverage; fails fast if
  anything required for exact parity is missing
- `local_subset_debug`: runs with the checked-in local subset and writes an
  omission report so missing methods are explicit

## Example Outputs

The reference local-subset run produced by this package lives under:

- `wenkai/scE2G_benchmark/outputs/reference_local_subset_debug`

This includes:

- per-panel plots
- a combined paper-style `2d-h` figure
- benchmark manifests
- local-subset debug notes

## Troubleshooting

- If `paper_parity` fails, open `missing_requirements.tsv` first.
- If a prediction TSV is rejected, compare its columns against
  `configs/normalize_predictions.example.json`.
- If biosample names do not match manuscript names, extend
  `configs/biosample_aliases.tsv`.
- If real scoring fails with a background-file error, provide
  `--bg-variants` or set `SCE2G_BG_VARIANTS`.
- If real scoring fails without `bedtools`, install `bedtools` or use a much
  smaller background BED for toy runs; the pure-Python fallback is intended for
  small fixtures, not millions of SNPs.
- Exact paper parity is still expected to fail locally because the checked-in
  fine-grained eQTL artifacts are missing paper methods `6`, `11`, and a
  standalone paper-style `15`.

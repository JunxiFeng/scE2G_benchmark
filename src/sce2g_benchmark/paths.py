from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WENKAI_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WENKAI_ROOT.parent

SC_E2G_ANALYSIS_ROOT = WENKAI_ROOT / "scE2G_analysis"
FIGURE2_REPRO_ROOT = SC_E2G_ANALYSIS_ROOT / "3.Benchmarking" / "Figure2_reproduction"
EQTL_FINE_ROOT = SC_E2G_ANALYSIS_ROOT / "3.Benchmarking" / "eQTL" / "fine_grained_analysis"
GWAS_FINE_ROOT = SC_E2G_ANALYSIS_ROOT / "3.Benchmarking" / "GWAS" / "fine_grained_analysis"
EQTL_VARIANTS_FINE = SC_E2G_ANALYSIS_ROOT / "3.Benchmarking" / "eQTL" / "eQTL_Catalogue_v7" / "results" / "2024_1003_fine_tissues" / "eQTL_catalogue_v7.processed.PIP0.5.tsv.gz"
EQTL_TSS = SC_E2G_ANALYSIS_ROOT / "eQTLEnrichment" / "resources" / "genome_annotation" / "CollapsedGeneBounds.hg38.intGENCODEv43.TSS500bp.bed6"
EQTL_PARTITION = SC_E2G_ANALYSIS_ROOT / "eQTLEnrichment" / "resources" / "genome_annotation" / "PartitionCombined.bed"
EQTL_CHR_SIZES = SC_E2G_ANALYSIS_ROOT / "eQTLEnrichment" / "resources" / "genome_annotation" / "GRCh38_main.chrom.sizes.tsv"
GWAS_VARIANT_KEY = SC_E2G_ANALYSIS_ROOT / "GWAS_E2G_benchmarking" / "resources" / "UKBB_variant_key.tsv"
GWAS_VARIANT_ROOT = SC_E2G_ANALYSIS_ROOT / "GWAS_E2G_benchmarking"
GWAS_GENE_PRIORITIZATION = SC_E2G_ANALYSIS_ROOT / "GWAS_E2G_benchmarking" / "resources" / "UKBiobank.ABCGene.anyabc.tsv"
GWAS_TSS = SC_E2G_ANALYSIS_ROOT / "GWAS_E2G_benchmarking" / "resources" / "genome_annotation" / "CollapsedGeneBound.hg38.intGENCODEv43.TSS500bp.bed"
GWAS_CHR_SIZES = SC_E2G_ANALYSIS_ROOT / "GWAS_E2G_benchmarking" / "resources" / "genome_annotation" / "GRCh38_main.chrom.sizes.tsv"
SUPP_TABLE_S4 = SC_E2G_ANALYSIS_ROOT / "Supplementary Tables" / "Table S4 eQTL benchmark cell type pairings.tsv"
SUPP_TABLE_S5 = SC_E2G_ANALYSIS_ROOT / "Supplementary Tables" / "Table S5_GWAS benchmarking trait-cell type pairings.tsv"
PREDICTIONS_ROOT = SC_E2G_ANALYSIS_ROOT / "scE2Gpredictions"
PAPER_PDF = SC_E2G_ANALYSIS_ROOT / "sce2g.pdf"

CONFIGS_ROOT = PACKAGE_ROOT / "configs"
EXAMPLES_ROOT = PACKAGE_ROOT / "examples"
REFERENCES_ROOT = PACKAGE_ROOT / "references"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs"

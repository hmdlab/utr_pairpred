# UTR_PairPred

Code for:

> **Deciphering the comprehensive relationship between 5′ UTR and 3′ UTR sequences with deep learning**
> Kanta Suga, Keisuke Yamada, Michiaki Hamada
> *Bioinformatics* (accepted). See [Citation](#citation).

This repository contains the full pipeline used in the paper: RNA-language-model sequence
embedding, contrastive / supervised / random-forest models of the 5′–3′ UTR relationship,
cross-species evaluation, and the in-silico mutagenesis (window randomization) interpretability
analysis added during peer review.

## Installation
- Install required python libraries with `poetry install`

Basic requirements
```sh
python>=3.9.0
CUDA=11.8
torch=2.2.0
```

- If you want to preprocess the raw data yourself, also install:
	- `cd-hit`: https://github.com/weizhongli/cdhit
	- `ViennaRNA`: https://github.com/ViennaRNA/ViennaRNA

## Data preprocess
**Processed sequence embeddings & sequence CSV files can be downloaded from [here](https://waseda.box.com/v/utr-pairpred-data).**

1. Download the GENCODE `Protein-coding transcript sequences` FASTA file, e.g. from [here](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.pc_transcripts.fa.gz).

2. Create the sequence dataframe from the raw GENCODE FASTA file.
```sh
cd scripts
sh create_seqdf.sh
```
- This generates `gencode_v44(vM33)_utr_gene_unique.csv` and the corresponding `..._5utr(3utr).fa` files.
- To remove near-duplicate sequences, run `scripts/cd_hit.sh` on those FASTA files.

3. Get sequence embeddings (model inputs).
- With `RiNALMo`: `sh get_emb_rinalmo.sh`
- With `RNA-FM`: `sh get_emb_rnafm.sh`
- For the random-forest feature set: `sh get_rf_feature.sh`

* Processed sequence embedding & sequence CSV files can also be downloaded from [here](https://waseda.box.com/v/utr-pairpred-data) instead of running the above.

## Training prediction models
- Use `src/run_train_XX.py` for training (replace `XX` with a learning-method abbreviation below).
- Config files follow the naming rule `config/<SPECIES>_<LEARNING_METHOD>.yaml`.

| abb | full |
| ---- | ---- |
| cl | contrastive learning |
| sv | supervised learning |
| rf | random forest |

Run example:
```sh
poetry run python run_train_cl.py --cfg ../config/human_cl.yaml
```

## Cross-species evaluation
Train a model on one species from scratch and evaluate it on the other species' data (e.g.
train on human, evaluate on mouse, or vice versa). This workflow does not use cross-validation
(`kfold=1`).

**Important**: this trains models from scratch; it does not use pre-trained weights.

- For contrastive learning models:
```sh
cd scripts
sh run_cross_species_eval_cl.sh
```
This will:
1. Train a CL model on human data (60 epochs)
2. Evaluate the trained model on mouse data
3. Train a CL model on mouse data (60 epochs)
4. Evaluate the trained model on human data

- For supervised learning models:
```sh
cd scripts
sh run_cross_species_eval_sv.sh
```
This will:
1. Train an SV model on human data (100 epochs)
2. Evaluate the trained model on mouse data
3. Train an SV model on mouse data (100 epochs)
4. Evaluate the trained model on human data

- You can also run training and evaluation separately:
```sh
# Train on human
poetry run python ../src/run_train_cl.py --cfg ../config/human_cl_cross_species.yaml

# Evaluate on mouse
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/human_cl_cross_species.yaml \
    --model_path ../results/runs/cross_species_human_cl/best_model.pth \
    --eval_species mouse \
    --method cl
```

**Evaluation parameters**
| flag | meaning |
| ---- | ---- |
| `--cfg` | config file for the training species |
| `--model_path` | path to the trained model (`.pth`) |
| `--eval_species` | species to evaluate on (`human` or `mouse`) |
| `--method` | learning method (`cl` or `sv`) |

**Results**
- Training results: `results/runs/cross_species_{species}_{method}/`
- Cross-species evaluation: `results/runs/cross_species_{species}_{method}/cross_species_*_eval_{eval_species}/`

## In-silico mutagenesis (window randomization) analysis
Added in response to peer review to localize *which sequence regions drive the model's
predicted 5′–3′ UTR relationship*, rather than only reporting that pairs can be matched.

**Method.** For each **HRU** (Highly-Related UTR pair: a positive 5′/3′ UTR pair from the same
mRNA whose relation score — the cosine similarity of the contrastive model's L2-normalized
per-region embeddings — is ≥ 0.7; n = 157 for human) plus a low-score control group, a
contiguous window is slid along one UTR region while the paired region is kept intact:
- window length = 10% or 20% of the region length `L`; stride = 5% of `L`;
- the window is randomized `n_trials` (default 10) times and the resulting relation scores
  are averaged;
- **Δ relation score = baseline − randomized**. Δ > 0 means randomizing that window *lowers*
  the score, i.e. that region drives the predicted relationship.

Three randomization modes are implemented in `src/run_window_randomization.py`
(`--randomize_mode {uniform,shuffle,dinucleotide}`, one or more):

| mode | operation | what it destroys | isolates |
| ---- | ---- | ---- | ---- |
| `uniform` | replace the window with random A/C/G/U | base composition **and** order/structure | the window's full sequence contribution |
| `shuffle` | permute the window's existing bases | order/structure only (composition preserved) | order / secondary-structure contribution |
| `dinucleotide` | permute the window preserving its exact dinucleotide counts (Altschul–Erikson shuffle) | order/structure beyond dinucleotide composition (base pairing) | base-pairing propensity — contrast `shuffle − dinucleotide` |

### Pipeline
| step | script | what it does | compute | key outputs |
| ---- | ---- | ---- | ---- | ---- |
| 1 | [`scripts/run_window_randomization.sh`](./scripts/run_window_randomization.sh) → [`src/run_window_randomization.py`](./src/run_window_randomization.py) | selects HRUs/control, slides windows, re-embeds mutated sequences with RiNALMo under all three randomization modes, computes Δ | **GPU** | `results/window_randomization_human_w{20,10}/{window_randomization_results.csv, max_window_per_transcript.csv, position_aggregate.csv, analyzed_groups.csv, run.log}` + diagnostic PNGs |
| 2 | [`src/analyze_window_randomization_stats.py`](./src/analyze_window_randomization_stats.py) | per-transcript-then-per-bin averaging (avoids pseudo-replication), then per-position-bin HRU-vs-control Mann–Whitney U (or HRU-vs-0 Wilcoxon if there's no control group), BH/FDR corrected | CPU | `analysis/binned_per_transcript.csv`, `analysis/per_bin_stats.csv` |
| 3 | [`src/analyze_hru_top_sensitive.py`](./src/analyze_hru_top_sensitive.py) | ranks HRUs by their single most-sensitive window (`mode=uniform`); for outliers (Δ > mean + `n_sd`·SD) decomposes `uniform`/`shuffle`/`dinucleotide` into order/structure, base-pairing and composition contributions at that window; scans the outlier window sequences against known RBP/RNA-structure motifs | CPU | `analysis/hru_top_sensitive_uniform_w20.csv`, `analysis/hru_top_mode_decomposition_w20.csv`, `analysis/hru_top_motif_scan_w20.csv`, `docs/figures/hru_top_sensitive_profiles_w20.{png,svg}` |
| 4 | [`src/replot_window_randomization.py`](./src/replot_window_randomization.py) | regenerates the Δ-vs-position figure from `position_aggregate.csv` (drops `dinucleotide` from the plot, keeping `uniform`/`shuffle`) | CPU | `delta_vs_position_{utr5,utr3}.png` in `--result_dir` |
| 5 | [`src/plot_window_size_comparison.py`](./src/plot_window_size_comparison.py) | overlays window = 10% vs. 20% | CPU | `docs/figures/window_size_comparison.{png,svg}` |
| 6 | [`src/plot_ism_manuscript_figure.py`](./src/plot_ism_manuscript_figure.py) | the manuscript figure (main text + supplement) | CPU | `docs/figures/fig_ism_main.{png,svg}` (w20, main text), `fig_ism_w10.{png,svg}` (w10, supplement) |
| 7 | [`scripts/run_ism_pipeline.sh`](./scripts/run_ism_pipeline.sh) | driver that chains steps 2–6 (the CPU-only post-processing), with path/file sanity checks | CPU | — |

Step 1 is the only GPU-bound step (it re-embeds every mutated sequence with RiNALMo); steps
2–6 are CPU-only and operate on the CSVs step 1 produces.

> [`notebooks/window_randomization_analysis.ipynb`](./notebooks/window_randomization_analysis.ipynb)
> is the interactive companion to step 2: it reproduces the same per-bin statistics plus
> exploratory HRU × position heatmaps and annotated boxplots
> (`analysis/heatmap_hru_*.{png,pdf}`, `analysis/box_max_delta_*.{png,pdf}`). Step 2's script is
> what the rest of the pipeline (and `run_ism_pipeline.sh`) actually depends on; the notebook is
> for exploring the data by hand and is not required to reproduce the figures.

Example commands (step 1 — the wrapper already loops `--window_frac` over `0.2` then `0.1`,
writing `_w20`/`_w10` automatically; edit only the paths at the top of the file):
```sh
cd scripts
sh run_window_randomization.sh
```
or, running one window size directly:
```sh
poetry run python src/run_window_randomization.py \
    --cfg config/human_cl_cross_species_rinalmo.yaml \
    --model_path results/runs/cross_species_human_cl_rinalmo/best_model.pth \
    --seq_data data/human/gencode44_utr_gene_unique_cdhit09.csv \
    --cv_results_dir results/runs/contrastive_learning_10fold_rinalmo_whole_ave_seed1 \
    --emb_type rinalmo --input_dim 1280 \
    --relation_threshold 0.7 \
    --window_frac 0.2 --stride_frac 0.05 --n_trials 10 \
    --randomize_mode uniform shuffle dinucleotide \
    --regions utr5 utr3 \
    --control_high 0.1 --n_control 157 \
    --out_dir results/window_randomization_human_w20
# repeat with --window_frac 0.1 --out_dir results/window_randomization_human_w10
```

Steps 2–6 (no GPU needed), or run all of them at once with step 7:
```sh
# 2. per-position statistics
poetry run python src/analyze_window_randomization_stats.py --result_dir results/window_randomization_human_w20
poetry run python src/analyze_window_randomization_stats.py --result_dir results/window_randomization_human_w10

# 3. per-transcript outlier analysis (w20 only)
poetry run python src/analyze_hru_top_sensitive.py \
    --result_dir results/window_randomization_human_w20 \
    --seq_csv data/human/gencode44_utr_gene_unique_cdhit09.csv \
    --fig_dir docs/figures \
    --window_tag w20

# 4. re-plot the sensitivity figure
python src/replot_window_randomization.py --result_dir results/window_randomization_human_w10
python src/replot_window_randomization.py --result_dir results/window_randomization_human_w20

# 5. window-size comparison figure
python src/plot_window_size_comparison.py \
    --w10_dir results/window_randomization_human_w10 \
    --w20_dir results/window_randomization_human_w20 \
    --out_dir docs/figures

# 6. manuscript figure
python src/plot_ism_manuscript_figure.py \
    --w20_dir results/window_randomization_human_w20 \
    --w10_dir results/window_randomization_human_w10 \
    --out_dir docs/figures
```
```sh
# 7. or just chain steps 2-6 (from the scripts/ directory; paths overridable via env vars)
cd scripts
sh run_ism_pipeline.sh
```

## Downstream analysis
- [`crossval_analysis.ipynb`](./notebooks/crossval_analysis.ipynb): cross-validation analysis
  of result consistency; visualizes the distribution of cosine similarity and correlations
  across experiments.
- [`sequential_analysis.ipynb`](./notebooks/sequential_analysis.ipynb): basic sequence features
  (e.g. lengths of 5′UTR, 3′UTR, CDS, and MFE).
- [`expression_analysis.ipynb`](./notebooks/expression_analysis.ipynb): translation efficiency
  (TE) analysis using RNA-seq and Ribo-seq data for each cell line.
- [`RBP_interaction_analysis.ipynb`](./notebooks/RBP_interaction_analysis.ipynb): intersects
  relation-score-binned UTR regions with RBP CLIP-seq peaks (`bedtools intersect` against
  ENCORE RBP metadata) to test for RBP-binding enrichment across similarity bins.
- [`window_randomization_analysis.ipynb`](./notebooks/window_randomization_analysis.ipynb): see
  [In-silico mutagenesis](#in-silico-mutagenesis-window-randomization-analysis) above.

## Utils
**cd-hit**
- To eliminate similar sequences, use `scripts/cd_hit.sh`.
- After running cd-hit, you need to rebuild the sequence dataframe and embeddings for the
  representative set — use `create_represent_seq_df()` in `src/utils.py`.

## Repository layout
```
UTR_PairPred/
├── config/        # YAML configs for training / cross-species eval (per species × method)
├── data/          # (git-ignored) downloaded/generated sequence + embedding data
├── docs/
│   └── figures/   # (git-ignored) figures written by the analysis scripts
├── notebooks/      # analysis notebooks: cross-val, sequence stats, TE, RBP, window randomization
├── preprocess/     # GENCODE parsing, MFE calculation, sequence-embedding extraction
├── results/        # (git-ignored) training runs & analysis outputs, regenerated by the scripts/notebooks above
├── scripts/        # shell wrappers around the src/ and preprocess/ entry points
├── src/            # models, training, evaluation, and in-silico mutagenesis analysis code
├── LICENSE
├── pyproject.toml
└── README.md
```

## Reproducibility & outputs
- No image or tabular artifact is tracked. `results/`, `docs/figures/` and every
  `*.png` / `*.pdf` / `*.svg` / `*.csv` are git-ignored, so a fresh clone contains code
  only — every figure and table comes back by re-running the pipelines documented above.
  The scripts create their own output directories, so `docs/figures/` does not need to
  exist beforehand.
- Consequently the repository holds no rendered copy of the figures. Compare against the
  published article's figures when checking a reproduction.

## Citation
The paper has been accepted at *Bioinformatics*; the DOI, volume, and page numbers are not
yet assigned and will be filled in once published.

```bibtex
@article{suga_utr_pairpred,
  title   = {Deciphering the comprehensive relationship between 5' UTR and 3' UTR sequences with deep learning},
  author  = {Suga, Kanta and Yamada, Keisuke and Hamada, Michiaki},
  journal = {Bioinformatics},
  year    = {TBD},
  doi     = {TBD},
  note    = {Accepted for publication; DOI, year of issue, volume, and page numbers will be added once available.}
}
```

Corresponding author: Michiaki Hamada (mhamada@waseda.jp), Waseda University.

## License
[MIT](./LICENSE)

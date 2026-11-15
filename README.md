# Uzbek Paraphrase Detection Benchmark

We fine-tune pretrained encoders as cross-encoders on an Uzbek paraphrase dataset with three classes
(`paraphrase`, `near_miss`, `not_paraphrase`), compare them with logistic-regression baselines, and
evaluate them from several angles. The evaluation idea follows PARAPHRASUS (Michail, Clematide, Opitz, 2024).

> **Dataset not included.** The annotated data is private and will be released together with the paper.
> `make_synthetic_data.py` generates fake data with the same schema, so the whole pipeline can be run
> end to end without the real data.

## Results

Test-set results of the model selected on the dev set (mean dev macro-F1). Confidence intervals are
bootstrap intervals over whole source-sentence groups.

| Model | Dev macro-F1 | Test macro-F1 | 95% CI |
|---|---|---|---|
| Logistic regression (char n-gram TF-IDF) | [fill in] | [fill in] | [fill in] |
| mBERT | [fill in] | [fill in] | [fill in] |
| XLM-RoBERTa base | [fill in] | [fill in] | [fill in] |
| [other models] | [fill in] | [fill in] | [fill in] |

XLM-RoBERTa gave the best test results. Figures (confusion matrices, model comparison, bootstrap CIs)
are in `figs/`, and all tables are in `results/`.

## Repository contents

| File | What it is |
|---|---|
| `paraphrase.ipynb` | The whole experiment, in numbered cells (see below) |
| `download_models.py` | Downloads the model weights once into `models/<name>` |
| `make_synthetic_data.py` | Writes fake data with the same schema, for testing without the real data |
| `results/` | Metrics only: result tables, summaries, breakdowns, bootstrap CIs, report |
| `figs/` | Confusion matrices, model comparison, strictness view, sub-strategy heatmap, bootstrap CIs |

Notebook cells: 1 configuration, 2 environment check, 3 load and clean data, 4 data checks (counts only),
5 evaluation helpers, 6 baselines, 7 tokenizer and apostrophe test, 8 training function, 9 run loop,
10 results tables, 11 plots, 12 bootstrap CIs and model selection, 13 error-analysis export,
14 report.

The notebook does not depend on scikit-learn or scipy. Metrics are computed with NumPy, the character
n-gram TF-IDF is a small implementation of our own, and the logistic-regression baselines are trained
with PyTorch.

## Data and privacy

The dataset and the error-analysis files (which contain sentences) are kept private and are not part of
this repository. The notebook prints and saves only counts and metrics. Training and evaluation run
offline; only `download_models.py` uses the internet, to download model weights.

## Setup

- Python 3.10+ with PyTorch (CUDA recommended) and Hugging Face Transformers.
- Install dependencies with `python -m pip install -r requirements.txt` (add this file if you publish one).

## How to run

1. **Download the models** (once, needs internet):
   `python download_models.py` (or only some: `python download_models.py mbert xlmr-base`)
2. **Optional pipeline test on fake data:**
   `python make_synthetic_data.py <fake_data_dir>`, then in cell 1 set `DATA_DIR` to that folder,
   `PRIVATE_OUT` to a separate output folder, and `QUICK = True`, and run all cells. Every run stores a
   fingerprint (hash) of the data, so fake results are replaced automatically when you switch to real data.
3. **Edit cell 1** for your data: `DATA_DIR`, `PRIVATE_OUT` (both outside the project), `QUICK = True`.
4. **Run cells 1 to 4** (environment and data checks, counts only). Check the source-ID overlap, the
   `label` vs `verified_label` crosstab and the cleaning report.
5. **Quick run:** with `QUICK = True`, run all cells (2,000 training rows, 1 epoch, 1 seed per model).
   This checks that the pipeline works. Results go to `results/results_quick.csv`.
6. **Full run:** set `QUICK = False` and run all cells (full training set, 4 epochs, seeds 42/43/44).
   Results go to `results/results_all.csv`. If the kernel crashes, run the cells again: finished runs are skipped.
7. **Large models** (`xlmr-large`, `bge-m3`) need more GPU memory than a 12 GB card. Run them on a larger GPU
   with `MODELS_TO_RUN = ["xlmr-large", "bge-m3"]`, then copy the resulting `results/runs_all/*` files back
   and re-run cells 1-5 and 10-13 to get one combined table. Runs are combined only when the data
   fingerprint printed in cell 3 is identical on both machines.

## Method notes

- **Gold label** for evaluation is `verified_label` if present and valid, otherwise `label`. Training uses `label`.
- **Model selection** uses mean dev macro-F1. The selected model's test score is reported once.
- **Confidence intervals:** bootstrap over whole `source_sentence_id` groups. Differences whose paired
  bootstrap interval contains 0 are reported as ties.
- **Collapsed runs** (dev macro-F1 below 0.40, which can happen with large models) are kept, counted and
  included in the means.
- **Apostrophes:** all variants are normalized to `'`. Then, per model, the form with the fewest `[UNK]`
  tokens, then the fewest tokens, is used for training (cell 7).
- No model checkpoints are written: the test logits of the best dev epoch are kept in memory instead.

## Reference

Michail, Clematide, Opitz (2024). PARAPHRASUS: a benchmark for evaluating paraphrase detection models.

## Authors

Samira Ikromova and [Professor Name], Inha University in Tashkent.

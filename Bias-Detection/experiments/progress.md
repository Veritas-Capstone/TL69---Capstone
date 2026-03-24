# Veritas — Political Bias Detection Pipeline

Browser extension for detecting political bias in news articles using transformer models.

## Models

| Model | Purpose | Size | Paper |
|---|---|---|---|
| [matous-volf/political-leaning-deberta-large](https://huggingface.co/matous-volf/political-leaning-deberta-large) | Bias classification (Left/Center/Right) | 400M params | [arXiv:2507.13913](https://arxiv.org/abs/2507.13913) |
| [mlburnham/Political_DEBATE_large_v1.0](https://huggingface.co/mlburnham/Political_DEBATE_large_v1.0) | Politicalness filter (NLI-based) | 400M params | [arXiv:2409.02078](https://arxiv.org/abs/2409.02078) |

## Pipeline Architecture

```
Article text
    |
    v
+----------------------------+
|  Politicalness Filter      |  NLI with 4 hypotheses, take max
|  (Political DEBATE)        |  -> is_political (bool)
+----------------------------+
    | if political
    v
+----------------------------+
|  Document-Level Bias       |  Full article -> single label
|  Prediction (DeBERTa)      |  -> Left / Center / Right
+----------------------------+
    | optional (for UI)
    v
+----------------------------+
|  Explainability            |  Gradient x Input on top-k
|  (sentence-level)          |  biased sentences -> key tokens
+----------------------------+
```

**Key design decision:** The bias model was trained on full articles, so we feed full articles for prediction (not sentence-by-sentence). Sentence-level analysis is only used for explainability in the browser extension UI.

## Setup

```bash
# 1. Clone BASIL dataset (evaluation benchmark)
git clone https://github.com/launchnlp/BASIL.git

# 2. Install dependencies
pip install transformers torch scikit-learn

# 3. Download models locally
python download_models.py

# 4. Quick test
python pipeline.py --demo

# 5. Full evaluation report
python run_report.py --basil-dir BASIL
```

## Project Structure

### Active Files

| File | What it does |
|---|---|
| `pipeline.py` | **Core pipeline** — BiasDetector class with two-pass architecture + Gradient×Input explainability |
| `basil_loader.py` | BASIL dataset loader (300 articles, 100 events × 3 sources) |
| `download_models.py` | Downloads both models from HuggingFace and saves locally |
| `run_report.py` | **Run this** — evaluates everything and saves results to `results/` |
| `threshold.py` | Grid search over logit calibration parameters (temperature, bias offsets) |
| `finetune_basil.py` | LoRA fine-tuning on BASIL (needs GPU with >6GB VRAM — run on school servers) |
| `load_finetuned.py` | Loads LoRA-adapted model back into the pipeline |
| `topic_model.py` | Experimental: topic-based PMI analysis for agenda-setting bias detection |

### Deprecated (kept for reference)

These are earlier iterations that `pipeline.py` replaced:

| File | Why replaced |
|---|---|
| `test_attention.py` | v1 — used raw attention weights (not faithful attributions) |
| `explain_test.py` | v2 — used Integrated Gradients (accurate but ~30x slower) |
| `full_article.py` | v2.5 — intermediate version before pipeline.py consolidated everything |
| `parralel.py` | Used old attention-based approach |
| `generalization_test.py` | Old AllSides evaluation script; replaced by `run_report.py` |

## Current Results (Pre-Fine-Tuning)

Demo examples (hand-picked): **4/4 correct** — the model handles overt political signals well.

Evaluated on BASIL (out-of-distribution — the model was NOT trained on this data):

| Benchmark | Accuracy | Macro F1 |
|---|---|---|
| BASIL (source labels: HPO=Left, NYT=Center, FOX=Right) | 47.7% | 0.404 |
| BASIL (annotation labels: expert-annotated stance) | 34.3% | 0.307 |
| Calibration (best threshold tuning) | 52.0% | 0.451 |

### Per-Class Breakdown (Source Labels)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Left | 0.419 | 0.960 | 0.584 | 100 |
| Center | 0.048 | 0.010 | 0.017 | 100 |
| Right | 0.920 | 0.460 | 0.613 | 100 |

The model has a strong Left-prediction bias: 98 out of 100 Center articles get misclassified as Left, and 35 out of 100 Right articles do too. Right precision is excellent (0.92) but recall is low — when the model says Right, it's almost always correct, but it misses over half of them.

### Within-Event Analysis

BASIL has 100 events where the same story is covered by all 3 outlets (HPO, NYT, FOX):

- All 3 correct: 0/100 (0%) — the model never gets all three right for the same event
- Partially correct: 100/100 (100%) — it always gets at least one right
- Model sees a difference (>1 unique prediction across the 3 outlets): 65/100 (65%)

### Why the numbers look low

The Volf model achieves 83.7% F1 on its training data, but the paper itself shows out-of-distribution accuracy drops to 38–67% depending on the dataset. BASIL articles are mainstream news (NYT, HuffPost, Fox) covering the *same events* — the bias differences are subtle framing choices, not the overt ideological signals the model was trained on. The model essentially can't distinguish Center from Left in this domain.

### What calibration showed

Grid search over logit bias offsets and temperature scaling maxed out at 52.0% accuracy / 0.451 macro F1 (bias_L=0.50, bias_C=-2.00, T=0.50). The Center class F1 stays near 0.0 regardless of thresholds — this confirms the model genuinely can't detect centrist reporting in BASIL without weight updates (fine-tuning).

## Next Step: Fine-Tuning

```bash
# Cross-validation (needs school GPU server)
python finetune_basil.py --basil-dir BASIL

# Train final model
python finetune_basil.py --basil-dir BASIL --train-final --final-epochs 6

# Merge into standalone model
python load_finetuned.py --merge

# Re-evaluate with fine-tuned model
python run_report.py --basil-dir BASIL --bias-model models/bias_detector_merged
```

Fine-tuning uses LoRA (Low-Rank Adaptation) — only ~0.3% of parameters are trainable, which prevents overfitting on the 300-sample BASIL dataset. Event-level cross-validation ensures articles from the same news story stay together in train/val splits.

## Explainability

The pipeline uses **Gradient × Input** attribution (1 forward + 1 backward pass) to identify which tokens drive the bias prediction. This replaced the earlier Integrated Gradients approach which required ~30 forward passes per sentence.

Stop words are filtered from the token list since the model sometimes assigns high attribution to function words (e.g., "the", "said") that correlate with political writing style but aren't meaningful to users.

## References

- Volf, M. & Simko, J. (2025). *Political Leaning and Politicalness Classification of Texts*. [arXiv:2507.13913](https://arxiv.org/abs/2507.13913)
- Burnham, M. (2024). *Political DEBATE*. [arXiv:2409.02078](https://arxiv.org/abs/2409.02078)
- Fan, L. et al. (2019). *In Plain Sight: Media Bias Through the Lens of Factual Reporting*. EMNLP 2019.

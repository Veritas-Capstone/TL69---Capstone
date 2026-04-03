Early-stage exploration scripts used to evaluate existing political bias detection models before building our own pipeline. We used the best performing one for the POC demo in Nov 2025 to give an idea for how the finished product would look.

Each baseline_modelx.py file loads a different model that we tested and later evaluated on a full dataset and compared each model with each other:
baseline_model1.py — BERT (politicalBiasBERT)
baseline_model2.py — RoBERTa (political-bias)
baseline_model3.py — RoBERTa (Volf POLITICS)
baseline_model4.py — DeBERTa-v3-large (Volf)

The final pipeline for the demo was sentence splitting -> merging closely related sentences -> predict bias -> downweighting -> aggregation for final score

The BABE and Qbias datasets were used as our evaluation benchmark.
# Project Veritas - Group 19
A chrome extension for sentence-level claim verification, partisan bias detection with rationales, and concise, evidence-tagged counterpoints. Core prediction relies on purpose-trained classifiers; generation is used only for presentation and grounded by retrieved sources.

Access our demo video here: [:movie_camera:](https://www.macvideo.ca/media/Veritas+-+ComputingSoftware.mp4/1_2ajmmnum/397464013)

## User Guide
- Download the .output directory from extension/wxt-dev-wxt
- Open up manage extension on chrome and turn on developer mode
- Click load unpacked and from the .output folder from above, find and select the chrome-mv3 folder
- Veritas should now be added to extensions.
  
## Overview
Given a news article (URL or pasted text), the system:
- Extracts checkable, sentence-level claims.
- Retrieves supporting evidence from reputable sources and verifies each claim (Supported / Refuted / Not Enough Evidence).
- Detects partisan bias at claim and article levels and highlights rationale phrases.
- Presents concise counterpoints grounded in retrieved evidence, clearly tagged and checked for support.

## Objectives
- Deliver trustworthy, transparent analysis of claims in news articles.
- Provide clear rationales, evidence links, and calibrated confidence for each output.
- Keep generation strictly grounded in retrieved sources and clearly labeled.

## Key Features
- Article ingestion via URL or raw text with paywall awareness.
- Claim extraction for atomic, check-worthy statements.
- Evidence retrieval and reranking across static and recent sources.
- Structured claim verification with confidence and evidence snippets.
- Bias detection with rationale highlighting and article-level aggregation.
- Evidence-based counterpoints with explicit source tags and support checks.

## How It Works (High Level)
1. Ingest: Fetch and clean article content; segment into sentences and paragraphs.
2. Extract: Identify concise factual claims suitable for verification.
3. Route: Use static retrieval first; fall back to real-time search for low-similarity or time-sensitive claims.
4. Retrieve & Rerank: Collect candidate passages and prioritize the most relevant evidence.
5. Verify & Classify: Produce a verdict with confidence; predict claim-level and article-level bias with rationale spans.
6. Counterpoints: Generate concise, evidence-tagged counterpoints grounded in retrieved snippets; reject unsupported text.
7. Output: Return a structured JSON payload with claims, verdicts, evidence, bias, counterpoints, and diagnostics.

## Datasets

### Claim verification
- FEVER — Wikipedia-derived claims with labels and evidence sentences for supervised verification.
- AVeriTeC — Real-world claims linked to online evidence and textual justifications.

### Bias detection
- article_bias_prediction — News articles labelled by political leaning from various outlets.
- dem_rep_party_platform_topics — Excerpts from Democratic and Republican party platforms.
- gpt4_political_bias — GPT-4 generated politically biased text samples.
- gpt4_political_ideologies — GPT-4 generated text representing different political ideologies.
- political_tweets — Tweets labelled by political leaning.
- qbias — Articles curated for balanced perspectives and bias research.
- webis_bias_flipper_18 — News articles with bias labels and neutralized rewrites.
- webis_news_bias_20 — News sentences annotated for biased and neutral language.

### Retrieval and real-time sources
- Wikipedia passages — Static corpus for broad coverage and reproducible retrieval.
- Real-time news search — General news API integration for fresh, time-sensitive evidence when static similarity is low.

## Principles & Ethics
- No paywall bypass; request accessible text if content is restricted.
- Evidence-first transparency: show sources, snippets, and rationale highlights.
- Generation is clearly marked, grounded, and checked; unverifiable outputs are rejected.
- Decision support only; human oversight is expected for critical use.

# Coding and Documentation Standards
- For python backend, we should follow Pep 8 guidelines
- With React/Node, we can follow the Google Typescript guide
- For API building and usage, we can follow REST API although this is a lot more lenient
- Document security measures, login credentials for hosting or any infrastructure

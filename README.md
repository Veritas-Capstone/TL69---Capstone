# Project Veritas - Group 19
A web application for sentence-level claim verification, partisan bias detection with rationales, and concise, evidence-tagged counterpoints. Core prediction relies on purpose-trained classifiers; generation is used only for presentation and grounded by retrieved sources.

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
- FEVEROUS — Claims with evidence from both text and tables for multi-hop reasoning.
- MultiFC — Multi-domain claims aggregated from fact-checking sites with real-world variability.
- AVeriTeC — Real-world claims linked to online evidence and textual justifications.
- LIAR — PolitiFact statements with fine-grained truth labels and metadata.
- PolitiFact archives — Fact-check articles and verdicts for political claims.

### Bias detection
- MBIC — Media bias annotations with sentence/word-level labels and annotator context.
- BABE — Expert-annotated sentences with bias labels and rationale spans.
- AllSides ratings — Outlet-level Left/Center/Right labels for distant supervision and aggregation.
- QBias — Articles curated for balanced perspectives and bias research.
- NewsMediaBias (full) — Multi-dimensional bias labels for auxiliary training and analysis.

### Retrieval and real-time sources
- Wikipedia passages — Static corpus for broad coverage and reproducible retrieval.
- Multi-site fact checks — Sources aggregated across fact-checking outlets to reduce single-site bias.
- Real-time news search — General news API integration for fresh, time-sensitive evidence when static similarity is low.

## Principles & Ethics
- No paywall bypass; request accessible text if content is restricted.
- Evidence-first transparency: show sources, snippets, and rationale highlights.
- Generation is clearly marked, grounded, and checked; unverifiable outputs are rejected.
- Decision support only; human oversight is expected for critical use.

## Tech Stack (Planned)  
- **Frontend:** React / Next.js or Flutter Web  
- **Backend:** FastAPI (Python)  
- **Database:** MongoDB or PostgreSQL (for storing user history, claims, and bias results)  
- **ML Models:**  
  - Claim extraction: fine-tuned transformer (e.g., BERT, RoBERTa, or open-source GPT variant)  
  - Bias detection: classifier trained on labeled partisan datasets  
  - Claim verification: retrieval + NLI pipeline with optional LLM fallback  
- **Deployment:** Docker + cloud hosting (AWS/GCP/Azure)

# Coding and Documentation Standards
- For python backend, we should follow Pep 8 guidelines
- With React/Node, we can follow the Google Typescript guide
- For API building and usage, we can follow REST API although this is a lot more lenient
- Document security measures, login credentials for hosting or any infrastructure

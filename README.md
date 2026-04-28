<h1 align="center">Spurious Memory Benchmark</h1>

<p align="center">
  <b>Benchmark pipeline to test whether LLMs can distinguish real causal relationships from spurious correlations in multi-turn memory settings</b>
</p>


## Overview

For each dataset sample the pipeline automatically:
1. Discovers causal structure (T, Y, X, M, DAG)
2. Generates 3 types of spurious correlations
3. Validates spurious features
4. Generates 4 query versions (Q0/Q1/Q2/Q3)
5. Evaluates 3 memory systems with token and time stats

---

## Three Types of Spurious Correlations

- Type 1 Confounding  (S <- C -> Y): S and Y share hidden common cause C
- Type 2 Collider     (T -> C <- S): Both T and S cause collider C
- Type 3 Proxy        (S <- X -> Y): S is downstream effect of covariate X

---

## 4 Query Types

- Q0: Does T cause Y?   answer YES (true causal baseline)
- Q1: Does S1 cause Y?  answer NO  (confounding spurious)
- Q2: Does S2 cause Y?  answer NO  (collider spurious)
- Q3: Does S3 cause Y?  answer NO  (proxy spurious)

---

## Datasets

| Dataset | Instances | Task Type | Source |
|---------|-----------|-----------|--------|
| TravelPlanner | 225 | Multi-turn travel planning | ICML 2024 |
| InterCode-CTF | 100 | Multi-turn bash interaction | NeurIPS 2023 |

---

## Project Structure

    spurious-memory-benchmark/
    |
    +-- pipeline/
    |   +-- step0_parse_travelplanner.py
    |   +-- step0_parse_intercode_ctf.py
    |   +-- step1_discover_causal.py
    |   +-- step2_generate_spurious.py
    |   +-- step3_validate_spurious.py
    |   +-- step4_generate_queries.py
    |   +-- step5_evaluate.py
    |
    +-- data/
    |   +-- travelplanner/
    |   +-- intercode_ctf/
    |
    +-- results/
    |   +-- results_travelplanner.output
    |   +-- results_intercode_ctf.output
    |   +-- logs/
    |
    +-- utils/
    |   +-- model.py
    |
    +-- venv/                           (Python virtual environment)
    +-- generate_appendix.py
    +-- appendix_a_output.txt
    +-- requirements.txt
    +-- README.md

---

## Setup

    git clone https://github.com/rupali559/spurious-memory-benchmark.git
    cd spurious-memory-benchmark
    . venv/bin/activate
    pip install -r requirements.txt

---

## How To Run - TravelPlanner

    # Step 0 — Parse raw TravelPlanner dataset into standard format
    python3 pipeline/step0_parse_travelplanner.py

    # Step 1 — Discover causal structure (T, Y, X, M, DAG) using Qwen
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step1_discover_causal.py \
        --input_file data/travelplanner/parsed.json \
        --output_file data/travelplanner/causal_structures.json --limit 225

    # Step 2 — Generate 3 spurious types (confounding, collider, proxy)
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step2_generate_spurious.py \
        --input_file data/travelplanner/causal_structures.json \
        --output_file data/travelplanner/spurious_types.json --limit 225

    # Step 3 — Validate spurious features (structural + counterfactual checks)
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step3_validate_spurious.py \
        --input_file data/travelplanner/spurious_types.json \
        --output_file data/travelplanner/spurious_validated.json --limit 225

    # Step 4 — Generate 4 query versions (Q0 causal, Q1/Q2/Q3 spurious)
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step4_generate_queries.py \
        --input_file data/travelplanner/spurious_validated.json \
        --output_file data/travelplanner/queries.json --limit 225

    # Step 5 — Evaluate 3 systems with token and time tracking
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step5_evaluate.py \
        --input_file data/travelplanner/queries.json \
        --output_file results/results_travelplanner.output \
        --log_dir results/logs/travelplanner \
        --dataset TravelPlanner --systems qwen,mem0,amem --limit 225

---

## How To Run - InterCode-CTF

    # Step 0 — Parse raw InterCode-CTF dataset into standard format
    python3 pipeline/step0_parse_intercode_ctf.py

    # Step 1 — Discover causal structure (T, Y, X, M, DAG) using Qwen
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step1_discover_causal.py \
        --input_file data/intercode_ctf/parsed.json \
        --output_file data/intercode_ctf/causal_structures.json --limit 100

    # Step 2 — Generate 3 spurious types (confounding, collider, proxy)
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step2_generate_spurious.py \
        --input_file data/intercode_ctf/causal_structures.json \
        --output_file data/intercode_ctf/spurious_types.json --limit 100

    # Step 3 — Validate spurious features (structural + counterfactual checks)
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step3_validate_spurious.py \
        --input_file data/intercode_ctf/spurious_types.json \
        --output_file data/intercode_ctf/spurious_validated.json --limit 100

    # Step 4 — Generate 4 query versions (Q0 causal, Q1/Q2/Q3 spurious)
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step4_generate_queries.py \
        --input_file data/intercode_ctf/spurious_validated.json \
        --output_file data/intercode_ctf/queries.json --limit 100

    # Step 5 — Evaluate 3 systems with token and time tracking
    CUDA_VISIBLE_DEVICES=2 python3 pipeline/step5_evaluate.py \
        --input_file data/intercode_ctf/queries.json \
        --output_file results/results_intercode_ctf.output \
        --log_dir results/logs/intercode_ctf \
        --dataset InterCode-CTF --systems qwen,mem0,amem --limit 100

---

## Causal Graph Memory Design

For every query a causal graph memory is built automatically:

    [CAUSAL GRAPH MEMORY]
    Context: Could you help me plan a trip from Daytona Beach to Texas?

    Outcome: A valid 7-day travel plan within $3,600 budget

    Candidate relationships observed (shuffled, no labels):
      - Spending habits
      - Weather Conditions
      - Airbnb accommodation listings
      - traveling to 3 cities from Daytona Beach

    Task: Determine which candidate directly causes the outcome.
    Note: Some candidates may be correlated but not causal.



---

## Model Used

Qwen/Qwen2.5-1.5B-Instruct (local, via HuggingFace transformers)

---

## References

- TravelPlanner: Xie et al., ICML 2024. GitHub: https://github.com/OSU-NLP-Group/TravelPlanner
- InterCode-CTF: Yang et al., NeurIPS 2023. GitHub: https://github.com/princeton-nlp/intercode
- Base paper: CAUSAL-MEMORY-ARENA, NeurIPS 2026 submission

# Spurious Memory Benchmark

> **Can LLMs tell the difference between a real cause and a lucky coincidence — even across multi-turn interactions with memory?**

This benchmark systematically tests whether large language models can distinguish **genuine causal relationships** from **spurious correlations** in realistic multi-turn task settings, with and without external memory systems.

---

## Key Questions

- **(i) Identify spuriousness** — Can the model correctly reject fake correlations injected into its context?
- **(ii) Disentangle** — Can it handle both causal and spurious queries correctly for the *same* instance?

---

## Datasets

| Dataset | Instances | Task Type | Venue | Source |
|---|---|---|---|---|
| **TravelPlanner** | 225 | Multi-turn travel planning | ICML 2024 | [GitHub](https://github.com/OSU-NLP-Group/TravelPlanner) |
| **InterCode-CTF** | 100 | Multi-turn bash interaction | NeurIPS 2023 | [GitHub](https://github.com/princeton-nlp/intercode) |

Both datasets involve extended, multi-turn task completion — making them ideal for stress-testing memory systems that accumulate context over time.

---

## How It Works

The pipeline runs five automatic stages per dataset instance:
Parse → Discover Causal Structure → Generate Spurious → Validate → Generate Queries → Evaluate

### Three Types of Spurious Correlations

| Type | Structure | Description |
|---|---|---|
| **Confounding** | S ← C → Y | S and Y share a hidden common cause C |
| **Collider** | T → C ← S | Both T and S cause collider C |
| **Proxy** | S ← X → Y | S is a downstream effect of covariate X |

### Four Query Types per Instance

| Query | Question | Expected Answer |
|---|---|---|
| **Q0** | Does T cause Y? | **YES** — true causal baseline |
| **Q1** | Does S1 cause Y? | **NO** — confounding spurious |
| **Q2** | Does S2 cause Y? | **NO** — collider spurious |
| **Q3** | Does S3 cause Y? | **NO** — proxy spurious |

---

## Project Structure

    spurious-memory-benchmark/

      pipeline/
        step0_parse_travelplanner.py    Parse raw TravelPlanner dataset
        step0_parse_intercode_ctf.py    Parse raw InterCode-CTF dataset
        step1_discover_causal.py        Discover causal structure (T, Y, X, M, DAG)
        step2_generate_spurious.py      Generate 3 spurious correlation types
        step3_validate_spurious.py      Validate spurious features
        step4_generate_queries.py       Generate Q0-Q3 query versions
        step5_evaluate.py               Evaluate systems with token and time tracking

      data/
        travelplanner/
        intercode_ctf/

      results/
        results_travelplanner.output
        results_intercode_ctf.output
        logs/

      utils/
        model.py

      venv/
      generate_appendix.py
      appendix_a_output.txt
      requirements.txt
      README.md

---

## Setup

```bash
git clone https://github.com/rupali559/spurious-memory-benchmark.git
cd spurious-memory-benchmark
. venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Pipeline — TravelPlanner

### Step 0 — Parse raw dataset

```bash
python3 pipeline/step0_parse_travelplanner.py
```

### Step 1 — Discover causal structure

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step1_discover_causal.py \
    --input_file data/travelplanner/parsed.json \
    --output_file data/travelplanner/causal_structures.json \
    --limit 225
```

### Step 2 — Generate spurious correlations

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step2_generate_spurious.py \
    --input_file data/travelplanner/causal_structures.json \
    --output_file data/travelplanner/spurious_types.json \
    --limit 225
```

### Step 3 — Validate spurious features

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step3_validate_spurious.py \
    --input_file data/travelplanner/spurious_types.json \
    --output_file data/travelplanner/spurious_validated.json \
    --limit 225
```

### Step 4 — Generate queries (Q0–Q3)

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step4_generate_queries.py \
    --input_file data/travelplanner/spurious_validated.json \
    --output_file data/travelplanner/queries.json \
    --limit 225
```

### Step 5 — Evaluate

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step5_evaluate.py \
    --input_file data/travelplanner/queries.json \
    --output_file results/results_travelplanner.output \
    --log_dir results/logs/travelplanner \
    --dataset TravelPlanner \
    --systems qwen,mem0,amem \
    --limit 225
```

---

## Running the Pipeline — InterCode-CTF

### Step 0 — Parse raw dataset

```bash
python3 pipeline/step0_parse_intercode_ctf.py
```

### Step 1 — Discover causal structure

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step1_discover_causal.py \
    --input_file data/intercode_ctf/parsed.json \
    --output_file data/intercode_ctf/causal_structures.json \
    --limit 100
```

### Step 2 — Generate spurious correlations

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step2_generate_spurious.py \
    --input_file data/intercode_ctf/causal_structures.json \
    --output_file data/intercode_ctf/spurious_types.json \
    --limit 100
```

### Step 3 — Validate spurious features

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step3_validate_spurious.py \
    --input_file data/intercode_ctf/spurious_types.json \
    --output_file data/intercode_ctf/spurious_validated.json \
    --limit 100
```

### Step 4 — Generate queries (Q0–Q3)

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step4_generate_queries.py \
    --input_file data/intercode_ctf/spurious_validated.json \
    --output_file data/intercode_ctf/queries.json \
    --limit 100
```

### Step 5 — Evaluate

```bash
CUDA_VISIBLE_DEVICES=2 python3 pipeline/step5_evaluate.py \
    --input_file data/intercode_ctf/queries.json \
    --output_file results/results_intercode_ctf.output \
    --log_dir results/logs/intercode_ctf \
    --dataset InterCode-CTF \
    --systems qwen,mem0,amem \
    --limit 100
```

---

## Causal Graph Memory Format

For every query, a causal graph is automatically built and injected as memory — with no labels and shuffled candidates, so the model must reason from structure alone:
[CAUSAL GRAPH MEMORY]
Context: Could you help me plan a trip from Daytona Beach to Texas?
Outcome: A valid 7-day travel plan within $3,600 budget
Candidate relationships observed (shuffled, no labels):

Spending habits
Weather Conditions
Airbnb accommodation listings
Traveling to 3 cities from Daytona Beach

Task: Determine which candidate directly causes the outcome.
Note: Some candidates may be correlated but not causal.

**Key design principles:**
- No true/spurious labels are shown to the model
- Candidates are shuffled randomly each time
- The model must reason entirely from causal structure

---

## Models and Systems Used

- **Qwen2.5-1.5B-Instruct** — Local LLM for all pipeline steps. [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Mem0** — Memory system for LLM agents. [GitHub](https://github.com/mem0ai/mem0)
- **A-MEM** — Agentic memory system. [GitHub](https://github.com/WujiangXu/A-MEM)

---

## References

1. **Xie et al.** — TravelPlanner: A Benchmark for Real-World Planning with Language Agents, ICML 2024. [GitHub](https://github.com/OSU-NLP-Group/TravelPlanner)
2. **Yang et al.** — InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback, NeurIPS 2023. [GitHub](https://github.com/princeton-nlp/intercode)
3. **Base paper** — CAUSAL-MEMORY-ARENA, NeurIPS 2026 submission.

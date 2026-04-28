"""
CAUSAL-MEMORY-ARENA — Step 1: Discover Causal Structure

For each sample, identifies:
  T (Treatment)  : true direct cause
  X (Covariate)  : background variable
  M (Mediator)   : intermediate variable between T and Y
  Y (Outcome)    : the effect
  DAG edges      : directed causal graph

Pipeline (from paper Section 3.1):
  Phase 1: Variable extraction with majority voting (K=3)
  Phase 2: Pairwise direction judgment
  Phase 3: DAG consistency verification
"""
import json
import argparse
import sys
import os
import time
sys.path.insert(0, os.path.expanduser("~/causal-memory-arena"))
from utils.model import call_llm, extract_json, majority_vote

# ── Prompts ──────────────────────────────────────────────────────────────────

VARIABLE_EXTRACTION_PROMPT = """You are a causal reasoning expert.

Scenario:
Context   : {premise}
Outcome   : {hypothesis}
Known cause: {causal_feature}

Identify the causal variables. Return ONLY this JSON with no explanation:
{{
  "T": "treatment — the direct action or cause",
  "Y": "outcome — the effect",
  "X": "covariate — background variable before T, or null",
  "M": "mediator — intermediate variable between T and Y, or null",
  "dag_edges": ["T->Y"],
  "explanation": "one sentence"
}}"""

PAIRWISE_PROMPT = """Given this scenario:
Context: {premise}
Variable A: {var_a}
Variable B: {var_b}

What is the causal direction? Return ONLY this JSON:
{{
  "direction": "A->B or B->A or none",
  "confidence": "high or medium or low",
  "reason": "one sentence"
}}"""

DAG_CONSISTENCY_PROMPT = """Check this causal DAG for validity.
Context: {premise}
T={T}, X={X}, M={M}, Y={Y}
Edges: {edges}

Return ONLY this JSON:
{{
  "is_valid": true,
  "has_cycle": false,
  "issues": [],
  "corrected_edges": ["T->Y"],
  "explanation": "one sentence"
}}"""

# ── Core function ─────────────────────────────────────────────────────────────

def discover_causal_structure(sample):
    premise        = sample.get("premise", "")[:400]
    hypothesis     = sample.get("hypothesis", "")[:200]
    causal_feature = sample.get("causal_feature", "")

    # Phase 1 — Variable extraction
    p1 = VARIABLE_EXTRACTION_PROMPT.format(
        premise=premise,
        hypothesis=hypothesis,
        causal_feature=causal_feature
    )
    cs = majority_vote(p1, K=3)
    if not cs:
        cs = {
            "T": causal_feature,
            "Y": hypothesis[:100],
            "X": None,
            "M": None,
            "dag_edges": ["T->Y"],
            "explanation": "direct causal relationship"
        }

    T = cs.get("T", causal_feature)
    Y = cs.get("Y", hypothesis[:100])
    X = cs.get("X")
    M = cs.get("M")

    # Phase 2 — Pairwise direction T->Y
    p2 = PAIRWISE_PROMPT.format(premise=premise, var_a=T, var_b=Y)
    pairwise = majority_vote(p2, K=3)

    # Phase 3 — DAG consistency
    p3 = DAG_CONSISTENCY_PROMPT.format(
        premise=premise,
        T=T, X=X, M=M, Y=Y,
        edges=cs.get("dag_edges", ["T->Y"])
    )
    dag_check = majority_vote(p3, K=3)

    # Apply corrections if needed
    if dag_check and not dag_check.get("is_valid", True):
        corrected = dag_check.get("corrected_edges")
        if corrected:
            cs["dag_edges"] = corrected

    return {
        "id"              : sample.get("id", ""),
        "premise"         : sample.get("premise", ""),
        "causal_feature"  : causal_feature,
        "hypothesis"      : hypothesis,
        "causal_structure": cs,
        "pairwise_check"  : pairwise,
        "dag_validation"  : dag_check
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int,   default=10)
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)
    data = data[:args.limit]

    print(f"Processing {len(data)} samples — causal structure discovery")

    results = []
    failed  = 0

    for i, sample in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {sample.get('id','')}")
        try:
            result = discover_causal_structure(sample)
            results.append(result)
            cs = result["causal_structure"]
            print(f"  T   : {str(cs.get('T',''))[:60]}")
            print(f"  Y   : {str(cs.get('Y',''))[:60]}")
            print(f"  X   : {str(cs.get('X',''))[:60]}")
            print(f"  M   : {str(cs.get('M',''))[:60]}")
            print(f"  DAG : {cs.get('dag_edges',[])}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        time.sleep(args.sleep)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Processed : {len(results)}/{len(data)}")
    print(f"Failed    : {failed}")
    print(f"Saved     -> {args.output_file}")

if __name__ == "__main__":
    main()

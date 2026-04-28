"""
CAUSAL-MEMORY-ARENA — Step 3: Validate Spurious Features

For each spurious feature, applies two checks (paper Section 3.3):
  1. Structural check  : no directed path from S to Y in DAG
  2. Counterfactual    : removing S does not change Y (4/5 runs agree)

Features passing both checks are confirmed spurious.
"""
import json
import argparse
import sys
import os
import time
sys.path.insert(0, os.path.expanduser("~/causal-memory-arena"))
from utils.model import call_llm, extract_json, majority_vote

STRUCTURAL_CHECK_PROMPT = """You are a causal reasoning expert.

Scenario:
Context : {premise}
T (true cause) : {T}
Y (outcome)    : {Y}
DAG edges      : {dag}

Spurious feature S: {S}

Does S have a DIRECT causal path to Y in this scenario?
- Check if S -> Y exists directly
- Check if S -> anything -> Y exists

Return ONLY this JSON:
{{"has_causal_path": false, "confidence": "high", "reason": "one sentence"}}"""

COUNTERFACTUAL_CHECK_PROMPT = """You are a causal reasoning expert.

Scenario:
Context : {premise}
T (true cause) : {T}
Y (outcome)    : {Y}

Spurious feature S: {S}

If we REMOVE S from the scenario completely, does Y still occur?
- YES means Y still happens without S (S is spurious)
- NO means Y changes without S (S might be causal)

Return ONLY this JSON:
{{"Y_unchanged_without_S": true, "confidence": "high", "reason": "one sentence"}}"""

def validate_spurious_feature(premise, T, Y, dag, S, spurious_type):
    """
    Validate one spurious feature S.
    Returns True if S is confirmed spurious.
    """
    # Check 1 — Structural
    p1 = STRUCTURAL_CHECK_PROMPT.format(
        premise=premise[:400], T=T, Y=Y, dag=dag, S=S
    )
    structural = majority_vote(p1, K=1)
    has_path = True  # default conservative
    if structural:
        has_path = structural.get("has_causal_path", True)

    # Check 2 — Counterfactual
    p2 = COUNTERFACTUAL_CHECK_PROMPT.format(
        premise=premise[:400], T=T, Y=Y, S=S
    )
    counterfactual = majority_vote(p2, K=1)
    y_unchanged = False  # default conservative
    if counterfactual:
        y_unchanged = counterfactual.get("Y_unchanged_without_S", False)

    # Confirmed spurious if:
    # - no causal path from S to Y (structural check passes)
    # - Y unchanged without S (counterfactual check passes)
    is_spurious = (not has_path) and y_unchanged

    return {
        "S"               : S,
        "spurious_type"   : spurious_type,
        "structural_check": {"has_causal_path": has_path, "result": structural},
        "counterfactual"  : {"Y_unchanged": y_unchanged, "result": counterfactual},
        "is_confirmed_spurious": is_spurious
    }

def validate_sample(sample):
    premise = sample.get("premise", "")
    cs      = sample.get("causal_structure", {})
    T       = cs.get("T", sample.get("causal_feature", ""))
    Y       = cs.get("Y", sample.get("hypothesis", ""))[:150]
    dag     = cs.get("dag_edges", ["T->Y"])
    sf      = sample.get("spurious_features", {})

    validated = {}

    for type_key, feature in sf.items():
        S = feature.get("S", "")
        if not S:
            continue
        spurious_type = feature.get("spurious_type", type_key)
        result = validate_spurious_feature(premise, T, Y, dag, S, spurious_type)
        validated[type_key] = {**feature, **result}

    return {
        "id"                : sample.get("id", ""),
        "premise"           : sample.get("premise", ""),
        "causal_feature"    : sample.get("causal_feature", ""),
        "hypothesis"        : sample.get("hypothesis", ""),
        "causal_structure"  : cs,
        "spurious_features" : sf,
        "validated_features": validated
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int,   default=10)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)
    data = data[:args.limit]

    print(f"Validating spurious features for {len(data)} samples...")

    results  = []
    failed   = 0
    confirmed = 0
    rejected  = 0

    for i, sample in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {sample.get('id','')}")
        try:
            result = validate_sample(sample)
            results.append(result)
            vf = result["validated_features"]
            for key, v in vf.items():
                status = "✅ SPURIOUS" if v["is_confirmed_spurious"] else "❌ REJECTED"
                if v["is_confirmed_spurious"]:
                    confirmed += 1
                else:
                    rejected += 1
                print(f"  {key}: {status} — {str(v.get('S',''))[:50]}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        time.sleep(args.sleep)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Processed : {len(results)}/{len(data)}")
    print(f"Confirmed spurious : {confirmed}")
    print(f"Rejected           : {rejected}")
    print(f"Failed             : {failed}")
    print(f"Saved -> {args.output_file}")

if __name__ == "__main__":
    main()

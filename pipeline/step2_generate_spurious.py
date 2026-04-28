"""
CAUSAL-MEMORY-ARENA — Step 2: Generate 3 Types of Spurious Correlations
"""
import json
import argparse
import sys
import os
import time
sys.path.insert(0, os.path.expanduser("~/causal-memory-arena"))
from utils.model import call_llm, extract_json, majority_vote

TYPE1_PROMPT = """You are building a causal reasoning benchmark.

Read this specific scenario carefully:
Context: {premise}
T (true cause): {T}
Y (outcome): {Y}

Generate a TYPE 1 SPURIOUS feature (Confounding) SPECIFIC to this scenario:
- Find a hidden common cause C mentioned in THIS context
- C causes both a spurious feature S AND the outcome Y
- S must be directly related to THIS specific scenario
- Do NOT use generic examples like motivation or parental involvement

Return ONLY valid JSON like this example:
{{"spurious_type": "confounding", "S": "spurious feature here", "C": "common cause here", "C_in_context": true, "answer": "no", "explanation": "reason here"}}"""

TYPE2_PROMPT = """You are building a causal reasoning benchmark.

Read this specific scenario carefully:
Context: {premise}
T (true cause): {T}
Y (outcome): {Y}

Generate a TYPE 2 SPURIOUS feature (Collider Bias) SPECIFIC to this scenario:
- Find a collider C caused by BOTH T and a spurious feature S
- S must be directly related to THIS specific scenario
- Do NOT use generic examples

Return ONLY valid JSON like this example:
{{"spurious_type": "collider", "S": "spurious feature here", "C": "collider effect here", "answer": "no", "explanation": "reason here"}}"""

TYPE3_PROMPT = """You are building a causal reasoning benchmark.

Read this specific scenario carefully:
Context: {premise}
T (true cause): {T}
Y (outcome): {Y}
X (covariate): {X}

Generate a TYPE 3 SPURIOUS feature (Proxy) SPECIFIC to this scenario:
- Find a proxy S that is a visible effect of covariate X in THIS scenario
- X causes both S and Y
- S must be directly related to THIS specific scenario

Return ONLY valid JSON like this example:
{{"spurious_type": "proxy", "S": "spurious proxy here", "X": "true covariate here", "answer": "no", "explanation": "reason here"}}"""

def generate_spurious_features(sample):
    premise = sample.get("premise", "")[:500]
    cs      = sample.get("causal_structure", {})
    T       = cs.get("T", sample.get("causal_feature", ""))
    Y       = cs.get("Y", sample.get("hypothesis", ""))[:150]
    X       = cs.get("X") or "background context"

    spurious_features = {}

    # Type 1 — Confounding
    p1 = TYPE1_PROMPT.format(premise=premise, T=T, Y=Y)
    type1 = majority_vote(p1, K=3)
    if type1 and "S" in type1:
        spurious_features["type1_confounding"] = type1

    # Type 2 — Collider
    p2 = TYPE2_PROMPT.format(premise=premise, T=T, Y=Y)
    type2 = majority_vote(p2, K=3)
    if type2 and "S" in type2:
        spurious_features["type2_collider"] = type2

    # Type 3 — Proxy
    p3 = TYPE3_PROMPT.format(premise=premise, T=T, Y=Y, X=X)
    type3 = majority_vote(p3, K=3)
    if type3 and "S" in type3:
        spurious_features["type3_proxy"] = type3

    return {
        "id"               : sample.get("id", ""),
        "premise"          : sample.get("premise", ""),
        "causal_feature"   : sample.get("causal_feature", ""),
        "hypothesis"       : sample.get("hypothesis", ""),
        "causal_structure" : cs,
        "spurious_features": spurious_features
    }

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

    print(f"Generating spurious features for {len(data)} samples...")

    results = []
    failed  = 0

    for i, sample in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {sample.get('id','')}")
        try:
            result = generate_spurious_features(sample)
            results.append(result)
            sf = result["spurious_features"]
            if "type1_confounding" in sf:
                print(f"  T1: {str(sf['type1_confounding'].get('S',''))[:70]}")
            if "type2_collider" in sf:
                print(f"  T2: {str(sf['type2_collider'].get('S',''))[:70]}")
            if "type3_proxy" in sf:
                print(f"  T3: {str(sf['type3_proxy'].get('S',''))[:70]}")
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

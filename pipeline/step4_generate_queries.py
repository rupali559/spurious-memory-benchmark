"""
CAUSAL-MEMORY-ARENA — Step 4: Generate 4 Query Versions
"""
import json
import argparse
import sys
import os
import time
sys.path.insert(0, os.path.expanduser("~/causal-memory-arena"))
from utils.model import call_llm, extract_json, majority_vote

Q0_PROMPT = """Given this scenario:
Context: {premise}
True cause (T): {T}
Outcome (Y): {Y}

Write a natural YES/NO question asking if T causes Y.
Return ONLY this JSON:
{{"query_type": "Q0_causal", "query": "natural question here using actual T and Y", "answer": "yes", "explanation": "why T causes Y"}}"""

Q1_PROMPT = """Given this scenario:
Context: {premise}
True cause (T): {T}
Outcome (Y): {Y}
Spurious feature (S): {S}
Common cause (C): {C}

Write a natural YES/NO question asking if S causes Y.
The question should sound plausible but answer is NO because C causes both S and Y.
Return ONLY this JSON:
{{"query_type": "Q1_confounding", "query": "natural question using actual S and Y", "answer": "no", "trap": "C is visible in context so agent thinks S causes Y", "explanation": "C causes both S and Y not S causing Y directly"}}"""

Q2_PROMPT = """Given this scenario:
Context: {premise}
True cause (T): {T}
Outcome (Y): {Y}
Spurious feature (S): {S}
Collider (C): {C}

Write a natural YES/NO question asking if S causes Y.
The question should sound plausible but answer is NO because of collider bias through C.
Return ONLY this JSON:
{{"query_type": "Q2_collider", "query": "natural question using actual S and Y", "answer": "no", "trap": "conditioning on C makes S appear correlated with Y", "explanation": "S and Y are only associated through collider C"}}"""

Q3_PROMPT = """Given this scenario:
Context: {premise}
True cause (T): {T}
Outcome (Y): {Y}
Spurious proxy (S): {S}
True covariate (X): {X}

Write a natural YES/NO question asking if S causes Y.
The question should sound plausible but answer is NO because S is just a proxy for X.
Return ONLY this JSON:
{{"query_type": "Q3_proxy", "query": "natural question using actual S and Y", "answer": "no", "trap": "S and X share surface vocabulary making them hard to distinguish", "explanation": "X causes both S and Y so S is just a proxy"}}"""

def generate_queries(sample):
    premise = sample.get("premise", "")[:400]
    cs      = sample.get("causal_structure", {})
    T       = cs.get("T", sample.get("causal_feature", ""))
    Y       = cs.get("Y", sample.get("hypothesis", ""))[:150]
    vf      = sample.get("validated_features", {})

    queries = {}

    # Q0 — True causal baseline
    p0 = Q0_PROMPT.format(premise=premise, T=T, Y=Y)
    q0 = majority_vote(p0, K=1)
    if q0:
        q0["T"] = T
        q0["Y"] = Y
        queries["Q0"] = q0

    # Q1 — Type 1 confounding
    t1 = vf.get("type1_confounding", {})
    if t1.get("is_confirmed_spurious"):
        S = t1.get("S", "")
        C = t1.get("C", "unknown common cause")
        p1 = Q1_PROMPT.format(premise=premise, T=T, Y=Y, S=S, C=C)
        q1 = majority_vote(p1, K=1)
        if q1:
            q1["S"] = S
            q1["C"] = C
            queries["Q1"] = q1

    # Q2 — Type 2 collider
    t2 = vf.get("type2_collider", {})
    if t2.get("is_confirmed_spurious"):
        S = t2.get("S", "")
        C = t2.get("C", "unknown collider")
        p2 = Q2_PROMPT.format(premise=premise, T=T, Y=Y, S=S, C=C)
        q2 = majority_vote(p2, K=1)
        if q2:
            q2["S"] = S
            q2["C"] = C
            queries["Q2"] = q2

    # Q3 — Type 3 proxy
    t3 = vf.get("type3_proxy", {})
    if t3.get("is_confirmed_spurious"):
        S = t3.get("S", "")
        X = t3.get("X", cs.get("X", "unknown covariate"))
        p3 = Q3_PROMPT.format(premise=premise, T=T, Y=Y, S=S, X=X)
        q3 = majority_vote(p3, K=1)
        if q3:
            q3["S"] = S
            q3["X"] = X
            queries["Q3"] = q3

    return {
        "id"              : sample.get("id", ""),
        "premise"         : sample.get("premise", ""),
        "causal_feature"  : sample.get("causal_feature", ""),
        "hypothesis"      : sample.get("hypothesis", ""),
        "causal_structure": cs,
        "queries"         : queries
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

    print(f"Generating queries for {len(data)} samples...")

    results  = []
    failed   = 0
    q_counts = {"Q0": 0, "Q1": 0, "Q2": 0, "Q3": 0}

    for i, sample in enumerate(data):
        print(f"\n[{i+1}/{len(data)}] {sample.get('id','')}")
        try:
            result = generate_queries(sample)
            results.append(result)
            qs = result["queries"]
            for qk in ["Q0","Q1","Q2","Q3"]:
                if qk in qs:
                    q_counts[qk] += 1
                    print(f"  {qk}: {str(qs[qk].get('query',''))[:70]}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        time.sleep(args.sleep)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Processed        : {len(results)}/{len(data)}")
    for qk, cnt in q_counts.items():
        print(f"{qk}             : {cnt}")
    print(f"Failed           : {failed}")
    print(f"Saved -> {args.output_file}")

if __name__ == "__main__":
    main()

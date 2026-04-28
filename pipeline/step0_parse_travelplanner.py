"""
Parse TravelPlanner dataset into causal-memory-arena format.

TravelPlanner task: plan a trip given constraints
Causal structure:
  T (Treatment) : travel constraints (budget, days, people)
  Y (Outcome)   : successful travel plan
  X (Covariate) : origin, destination, dates
"""
import json
import argparse
import os

def parse_sample(d, idx):
    # Build premise from query + constraints
    constraints = []
    local = d.get("local_constraint", {})
    if isinstance(local, str):
        import ast
        try:
            local = ast.literal_eval(local)
        except:
            local = {}

    if local.get("house rule"):
        constraints.append(f"house rule: {local['house rule']}")
    if local.get("cuisine"):
        constraints.append(f"cuisine preference: {local['cuisine']}")
    if local.get("room type"):
        constraints.append(f"room type: {local['room type']}")
    if local.get("transportation"):
        constraints.append(f"transportation: {local['transportation']}")

    constraint_str = ", ".join(constraints) if constraints else "no specific constraints"

    premise = (
        f"{d.get('query', '')} "
        f"Budget: ${d.get('budget', 'N/A')}. "
        f"People: {d.get('people_number', 1)}. "
        f"Constraints: {constraint_str}."
    )

    causal_feature = (
        f"planning a {d.get('days', 3)}-day trip from "
        f"{d.get('org', '')} to {d.get('dest', '')} "
        f"within budget of ${d.get('budget', 'N/A')} "
        f"for {d.get('people_number', 1)} person(s)"
    )

    hypothesis = (
        f"A valid travel plan can be created from "
        f"{d.get('org', '')} to {d.get('dest', '')} "
        f"for {d.get('days', 3)} days within the given constraints"
    )

    return {
        "id"             : f"travelplanner_{idx}",
        "premise"        : premise,
        "causal_feature" : causal_feature,
        "hypothesis"     : hypothesis,
        "metadata"       : {
            "org"    : d.get("org", ""),
            "dest"   : d.get("dest", ""),
            "days"   : d.get("days", 3),
            "budget" : d.get("budget", 0),
            "people" : d.get("people_number", 1),
            "level"  : d.get("level", ""),
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  default="/home/rupali/causal-memory-arena/data/travelplanner/raw.json")
    parser.add_argument("--output_file", default="/home/rupali/causal-memory-arena/data/travelplanner/parsed.json")
    parser.add_argument("--limit", type=int, default=225)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)
    data = data[:args.limit]

    results = []
    for i, d in enumerate(data):
        sample = parse_sample(d, i)
        results.append(sample)
        print(f"[{i+1}/{len(data)}] {sample['id']} — {d.get('org','')} -> {d.get('dest','')}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Parsed  : {len(results)} instances")
    print(f"Saved  -> {args.output_file}")

if __name__ == "__main__":
    main()

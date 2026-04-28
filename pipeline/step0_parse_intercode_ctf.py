"""
Parse InterCode-CTF dataset into causal-memory-arena format.

InterCode-CTF task: find hidden flag using Bash commands
Causal structure:
  T (Treatment) : correct sequence of bash commands to find flag
  Y (Outcome)   : successfully capturing the flag
  X (Covariate) : file access, task environment
"""
import json
import argparse
import os

def parse_sample(d):
    premise = (
        f"CTF Challenge: {d.get('query', '')} "
        f"Category: {', '.join(d.get('tags', []))}. "
        f"The agent must interact with a Bash shell to find the hidden flag."
    )

    causal_feature = (
        f"executing the correct sequence of bash commands "
        f"to analyze and extract the hidden flag from the challenge files"
    )

    hypothesis = (
        f"The agent successfully captures the flag "
        f"by correctly analyzing the challenge environment"
    )

    return {
        "id"             : f"ctf_{d.get('task_id', 0)}",
        "premise"        : premise,
        "causal_feature" : causal_feature,
        "hypothesis"     : hypothesis,
        "metadata"       : {
            "task_id" : d.get("task_id", 0),
            "gold"    : d.get("gold", ""),
            "source"  : d.get("source", ""),
            "tags"    : d.get("tags", []),
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  default="/home/rupali/causal-memory-arena/data/intercode_ctf/raw.json")
    parser.add_argument("--output_file", default="/home/rupali/causal-memory-arena/data/intercode_ctf/parsed.json")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = json.load(f)
    data = data[:args.limit]

    results = []
    for d in data:
        sample = parse_sample(d)
        results.append(sample)
        print(f"[{d.get('task_id',0)}] {sample['id']} — {', '.join(d.get('tags',[]))}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Parsed  : {len(results)} instances")
    print(f"Saved  -> {args.output_file}")

if __name__ == "__main__":
    main()

"""
CAUSAL-MEMORY-ARENA — Step 5 with Token + Time tracking

Tracks per query:
  - Input tokens
  - Output tokens  
  - Total tokens
  - Running time (seconds)
"""
import json
import argparse
import sys
import os
import time
sys.path.insert(0, os.path.expanduser("~/causal-memory-arena"))
from utils.model import call_llm, extract_json

EVAL_PROMPT_NO_MEMORY = """You are a causal reasoning expert.

Context: {premise}

Question: {query}

Answer YES or NO based on whether the stated relationship is genuinely causal.
Consider carefully — some relationships may be correlations, not causes.

Return ONLY this JSON:
{{"answer": "yes or no", "confidence": "high or medium or low", "reasoning": "one sentence"}}"""

EVAL_PROMPT_WITH_MEMORY = """You are a causal reasoning expert with memory of past interactions.

Memory from past interactions:
{memory}

Context: {premise}

Question: {query}

Answer YES or NO based on whether the stated relationship is genuinely causal.
Consider carefully — some relationships may be correlations, not causes.

Return ONLY this JSON:
{{"answer": "yes or no", "confidence": "high or medium or low", "reasoning": "one sentence"}}"""

def count_tokens(text):
    """Approximate token count — 1 token ~ 4 chars."""
    return len(text) // 4

def build_memory(sample):
    cs = sample.get('causal_structure', {})
    T  = cs.get('T', sample.get('causal_feature', ''))
    Y  = cs.get('Y', sample.get('hypothesis', ''))[:150]
    premise = sample.get('premise', '')[:300]
    queries = sample.get('queries', {})
    candidates = [T]
    for qk in ['Q1','Q2','Q3']:
        q = queries.get(qk, {})
        S = q.get('S','')
        if S and S not in candidates:
            candidates.append(S)
    import random
    random.shuffle(candidates)
    memory = f"[CAUSAL GRAPH MEMORY]\nContext: {premise}\n\nOutcome: {Y}\n\nCandidates:\n"
    for c in candidates:
        memory += f"  - {c}\n"
    memory += "\nTask: Determine which candidate directly causes the outcome."
    return memory

def extract_answer(response):
    if not response:
        return "unknown"
    parsed = extract_json(response)
    if parsed and "answer" in parsed:
        return parsed["answer"].lower().strip()
    text = response.lower()
    if "yes" in text[:50]: return "yes"
    if "no" in text[:50]: return "no"
    return "unknown"

def evaluate_sample(sample, system):
    premise = sample.get('premise', '')[:400]
    queries = sample.get('queries', {})
    results = {}
    memory = build_memory(sample) if system != "qwen_alone" else None

    for qk, query_data in queries.items():
        query    = query_data.get('query', '')
        expected = query_data.get('answer', '').lower().strip()
        if not query:
            continue

        if system == "qwen_alone":
            prompt = EVAL_PROMPT_NO_MEMORY.format(premise=premise, query=query)
        else:
            prompt = EVAL_PROMPT_WITH_MEMORY.format(premise=premise, query=query, memory=memory)

        # Count input tokens
        input_tokens = count_tokens(prompt)

        # Time the call
        start = time.time()
        response = call_llm(prompt)
        elapsed = time.time() - start

        # Count output tokens
        output_tokens = count_tokens(response) if response else 0
        total_tokens  = input_tokens + output_tokens

        predicted = extract_answer(response)
        correct   = (predicted == expected)

        results[qk] = {
            "query"        : query,
            "expected"     : expected,
            "predicted"    : predicted,
            "correct"      : correct,
            "input_tokens" : input_tokens,
            "output_tokens": output_tokens,
            "total_tokens" : total_tokens,
            "time_seconds" : round(elapsed, 3),
            "response"     : response[:200] if response else ""
        }

    return results

def compute_metrics(all_results, system):
    counts = {"Q0":[0,0],"Q1":[0,0],"Q2":[0,0],"Q3":[0,0]}
    all_tokens = []
    all_times  = []

    for sample_results in all_results:
        r = sample_results.get(system, {})
        for qk, qr in r.items():
            if qk in counts:
                counts[qk][1] += 1
                if qr.get('correct'):
                    counts[qk][0] += 1
                all_tokens.append(qr.get('total_tokens', 0))
                all_times.append(qr.get('time_seconds', 0))

    metrics = {}
    for qk, (correct, total) in counts.items():
        metrics[qk] = {
            "correct" : correct,
            "total"   : total,
            "accuracy": round(100*correct/total, 1) if total > 0 else 0
        }

    metrics["avg_tokens_per_query"] = round(sum(all_tokens)/len(all_tokens), 1) if all_tokens else 0
    metrics["avg_time_per_query"]   = round(sum(all_times)/len(all_times), 3) if all_times else 0
    metrics["total_tokens"]         = sum(all_tokens)
    metrics["total_time_seconds"]   = round(sum(all_times), 1)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--log_dir",     required=True)
    parser.add_argument("--dataset",     default="DATASET")
    parser.add_argument("--systems",     default="qwen,mem0,amem")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()

    systems_map = {"qwen":"qwen_alone","mem0":"mem0","amem":"amem"}
    systems = [systems_map[s] for s in args.systems.split(",")]

    with open(args.input_file) as f:
        data = json.load(f)
    data = [d for d in data if d.get('queries',{}).get('Q0')]
    data = data[:args.limit]

    print(f"Evaluating {len(data)} samples on {args.dataset}")
    os.makedirs(args.log_dir, exist_ok=True)

    all_results = []
    for i, sample in enumerate(data):
        print(f"[{i+1}/{len(data)}] {sample.get('id','')}")
        sample_result = {"id": sample.get("id","")}
        for system in systems:
            results = evaluate_sample(sample, system)
            sample_result[system] = results
        all_results.append(sample_result)
        time.sleep(args.sleep)

    # Save logs
    for system in systems:
        log_path = os.path.join(args.log_dir, f"per_query_{system}.json")
        logs = []
        for sr in all_results:
            for qk, qr in sr.get(system,{}).items():
                logs.append({"id": sr["id"], "query_type": qk, **qr})
        with open(log_path,"w") as f:
            json.dump(logs, f, indent=2)

    # Print results
    lines = [f"=== {args.dataset} EVALUATION RESULTS ===\n"]
    for system in systems:
        m = compute_metrics(all_results, system)
        lines.append(f"--- {system} ---")
        for qk in ["Q0","Q1","Q2","Q3"]:
            qm = m.get(qk,{})
            lines.append(f"  {qk}: {qm.get('accuracy',0)}% ({qm.get('correct',0)}/{qm.get('total',0)})")
        lines.append(f"  Avg tokens/query : {m['avg_tokens_per_query']}")
        lines.append(f"  Avg time/query   : {m['avg_time_per_query']}s")
        lines.append(f"  Total tokens     : {m['total_tokens']}")
        lines.append(f"  Total time       : {m['total_time_seconds']}s")
        lines.append("")

    output_text = "\n".join(lines)
    print("\n" + output_text)
    with open(args.output_file,"w") as f:
        f.write(output_text)
    print(f"Saved -> {args.output_file}")

if __name__ == "__main__":
    main()

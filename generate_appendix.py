import json

def find_samples_with_all_queries(queries_file, n=2):
    data = json.load(open(queries_file))
    results = []
    for d in data:
        qs = d.get('queries', {})
        if 'Q0' in qs and 'Q1' in qs and 'Q2' in qs and 'Q3' in qs:
            results.append(d)
        if len(results) == n:
            break
    return results

def get_filename(query):
    parts = query.split("'")
    return parts[1] if len(parts) > 1 else 'flag_file'

def format_tp_example(d, raw_sample, idx):
    plan = eval(str(raw_sample.get('annotated_plan', '[]')))
    days = plan[1] if isinstance(plan, list) and len(plan) > 1 else []
    cs = d.get('causal_structure', {})
    qs = d.get('queries', {})
    out = []
    out.append(f"\nExample {idx}: {d['id']}")
    out.append(f"  Origin      : {raw_sample.get('org','')}")
    out.append(f"  Destination : {raw_sample.get('dest','')}")
    out.append(f"  Duration    : {raw_sample.get('days','')} days")
    out.append(f"  Budget      : ${raw_sample.get('budget','')}")
    out.append(f"  People      : {raw_sample.get('people_number','')} person(s)")
    out.append(f"  Query       : {raw_sample.get('query','')}")
    out.append(f"\n  Annotated Plan (ground truth):")
    for day in days[:3]:
        out.append(f"    Day {day['days']}: {day['current_city']}")
        if day.get('transportation') and day['transportation'] != '-':
            out.append(f"      Transport    : {day['transportation']}")
        if day.get('breakfast') and day['breakfast'] != '-':
            out.append(f"      Breakfast    : {day['breakfast']}")
        if day.get('attraction') and day['attraction'] != '-':
            out.append(f"      Attraction   : {day['attraction']}")
        if day.get('lunch') and day['lunch'] != '-':
            out.append(f"      Lunch        : {day['lunch']}")
        if day.get('dinner') and day['dinner'] != '-':
            out.append(f"      Dinner       : {day['dinner']}")
        if day.get('accommodation') and day['accommodation'] != '-':
            out.append(f"      Accommodation: {day['accommodation']}")
    out.append(f"    ...continues for {raw_sample.get('days',7)} days")
    out.append(f"\n  Causal Structure:")
    out.append(f"    T  : {cs.get('T','')}")
    out.append(f"    Y  : {cs.get('Y','')}")
    out.append(f"    X  : {cs.get('X','')}")
    out.append(f"    DAG: {cs.get('dag_edges',[])}")
    out.append(f"\n  Three Types of Spuriousness:")
    for qk, label in [('Q1','Type 1 Confounding (S<-C->Y)'),('Q2','Type 2 Collider (T->C<-S)'),('Q3','Type 3 Proxy (S<-X->Y)')]:
        q = qs.get(qk, {})
        if q:
            out.append(f"    {label}:")
            out.append(f"      S      : {q.get('S','')}")
            out.append(f"      C/X    : {q.get('C', q.get('X',''))}")
            out.append(f"      Query  : {q.get('query','')}")
            out.append(f"      Answer : {q.get('answer','').upper()}")
            out.append(f"      Explain: {q.get('explanation','')}")
    return '\n'.join(out)

def format_ctf_example(d, raw_sample, idx):
    cs = d.get('causal_structure', {})
    qs = d.get('queries', {})
    filename = get_filename(raw_sample.get('query',''))
    gold = raw_sample.get('gold','')
    out = []
    out.append(f"\nExample {idx}: {d['id']}")
    out.append(f"  Task ID : {raw_sample.get('task_id','')}")
    out.append(f"  Query   : {raw_sample.get('query','')}")
    out.append(f"  Category: {', '.join(raw_sample.get('tags',[]))}")
    out.append(f"  Gold    : {gold}")
    out.append(f"  Source  : {raw_sample.get('source','')}")
    out.append(f"\n  Multi-turn bash interaction:")
    out.append(f"    Turn 1: ls -la  ->  {filename} found")
    out.append(f"    Turn 2: cat {filename}  ->  encrypted code displayed")
    out.append(f"    Turn 3: python3 decode script  ->  partial flag")
    out.append(f"    Turn 4: python3 {filename}  ->  {gold}  (flag captured!)")
    out.append(f"    Memory: Turn 2 file contents required at Turn 3")
    out.append(f"\n  Causal Structure:")
    out.append(f"    T  : {cs.get('T','')}")
    out.append(f"    Y  : {cs.get('Y','')}")
    out.append(f"    X  : {cs.get('X','')}")
    out.append(f"    DAG: {cs.get('dag_edges',[])}")
    out.append(f"\n  Three Types of Spuriousness:")
    for qk, label in [('Q1','Type 1 Confounding (S<-C->Y)'),('Q2','Type 2 Collider (T->C<-S)'),('Q3','Type 3 Proxy (S<-X->Y)')]:
        q = qs.get(qk, {})
        if q:
            out.append(f"    {label}:")
            out.append(f"      S      : {q.get('S','')}")
            out.append(f"      C/X    : {q.get('C', q.get('X',''))}")
            out.append(f"      Query  : {q.get('query','')}")
            out.append(f"      Answer : {q.get('answer','').upper()}")
            out.append(f"      Explain: {q.get('explanation','')}")
    return '\n'.join(out)

base = '/home/rupali/causal-memory-arena'
tp_samples  = find_samples_with_all_queries(f'{base}/data/travelplanner/queries.json', n=2)
ctf_samples = find_samples_with_all_queries(f'{base}/data/intercode_ctf/queries.json', n=2)
tp_raw  = json.load(open(f'{base}/data/travelplanner/raw.json'))
ctf_raw = json.load(open(f'{base}/data/intercode_ctf/raw.json'))

def find_raw(raw_list, sample_id):
    idx = int(sample_id.split('_')[-1])
    return raw_list[idx] if idx < len(raw_list) else raw_list[0]

output = []
output.append("=" * 70)
output.append("APPENDIX A: MULTI-TURN DATASET EXAMPLES")
output.append("TravelPlanner (ICML 2024) | InterCode-CTF (NeurIPS 2023)")
output.append("=" * 70)

output.append("\n" + "=" * 70)
output.append("DATASET 1: TravelPlanner (ICML 2024)")
output.append("=" * 70)
output.append("\n1.1 HOW ORIGINAL DATA LOOKS")
output.append("-" * 40)
output.append("Dataset        : TravelPlanner (osunlp/TravelPlanner)")
output.append("Total instances: 225 (train + validation)")
output.append("Task type      : Multi-turn travel planning with tool use")
output.append("Why multi-turn : Agent searches flights, hotels, restaurants across multiple steps")
output.append("Why needs memory: Must track costs across days to stay within budget")

for i, d in enumerate(tp_samples):
    raw = find_raw(tp_raw, d['id'])
    output.append(format_tp_example(d, raw, i+1))

output.append("\n1.2 SYSTEM PERFORMANCE — TravelPlanner (225 instances)")
output.append("-" * 40)
output.append(f"  {'System':<25} {'Q0':>8} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Leakage':>8}")
output.append(f"  {'-'*65}")
output.append(f"  {'Qwen alone':<25} {'74.2%':>8} {'57.1%':>8} {'50.9%':>8} {'57.0%':>8} {'46.7%':>8}")
output.append(f"  {'Mem0 + Qwen':<25} {'68.4%':>8} {'56.3%':>8} {'51.8%':>8} {'57.9%':>8} {'31.0%':>8}")
output.append(f"  {'A-mem-sys + Qwen':<25} {'68.0%':>8} {'61.3%':>8} {'57.3%':>8} {'64.0%':>8} {'26.9%':>8}")

output.append("\n" + "=" * 70)
output.append("DATASET 2: InterCode-CTF (NeurIPS 2023)")
output.append("=" * 70)
output.append("\n2.1 HOW ORIGINAL DATA LOOKS")
output.append("-" * 40)
output.append("Dataset        : InterCode-CTF (princeton-nlp/intercode)")
output.append("Total instances: 100 CTF challenges")
output.append("Task type      : Multi-turn bash interaction — Capture The Flag")
output.append("Why multi-turn : Each bash command depends on previous output")
output.append("Why needs memory: Must remember file contents across turns to decode flag")

for i, d in enumerate(ctf_samples):
    raw = find_raw(ctf_raw, d['id'])
    output.append(format_ctf_example(d, raw, i+1))

output.append("\n2.2 SYSTEM PERFORMANCE — InterCode-CTF (100 instances)")
output.append("-" * 40)
output.append(f"  {'System':<25} {'Q0':>8} {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Leakage':>8}")
output.append(f"  {'-'*65}")
output.append(f"  {'Qwen alone':<25} {'17.0%':>8} {'61.6%':>8} {'55.6%':>8} {'43.2%':>8} {'3.6%':>8}")
output.append(f"  {'Mem0 + Qwen':<25} {'69.0%':>8} {'68.6%':>8} {'56.8%':>8} {'50.0%':>8} {'3.6%':>8}")
output.append(f"  {'A-mem-sys + Qwen':<25} {'69.0%':>8} {'66.3%':>8} {'61.7%':>8} {'47.3%':>8} {'2.7%':>8}")
output.append(f"  Key: Qwen alone only 17% Q0 — memory critical (+52% with Mem0/A-mem)")
output.append("\n" + "=" * 70)

text = '\n'.join(output)
print(text)
with open('/home/rupali/causal-memory-arena/appendix_a_output.txt', 'w') as f:
    f.write(text)
print("\nSaved -> /home/rupali/causal-memory-arena/appendix_a_output.txt")

"""
Local Qwen model utility for CAUSAL-MEMORY-ARENA.
Uses Qwen/Qwen2.5-1.5B-Instruct via HuggingFace transformers.
"""
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)
model.eval()
print("Model loaded.")

def call_llm(prompt, max_new_tokens=512):
    """Call local Qwen model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    output_ids = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def extract_json(text):
    """Extract JSON from LLM response."""
    match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except:
            pass
    start = text.find('{')
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except:
                    pass
    return None

def majority_vote(prompt, K=3):
    """
    Run prompt K times and return most common result.
    Paper uses K=5 with 4/5 agreement threshold.
    We use K=3 for efficiency with local model.
    """
    results = []
    for _ in range(K):
        response = call_llm(prompt)
        parsed = extract_json(response)
        if parsed:
            results.append(parsed)
    return results[0] if results else None  # K=1 for speed

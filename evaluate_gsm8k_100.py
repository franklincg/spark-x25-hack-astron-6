import json, random, re, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "gsm8k-test-3101c7d.jsonl"
OUT = ROOT / "results_gsm8k_100_seed20260907.jsonl"
SUMMARY = ROOT / "summary_gsm8k_100.json"
URL = "http://127.0.0.1:30000/v1/chat/completions"
SEED = 20260907
N = 100

rows = [json.loads(x) for x in DATASET.read_text(encoding="utf-8").splitlines() if x.strip()]
rng = random.Random(SEED)
indices = sorted(rng.sample(range(len(rows)), N))

def normalize(value):
    if value is None:
        return None
    value = str(value).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        n = float(value)
        return str(int(n)) if n.is_integer() else ("%.10g" % n)
    except Exception:
        return value.lower()

def gold(answer):
    return normalize(answer.rsplit("####", 1)[-1])

def pred(text):
    hits = re.findall(r"FINAL_ANSWER\s*:\s*(?:<answer>)?\s*([-+]?[$]?[0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*</answer>)?", text, re.I)
    return normalize(hits[-1]) if hits else None

def call_model(question):
    payload = {
        "model": "spark25-math",
        "messages": [{"role": "user", "content": (
            "Solve the following grade-school math problem. Give a concise solution in 1 to 4 sentences. "
            "Do not use external tools. End with exactly FINAL_ANSWER: <answer>.\n\n" + question
        )}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": 256, "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type":"application/json"})
    err = None
    for attempt in range(3):
        try:
            started = time.time()
            with urllib.request.urlopen(req, timeout=180) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            return obj, time.time() - started
        except Exception as exc:
            err = repr(exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(err)

completed = {}
if OUT.exists():
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            completed[item["index"]] = item

for pos, idx in enumerate(indices, 1):
    if idx in completed:
        print(f"[{pos}/{N}] index={idx} already complete", flush=True)
        continue
    row = rows[idx]
    response, latency = call_model(row["question"])
    message = response["choices"][0].get("message", {})
    content = message.get("content") or ""
    prediction = pred(content)
    target = gold(row["answer"])
    item = {"index": idx, "question": row["question"], "gold": target,
            "prediction": prediction, "correct": prediction == target,
            "latency_s": latency, "raw_message": message, "usage": response.get("usage")}
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    completed[idx] = item
    print(f"[{pos}/{N}] index={idx} pred={prediction!r} gold={target!r} correct={item['correct']} {latency:.1f}s", flush=True)

items = [completed[i] for i in indices if i in completed]
correct = sum(bool(x["correct"]) for x in items)
unparseable = sum(x.get("prediction") is None for x in items)
summary = {"dataset":"openai/grade-school-math GSM8K test",
           "dataset_revision":"3101c7d5072418e28b9008a6636bde82a006892c",
           "model":"Spark-X2.5-1.7B Q8_0 GGUF", "sample_seed":SEED,
           "sample_indices":indices, "n":len(items), "correct":correct,
           "accuracy": correct/len(items) if items else 0,
           "unparseable": unparseable,
           "mean_latency_s": sum(x["latency_s"] for x in items)/len(items) if items else None,
           "decoding":{"thinking":False,"temperature":0.0,"top_p":1.0,"max_tokens":256,"pass_at":"pass@1"}}
SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)

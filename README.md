# [HER Hack-Astron #6] Spark-X2.5-1.7B Q8_0 — Reproducible GSM8K 100-sample CPU evaluation

Challenge: https://github.com/XHToken/Spark-X2.5/issues/9

Public reproducibility repository: https://github.com/franklincg/spark-x25-hack-astron-6

## Result

I evaluated **Spark-X2.5-1.7B Q8_0 GGUF** on a fixed, seeded 100-item subset of the **GSM8K test split** with no external tools and deterministic pass@1 decoding.

- **Accuracy:** 70/100 = **70.0%**
- **Unparseable outputs:** 1/100
- **Mean end-to-end request latency:** 8.123 s/sample
- **Pass:** pass@1
- **Subset seed:** `20260907`
- **Thinking mode:** disabled
- **Temperature:** `0.0`
- **top_p:** `1.0`
- **max_tokens:** `256`

This is intentionally reported as a **seeded 100-item subset**, not a full-GSM8K score.

## Exact model artifact

- Official GGUF repo: `XHToken/Spark-X2.5-1.7B-GGUF`
- Repo revision: `23e1fcac55e7dd71e4c12a23723cc228ba0e5e85`
- File: `Spark-X2.5-1.7B-Q8_0.gguf`
- File size: `1,820,112,704` bytes
- SHA-256: `cd77c03185a834bb1162a4b7713520be5838058bfc54873645beff470bb24442`
- Upstream model repo: `XHToken/Spark-X2.5-1.7B`
- Upstream revision observed for this run: `448e61eb392c00f2c403185c5b56d5e0665bfaab`

The local GGUF SHA-256 exactly matched the LFS SHA-256 exposed by the public Hugging Face API for the official `XHToken/Spark-X2.5-1.7B-GGUF` Q8_0 file.

## Dataset provenance

- Dataset: `openai/grade-school-math` / GSM8K
- Split: `test`
- Dataset revision: `3101c7d5072418e28b9008a6636bde82a006892c`
- Local test JSONL SHA-256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
- Sampling method: Python `random.Random(20260907).sample(range(len(test)), 100)`, then sorted by source index

The exact selected source indices are preserved in `summary_gsm8k_100.json`.

## Prompt

Every item used the same user prompt:

```text
Solve the following grade-school math problem. Give a concise solution in 1 to 4 sentences. Do not use external tools. End with exactly FINAL_ANSWER: <answer>.

{question}
```

No few-shot examples and no tool calls were used.

## Runtime and hardware

- Runtime: XHToken llama.cpp fork
- `llama-server` version: `0.1.2-dev`, build `1`, commit `4a3635c`
- Build toolchain: MSVC 19.44.35228.0 / Windows AMD64
- Inference backend used for this benchmark: **CPU only**
- CPU: 11th Gen Intel Core i9-11900K @ 3.50 GHz
- Threads: 8
- Context: 4096
- Host: Windows x64
- NVIDIA RTX 4090 and GTX 1080 Ti were installed in the machine, but **were not used** by this llama.cpp build; `llama-server --list-devices` returned no accelerator devices.

Sanitized server invocation:

```bash
llama-server -m Spark-X2.5-1.7B-Q8_0.gguf \
  --host 127.0.0.1 --port 30000 \
  -c 4096 -t 8 -tb 8 \
  --alias spark25-math --jinja
```

Evaluator invocation:

```bash
python evaluate_gsm8k_100.py
```

## Scoring

The evaluator extracted the last numeric value following the required marker with this regular expression:

```python
r"FINAL_ANSWER\s*:\s*(?:<answer>)?\s*([-+]?[$]?[0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*</answer>)?"
```

Both prediction and GSM8K gold answer were normalized by stripping whitespace, commas, `$`, and a trailing period, then normalizing numeric values (`60.0` -> `60`). Exact normalized equality counted as correct. A response without a parseable `FINAL_ANSWER` counted as incorrect. There was **1 unparseable response**.

## Representative successes

### Source index 14 — correct

Question: In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?

Model output:

> There are 20 students total. 20% are in contemporary dance, so 0.20 × 20 = 4 students. The remaining 16 students are in jazz dance, which is 25% of 16, or 0.25 × 16 = 4 students. The rest, 16 − 4 = 12 students, are in hip-hop dance. To find the percentage, divide 12 by 20 and multiply by 100: (12/20) × 100 = 60%.
>
> FINAL_ANSWER: `<answer>60%</answer>`

Gold: `60` — Prediction: `60`

### Source index 29 — correct

Question: Gloria is shoe shopping when she comes across a pair of boots that fit her shoe budget. However, she has to choose between the boots and two pairs of high heels that together cost five dollars less than the boots. If one pair of heels costs $33 and the other costs twice as much, how many dollars are the boots?

Model output:

> The two pairs of heels cost $33 and twice as much, so the other pair costs $66. Together, the heels cost $99. The heels cost five dollars less than the boots, so the boots cost $104.
>
> FINAL_ANSWER: `<answer>104</answer>`

Gold: `104` — Prediction: `104`

## Representative failures

### Source index 8 — state/direction tracking failure

Question: John first drove 3 hours at 60 mph away from home, then turned around. During the 4-hour return window he spent 2 hours stopped, 0.5 hour at 30 mph, and the remaining 1.5 hours at 80 mph. How far was he from home afterward?

The model incorrectly treated the two stationary hours as 120 additional miles and summed outbound and return travel instead of tracking signed distance.

Gold: `45` — Prediction: `435`

### Source index 28 — positional relation failure

Question: Henry made two stops during a 60-mile bike trip. The first was after 20 miles. The second was 15 miles before the end. How many miles were between the stops?

The second stop is at mile 45, so the interval is 25 miles. The model instead subtracted 15 directly from 20.

Gold: `25` — Prediction: `5`

### Source index 94 — relational-language failure

Question: The number of rabbits is twelve less than the combined number of dogs and cats. There are 60 dogs and two cats per dog. How many pets are there in total?

The model interpreted “twelve less than the combined number” as “12 rabbits” instead of `180 - 12 = 168` rabbits.

Gold: `348` — Prediction: `192`

### Source index 12 — generation-length / unresolved-reasoning failure

The model did not emit the required `FINAL_ANSWER` within the 256-token generation limit. This was scored as unparseable and incorrect, not manually repaired.

Gold: `13` — Prediction: unparseable

## Failure observations

The 30 failed items were not hidden or post-corrected. The examples above show recurring weaknesses in this sample: translating relational wording into equations, tracking direction/state across multi-stage word problems, and occasional arithmetic/setup mistakes. One response exhausted the output budget without producing the required final-answer marker.

## Reproducibility artifacts

Files generated for the run:

| File | Purpose | SHA-256 |
|---|---|---|
| `evaluate_gsm8k_100.py` | sampling, inference, extraction and scoring | `1702d64e74421766d9c93a11eafe5de01e1046e1832613fe2e83d40c76c87c29` |
| `results_gsm8k_100_seed20260907.jsonl` | all 100 raw per-item records, outputs, predictions and latency | `7da9e05ccff76844c247ee8b9e5d217b3614bf2e6ba35feae426dbaf760e01dd` |
| `summary_gsm8k_100.json` | exact sample indices and aggregate metrics | `e0414e28f98c5495842732e8e65bca5dcb8f4453ab47db0f0b59a58ccc5e3e9d` |
| `artifact_metadata.json` | model/dataset/runtime provenance | `344f81e1208a16dba8361bcec6a02c6f9891ded914df2997fb903c94416aff1a` |

For verification, re-run the evaluator against the same official GGUF and GSM8K revision with the same seed/settings; the output JSONL preserves each selected source index, question, raw message, gold answer, parsed prediction, correctness flag, latency, and token usage returned by the local server.

## Limitations

- This is a **100-item seeded subset**, not the full 1,319-item GSM8K test split.
- The model is the **Q8_0 GGUF quantization**, not unquantized FP16/BF16 weights.
- This run used a CPU-only llama.cpp backend, so latency is not representative of GPU inference.
- Scoring is exact-match on the normalized numeric final answer; it does not award partial credit for reasoning.
- Thinking mode was disabled and no code interpreter, calculator, Python, SymPy, retrieval, or other external tool was available to the model.
- Therefore this 70.0% figure should not be compared as if it were a full-dataset, unquantized, tool-assisted benchmark.

## Bottom line

On this reproducible 100-item GSM8K subset, Spark-X2.5-1.7B Q8_0 reached **70.0% pass@1** with deterministic no-tool decoding. It handled many direct proportional, percentage, and arithmetic word problems cleanly, while the observed errors were concentrated in multi-stage state tracking, relational-language interpretation, and setup mistakes. All failures remain in the raw output set for inspection.

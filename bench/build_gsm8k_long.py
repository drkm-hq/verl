#!/usr/bin/env python3
"""Build the synthetic long-prompt GSM8K variant (runs inside the Modal container, in verl's uv venv).

For every row of the verl-preprocessed GSM8K parquet (examples/data_preprocess/gsm8k.py schema) a
deterministic filler passage is prepended to the user turn so that the chat-templated prompt is
~TARGET tokens of the given tokenizer; the original question (with verl's instruction suffix) stays
last, so the task, the reward (rule on `#### answer`) and the schema are unchanged. The filler is a
public-domain book (Project Gutenberg #1342) read as a ring of words; row i starts at word
(i * 7919) % n_words so the prefix differs per row (no cross-row vLLM prefix-cache gift) while the
n rollouts of one prompt still share it. Everything is recorded in <dst>/build.json.

    python3 build_gsm8k_long.py --src /hf-cache/data/gsm8k --dst /hf-cache/data/gsm8k-long \
        --tokenizer /hf-cache/hf_home/hub/models--Qwen--Qwen3-1.7B/snapshots/<sha> --target-tokens 6000
"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request

FILLER_URLS = ["https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "https://www.gutenberg.org/files/1342/1342-0.txt"]
STRIDE = 7919  # prime: distinct start offsets for consecutive rows


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_filler(cache_path: str) -> tuple[list[str], dict]:
    """Return the filler as a word list plus provenance; cached on the volume so a rebuild is byte-identical."""
    if os.path.exists(cache_path):
        raw = open(cache_path, encoding="utf-8").read()
        src = "cache"
    else:
        raw, src = None, None
        for url in FILLER_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "verl-contrib-bench/1.0 (dataset author)"})
                raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="replace")
                src = url
                break
            except Exception as exc:  # noqa: BLE001
                print(f"filler fetch failed {url}: {exc}", flush=True)
        if raw is None:  # offline fallback: wikitext-2 (CC BY-SA) via the HF hub, recorded as such
            import datasets
            ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            raw = "\n".join(t for t in ds["text"] if t.strip() and not t.startswith(" ="))
            src = "hf:wikitext/wikitext-2-raw-v1/train"
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(raw)
    body = raw
    m = re.search(r"\*\*\* START OF [^\n]*\*\*\*(.*)\*\*\* END OF", raw, flags=re.S)
    if m:
        body = m.group(1)
    words = body.split()
    return words, {"source": src, "sha256": sha256_file(cache_path), "n_words": len(words), "cache": cache_path}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--target-tokens", type=int, default=6000)
    ap.add_argument("--tolerance", type=int, default=8)
    ap.add_argument("--num-proc", type=int, default=16)
    ap.add_argument("--header", default="Reference material (background reading, not needed to answer the question):")
    a = ap.parse_args()
    import datasets
    from transformers import AutoTokenizer

    os.makedirs(a.dst, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    words, prov = load_filler(os.path.join(a.dst, "filler.txt"))
    n_words = len(words)
    sample_text = " ".join(words[:4000])
    ratio = 4000 / max(1, len(tok(sample_text, add_special_tokens=False)["input_ids"]))  # words per token

    def n_tokens(messages) -> int:
        # same count as verl's RLHFDataset filter (rl_dataset.py: apply_chat_template(..., tokenize=True) ->
        # normalize_token_ids): transformers 5.x returns a BatchEncoding here, older versions a list of ids
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return len(ids)

    def filler(idx: int, n: int) -> str:
        start = (idx * STRIDE) % n_words
        out = words[start:start + n]
        while len(out) < n:
            out += words[: n - len(out)]
        return " ".join(out)

    def process(example, idx):
        msgs = [dict(m) for m in example["prompt"]]
        orig = msgs[-1]["content"]
        base = n_tokens(msgs)
        n = max(1, int((a.target_tokens - base) * ratio))
        for it in range(8):
            msgs[-1]["content"] = f"{a.header}\n\n{filler(idx, n)}\n\n{orig}"
            m = n_tokens(msgs)
            if abs(m - a.target_tokens) <= a.tolerance:
                break
            n = max(1, n + int(round((a.target_tokens - m) * ratio)))
        example["prompt"] = msgs
        example["_n_tokens"] = m  # temporary: stats are computed from the column (map runs in num_proc workers)
        example["_iters"] = it + 1
        return example

    out = {"target_tokens": a.target_tokens, "tolerance": a.tolerance, "tokenizer": a.tokenizer, "header": a.header,
           "stride": STRIDE, "filler": prov, "words_per_token": ratio, "num_proc": a.num_proc, "splits": {}}
    for split in ("train", "test"):
        t0 = time.time()
        ds = datasets.load_dataset("parquet", data_files=os.path.join(a.src, f"{split}.parquet"), split="train")
        ds = ds.map(process, with_indices=True, load_from_cache_file=False, num_proc=a.num_proc)
        toks, iters = list(ds["_n_tokens"]), list(ds["_iters"])
        ds = ds.remove_columns(["_n_tokens", "_iters"])
        dst = os.path.join(a.dst, f"{split}.parquet")
        ds.to_parquet(dst)
        out["splits"][split] = {"rows": len(toks), "iters": sum(iters), "tokens_min": min(toks), "tokens_max": max(toks),
                                "tokens_mean": sum(toks) / max(1, len(toks)),
                                "off_target": sum(1 for t in toks if abs(t - a.target_tokens) > a.tolerance),
                                "over_6144": sum(1 for t in toks if t > 6144), "sha256": sha256_file(dst),
                                "seconds": round(time.time() - t0, 1), "columns": ds.column_names}
        print(f"{split}: {json.dumps(out['splits'][split])}", flush=True)
    ex = datasets.load_dataset("parquet", data_files=os.path.join(a.dst, "test.parquet"), split="train")[0]
    out["example_tail"] = ex["prompt"][-1]["content"][-400:]
    out["example_ground_truth"] = ex["reward_model"]["ground_truth"]
    with open(os.path.join(a.dst, "build.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("BUILD_OK " + json.dumps({k: v for k, v in out.items() if k != "example_tail"}), flush=True)


if __name__ == "__main__":
    main()

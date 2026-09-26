"""Client half of `gpu_ab_overlap_sched.sh`: C concurrent greedy chat requests against a running
server, twice (a warm-up wave, then the measured wave). Prints decode throughput and writes each
request's text so the two server runs (SEED_OVERLAP_SCHED=0/1) can be diffed for identity.

    python3 scratchpad/overlap_probe.py --port 30000 --concurrency 48 --max-tokens 256 --out F
"""

import argparse
import json
import sys
import threading
import time
import urllib.request


def one(url: str, i: int, max_tokens: int, ignore_eos: bool, out: list) -> None:
    body = {
        "messages": [
            {"role": "user", "content": f"Request {i}: write a long story about topic {i}."}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": ignore_eos,
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.loads(r.read())
    out[i] = (resp["choices"][0]["message"]["content"], resp["usage"]["completion_tokens"])


def wave(url: str, n: int, max_tokens: int, ignore_eos: bool) -> tuple[list, float]:
    out: list = [None] * n
    threads = [
        threading.Thread(target=one, args=(url, i, max_tokens, ignore_eos, out)) for i in range(n)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
    wave(url, a.concurrency, 16, True)  # warm-up
    # ignore_eos: every request decodes exactly max_tokens, so the batch stays full (the
    # steady-state b48 decode step the prediction is about).
    full, dt = wave(url, a.concurrency, a.max_tokens, True)
    toks = sum(n for _, n in full)
    # Natural stops (EOS mid-batch, the lookahead-discard path), for the identity diff.
    natural, _ = wave(url, a.concurrency, a.max_tokens, False)
    result = {
        "decode_tok_s": toks / dt,
        "wall_s": dt,
        "tokens": toks,
        "texts_full": [t for t, _ in full],
        "texts_natural": [t for t, _ in natural],
    }
    with open(a.out, "w") as f:
        json.dump(result, f)
    print(f"tok/s={toks / dt:.1f} wall_s={dt:.1f} tokens={toks}", file=sys.stderr)


if __name__ == "__main__":
    main()

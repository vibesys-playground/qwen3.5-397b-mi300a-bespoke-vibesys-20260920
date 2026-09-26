"""Idle-server turn-2 TTFT probe: each turn 2 appends ~N new tokens to a short cached turn 1.

    python3 ttft_probe.py [out.json]   # server on 127.0.0.1:30000
"""
import json
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30000"
WORDS = "alpha river stone cloud market engine yellow garden window silver music number table forest planet signal copper motion winter candle bridge orange paper rocket island shadow".split()


def send(messages, max_tokens=2):
    body = {
        "model": "m",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            p = line[6:]
            if p == "[DONE]":
                break
            o = json.loads(p)
            if o.get("usage"):
                usage = o["usage"]
            ch = o.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content") and ttft is None:
                ttft = (time.perf_counter() - t0) * 1e3
            if ch and ch[0].get("delta", {}).get("content"):
                text.append(ch[0]["delta"]["content"])
    return ttft, usage, "".join(text)


def main():
    out = []
    rng = random.Random(0)
    for n in (128, 256, 400, 512, 768, 1024, 2048):
        for rep in range(3):
            u1 = f"Session {n}-{rep}-{rng.random()}: say ok."
            _, _, r1 = send([{"role": "user", "content": u1}], 2)
            words = " ".join(rng.choice(WORDS) for _ in range(int(n * 0.95)))
            msgs = [
                {"role": "user", "content": u1},
                {"role": "assistant", "content": r1},
                {"role": "user", "content": "Count the words: " + words},
            ]
            ttft, usage, _ = send(msgs, 2)
            new = usage["prompt_tokens"] - (usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            )
            row = {"target": n, "rep": rep, "new_tokens": new, "ttft_ms": round(ttft, 1)}
            print("PROBE", json.dumps(row), flush=True)
            out.append(row)
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"))


main()

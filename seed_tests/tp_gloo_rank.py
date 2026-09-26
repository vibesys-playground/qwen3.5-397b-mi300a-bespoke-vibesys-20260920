"""One rank of the gloo-backed multi-process check. Started by test_tensor_parallel.py.

Not a test: it is the program that test starts once per rank, the same shape `server.py`
uses in production (re-exec this file per rank, rank 0 drives). It goes through the
production entry points, `tp.init`, `tp.plan`, `model.Model` and `tp_driver`, so the check
exercises real `torch.distributed` process-group setup and real collectives rather than a
stand-in for them. The backend is gloo because that one runs on CPU; RCCL is not exercised
here and is not exercised anywhere without the cluster.

    python tp_gloo_rank.py --checkpoint DIR --rank R --world 4 --port P \
        --prompt 3,4,5 --new 6 [--out FILE]

Rank 0 writes {"tokens": [...], "logits": [...]} to `--out`; the other ranks write nothing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tp  # noqa: E402
from model import Model, load_cfg  # noqa: E402
from tp_driver import Broadcaster, Channel, serve_worker  # noqa: E402

MAX_SEQ = 64
PREFILL_CHUNK = 4  # small, so a multi-chunk prefill is exercised on a short prompt


def greedy(runner: Any, prompt: list[int], new: int) -> tuple[list[int], list[float]]:
    """Greedy-decode `new` tokens through the `Runner` surface the scheduler uses.

    This is the scheduler's call sequence (begin, chunked prefill, save_snapshot, then one
    decode step per token) reduced to one sequence in slot 0, so that driving it through
    `Broadcaster` puts every rank through the same calls in the same order.
    """
    runner.begin(0)
    logits = None
    for s in range(0, len(prompt), PREFILL_CHUNK):
        logits = runner.prefill(0, prompt[s : s + PREFILL_CHUNK], s)
    runner.save_snapshot(0, 0)
    token, pos, out = runner.sample_batch(logits, [0.0])[0], len(prompt), []
    out.append(token)
    for _ in range(new - 1):
        logits = runner.decode([0], [token], [pos])
        token = runner.sample_batch(logits, [0.0])[0]
        out.append(token)
        pos += 1
    return out, logits.reshape(-1).tolist()


def build(checkpoint: str, rank: int, world: int, reduce: tp.Reducer | None) -> Model:
    plan = tp.plan(load_cfg(checkpoint), rank, world)
    handle = tp.TP(plan, "cpu", reduce)
    return Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, max_batch=2, tp=handle)


def main(argv: list[str] | None = None) -> None:
    import torch.distributed as dist

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--new", type=int, required=True)
    p.add_argument("--out", default="")
    p.add_argument(
        "--score-start",
        type=int,
        default=None,
        help="If set, also drive Op.SCORE (`/v1/score`'s broadcast op) over the prompt, "
        "scoring positions [score_start, len(prompt)). Regression coverage for the bug where "
        "a direct, unbroadcast Model.forward call hangs every other rank; see server.py's "
        "score_continuation.",
    )
    a = p.parse_args(argv)

    reduce = tp.init(a.rank, a.world, port=a.port, backend="gloo")
    model = build(a.checkpoint, a.rank, a.world, reduce)
    channel = Channel(torch.device("cpu"))
    prompt = [int(t) for t in a.prompt.split(",")]
    try:
        if a.rank:
            serve_worker(model, channel.recv)
        else:
            driver = Broadcaster(model, channel.send)
            tokens, logits = greedy(driver, prompt, a.new)
            result = {"tokens": tokens, "logits": logits}
            if a.score_start is not None:
                driver.begin(0)
                score_logits = driver.score(0, prompt, a.score_start)
                driver.stop()
                result["score_logits"] = score_logits.reshape(-1).tolist()
                result["score_shape"] = list(score_logits.shape)
            else:
                driver.stop()
            Path(a.out).write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

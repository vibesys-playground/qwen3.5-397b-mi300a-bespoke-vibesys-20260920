"""Four gloo ranks serving MTP rounds (`SEED_MTP_SERVE=1`) through `Broadcaster`/`serve_worker`.

Checks, eager and captured (`GraphMTPRunner`, eager capture backend):

- tokens and finish events equal the unsharded plain greedy `generate`, with stops landing
  inside accepted drafts (oracle drafter, see `test_mtp_serve.py`);
- every rank applied the identical op sequence, and every `SPECULATIVE_DECODE` is preceded by
  the `EXTEND_BLOCKS` that reserved its lanes whenever a lane crossed a block boundary;
- every rank ends with identical block tables, and no worker ever allocated a block itself
  (its own allocator is untouched: rank 0's scheduler is the only allocator).

    /tmp/torchenv/bin/python -m pytest seed_tests/test_mtp_serve_tp.py -q -o addopts=
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mtp  # noqa: E402
from test_mtp import prompt_of, write_mtp_shard  # noqa: E402
from test_mtp_serve import Oracle  # noqa: E402
from test_seed_parity import build_hf, tiny_config, write_checkpoint  # noqa: E402
from test_tensor_parallel import TP_AXES, WORLD, free_port  # noqa: E402
from tp_driver import Op  # noqa: E402

K = 2


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tp-mtp")
    cfg = tiny_config(**TP_AXES)
    write_checkpoint(build_hf(cfg=cfg), out, mxfp4=False)
    write_mtp_shard(cfg, out)
    return out


def run_ranks(
    checkpoint: Path, tmp: Path, specs: list, oracle: Oracle, graph: bool, overlap: bool = False
) -> tuple:
    specs_file, oracle_file = tmp / "specs.json", tmp / "oracle.json"
    specs_file.write_text(json.dumps(specs))
    # The oracle's corruption is applied here, once, so every rank reads identical drafts.
    rows = []
    for pos, tok in oracle.table:
        rows.append([pos, tok, oracle.drafts([tok], [pos])[0]])
    oracle_file.write_text(json.dumps(rows))
    script = Path(__file__).resolve().parent / "mtp_gloo_rank.py"
    out = tmp / f"out-{int(graph)}.json"
    base = [sys.executable, str(script), "--checkpoint", str(checkpoint), "--world", str(WORLD)]
    base += ["--port", str(free_port()), "--k", str(K), "--specs", str(specs_file)]
    base += ["--oracle", str(oracle_file), "--graph", str(int(graph))]
    base += ["--overlap", str(int(overlap))]
    procs = [
        subprocess.Popen(  # noqa: S603
            [
                *base,
                "--rank",
                str(r),
                "--trace",
                str(tmp / f"trace-{int(graph)}-{r}.json"),
                *(["--out", str(out)] if r == 0 else []),
            ]
        )
        for r in range(WORLD)
    ]
    try:
        codes = [p.wait(timeout=600) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    assert codes == [0] * WORLD, f"rank exit codes {codes}"
    traces = [json.loads((tmp / f"trace-{int(graph)}-{r}.json").read_text()) for r in range(WORLD)]
    return json.loads(out.read_text()), traces


@pytest.mark.parametrize(("graph", "overlap"), [(False, False), (True, False), (True, True)])
def test_four_gloo_ranks_mtp_matches_plain_greedy(
    checkpoint: Path, tmp_path: Path, graph: bool, overlap: bool
) -> None:
    """`overlap`: `SEED_OVERLAP_SCHED` rounds (`SPECULATIVE_LAUNCH`, mtp_overlap.py)."""
    import model as seed_model
    from tp_gloo_rank import MAX_SEQ

    old = mtp.MTP_ENABLED
    mtp.MTP_ENABLED = False
    try:
        ref = seed_model.Model(checkpoint, ["cpu"], torch.float32, MAX_SEQ, 1)
    finally:
        mtp.MTP_ENABLED = old
    base = [prompt_of(700 + i, 5 + i) for i in range(4)]
    oracle = Oracle(K)
    greedy = []
    for p in base:
        full = list(ref.generate(p, 12 + K, 0.0, frozenset()))
        oracle.add(p + full, len(p) - 1)
        greedy.append(full)
    specs = [
        [base[0], 12, [], 0],
        [base[1], 10, [greedy[1][5]], 0],  # stop somewhere inside a round
        [base[2], 9, [greedy[2][3]], 1],  # waits for a lane (max_batch 2)
        [base[3], 7, [], 3],
    ]
    want = [list(ref.generate(p, n, 0.0, frozenset(s))) for p, n, s, _ in specs]
    got, traces = run_ranks(checkpoint, tmp_path, specs, oracle, graph, overlap)

    for i, ((_, _, stop, _), ref_tokens) in enumerate(zip(specs, want, strict=True)):
        assert got[i][0] == [t for t in ref_tokens if t not in stop], f"req {i} diverged"
        assert got[i][1][0] == "end", got[i][1]

    ops = traces[0]["ops"]
    round_op = Op.SPECULATIVE_LAUNCH if overlap else Op.SPECULATIVE_DECODE
    rounds = ops.count(round_op)
    assert rounds, "no MTP round ran"
    decoded = sum(max(end[1][1] - 1, 0) for _, end in got)  # tokens after each first token
    assert rounds < decoded / 2, "rounds did not advance lanes by more than one token"
    assert any(
        ops[j] == Op.EXTEND_BLOCKS and ops[j + 1] == round_op
        for j in range(len(ops) - 1)
    ), "no round was preceded by its rank-0 block reservation"
    for r in range(1, WORLD):
        assert traces[r]["ops"] == ops, f"rank {r} applied a different op sequence"
        assert traces[r]["tables"] == traces[0]["tables"], f"rank {r} block tables differ"
        assert traces[r]["local_free"] == traces[r]["usable"], f"rank {r} allocated locally"

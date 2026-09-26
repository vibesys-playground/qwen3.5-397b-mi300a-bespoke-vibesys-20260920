# Campaign artifacts

The repository root is the agent's workspace at the end of round 15: the engine (`server.py` and its
modules), `accuracy_checker/`, `benchmark/`, `reference/`, `seed_tests/`, and the agent's probes and notes.
This directory adds what lived outside the workspace.

| File | Contents |
|---|---|
| `launch.sh` | Boots the engine with the final production flags and runs one quick C96 measurement |
| `WRITEUP.md` | Experiment description, trajectory, main optimizations, SGLang comparison |
| `results/bespoke_runs_r15.csv` | Every recorded throughput measurement, 271 rows, including rejected and broken runs |
| `results/trajectory_styled_r15*.png` | Best-so-far goodput trajectory (full and reduced annotations) |
| `results/traj4.py`, `results/traj5.py` | Plot scripts; run from `results/` |

## Final result

C96 goodput 2,242.4 and 2,201.4 tok/s in two same-node pairs (1,905.4 and 1,898.3 without MTP on the same
node), numerics gate passed, zero failed turns. SGLang baseline: 961.013 tok/s.

## Reproducing

- Hardware: one node with 4x AMD MI300A (gfx942), ROCm.
- `launch.sh` needs `MODEL_PATH`, `AITER_JIT_DIR`, and `CACHE_SEED`.
- `CACHE_SEED` is a directory with `hipcache/` (compiled HIP kernels) and `blascache/` (tuned hipBLASLt
  solutions) and is not checked in. Without a seed the engine compiles kernels and tunes BLAS at boot; cold
  BLAS tuning is slow and can hang, so run one boot to populate a cache, then reuse a copy of it.
- Throughput varies by about ±10% across nodes. Compare builds only as same-node back-to-back pairs.
- Correctness gate: `python3 accuracy_checker/checker.py` (see `accuracy_checker/README.md`).

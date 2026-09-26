"""A custom one-shot all-reduce for small (<=1.5 MiB) bf16 row-parallel partials.

`tp.py`'s `all_reduce` goes through RCCL, which at these sizes is latency-bound: ~40-100us a
call regardless of whether the payload is 16 KiB or 2 MiB (see `rccl-bench/size_bf16_results
.jsonl`), because a ring/tree collective pays several kernel launches and inter-rank round
trips no matter how little data moves. On one node with a full XGMI mesh (every GPU pair is
one hop, see `rocm-smi --showtopo`) and peer access already enabled between all four devices,
that per-call cost is mostly the RCCL protocol, not the wire.

This module replaces RCCL for that regime with a *one-shot direct* all-reduce: every rank
publishes its partial into a buffer the other three can read directly over XGMI, then every
rank reads all four buffers (its own included) and reduces locally. One round trip, two
kernel launches, no ring hops.

**Memory.** Each rank `hipMalloc`s one buffer holding two double-buffered payload slots (so a
publish for call `k+2` cannot race a still-in-flight read of call `k`'s slot, see below) plus
one flag `int32` per slot, and exports it with `hipIpcGetMemHandle`. Handles are exchanged
once, at construction, over `torch.distributed.all_gather_object` on the default process
group; every rank then `hipIpcOpenMemHandle`s the other three and keeps the four resulting
device pointers (its own plus three peers') in a small on-device pointer table built once.
Nothing here allocates or opens a handle again after `__init__`, which is what makes every
buffer and pointer a graph-capture-time constant: replay only ever re-issues the two kernels
against addresses fixed at construction.

**Synchronization, and why it survives CUDA graph capture and replay.** There is no barrier
primitive here beyond a spin on a flag; the correctness argument is call-position based:

- Both kernels below run as a single block (so no grid-wide sync is needed to know "every
  thread has finished writing" before the flag is set).
- Publish writes this call's slot (`call_index % 2`), fences system-wide
  (`__threadfence_system`, required for the write to be visible to a *different* GPU, not
  just this one), then stores `call_index + 1` (never 0, which is the buffer's zero-
  initialized rest state) into that slot's flag.
- Gather spins until each peer's flag for this slot equals this call's expected value, then
  reduces in fp32 and writes the bf16 result in place.
- `call_index` is a plain Python counter on this object, incremented once per `all_reduce`
  call and baked into that call's kernel arguments. Under `TP`'s lockstep invariant (every
  rank issues the same sequence of collectives in the same order -- see `tp.py`'s module
  docstring), rank r's k-th call is always concurrent with every other rank's k-th call, so
  the counter is implicitly synchronized across ranks without ever being communicated.
  Because each rank's own two kernels for call k+1 cannot even be *enqueued* on its stream
  until call k's kernels (publish, then the spin-and-read in gather) have completed in
  program order, no rank can reach the point of overwriting a slot with call k+2's data until
  every rank -- including whichever one is slowest -- has already finished reading that slot's
  call-k data. That argument doesn't depend on `call_index` counting from 1 each graph replay
  (it doesn't: replay re-issues the exact values baked in at capture time, so a replayed graph
  emits the same small integers every time) or on eager and captured calls sharing one
  incrementing counter (they do, and the argument doesn't care what the numbers are, only that
  each position's value is unique within a two-position window on that slot, which holds
  because `call_index` increases by exactly 1 every call).

**What this does not do.** No cross-node case (this node's 4 ranks only), no non-power-of-two
world size handling beyond "loop over `world`", no fallback inside the kernel for a hung peer
(see `SPIN_TIMEOUT_S` and the module docstring in `test_custom_allreduce.py` for the watchdog this
relies on at the test level instead). `TP.all_reduce` only ever routes here for a bf16 tensor
within `MAX_BYTES`; anything else (fp32 votes, a payload over the cap) takes the RCCL path
unchanged.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

AR_WIDE = os.environ.get("SEED_AR_WIDE", "0") not in ("0", "", "false", "False")
"""`SEED_AR_WIDE=1` (off by default): 4x larger slots, so prefill and mixed steps up to 1024
rows at hidden 4096 (8 MiB bf16) take the one-shot kernel too. Without it a step over 256 rows
falls back to RCCL with an fp32 upcast (twice the wire bytes plus two cast kernels), measured
as the largest per-step term of a 512-row step (see `PREFILL_COST.md`). Costs 12 MiB more of
IPC buffer per rank."""

MAX_ELEMS = (1 << 22) if AR_WIDE else (1 << 20)
"""Elements per slot (2 MiB of bf16, 8 MiB under `SEED_AR_WIDE`), comfortably above the
1.5 MiB/786,432-elem test case."""

MAX_BYTES = MAX_ELEMS * 2

MAX_FLAG_BLOCKS = 256
"""Per-slot flag count of the graph-safe kernels; mirrors `MAX_FLAG_BLOCKS` in `_HIP_SRC`. The
fused residual-norm kernel launches one block per row, so `rows` must not exceed this."""

GRAPH_SAFE = os.environ.get("SEED_AR_GRAPH_SAFE", "0") not in ("0", "false", "False")
"""`SEED_AR_GRAPH_SAFE=1` (default off): one kernel launch per call with a device-resident call
counter (`oneshot_ar_kernel`), instead of copy + flag + gather launches keyed by a host counter
baked into each launch. Safe under any interleaving of eager calls and graph replays; see the
kernel's comment in `_HIP_SRC`."""

SP = os.environ.get("SEED_AR_SP", "0") not in ("0", "", "false", "False")
"""`SEED_AR_SP=1` (off by default): the mixed step's wide all-reduces run as reduce-scatter +
residual add + RMSNorm + all-gather in one kernel, with the residual stream kept sharded by
rows (`CustomAllReduce.sp_ar_add_rmsnorm`, `rmsnorm_fused._sp_ar_add_rmsnorm_kernel`). Costs
`SP_BYTES` more of IPC buffer per rank."""

SP_MAX_TOKENS = int(os.environ.get("SEED_AR_SP_MAX_TOKENS", "1024"))
"""Total rows (tokens) one SP call may carry, a multiple of 4. The default 1024 leaves the
1536- and 2048-token captured prefill shapes (`SEED_PREFILL_ACCUM`) on the replicated
all-reduce path; 2048 row-shards them too at 64 MiB of IPC buffer per rank instead of 32."""

SP_SLOT_ELEMS = SP_MAX_TOKENS * 4096
"""Per-slot capacity of the reduce-scatter and all-gather receive buffers: `SP_MAX_TOKENS`
rows at hidden 4096 (8 MiB bf16 at 1024). Two slots each: 32 MiB per rank at 1024."""

SP_MAX_ROWS = SP_MAX_TOKENS // 4
"""Rows per rank (`S`) the SP kernel's flag table holds: `SP_MAX_TOKENS` total at world 4."""

SP_WARPS = 4
"""Warps per program of the SP kernel. Must equal `add_rmsnorm`'s and `ar_add_rmsnorm`'s (Triton's
default, 4): `tl.sum`'s reduction order depends on it, and 8 warps measured rare 1-ulp norm
differences (same speed)."""

SP_BYTES = 2 * 2 * SP_SLOT_ELEMS * 2 + 2 * 2 * 4 * SP_MAX_ROWS * 4

PUSH = os.environ.get("SEED_AR_PUSH", "0") not in ("0", "", "false", "False")
"""`SEED_AR_PUSH=1` (off by default): `ar_add_rmsnorm` runs the push variant
(`rmsnorm_fused._push_ar_add_rmsnorm_kernel`): each rank writes its partial rows into the
peers' receive buffers and spins on local flags, instead of publishing locally and reading the
peers' buffers over xGMI. Bit-identical outputs. Costs `PUSH_BYTES` more of IPC buffer."""

PUSH_ALLOC = PUSH

PUSH_SYS_FENCE = True
"""Diagnostic toggle: False drops the system-scope release/acquire on the push kernel's flags
(no L2 writeback/invalidate). Not a production setting until shown safe."""
"""Whether construction allocates the push region (fixed at import, so a diagnostic can
toggle `PUSH` per capture after construction)."""

PUSH_ELEMS = MAX_FLAG_BLOCKS * 4096
"""Per-(slot, source rank) receive capacity: `MAX_FLAG_BLOCKS` rows at hidden 4096."""

PUSH_BYTES = 2 * 4 * PUSH_ELEMS * 2 + 2 * 4 * MAX_FLAG_BLOCKS * 4

_HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#define THREADS 1024

// Bounded spin: a hang is a wrong-answer replay, not a wedged GPU. Originally a fixed spin-
// count (SPIN_LIMIT = 2e8), which on MI300A (gfx942) graph capture could trip early: per-
// bucket Triton compiles skew ranks by 58-77s of *host* time during capture, and that skew
// shows up as extra spin iterations on whichever rank published first, occasionally blowing
// through 2e8 fast spins well before any peer actually hung. wall_clock64() (a monotonic
// counter at a fixed, queryable rate -- hipDeviceAttributeWallClockRate reports it in kHz;
// 100 MHz on gfx942) makes the watchdog measure wall-clock time instead of iteration count,
// so it is immune to that skew: 600s is generous even for a slow capture, but still finite.
#define WALL_CLOCK_HZ 100000000LL
#define SPIN_TIMEOUT_S 600
#define SPIN_TIMEOUT_TICKS (WALL_CLOCK_HZ * SPIN_TIMEOUT_S)

#define VEC 8  // bf16 elements per 16-byte (uint4) transaction
#define MAX_BLOCKS 64

// A single block (THREADS wide) tops out at one CU's worth of bandwidth: the first version
// of this file measured ~280 MB/s (scaling exactly linearly with payload size, so genuinely
// bandwidth-, not latency-, bound), and THREADS=1024 plus vectorized 16-byte transactions
// (below) still only brought 384 KiB and 1.5 MiB to a few hundred us -- fine at 48 KiB, not at
// production's 384 KiB. Spreading the copy and the reduce across many blocks (many CUs) is
// the rest of that fix. Multiple blocks means the flag can no longer be set from inside the
// same launch that does the copy (no block knows when *every* block's writes have landed
// without a grid-wide barrier, which HIP does not give without cooperative-groups plumbing
// this file skips): `publish_copy_kernel` only copies, and a separate, tiny
// `publish_flag_kernel` sets the flag, relying on HIP's own per-stream ordering (kernel N+1
// never starts until every block of kernel N has retired) instead of an in-kernel barrier.
// `gather_kernel` avoids the same problem the other way: every block does its own redundant
// spin-wait on the peer flags before touching any data, which costs nothing once a flag is
// already set (the common case) and is simpler than coordinating one "leader" block across
// blocks that may not even be resident at the same time.

extern "C" __global__ void publish_copy_kernel(
    const __hip_bfloat16* __restrict__ local_in,
    __hip_bfloat16* __restrict__ my_buf,   // [2][MAX_ELEMS]
    int n_elems, int slot, int max_elems)
{
    __hip_bfloat16* dst = my_buf + (size_t)slot * max_elems;
    int n_vec = n_elems / VEC;
    const uint4* src_v = reinterpret_cast<const uint4*>(local_in);
    uint4* dst_v = reinterpret_cast<uint4*>(dst);
    int stride = gridDim.x * THREADS;
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < n_vec; i += stride) dst_v[i] = src_v[i];
    for (int i = n_vec * VEC + blockIdx.x * THREADS + threadIdx.x; i < n_elems; i += stride)
        dst[i] = local_in[i];
}

extern "C" __global__ void publish_flag_kernel(int32_t* __restrict__ my_flag, int slot, int expected)
{
    // Runs after publish_copy_kernel on the same stream, so every one of that kernel's
    // blocks has already retired: this is the barrier, not an in-kernel one.
    __threadfence_system();
    atomicExch(&my_flag[slot], expected);
}

extern "C" __global__ void gather_kernel(
    const int64_t* __restrict__ buf_ptrs,   // [world] device pointers to each rank's buffer
    const int64_t* __restrict__ flag_ptrs,  // [world] device pointers to each rank's flags
    __hip_bfloat16* __restrict__ out,
    int32_t* __restrict__ error_flag,       // [1] set to 1 if any peer's flag never arrived
    int n_elems, int slot, int expected, int world, int self_rank, int max_elems)
{
    if (threadIdx.x == 0) {
        for (int r = 0; r < world; r++) {
            if (r == self_rank) continue;
            volatile int32_t* f = (volatile int32_t*)(flag_ptrs[r]) + slot;
            long long spin_start = wall_clock64();
            while (*f != expected) {
                if (wall_clock64() - spin_start > SPIN_TIMEOUT_TICKS) {
                    // Watchdog: a peer never published within SPIN_TIMEOUT_S wall-clock
                    // seconds. Continuing would reduce whatever garbage sits in its slot, so
                    // trap: the kernel aborts, the next sync on this rank raises a HIP error,
                    // and the rank dies loudly (a dead rank shows on /health). `error_flag` is
                    // set first for `check_errors` (tests, bench).
                    atomicExch(error_flag, 1); __builtin_trap();
                    break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
    int n_vec = n_elems / VEC;
    int stride = gridDim.x * THREADS;
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < n_vec; i += stride) {
        float acc[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int r = 0; r < world; r++) {
            const __hip_bfloat16* buf = (const __hip_bfloat16*)(buf_ptrs[r]) + (size_t)slot * max_elems;
            uint4 v = reinterpret_cast<const uint4*>(buf)[i];
            const __hip_bfloat16* bf = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
            for (int k = 0; k < VEC; k++) acc[k] += __bfloat162float(bf[k]);
        }
        uint4 outv;
        __hip_bfloat16* obf = reinterpret_cast<__hip_bfloat16*>(&outv);
#pragma unroll
        for (int k = 0; k < VEC; k++) obf[k] = __float2bfloat16(acc[k]);
        reinterpret_cast<uint4*>(out)[i] = outv;
    }
    for (int i = n_vec * VEC + blockIdx.x * THREADS + threadIdx.x; i < n_elems; i += stride) {
        float acc = 0.0f;
        for (int r = 0; r < world; r++) {
            const __hip_bfloat16* buf = (const __hip_bfloat16*)(buf_ptrs[r]) + (size_t)slot * max_elems;
            acc += __bfloat162float(buf[i]);
        }
        out[i] = __float2bfloat16(acc);
    }
}

static int nblocks_for(int n_elems) {
    int n_vec = n_elems / VEC;
    int need = (n_vec + THREADS - 1) / THREADS;
    return need < 1 ? 1 : (need > MAX_BLOCKS ? MAX_BLOCKS : need);
}

extern "C" void launch_publish(
    int64_t local_in, int64_t my_buf, int64_t my_flag,
    int n_elems, int slot, int expected, int max_elems, int64_t stream)
{
    // also reused by TwoShotAllReduce's phase 0
    int blocks = nblocks_for(n_elems);
    hipLaunchKernelGGL(publish_copy_kernel, dim3(blocks), dim3(THREADS), 0, (hipStream_t)stream,
        (const __hip_bfloat16*)local_in, (__hip_bfloat16*)my_buf, n_elems, slot, max_elems);
    hipLaunchKernelGGL(publish_flag_kernel, dim3(1), dim3(1), 0, (hipStream_t)stream,
        (int32_t*)my_flag, slot, expected);
}

extern "C" void launch_gather(
    int64_t buf_ptrs, int64_t flag_ptrs, int64_t out, int64_t error_flag,
    int n_elems, int slot, int expected, int world, int self_rank, int max_elems, int64_t stream)
{
    int blocks = nblocks_for(n_elems);
    hipLaunchKernelGGL(gather_kernel, dim3(blocks), dim3(THREADS), 0, (hipStream_t)stream,
        (const int64_t*)buf_ptrs, (const int64_t*)flag_ptrs, (__hip_bfloat16*)out,
        (int32_t*)error_flag, n_elems, slot, expected, world, self_rank, max_elems);
}

// -- fused gather + residual add + RMSNorm: see CustomAllReduce.residual_norm. --
//
// Idea: TensorRT-LLM's and vLLM's custom-all-reduce kernels both fold the residual add and
// the following RMSNorm into the reduction kernel itself ("all-reduce + residual + norm"
// fusion), instead of writing the reduced partial to HBM and having two more kernels read it
// back -- cited for the shape of the fusion; nothing here is their code. `Model.decode_layer`/
// `graph_decode.decode_layer_static` do exactly "all_reduce, then `x = residual + reduced`,
// then `rmsnorm(x, w, eps)`" twice a layer (once after the mixer, once after MoE), so this
// kernel does all three steps' work in the same single read of every rank's buffer the plain
// `gather_kernel` already does, and never round-trips the reduced-but-unresidualed partial or
// the residualed-but-unnormalized `x` through HBM to get there.
//
// One block per row (token): `hidden` is the RMSNorm reduction width, and a block-wide
// reduction of `hidden` values needs every value live in one block, which the plain kernel's
// grid-stride-over-the-whole-flat-tensor layout does not guarantee (a block there may span a
// row boundary). `THREADS_NORM` divides `hidden` (checked by the Python wrapper -- callers use
// the model's own `hidden`, always a multiple of `VEC` in practice) so every thread owns a
// whole `VEC`-wide chunk with no tail case.
//
// Two passes over each row's `VEC`-chunks, both by the same threads at the same offsets, never
// re-reading anything another block wrote: pass 1 reduces this call's `world` partials for a
// chunk, rounds to bf16 (matching what `gather_kernel` would have produced, so the *residual
// add* below reproduces `all_reduce`'s output bit for bit -- see the Python wrapper's
// docstring for why that round-trip through bf16 is deliberate, not dropped for one fewer
// rounding), adds the bf16 residual in fp32, and stores the (still just single-rounded, same
// as the unfused `x = residual + reduced` today) result as `new_residual`, while accumulating
// each element's square into this thread's partial sum. A block-wide tree reduction over
// shared memory turns those `THREADS_NORM` partials into the row's total sum of squares.
// Pass 2 re-reads `new_residual` (now sitting in this block's own recent writes, near-free)
// and produces the normalized output with `model.rmsnorm`'s exact fp32 chain and single
// rounding at the store.
#define THREADS_NORM 512

extern "C" __global__ void gather_residual_norm_kernel(
    const int64_t* __restrict__ buf_ptrs, const int64_t* __restrict__ flag_ptrs,
    const __hip_bfloat16* __restrict__ residual, const float* __restrict__ norm_w,
    __hip_bfloat16* __restrict__ new_residual, __hip_bfloat16* __restrict__ normed_out,
    int32_t* __restrict__ error_flag,
    int hidden, int slot, int expected, int world, int self_rank, int max_elems, float eps)
{
    int row = blockIdx.x;
    if (threadIdx.x == 0) {
        for (int r = 0; r < world; r++) {
            if (r == self_rank) continue;
            volatile int32_t* f = (volatile int32_t*)(flag_ptrs[r]) + slot;
            long long spin_start = wall_clock64();
            while (*f != expected) {
                // Same watchdog as gather_kernel: flag for check_errors, then trap.
                if (wall_clock64() - spin_start > SPIN_TIMEOUT_TICKS) {
                    atomicExch(error_flag, 1); __builtin_trap(); break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();

    __shared__ float sh[THREADS_NORM];
    int row_off = row * hidden;
    int n_vec = hidden / VEC;
    float local_sumsq = 0.0f;

    for (int vi = threadIdx.x; vi < n_vec; vi += THREADS_NORM) {
        float acc[VEC];
#pragma unroll
        for (int kk = 0; kk < VEC; kk++) acc[kk] = 0.0f;
        for (int r = 0; r < world; r++) {
            const __hip_bfloat16* buf = (const __hip_bfloat16*)(buf_ptrs[r]) + (size_t)slot * max_elems;
            uint4 v = reinterpret_cast<const uint4*>(buf + row_off)[vi];
            const __hip_bfloat16* bf = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
            for (int kk = 0; kk < VEC; kk++) acc[kk] += __bfloat162float(bf[kk]);
        }
        uint4 rv = reinterpret_cast<const uint4*>(residual + row_off)[vi];
        const __hip_bfloat16* rbf = reinterpret_cast<const __hip_bfloat16*>(&rv);

        uint4 outv;
        __hip_bfloat16* obf = reinterpret_cast<__hip_bfloat16*>(&outv);
#pragma unroll
        for (int kk = 0; kk < VEC; kk++) {
            // Round the reduced partial to bf16 first, exactly like `all_reduce`'s own
            // `gather_kernel` store does, *then* add the (already-bf16) residual in fp32:
            // this reproduces the unfused path's double rounding (round after reduce, round
            // again at the `x = residual + reduced` store) bit for bit, not a new, single-
            // rounded (more precise, but different) value. See the module docstring.
            float reduced = __bfloat162float(__float2bfloat16(acc[kk]));
            float xval = reduced + __bfloat162float(rbf[kk]);
            local_sumsq += xval * xval;
            obf[kk] = __float2bfloat16(xval);
        }
        reinterpret_cast<uint4*>(new_residual + row_off)[vi] = outv;
    }

    sh[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int stride = THREADS_NORM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    float rms = rsqrtf(sh[0] / (float)hidden + eps);

    for (int vi = threadIdx.x; vi < n_vec; vi += THREADS_NORM) {
        uint4 nv = reinterpret_cast<const uint4*>(new_residual + row_off)[vi];
        const __hip_bfloat16* nbf = reinterpret_cast<const __hip_bfloat16*>(&nv);
        uint4 outv2;
        __hip_bfloat16* obf2 = reinterpret_cast<__hip_bfloat16*>(&outv2);
#pragma unroll
        for (int kk = 0; kk < VEC; kk++) {
            float wv = norm_w[vi * VEC + kk];
            float y = __bfloat162float(nbf[kk]) * rms;
            obf2[kk] = __float2bfloat16(y * (1.0f + wv));
        }
        reinterpret_cast<uint4*>(normed_out + row_off)[vi] = outv2;
    }
}

extern "C" void launch_gather_residual_norm(
    int64_t buf_ptrs, int64_t flag_ptrs, int64_t residual, int64_t norm_w,
    int64_t new_residual, int64_t normed_out, int64_t error_flag,
    int rows, int hidden, int slot, int expected, int world, int self_rank, int max_elems,
    float eps, int64_t stream)
{
    hipLaunchKernelGGL(gather_residual_norm_kernel, dim3(rows), dim3(THREADS_NORM), 0, (hipStream_t)stream,
        (const int64_t*)buf_ptrs, (const int64_t*)flag_ptrs,
        (const __hip_bfloat16*)residual, (const float*)norm_w,
        (__hip_bfloat16*)new_residual, (__hip_bfloat16*)normed_out, (int32_t*)error_flag,
        hidden, slot, expected, world, self_rank, max_elems, eps);
}

// -- graph-safe single-launch one-shot (`SEED_AR_GRAPH_SAFE=1`): see `CustomAllReduce`. --
//
// The kernels above take `slot`/`expected` as launch arguments computed from a host counter,
// so a captured graph replays the values baked in at capture. Two hazards follow once graphs
// (one per bucket, prefill graphs, MTP graphs) and eager calls interleave in arbitrary order:
// (1) slot parity: the call before a replay and the replay's first call can land on the same
// slot, so a rank overwrites a slot a slow peer may still be reading; (2) the flag protocol
// assumes a value unique within a two-call window per slot, which replays repeat. Here the
// call counter lives on the device (`ctr[0]`, per rank, never read by the host): every block
// reads it at kernel start, and the last block to finish (`ctr[1]` counts retired blocks)
// advances it, so the next kernel on the stream sees the next value whether it was issued
// eagerly or by a replay. Idea: device-resident barrier counters in the one-shot all-reduce
// designs of TensorRT-LLM / vLLM custom all-reduce (cited for the idea; no code copied).
//
// One launch per call instead of three (copy, flag, gather): block b publishes only its own
// partition of the payload, sets its own flag `flags[slot][b]`, then waits for every peer's
// block b flag and reduces that same partition. A block only ever reads data its peers'
// same-index blocks wrote, so no grid-wide barrier is needed. Every rank launches the same
// grid for the same call (grid size is a function of the payload size only), so the
// partitions agree across ranks. Slot reuse is safe for the same reason as the host-counter
// version: rank r's call k runs only after its call k-1 finished, which required every peer
// to have started call k-1, so every peer has finished reading call k-2's slot.
#define AR_THREADS 512
#define AR_MAX_BLOCKS 64
#define MAX_FLAG_BLOCKS 256  // flags per slot; >= AR_MAX_BLOCKS and >= rows of the norm kernel

__device__ __forceinline__ void ar_signal_and_wait(
    const int64_t* __restrict__ flag_ptrs, int32_t* __restrict__ error_flag,
    int slot, int value, int world, int self_rank)
{
    // Caller: every thread has stored its partition to this rank's slot. The barrier waits
    // for those stores to reach L2; thread 0's system-scope release (an L2 writeback on
    // gfx942) then publishes the whole block's stores before the flag. One writeback per
    // block, not per wave: a per-thread fence measured ~40% slower at 48 rows.
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        int32_t* mine = (int32_t*)(flag_ptrs[self_rank]) + slot * MAX_FLAG_BLOCKS + blockIdx.x;
        __hip_atomic_store(mine, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        for (int r = 0; r < world; r++) {
            if (r == self_rank) continue;
            int32_t* f = (int32_t*)(flag_ptrs[r]) + slot * MAX_FLAG_BLOCKS + blockIdx.x;
            long long spin_start = wall_clock64();
            while (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != value) {
                if (wall_clock64() - spin_start > SPIN_TIMEOUT_TICKS) {
                    atomicExch(error_flag, 1); __builtin_trap(); break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __threadfence_system();
    }
    __syncthreads();
}

__device__ __forceinline__ void ar_retire(uint32_t* __restrict__ ctr, uint32_t e)
{
    // Last block of this launch advances the per-rank call counter for the next launch.
    // Every block read ctr[0] == e at start; stream order makes the store visible to the
    // next kernel, eager or replayed.
    if (threadIdx.x == 0) {
        uint32_t done = atomicAdd(&ctr[1], 1u);
        if (done == gridDim.x - 1) {
            ctr[1] = 0u;
            ctr[0] = e + 1u;
        }
    }
}

extern "C" __global__ void oneshot_ar_kernel(
    const int64_t* __restrict__ buf_ptrs, const int64_t* __restrict__ flag_ptrs,
    const __hip_bfloat16* in, __hip_bfloat16* out,  // may alias: each block owns its partition
    uint32_t* __restrict__ ctr, int32_t* __restrict__ error_flag,
    int n_elems, int world, int self_rank, int max_elems)
{
    __shared__ uint32_t s_e;
    if (threadIdx.x == 0) s_e = __hip_atomic_load(&ctr[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    uint32_t e = s_e;
    int slot = (int)(e & 1u);
    int value = (int)(e + 1u);  // never 0 (the zeroed rest state) until 2^32 calls wrap
    int n_vec = n_elems / VEC;
    int stride = gridDim.x * AR_THREADS;
    int t0 = blockIdx.x * AR_THREADS + threadIdx.x;

    __hip_bfloat16* mine = (__hip_bfloat16*)(buf_ptrs[self_rank]) + (size_t)slot * max_elems;
    const uint4* src_v = reinterpret_cast<const uint4*>(in);
    uint4* dst_v = reinterpret_cast<uint4*>(mine);
    for (int i = t0; i < n_vec; i += stride) dst_v[i] = src_v[i];
    for (int i = n_vec * VEC + t0; i < n_elems; i += stride) mine[i] = in[i];

    ar_signal_and_wait(flag_ptrs, error_flag, slot, value, world, self_rank);

    for (int i = t0; i < n_vec; i += stride) {
        float acc[VEC] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int r = 0; r < world; r++) {
            const __hip_bfloat16* buf = (const __hip_bfloat16*)(buf_ptrs[r]) + (size_t)slot * max_elems;
            uint4 v = reinterpret_cast<const uint4*>(buf)[i];
            const __hip_bfloat16* bf = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
            for (int k = 0; k < VEC; k++) acc[k] += __bfloat162float(bf[k]);
        }
        uint4 outv;
        __hip_bfloat16* obf = reinterpret_cast<__hip_bfloat16*>(&outv);
#pragma unroll
        for (int k = 0; k < VEC; k++) obf[k] = __float2bfloat16(acc[k]);
        reinterpret_cast<uint4*>(out)[i] = outv;
    }
    for (int i = n_vec * VEC + t0; i < n_elems; i += stride) {
        float acc = 0.0f;
        for (int r = 0; r < world; r++) {
            const __hip_bfloat16* buf = (const __hip_bfloat16*)(buf_ptrs[r]) + (size_t)slot * max_elems;
            acc += __bfloat162float(buf[i]);
        }
        out[i] = __float2bfloat16(acc);
    }
    ar_retire(ctr, e);
}

// Same as gather_residual_norm_kernel (same numerics, same reduction order), but one launch:
// block `row` publishes its own row of `mixer` first, then waits on the peers' row flags.
extern "C" __global__ void oneshot_residual_norm_kernel(
    const int64_t* __restrict__ buf_ptrs, const int64_t* __restrict__ flag_ptrs,
    const __hip_bfloat16* __restrict__ mixer,
    const __hip_bfloat16* __restrict__ residual, const float* __restrict__ norm_w,
    __hip_bfloat16* __restrict__ new_residual, __hip_bfloat16* __restrict__ normed_out,
    uint32_t* __restrict__ ctr, int32_t* __restrict__ error_flag,
    int hidden, int world, int self_rank, int max_elems, float eps)
{
    __shared__ uint32_t s_e;
    __shared__ float sh[THREADS_NORM];
    if (threadIdx.x == 0) s_e = __hip_atomic_load(&ctr[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    uint32_t e = s_e;
    int slot = (int)(e & 1u);
    int value = (int)(e + 1u);
    int row = blockIdx.x;
    int row_off = row * hidden;
    int n_vec = hidden / VEC;

    __hip_bfloat16* mine = (__hip_bfloat16*)(buf_ptrs[self_rank]) + (size_t)slot * max_elems;
    for (int vi = threadIdx.x; vi < n_vec; vi += THREADS_NORM)
        reinterpret_cast<uint4*>(mine + row_off)[vi] = reinterpret_cast<const uint4*>(mixer + row_off)[vi];

    ar_signal_and_wait(flag_ptrs, error_flag, slot, value, world, self_rank);

    float local_sumsq = 0.0f;
    for (int vi = threadIdx.x; vi < n_vec; vi += THREADS_NORM) {
        float acc[VEC];
#pragma unroll
        for (int kk = 0; kk < VEC; kk++) acc[kk] = 0.0f;
        for (int r = 0; r < world; r++) {
            const __hip_bfloat16* buf = (const __hip_bfloat16*)(buf_ptrs[r]) + (size_t)slot * max_elems;
            uint4 v = reinterpret_cast<const uint4*>(buf + row_off)[vi];
            const __hip_bfloat16* bf = reinterpret_cast<const __hip_bfloat16*>(&v);
#pragma unroll
            for (int kk = 0; kk < VEC; kk++) acc[kk] += __bfloat162float(bf[kk]);
        }
        uint4 rv = reinterpret_cast<const uint4*>(residual + row_off)[vi];
        const __hip_bfloat16* rbf = reinterpret_cast<const __hip_bfloat16*>(&rv);
        uint4 outv;
        __hip_bfloat16* obf = reinterpret_cast<__hip_bfloat16*>(&outv);
#pragma unroll
        for (int kk = 0; kk < VEC; kk++) {
            float reduced = __bfloat162float(__float2bfloat16(acc[kk]));
            float xval = reduced + __bfloat162float(rbf[kk]);
            obf[kk] = __float2bfloat16(xval);
            // Sum squares of the *stored* (bf16-rounded) residual, which is what the unfused
            // path's rmsnorm reads back.
            float xr = __bfloat162float(obf[kk]);
            local_sumsq += xr * xr;
        }
        reinterpret_cast<uint4*>(new_residual + row_off)[vi] = outv;
    }
    sh[threadIdx.x] = local_sumsq;
    __syncthreads();
    for (int s = THREADS_NORM / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sh[0] / (float)hidden + eps);
    for (int vi = threadIdx.x; vi < n_vec; vi += THREADS_NORM) {
        uint4 nv = reinterpret_cast<const uint4*>(new_residual + row_off)[vi];
        const __hip_bfloat16* nbf = reinterpret_cast<const __hip_bfloat16*>(&nv);
        uint4 outv2;
        __hip_bfloat16* obf2 = reinterpret_cast<__hip_bfloat16*>(&outv2);
#pragma unroll
        for (int kk = 0; kk < VEC; kk++) {
            float wv = norm_w[vi * VEC + kk];
            float y = __bfloat162float(nbf[kk]) * rms;
            obf2[kk] = __float2bfloat16(y * (1.0f + wv));
        }
        reinterpret_cast<uint4*>(normed_out + row_off)[vi] = outv2;
    }
    ar_retire(ctr, e);
}

extern "C" void launch_oneshot(
    int64_t buf_ptrs, int64_t flag_ptrs, int64_t in, int64_t out, int64_t ctr, int64_t error_flag,
    int n_elems, int world, int self_rank, int max_elems, int64_t stream)
{
    int n_vec = n_elems / VEC;
    int need = (n_vec + AR_THREADS - 1) / AR_THREADS;
    int blocks = need < 1 ? 1 : (need > AR_MAX_BLOCKS ? AR_MAX_BLOCKS : need);
    hipLaunchKernelGGL(oneshot_ar_kernel, dim3(blocks), dim3(AR_THREADS), 0, (hipStream_t)stream,
        (const int64_t*)buf_ptrs, (const int64_t*)flag_ptrs, (const __hip_bfloat16*)in,
        (__hip_bfloat16*)out, (uint32_t*)ctr, (int32_t*)error_flag, n_elems, world, self_rank,
        max_elems);
}

extern "C" void launch_oneshot_residual_norm(
    int64_t buf_ptrs, int64_t flag_ptrs, int64_t mixer, int64_t residual, int64_t norm_w,
    int64_t new_residual, int64_t normed_out, int64_t ctr, int64_t error_flag,
    int rows, int hidden, int world, int self_rank, int max_elems, float eps, int64_t stream)
{
    hipLaunchKernelGGL(oneshot_residual_norm_kernel, dim3(rows), dim3(THREADS_NORM), 0, (hipStream_t)stream,
        (const int64_t*)buf_ptrs, (const int64_t*)flag_ptrs, (const __hip_bfloat16*)mixer,
        (const __hip_bfloat16*)residual, (const float*)norm_w, (__hip_bfloat16*)new_residual,
        (__hip_bfloat16*)normed_out, (uint32_t*)ctr, (int32_t*)error_flag, hidden, world,
        self_rank, max_elems, eps);
}

// -- two-shot (reduce-scatter + all-gather), comparison-only: see TwoShotAllReduce below. --

extern "C" __global__ void reduce_scatter_kernel(
    const int64_t* __restrict__ buf_ptrs, const int64_t* __restrict__ flag_ptrs,
    __hip_bfloat16* __restrict__ my_shard, int32_t* __restrict__ my_shard_flag,
    int32_t* __restrict__ error_flag,
    int shard_elems, int slot, int expected, int world, int self_rank, int max_elems,
    int max_shard_elems)
{
    if (threadIdx.x == 0) {
        for (int r = 0; r < world; r++) {
            if (r == self_rank) continue;
            volatile int32_t* f = (volatile int32_t*)(flag_ptrs[r]) + slot;
            long long spin_start = wall_clock64();
            while (*f != expected) {
                if (wall_clock64() - spin_start > SPIN_TIMEOUT_TICKS) {
                    atomicExch(error_flag, 1); __builtin_trap(); break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
    int base = self_rank * shard_elems;
    // Stride by the *allocated* per-slot shard capacity, not the (possibly smaller) runtime
    // `shard_elems`: using the runtime count here made slot 1 alias slot 0's memory whenever
    // the caller's tensor was smaller than `max_elems` -- caught by `smoke_twoshot.py`, which
    // failed on exactly every odd (slot=1) call.
    __hip_bfloat16* dst = my_shard + (size_t)slot * max_shard_elems;
    for (int i = threadIdx.x; i < shard_elems; i += THREADS) {
        float acc = 0.0f;
        for (int r = 0; r < world; r++) {
            const __hip_bfloat16* buf = (const __hip_bfloat16*)(buf_ptrs[r]) + (size_t)slot * max_elems;
            acc += __bfloat162float(buf[base + i]);
        }
        dst[i] = __float2bfloat16(acc);
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) { __threadfence_system(); atomicExch(&my_shard_flag[slot], expected); }
}

extern "C" __global__ void allgather_kernel(
    const int64_t* __restrict__ shard_ptrs, const int64_t* __restrict__ shard_flag_ptrs,
    __hip_bfloat16* __restrict__ out, int32_t* __restrict__ error_flag,
    int shard_elems, int slot, int expected, int world,
    int self_rank, int max_shard_elems)
{
    if (threadIdx.x == 0) {
        for (int r = 0; r < world; r++) {
            if (r == self_rank) continue;
            volatile int32_t* f = (volatile int32_t*)(shard_flag_ptrs[r]) + slot;
            long long spin_start = wall_clock64();
            while (*f != expected) {
                if (wall_clock64() - spin_start > SPIN_TIMEOUT_TICKS) {
                    atomicExch(error_flag, 1); __builtin_trap(); break;
                }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
    for (int r = 0; r < world; r++) {
        const __hip_bfloat16* src = (const __hip_bfloat16*)(shard_ptrs[r]) + (size_t)slot * max_shard_elems;
        __hip_bfloat16* dst = out + r * shard_elems;
        for (int i = threadIdx.x; i < shard_elems; i += THREADS) dst[i] = src[i];
    }
}

extern "C" void launch_reduce_scatter(
    int64_t buf_ptrs, int64_t flag_ptrs, int64_t my_shard, int64_t my_shard_flag,
    int64_t error_flag,
    int shard_elems, int slot, int expected, int world, int self_rank, int max_elems,
    int max_shard_elems, int64_t stream)
{
    hipLaunchKernelGGL(reduce_scatter_kernel, dim3(1), dim3(THREADS), 0, (hipStream_t)stream,
        (const int64_t*)buf_ptrs, (const int64_t*)flag_ptrs, (__hip_bfloat16*)my_shard,
        (int32_t*)my_shard_flag, (int32_t*)error_flag, shard_elems, slot, expected, world,
        self_rank, max_elems, max_shard_elems);
}

extern "C" void launch_allgather(
    int64_t shard_ptrs, int64_t shard_flag_ptrs, int64_t out, int64_t error_flag,
    int shard_elems, int slot, int expected, int world, int self_rank, int max_shard_elems, int64_t stream)
{
    hipLaunchKernelGGL(allgather_kernel, dim3(1), dim3(THREADS), 0, (hipStream_t)stream,
        (const int64_t*)shard_ptrs, (const int64_t*)shard_flag_ptrs, (__hip_bfloat16*)out,
        (int32_t*)error_flag, shard_elems, slot, expected, world, self_rank, max_shard_elems);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <cstring>

extern "C" void launch_publish(int64_t, int64_t, int64_t, int, int, int, int, int64_t);
extern "C" void launch_gather(int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, int, int64_t);
extern "C" void launch_gather_residual_norm(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, int, float, int64_t);
extern "C" void launch_reduce_scatter(int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, int, int, int64_t);
extern "C" void launch_allgather(int64_t, int64_t, int64_t, int64_t, int, int, int, int, int, int, int64_t);
extern "C" void launch_oneshot(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, int, int, int, int64_t);
extern "C" void launch_oneshot_residual_norm(
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, float, int64_t);

static void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

// Allocates `nbytes`, returns (ptr as int64, 64-byte IPC handle as a python bytes object).
//
// Fine-grained, not `hipMalloc`'s default coarse-grained: coarse-grained device memory on
// MI300 only guarantees another agent (a peer GPU here) sees a write after some heavier
// synchronization than `__threadfence_system` actually provides for it, which is what the
// first version of this measured as ~170-5400us a call (linear in payload size -- consistent
// with the *fence*, not the copy, being the cost) instead of the low tens of microseconds a
// direct XGMI read should cost. Fine-grained memory is the AMD-documented way to get prompt
// cross-agent visibility for exactly this flag-and-payload signaling pattern.
std::pair<int64_t, py::bytes> alloc_ipc_buffer(int64_t nbytes) {
    void* p = nullptr;
    check(hipExtMallocWithFlags(&p, (size_t)nbytes, hipDeviceMallocFinegrained), "hipExtMallocWithFlags");
    check(hipMemset(p, 0, (size_t)nbytes), "hipMemset");
    hipIpcMemHandle_t handle;
    check(hipIpcGetMemHandle(&handle, p), "hipIpcGetMemHandle");
    return {(int64_t)p, py::bytes((const char*)&handle, sizeof(handle))};
}

// Opens a peer's handle, returns its pointer as int64 in this process's address space.
int64_t open_ipc_handle(py::bytes handle_bytes) {
    std::string s = handle_bytes;
    TORCH_CHECK(s.size() == sizeof(hipIpcMemHandle_t), "bad IPC handle size");
    hipIpcMemHandle_t handle;
    std::memcpy(&handle, s.data(), sizeof(handle));
    void* p = nullptr;
    check(hipIpcOpenMemHandle(&p, handle, hipIpcMemLazyEnablePeerAccess), "hipIpcOpenMemHandle");
    return (int64_t)p;
}

// Uploads a small int64 pointer table (world entries) to a freshly hipMalloc'd device array.
int64_t upload_ptr_table(std::vector<int64_t> values) {
    void* p = nullptr;
    size_t nbytes = values.size() * sizeof(int64_t);
    check(hipMalloc(&p, nbytes), "hipMalloc(ptr table)");
    check(hipMemcpy(p, values.data(), nbytes, hipMemcpyHostToDevice), "hipMemcpy(ptr table)");
    return (int64_t)p;
}

void publish(int64_t local_in, int64_t my_buf, int64_t my_flag,
             int64_t n_elems, int64_t slot, int64_t expected, int64_t max_elems, int64_t stream) {
    launch_publish(local_in, my_buf, my_flag, (int)n_elems, (int)slot, (int)expected, (int)max_elems, stream);
}

void gather(int64_t buf_ptrs, int64_t flag_ptrs, int64_t out, int64_t error_flag,
            int64_t n_elems, int64_t slot, int64_t expected, int64_t world, int64_t self_rank,
            int64_t max_elems, int64_t stream) {
    launch_gather(buf_ptrs, flag_ptrs, out, error_flag, (int)n_elems, (int)slot, (int)expected,
                  (int)world, (int)self_rank, (int)max_elems, stream);
}

void gather_residual_norm(
    int64_t buf_ptrs, int64_t flag_ptrs, int64_t residual, int64_t norm_w,
    int64_t new_residual, int64_t normed_out, int64_t error_flag,
    int64_t rows, int64_t hidden, int64_t slot, int64_t expected, int64_t world, int64_t self_rank,
    int64_t max_elems, double eps, int64_t stream) {
    launch_gather_residual_norm(
        buf_ptrs, flag_ptrs, residual, norm_w, new_residual, normed_out, error_flag,
        (int)rows, (int)hidden, (int)slot, (int)expected, (int)world, (int)self_rank,
        (int)max_elems, (float)eps, stream);
}

void reduce_scatter(int64_t buf_ptrs, int64_t flag_ptrs, int64_t my_shard, int64_t my_shard_flag,
                     int64_t error_flag,
                     int64_t shard_elems, int64_t slot, int64_t expected, int64_t world,
                     int64_t self_rank, int64_t max_elems, int64_t max_shard_elems, int64_t stream) {
    launch_reduce_scatter(buf_ptrs, flag_ptrs, my_shard, my_shard_flag, error_flag,
                          (int)shard_elems, (int)slot, (int)expected, (int)world,
                          (int)self_rank, (int)max_elems, (int)max_shard_elems, stream);
}

void allgather(int64_t shard_ptrs, int64_t shard_flag_ptrs, int64_t out, int64_t error_flag,
               int64_t shard_elems, int64_t slot, int64_t expected, int64_t world,
               int64_t self_rank, int64_t max_shard_elems, int64_t stream) {
    launch_allgather(shard_ptrs, shard_flag_ptrs, out, error_flag, (int)shard_elems, (int)slot,
                     (int)expected, (int)world, (int)self_rank, (int)max_shard_elems, stream);
}

void oneshot(int64_t buf_ptrs, int64_t flag_ptrs, int64_t in, int64_t out, int64_t ctr,
             int64_t error_flag, int64_t n_elems, int64_t world, int64_t self_rank,
             int64_t max_elems, int64_t stream) {
    launch_oneshot(buf_ptrs, flag_ptrs, in, out, ctr, error_flag, (int)n_elems, (int)world,
                   (int)self_rank, (int)max_elems, stream);
}

void oneshot_residual_norm(
    int64_t buf_ptrs, int64_t flag_ptrs, int64_t mixer, int64_t residual, int64_t norm_w,
    int64_t new_residual, int64_t normed_out, int64_t ctr, int64_t error_flag,
    int64_t rows, int64_t hidden, int64_t world, int64_t self_rank, int64_t max_elems,
    double eps, int64_t stream) {
    launch_oneshot_residual_norm(buf_ptrs, flag_ptrs, mixer, residual, norm_w, new_residual,
                                 normed_out, ctr, error_flag, (int)rows, (int)hidden, (int)world,
                                 (int)self_rank, (int)max_elems, (float)eps, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("oneshot", &oneshot);
    m.def("oneshot_residual_norm", &oneshot_residual_norm);
    m.def("alloc_ipc_buffer", &alloc_ipc_buffer);
    m.def("open_ipc_handle", &open_ipc_handle);
    m.def("upload_ptr_table", &upload_ptr_table);
    m.def("publish", &publish);
    m.def("gather", &gather);
    m.def("gather_residual_norm", &gather_residual_norm);
    m.def("reduce_scatter", &reduce_scatter);
    m.def("allgather", &allgather);
}
"""

_ext = None


def rmsnorm_raw(ptr: int, dtype: torch.dtype) -> object:
    """`rmsnorm_fused.RawPtr` (a Triton pointer argument over an IPC address), imported late so
    the RCCL-only path never imports Triton."""
    import rmsnorm_fused  # noqa: PLC0415

    return rmsnorm_fused.RawPtr(ptr, dtype)


def _build_dir(rank: int) -> str:
    """A fresh, unused directory for this rank's extension sources.

    All `world` ranks used to write `allreduce_ext.cpp`/`.hip` into one shared
    `$TMPDIR/custom_allreduce_src`: every rank runs this module in its own process (see
    `tp.py`'s "Ranks and devices" convention) but they share a node and therefore a `/tmp`, so
    concurrent `__init__` calls raced to create, write, and compile the same two files --
    whichever rank's writes lost the race could hand `load()` a source file mid-truncation.
    `tempfile.mkdtemp` hands back a directory nothing else has (its own random suffix), so two
    ranks -- or two attempts by the same rank after a failed build -- never share files even if
    the writes overlap. `rank` and the pid go in the prefix purely so a leftover directory
    under `/tmp` after a failed build is easy to attribute to who left it.
    """
    return tempfile.mkdtemp(prefix=f"custom_allreduce_src_r{rank}_p{os.getpid()}_")


def _load(rank: int) -> object:
    """Compile the extension from on-disk sources.

    `load` rather than `load_inline`: `load_inline` auto-generates its own pybind module
    wrapper around a `functions=` list, which collides with the `PYBIND11_MODULE` block
    `_CPP_SRC` writes itself (duplicate-definition build error) once the returns include
    non-trivial types like `std::pair<int64_t, py::bytes>`. Writing real `.cpp`/`.hip` files
    and calling `load` with our own `PYBIND11_MODULE` sidesteps that entirely.

    `rank` only selects this call's build directory (see `_build_dir`); the compiled module
    itself is cached process-wide in `_ext`; since one process is one rank, that cache is
    already per-rank.
    """
    global _ext
    if _ext is None:
        d = _build_dir(rank)
        cpp_path = os.path.join(d, "allreduce_ext.cpp")
        hip_path = os.path.join(d, "allreduce_ext_kernel.hip")
        with open(cpp_path, "w") as f:
            f.write(_CPP_SRC)
        with open(hip_path, "w") as f:
            f.write(_HIP_SRC)
        out_dir = _ext_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        _ext = load(
            name="custom_allreduce_ext",
            sources=[cpp_path, hip_path],
            build_directory=out_dir,
            with_cuda=True,
            verbose=False,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
    return _ext


def _ext_out_dir() -> str:
    """Where torch builds (and file-locks) the extension: `SEED_HIP_CACHE_DIR/custom_allreduce_
    <hash>` when set, else torch's default root, content-addressed like `moe_hip.build_dir`.

    It used to be torch's default `~/.cache/torch_extensions/<py>/custom_allreduce_ext`, one
    directory on the shared home filesystem for every job on every node. A process killed
    mid-build there leaves torch's `lock` file behind, and every later `load` on any node then
    waits on it forever before printing anything (seen as a decode A/B that hung silently for
    its whole 1500 s timeout). Under `SEED_HIP_CACHE_DIR` the lock is job-private, and the
    launch scripts already delete stale `lock` files there at startup; the hash keeps a
    source change from ever sharing a directory with an older build."""
    h = hashlib.sha256((_HIP_SRC + _CPP_SRC + torch.__version__ + str(torch.version.hip)).encode())
    root = os.environ.get("SEED_HIP_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    return os.path.join(root, f"custom_allreduce_{h.hexdigest()[:16]}")


def _vote_or_raise(
    local_ok: bool, local_error: Exception | None, device: torch.device, what: str
) -> None:
    """Every rank votes over the existing process group; raise on every rank if any failed.

    `local_ok` covers only this rank's own build/init work up to this call. This must be
    called at a point every rank is guaranteed to reach regardless of `local_ok` -- callers
    catch their own build exceptions and vote instead of letting them propagate, so a rank
    that failed never leaves its peers blocked on a *later* collective (handle exchange, the
    final barrier) they were never going to reach. `dist.all_reduce` with `MIN` over a 0/1
    flag is 0 if any rank failed; every rank then raises the same way, so `build()`'s
    `except Exception` falls back to RCCL on all ranks together, never on a subset (a subset
    would silently break the "bit-identical to the RCCL path" promise in the module
    docstring, since some ranks would be using a different collective than their peers).
    """
    ok_flag = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(ok_flag, op=dist.ReduceOp.MIN)
    if ok_flag.item() == 0:
        raise RuntimeError(
            f"{what}: build failed on at least one rank, falling back to RCCL on all ranks "
            f"(this rank's local error: {local_error!r})"
        )


def _check_error_flag(flag: torch.Tensor, what: str) -> None:
    """Raise if a collective since the last check hit `SPIN_TIMEOUT_S` waiting on a peer's flag.

    Reads a device-side flag on the host, which forces a sync -- call this at a point the
    caller already synchronizes (after `torch.cuda.synchronize()`, after a CUDA graph
    replay's own sync, or every few decode steps), not once per collective call. Checking
    every call would spend the one-shot design's whole latency argument (see the module
    docstring) on this check instead of the collective; checking occasionally still turns a
    wrong-answer replay into a loud, attributable failure instead of a silently corrupted
    decode. The flag is cleared on read, so a raised error is not re-raised by the next check.
    """
    if bool(flag.item()):
        flag.zero_()
        raise RuntimeError(
            f"{what}: a peer's publish flag never arrived within the SPIN_TIMEOUT_S (600s) "
            "watchdog; the "
            "result of at least one call since the last check_errors() is unreliable. This "
            "usually means a peer rank stalled or died; check the other ranks' health."
        )


class CustomAllReduce:
    """A `Reducer`-shaped one-shot all-reduce over bf16 tensors of up to `MAX_BYTES`.

    Construct once per rank, after `torch.distributed` is initialized and before any graph
    capture. `__call__(x)` sums `x` across the default process group's ranks in place and
    returns it, matching `tp.Reducer`'s contract, but unlike `tp.Reducer` it takes bf16 in and
    reduces in fp32 internally without ever materializing an fp32 copy on the wire.
    """

    def __init__(
        self, rank: int, world: int, device: torch.device, max_elems: int = MAX_ELEMS
    ) -> None:
        self.rank, self.world, self.device, self.max_elems = rank, world, device, max_elems
        self._pos = 0  # call position; see module docstring for why this need not be reset

        # -- Local-only build step. Everything here (compiling the extension, hipMalloc'ing
        # the IPC buffer) can fail independently per rank -- a bad compiler flag, a full disk,
        # one GPU out of fine-grained memory -- and none of it is a torch.distributed
        # collective yet. Catching failures here rather than letting them propagate is what
        # keeps `_vote_or_raise` below reachable by every rank no matter which ones failed: a
        # rank that raised out of a bare `try` would never reach the vote, and every other
        # rank would then block forever on it (or on the handle-exchange collective after it).
        local_ok: bool = True
        local_error: Exception | None = None
        ext = None
        ptr = handle = None
        try:
            ext = _load(rank)
            torch.cuda.set_device(device)

            slot_bytes = max_elems * 2  # bf16
            # Two int32 flags (one per slot) for the host-counter kernels, or, under
            # `GRAPH_SAFE`, `MAX_FLAG_BLOCKS` per slot (one per block); sized for the larger.
            flag_bytes = 2 * MAX_FLAG_BLOCKS * 4
            total = 2 * slot_bytes + flag_bytes
            self._flag_offset = 2 * slot_bytes
            self._sp_offset = total  # `SP` region: rs[2], ag[2], fa[2][4][S], fb[2][4][S]
            if SP:
                total += SP_BYTES
            self._push_offset = total  # `PUSH` region: recv[2][4][PUSH_ELEMS], flags[2][4][rows]
            if PUSH_ALLOC:
                total += PUSH_BYTES

            ptr, handle = ext.alloc_ipc_buffer(total)
        except Exception as exc:  # noqa: BLE001 -- reported through the vote below, not here
            local_ok, local_error = False, exc

        _vote_or_raise(local_ok, local_error, device, "CustomAllReduce")

        # Every rank got here only because every rank's local build step above succeeded, so
        # the handle-exchange collective below is safe: no rank is missing a `handle` to send.
        self._own_ptr = ptr

        handles = [None] * world
        dist.all_gather_object(handles, handle)

        # Opening peers' handles is also per-rank and can fail on one rank only, so it gets
        # its own vote (which doubles as the final barrier): all ranks use the custom path, or
        # all fall back to RCCL.
        local_ok, local_error = True, None
        try:
            buf_ptrs = [0] * world
            flag_ptrs = [0] * world
            for r in range(world):
                p = ptr if r == rank else ext.open_ipc_handle(handles[r])
                buf_ptrs[r] = p
                flag_ptrs[r] = p + self._flag_offset

            self._buf_ptr_table = ext.upload_ptr_table(buf_ptrs)
            self._flag_ptr_table = ext.upload_ptr_table(flag_ptrs)
            self._my_buf = buf_ptrs[rank]
            self._my_flag = flag_ptrs[rank]
            self._error_flag = torch.zeros(1, dtype=torch.int32, device=device)
            # [call counter, retired blocks of the current call]; see `oneshot_ar_kernel`.
            self._ctr = torch.zeros(2, dtype=torch.int32, device=device)
            self._ext = ext
            self._peer_ptrs = (buf_ptrs, flag_ptrs)
            if SP:
                slot_b = SP_SLOT_ELEMS * 2
                fbytes = 2 * 4 * SP_MAX_ROWS * 4
                sp = [p + self._sp_offset for p in buf_ptrs]
                self._sp_ptrs = (
                    [rmsnorm_raw(q, torch.bfloat16) for q in sp],
                    [rmsnorm_raw(q + 2 * slot_b, torch.bfloat16) for q in sp],
                    [rmsnorm_raw(q + 4 * slot_b, torch.int32) for q in sp],
                    [rmsnorm_raw(q + 4 * slot_b + fbytes, torch.int32) for q in sp],
                )
                self._sp_ctr = torch.zeros(2, dtype=torch.int32, device=device)
            if PUSH_ALLOC:
                pb = [p + self._push_offset for p in buf_ptrs]
                self._push_ptrs = (
                    [rmsnorm_raw(q, torch.bfloat16) for q in pb],
                    [rmsnorm_raw(q + 2 * 4 * PUSH_ELEMS * 2, torch.int32) for q in pb],
                )
        except Exception as exc:  # noqa: BLE001 -- reported through the vote below
            local_ok, local_error = False, exc
        _vote_or_raise(local_ok, local_error, device, "CustomAllReduce handle exchange")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16, "CustomAllReduce takes bf16 tensors"
        n = x.numel()
        assert n <= self.max_elems, (
            f"CustomAllReduce: {n} elems exceeds the {self.max_elems}-elem cap"
        )
        xc = x if x.is_contiguous() else x.contiguous()
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if GRAPH_SAFE:
            self._ext.oneshot(
                self._buf_ptr_table,
                self._flag_ptr_table,
                xc.data_ptr(),
                xc.data_ptr(),
                self._ctr.data_ptr(),
                self._error_flag.data_ptr(),
                n,
                self.world,
                self.rank,
                self.max_elems,
                stream,
            )
            if xc is not x:
                x.copy_(xc)
            return x
        slot = self._pos % 2
        expected = self._pos + 1  # never 0: the buffer's zero-initialized rest state
        self._pos += 1

        self._ext.publish(
            xc.data_ptr(), self._my_buf, self._my_flag, n, slot, expected, self.max_elems, stream
        )
        self._ext.gather(
            self._buf_ptr_table,
            self._flag_ptr_table,
            xc.data_ptr(),
            self._error_flag.data_ptr(),
            n,
            slot,
            expected,
            self.world,
            self.rank,
            self.max_elems,
            stream,
        )
        if xc is not x:
            x.copy_(xc)
        return x

    def check_errors(self) -> None:
        """Raise if a collective since the last check hit `SPIN_TIMEOUT_S`. See `_check_error_flag`."""
        _check_error_flag(self._error_flag, "CustomAllReduce")

    def ar_add_rmsnorm_ok(self, mixer: torch.Tensor, residual: torch.Tensor) -> bool:
        """Whether `ar_add_rmsnorm` can take this call (`SEED_AR_RMSNORM_FUSED`)."""
        rows = mixer.numel() // max(mixer.shape[-1], 1)
        return (
            GRAPH_SAFE
            and self.world == 4
            and mixer.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and mixer.shape == residual.shape
            and mixer.is_contiguous()
            and residual.stride(-1) == 1
            and mixer.numel() <= self.max_elems
            and rows <= MAX_FLAG_BLOCKS
        )

    def sp_ok(self, mixer: torch.Tensor, residual_shard: torch.Tensor) -> bool:
        """Whether `sp_ar_add_rmsnorm` can take this call (`SEED_AR_SP`)."""
        rows = mixer.numel() // max(mixer.shape[-1], 1)
        return (
            SP
            and self.world == 4
            and mixer.dtype == torch.bfloat16
            and residual_shard.dtype == torch.bfloat16
            and mixer.is_contiguous()
            and residual_shard.stride(-1) == 1
            and rows % 4 == 0
            and rows // 4 <= SP_MAX_ROWS
            and residual_shard.numel() * 4 == mixer.numel()
            and mixer.numel() <= SP_SLOT_ELEMS
        )

    def sp_ar_add_rmsnorm(
        self, mixer: torch.Tensor, residual_shard: torch.Tensor, norm_w: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(new residual shard, full normed)`: reduce-scatter of `mixer`, residual add and
        RMSNorm on this rank's `rows / 4` rows, all-gather of the normed rows, one launch
        (`SEED_AR_SP`, see `rmsnorm_fused._sp_ar_add_rmsnorm_kernel`). Bit-identical per row to
        `ar_add_rmsnorm`. Caller checks `sp_ok` first."""
        import rmsnorm_fused  # noqa: PLC0415

        rs, ag, fa, fb = self._sp_ptrs
        peers = (rs, ag, fa, fb, self._sp_ctr, self.rank, SP_SLOT_ELEMS, SP_MAX_ROWS, SP_WARPS)
        return rmsnorm_fused.sp_ar_add_rmsnorm(mixer, residual_shard, norm_w, eps, peers)

    def ar_add_rmsnorm(
        self, mixer: torch.Tensor, residual: torch.Tensor, norm_w: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(residual + self(mixer), rmsnorm(residual + self(mixer), norm_w, eps))`, one launch.

        The graph-safe replacement for `residual_norm` (`SEED_AR_RMSNORM_FUSED`): a Triton
        kernel (`rmsnorm_fused._ar_add_rmsnorm_kernel`) on this object's IPC buffers, flags and
        device call counter, so it interleaves with `oneshot_ar_kernel` calls, eager or
        replayed. The norm tail is `rmsnorm_fused.add_rmsnorm`'s own body, so this is bit-exact
        against `self(mixer)` + `add_rmsnorm` (checked on MI300A: every output of 1/16/48-row
        calls, eager and inside graphs interleaved with plain calls, and the full model's
        decode logits). The HIP `residual_norm` is not: its block-tree sum of squares rounds
        differently from `tl.sum`, and `SEED_FUSED_AR_NORM=1` failed the startup replay-vs-eager
        check (eager `Model.decode_layer` runs the second all-reduce unfused) at 3.1-3.6%
        relative logit error with no argmax change, the size a last-bit difference reaches
        once it flips a top-k expert choice (a non-bit-exact router GEMM measured the same).
        Measured in a 120-call graph: 9.3 vs 14.1 us/call at 16 rows, 16.9 vs 22.8 at 48.
        Caller checks `ar_add_rmsnorm_ok` first."""
        import rmsnorm_fused  # noqa: PLC0415 -- Triton import stays off the RCCL-only path

        if PUSH and PUSH_ALLOC and mixer.shape[-1] == 4096:
            recv, pflags = self._push_ptrs
            peers = (recv, pflags, self._ctr, self.rank, PUSH_ELEMS, MAX_FLAG_BLOCKS, PUSH_SYS_FENCE)
            return rmsnorm_fused.push_ar_add_rmsnorm(mixer, residual, norm_w, eps, peers)
        bufs, flags = self._peer_ptrs
        peers = (
            [rmsnorm_fused.RawPtr(p, torch.bfloat16) for p in bufs],
            [rmsnorm_fused.RawPtr(p, torch.int32) for p in flags],
            self._ctr,
            self.rank,
            self.max_elems,
            MAX_FLAG_BLOCKS,
        )
        return rmsnorm_fused.ar_add_rmsnorm(mixer, residual, norm_w, eps, peers)

    def residual_norm(
        self, mixer: torch.Tensor, residual: torch.Tensor, norm_w: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`all_reduce(mixer)` + `residual + reduced` + `rmsnorm(x, norm_w, eps)`, one kernel.

        `mixer` is this rank's row-parallel partial, `[rows, hidden]` or `[rows, 1, hidden]`
        bf16; `residual` is the same shape; `norm_w` is `[hidden]` (any float dtype -- the
        kernel reads it as fp32, matching `model.rmsnorm`'s own `w.float()`). Returns
        `(new_residual, normed)`, both fresh bf16 tensors shaped like `mixer`: `new_residual`
        is what `residual + self.all_reduce(mixer)` would have produced, `normed` is
        `rmsnorm(new_residual, norm_w, eps)`.

        `hidden` must be a multiple of 8 (bf16 elements per 16-byte vector load; every real
        model dimension is) and the whole call's `mixer.numel()` must fit `self.max_elems`,
        same cap `__call__` enforces -- callers check both before routing here (see
        `tp.TP.all_reduce_residual_norm`).

        **Numerics.** The reduction (world-partial sum, rounded to bf16) is byte-for-byte the
        same computation `gather_kernel` does for `__call__`, in the same per-rank order, so
        the "reduced" value inside this kernel is not a new, separately-rounded quantity. The
        residual add happens in fp32 against that already-bf16-rounded value and rounds once
        more at the `new_residual` store -- i.e. two roundings, in the same order the unfused
        `x = residual + self.all_reduce(mixer)` produces (bf16 `+` bf16 on an accelerator
        computes in an fp32 accumulator and rounds once, so adding an already-rounded bf16
        `reduced` to `residual` reproduces that). The norm pass mirrors `rmsnorm_fused`'s
        single fp32 chain with one rounding at the store; its column-wise reduction order
        (a block-wide tree over `hidden` chunks) is `tl.sum`'s, so the only remaining
        difference from the torch/Triton norm path is the fp32-reduction-order noise this
        module's other kernels already carry (see `test_rmsnorm_fused.py`'s tolerance and
        `deltanet_fused`'s module docstring for the precedent).
        """
        assert mixer.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
        hidden = mixer.shape[-1]
        if hidden % 8:
            raise ValueError(f"residual_norm needs hidden % 8 == 0, got {hidden}")
        mc = mixer.contiguous().reshape(-1, hidden)
        rc = residual.contiguous().reshape(-1, hidden)
        if mc.shape != rc.shape:
            raise ValueError(f"shape mismatch: mixer {tuple(mixer.shape)}, residual {tuple(residual.shape)}")
        rows = mc.shape[0]
        n = mc.numel()
        assert n <= self.max_elems, f"CustomAllReduce: {n} elems exceeds the {self.max_elems}-elem cap"
        if GRAPH_SAFE:
            assert rows <= MAX_FLAG_BLOCKS, f"residual_norm: {rows} rows > {MAX_FLAG_BLOCKS}"
            new_residual = torch.empty_like(mc)
            normed = torch.empty_like(mc)
            norm_w32 = norm_w if norm_w.dtype == torch.float32 else norm_w.float()
            self._ext.oneshot_residual_norm(
                self._buf_ptr_table,
                self._flag_ptr_table,
                mc.data_ptr(),
                rc.data_ptr(),
                norm_w32.data_ptr(),
                new_residual.data_ptr(),
                normed.data_ptr(),
                self._ctr.data_ptr(),
                self._error_flag.data_ptr(),
                rows,
                hidden,
                self.world,
                self.rank,
                self.max_elems,
                eps,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
            return new_residual.reshape(mixer.shape), normed.reshape(mixer.shape)

        slot = self._pos % 2
        expected = self._pos + 1
        self._pos += 1
        stream = torch.cuda.current_stream(self.device).cuda_stream

        self._ext.publish(mc.data_ptr(), self._my_buf, self._my_flag, n, slot, expected, self.max_elems, stream)
        new_residual = torch.empty_like(mc)
        normed = torch.empty_like(mc)
        norm_w32 = norm_w if norm_w.dtype == torch.float32 else norm_w.float()
        self._ext.gather_residual_norm(
            self._buf_ptr_table,
            self._flag_ptr_table,
            rc.data_ptr(),
            norm_w32.data_ptr(),
            new_residual.data_ptr(),
            normed.data_ptr(),
            self._error_flag.data_ptr(),
            rows,
            hidden,
            slot,
            expected,
            self.world,
            self.rank,
            self.max_elems,
            eps,
            stream,
        )
        return new_residual.reshape(mixer.shape), normed.reshape(mixer.shape)


class TwoShotAllReduce:
    """Reduce-scatter + all-gather, for the one-shot-vs-two-shot comparison at 384 KiB.

    Comparison-only: not wired into `tp.py`. Publishes into the same layout `CustomAllReduce`
    does (reusing `launch_publish`), then reduce-scatters each rank's `1/world` shard (each
    rank reads only the `world - 1` peers' contribution to *its own* shard, `shard_elems`
    wide, not the full tensor) and all-gathers the shards back. Same two synchronization
    rounds and the same per-slot-reuse argument as `CustomAllReduce` (see its module
    docstring); this class just has two flag rounds (`publish`+`reduce_scatter` share the
    first, `allgather` waits on the second) instead of one, and moves `1.5x n_elems` bytes
    of remote traffic per call instead of one-shot's `3x` (at `world=4`) -- at the cost of a
    third kernel launch. Whether that trade wins at 384 KiB is exactly what
    `bench_custom_allreduce.py --two-shot` measures; see its output for the number.
    """

    def __init__(
        self, rank: int, world: int, device: torch.device, max_elems: int = MAX_ELEMS
    ) -> None:
        assert max_elems % world == 0, "TwoShotAllReduce: max_elems must divide world"
        self.rank, self.world, self.device = rank, world, device
        self.max_elems, self.shard_elems = max_elems, max_elems // world
        self._pos = 0

        # See CustomAllReduce.__init__: local build step first, so a per-rank failure here
        # (compile, hipMalloc) never leaves a rank stuck before the vote below.
        local_ok: bool = True
        local_error: Exception | None = None
        ext = None
        ptr = handle = None
        try:
            ext = _load(rank)
            torch.cuda.set_device(device)

            slot_bytes = max_elems * 2
            flag0_bytes = 2 * 4
            shard_slot_bytes = self.shard_elems * 2
            flag1_bytes = 2 * 4
            total = 2 * slot_bytes + flag0_bytes + 2 * shard_slot_bytes + flag1_bytes
            self._flag0_off = 2 * slot_bytes
            self._shard_off = self._flag0_off + flag0_bytes
            self._flag1_off = self._shard_off + 2 * shard_slot_bytes

            ptr, handle = ext.alloc_ipc_buffer(total)
        except Exception as exc:  # noqa: BLE001 -- reported through the vote below, not here
            local_ok, local_error = False, exc

        _vote_or_raise(local_ok, local_error, device, "TwoShotAllReduce")

        self._own_ptr = ptr
        handles = [None] * world
        dist.all_gather_object(handles, handle)

        buf_ptrs, flag0_ptrs, shard_ptrs, flag1_ptrs = (
            [0] * world,
            [0] * world,
            [0] * world,
            [0] * world,
        )
        for r in range(world):
            p = ptr if r == rank else ext.open_ipc_handle(handles[r])
            buf_ptrs[r] = p
            flag0_ptrs[r] = p + self._flag0_off
            shard_ptrs[r] = p + self._shard_off
            flag1_ptrs[r] = p + self._flag1_off

        self._buf_table = ext.upload_ptr_table(buf_ptrs)
        self._flag0_table = ext.upload_ptr_table(flag0_ptrs)
        self._shard_table = ext.upload_ptr_table(shard_ptrs)
        self._flag1_table = ext.upload_ptr_table(flag1_ptrs)
        self._my_buf, self._my_flag0 = buf_ptrs[rank], flag0_ptrs[rank]
        self._my_shard, self._my_flag1 = shard_ptrs[rank], flag1_ptrs[rank]
        self._error_flag = torch.zeros(1, dtype=torch.int32, device=device)
        self._ext = ext
        dist.barrier()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16 and x.numel() % self.world == 0
        n = x.numel()
        assert n <= self.max_elems
        xc = x if x.is_contiguous() else x.contiguous()
        slot = self._pos % 2
        expected = self._pos + 1
        self._pos += 1
        stream = torch.cuda.current_stream(self.device).cuda_stream
        shard_elems = n // self.world

        self._ext.publish(
            xc.data_ptr(), self._my_buf, self._my_flag0, n, slot, expected, self.max_elems, stream
        )
        self._ext.reduce_scatter(
            self._buf_table,
            self._flag0_table,
            self._my_shard,
            self._my_flag1,
            self._error_flag.data_ptr(),
            shard_elems,
            slot,
            expected,
            self.world,
            self.rank,
            self.max_elems,
            self.shard_elems,
            stream,
        )
        self._ext.allgather(
            self._shard_table,
            self._flag1_table,
            xc.data_ptr(),
            self._error_flag.data_ptr(),
            shard_elems,
            slot,
            expected,
            self.world,
            self.rank,
            self.shard_elems,
            stream,
        )
        if xc is not x:
            x.copy_(xc)
        return x

    def check_errors(self) -> None:
        """Raise if a collective since the last check hit `SPIN_TIMEOUT_S`. See `_check_error_flag`."""
        _check_error_flag(self._error_flag, "TwoShotAllReduce")


def build(rank: int, world: int, device: torch.device) -> CustomAllReduce | None:
    """`CustomAllReduce`, or `None` if disabled or unavailable. Never raises."""
    if os.environ.get("SEED_CUSTOM_ALLREDUCE", "1") == "0":
        return None
    if world <= 1 or not torch.cuda.is_available():
        return None
    try:
        return CustomAllReduce(rank, world, device)
    except Exception as exc:  # pragma: no cover - defensive: fall back to RCCL on any failure
        print(f"[custom-allreduce] disabled, falling back to RCCL: {exc!r}", flush=True)
        return None

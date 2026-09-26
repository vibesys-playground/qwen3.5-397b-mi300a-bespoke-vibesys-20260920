// MXFP4 routed-expert decode GEMMs for gfx942, hand-written HIP (SEED_MOE_HIP=1).
// Design, budget and validation: see moe_hip.py. This file has three layers:
//   1. `MOE_HD` helpers (decode, lane mapping): compiled for the device and, with
//      MOE_HIP_HOST_TEST defined, for the host, where test_moe_hip_host.cpp checks them.
//   2. `moe_kernel`: one template for gate_up(+SiLU) and down(+routing weight).
//   3. extern "C" launchers, called from moe_hip_ext.cpp.
//
// Work unit (from mxfp4_gemv._bw_prep_kernel): unit[3u] = (local expert, first position in
// `sorted`, live count <= 16); `sorted` holds assignment ids grouped by expert.
//
// Wave tile: 16 weight rows x 512 K = 4 KB of fp4 + 256 B of e8m0. Lane l = r + 16 g owns
// row r and physical K [kw*512 + 128 g, +128): four 16-byte loads (64 contiguous bytes, one
// row's 256-byte segment spread over the four lanes of that row) plus one dword of scales.
// That is the MFMA 16x16x16 A-operand placement (lane = row + 16 * k-group), so decoded
// values feed the MFMA from registers.

#include <stdint.h>

#ifdef MOE_HIP_HOST_TEST
#define MOE_HD
#else
#include <hip/hip_runtime.h>
#define MOE_HD __host__ __device__
#endif

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint16_t u16x2 __attribute__((ext_vector_type(2)));

// v_perm_b32: byte i of the result = byte sel[i] of the 64-bit {hi:lo}. Host spelling follows
// the gfx9 ISA table (8-11 sign-replicate, 12 zero, 13+ 0xff); the decode only uses 0-7.
MOE_HD inline uint32_t moe_perm(uint32_t hi, uint32_t lo, uint32_t sel) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_perm(hi, lo, sel);
#else
    uint64_t v = ((uint64_t)hi << 32) | lo;
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t s = (sel >> (8 * i)) & 0xff, b;
        if (s < 8) b = (uint32_t)(v >> (8 * s)) & 0xff;
        else if (s < 12) b = ((v >> (16 * (s - 8) + 15)) & 1) ? 0xff : 0x00;
        else if (s == 12) b = 0x00;
        else b = 0xff;
        out |= b << (8 * i);
    }
    return out;
#endif
}

// Per-scale-block byte tables. Entry v (fp4 magnitude code 0..7) is bf16(fp4(v)) - bf16(1.0)
// as u16; adding (s << 7) to it gives bf16(fp4(v) * 2^(s-127)) exactly, for s in [2, 252]
// (mxfp4_gemv.bw_scales_ok). Entry 0 is forced to 0. The u16 entries are split into a low-
// byte table and a high-byte table so one v_perm_b32 looks up four values.
struct MoeTab {
    uint32_t lo0, lo1, hi0, hi1;
};

MOE_HD inline MoeTab moe_make_tab(uint32_t s) {
    uint32_t t = s * 0x00800080u;  // (s << 7) in both u16 halves
    u16x2 tt = __builtin_bit_cast(u16x2, t);
    uint32_t q0 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(u16x2, 0xff800000u) + tt);
    uint32_t q1 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(u16x2, 0x00400000u) + tt);
    uint32_t q2 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(u16x2, 0x00c00080u) + tt);
    uint32_t q3 = __builtin_bit_cast(uint32_t, __builtin_bit_cast(u16x2, 0x01400100u) + tt);
    q0 &= 0xffff0000u;  // magnitude 0 stays exactly 0
    MoeTab r;
    r.lo0 = moe_perm(q1, q0, 0x06040200u);
    r.lo1 = moe_perm(q3, q2, 0x06040200u);
    r.hi0 = moe_perm(q1, q0, 0x07050301u);
    r.hi1 = moe_perm(q3, q2, 0x07050301u);
    return r;
}

// One packed word (8 fp4, element n = nibble n, low nibble = even K) -> four bf16 pairs:
// p[0] = (e0, e2), p[1] = (e4, e6), p[2] = (e1, e3), p[3] = (e5, e7), low half first.
// p[0..1] is one MFMA's A operand (the even elements), p[2..3] the next MFMA's (odd).
MOE_HD inline void moe_decode_word(uint32_t w, const MoeTab& t, uint32_t p[4]) {
    uint32_t a = w & 0x07070707u;         // magnitudes of e0 e2 e4 e6
    uint32_t b = (w >> 4) & 0x07070707u;  // magnitudes of e1 e3 e5 e7
    uint32_t ha = moe_perm(t.hi1, t.hi0, a);
    uint32_t hb = moe_perm(t.hi1, t.hi0, b);
    uint32_t la = moe_perm(t.lo1, t.lo0, a);
    uint32_t lb = moe_perm(t.lo1, t.lo0, b);
    ha |= (w << 4) & 0x80808080u;  // low-nibble sign (bit 3) -> bf16 sign
    hb |= w & 0x80808080u;         // high-nibble sign is already at bit 7
    p[0] = moe_perm(ha, la, 0x05010400u);
    p[1] = moe_perm(ha, la, 0x07030602u);
    p[2] = moe_perm(hb, lb, 0x05010400u);
    p[3] = moe_perm(hb, lb, 0x07030602u);
}

// Eight bf16 activations x0..x7 (one 16-byte load, K order) -> the B operands matching
// moe_decode_word: a[0..1] = (x0, x2), (x4, x6); a[2..3] = (x1, x3), (x5, x7).
MOE_HD inline void moe_split_act(u32x4 d, uint32_t a[4]) {
    a[0] = moe_perm(d.y, d.x, 0x05040100u);
    a[1] = moe_perm(d.w, d.z, 0x05040100u);
    a[2] = moe_perm(d.y, d.x, 0x07060302u);
    a[3] = moe_perm(d.w, d.z, 0x07060302u);
}

MOE_HD inline uint16_t moe_f2bf(float f) {  // round to nearest even; inputs are finite
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

MOE_HD inline float moe_bf2f(uint16_t h) { return __builtin_bit_cast(float, (uint32_t)h << 16); }

// Lane placement. Byte offset of lane (r, g)'s 64 weight bytes within a tile whose first row
// starts at the tile base, for wave K-slice kw; its scale dword; its activation elements.
MOE_HD inline uint32_t moe_w_off(int r, int g, int kw, int K) { return r * (K / 2) + kw * 256 + g * 64; }
MOE_HD inline uint32_t moe_s_off(int r, int g, int kw, int K) { return r * (K / 32) + kw * 16 + g * 4; }
MOE_HD inline uint32_t moe_a_off(int g, int kw) { return kw * 512 + g * 128; }

// Work split. A segment is (unit, item range); each program runs whole segments, so a
// segment's activations are loaded once, before its tile loop. With n_units <= grid every
// program gets at most one segment: unit u's nt items are split evenly over the programs b
// with b * n_units / grid == u. With more units than programs, program b takes units
// b, b + grid, ... whole. Returns false when program b has no segment `seg`, and for every
// program when n_units is 0 (no live local assignment, e.g. a small batch routed entirely to
// other ranks' experts): the split below divides by n_units.
//
// MOE_FLAT=1 (SEED_MOE_HIP_WIDE): instead, the n_units * nt (unit, item) pairs are laid out
// flat, unit-major, and program b takes the contiguous range [b * total / grid, (b + 1) *
// total / grid), one segment per unit it touches. Every program gets the same item count
// (+-1), so no program runs a whole unit alone while others idle: with the default split a
// program's work is 1 unit / ceil(grid / n_units) when n_units <= grid, and up to
// ceil(n_units / grid) whole units otherwise, and one gate_up unit on one CU is ~0.2 ms of
// decode + MFMA. The cost is at most one extra activation load per unit boundary.
#ifndef MOE_FLAT
#define MOE_FLAT 0
#endif
MOE_HD inline bool moe_segment(int b, int grid, int n_units, int nt, int seg, int* u, int* it_lo,
                               int* it_hi) {
    if (n_units <= 0) return false;
    if (MOE_FLAT) {
        const int64_t total = (int64_t)n_units * nt;
        const int64_t start = (int64_t)b * total / grid, end = (int64_t)(b + 1) * total / grid;
        if (start >= end) return false;
        *u = (int)(start / nt) + seg;
        const int64_t base = (int64_t)*u * nt;
        if (base >= end) return false;
        *it_lo = (int)(start > base ? start - base : 0);
        *it_hi = (int)(end - base < nt ? end - base : nt);
        return *it_hi > *it_lo;
    }
    if (n_units <= grid) {
        if (seg > 0) return false;
        *u = (int)(((int64_t)b * n_units) / grid);
        const int b0 = (int)(((int64_t)*u * grid + n_units - 1) / n_units);
        const int b1 = (int)(((int64_t)(*u + 1) * grid + n_units - 1) / n_units);
        const int i = b - b0, c = b1 - b0;
        *it_lo = i * nt / c;
        *it_hi = (i + 1) * nt / c;
    } else {
        *u = b + seg * grid;
        if (*u >= n_units) return false;
        *it_lo = 0;
        *it_hi = nt;
    }
    return *it_hi > *it_lo;
}

// Weight row (within the expert) of tile j of an item starting at output row n0.
// gate_up: tiles 0..pt-1 are gate rows, pt..2pt-1 the matching up rows (at out_rows + n).
// down: wave rw of a program owns tiles rw * tpw .. rw * tpw + tpw - 1.
MOE_HD inline int moe_tile_row(bool gate_up, int out_rows, int pt, int tpw, int rw, int n0, int j) {
    if (gate_up) return (j < pt ? 0 : out_rows) + n0 + 16 * (j % pt);
    return n0 + 16 * (rw * tpw + j);
}

#ifndef MOE_HIP_HOST_TEST

#ifndef MOE_NT
#define MOE_NT 0  // 1: weight loads marked nontemporal (streamed once per step)
#endif

typedef short v4i16 __attribute__((ext_vector_type(4)));
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

struct MoeTile {
    u32x4 w[4];
    uint32_t s;
};

__device__ inline v4f32 moe_mfma(uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, v4f32 c) {
    u32x2 a = {a0, a1}, b = {b0, b1};
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        __builtin_bit_cast(v4i16, a), __builtin_bit_cast(v4i16, b), c, 0, 0, 0);
}

// GATE_UP: out[slot, n] = silu(gate[e, n] . x[slot / top_k]) * (up[e, n] . x[slot / top_k]);
//   item = (unit, 16 * PT neurons); wave kw of KW owns K-slice kw; tiles per wave per item
//   TPW = 2 * PT (PT gate tiles, then the PT matching up tiles). RW must be 1.
// !GATE_UP: out[slot, n] = a_weight[slot] * (down[e, n] . act[slot]);
//   item = (unit, 16 * RW * TPW rows); wave (kw, rw) owns K-slice kw and TPW row tiles.
// Static grid (graph-capturable); each program runs whole segments (see "Work split") and
// streams a segment's tiles through a DEPTH-deep register ring: tile s + DEPTH - 1 is issued before tile s
// is decoded, so every wave keeps DEPTH - 1 tiles (4 KB each) of loads in flight.
// NB (SEED_MOE_HIP_BT32): token groups of 16 per unit. A unit holds up to 16 * NB tokens of
// one expert; each decoded weight word feeds NB MFMAs (one per group), so an expert with 17-32
// tokens is loaded and decoded once instead of twice. Per token the MFMA sequence and the
// epilogue are the NB = 1 ones, so the results are bit-identical to NB = 1.
template <int K, int OUT_ROWS, int KW, int RW, int TPW, int DEPTH, bool GATE_UP, int NB = 1>
__global__ __launch_bounds__(64 * KW * RW, 1) void moe_kernel(
    const uint16_t* __restrict__ act,  // GATE_UP: x [tokens, K]; else inter [assignments, K]
    const uint8_t* __restrict__ wq,    // [E, W_ROWS, K / 2]
    const uint8_t* __restrict__ ws,    // [E, W_ROWS, K / 32]
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ unit,
    const int32_t* __restrict__ n_units_ptr,
    const float* __restrict__ a_weight,  // !GATE_UP only: fp32 routing weight [assignments]
    uint16_t* __restrict__ out,             // [assignments, OUT_ROWS]
    int top_k) {
    static_assert(K == 512 * KW, "one wave K-slice is 512");
    static_assert(DEPTH % TPW == 0, "tile-in-item index must be a compile-time constant");
    static_assert(!GATE_UP || (RW == 1 && TPW % 2 == 0), "gate_up: RW 1, gate/up tile pairs");
    constexpr int PT = TPW / 2;
    constexpr int NI = GATE_UP ? 16 * PT : 16 * RW * TPW;  // output rows per item
    static_assert(OUT_ROWS % NI == 0, "item rows must divide the output rows");
    constexpr int NT = OUT_ROWS / NI;  // items per unit
    constexpr int W_ROWS = GATE_UP ? 2 * OUT_ROWS : OUT_ROWS;
    constexpr int WAVES = KW * RW;
    // LDS partial sums, double-buffered by item parity: one barrier per item.
    constexpr int RED = GATE_UP ? WAVES * TPW : (KW - 1) * RW * TPW;
    __shared__ v4f32 red[2][RED > 0 ? RED : 1][64];

    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int kw = wave % KW, rw = wave / KW;
    const int lane = threadIdx.x & 63, r = lane & 15, g = lane >> 4;
    const uint32_t w_lane = moe_w_off(r, g, kw, K);
    const uint32_t s_lane = moe_s_off(r, g, kw, K);

    const int n_units = n_units_ptr[0];
    for (int seg = 0;; ++seg) {
        int u, it_lo, it_hi;
        if (!moe_segment(blockIdx.x, gridDim.x, n_units, NT, seg, &u, &it_lo, &it_hi)) break;
        const int S = (it_hi - it_lo) * TPW;  // > 0; empty programs exit before any load

        const int e = unit[3 * u], p0 = unit[3 * u + 1], cnt = unit[3 * u + 2];
        // this lane's token column in group b is r + 16 b; columns >= cnt repeat column 0 and
        // are never stored
        bool valid[NB];
        int slot[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            valid[b] = r + 16 * b < cnt;
            slot[b] = sorted[p0 + (valid[b] ? r + 16 * b : 0)];
        }

        auto issue = [&](MoeTile& t, int s) {
            s = s < S ? s : S - 1;  // past the end: reload the last tile (a cache hit), no branch
            const int item = it_lo + s / TPW, j = s % TPW;
            const int64_t row = (int64_t)e * W_ROWS + moe_tile_row(GATE_UP, OUT_ROWS, PT, TPW, rw, item * NI, j);
            const u32x4* wp = (const u32x4*)(wq + row * (K / 2) + w_lane);
#pragma unroll
            for (int i = 0; i < 4; ++i) t.w[i] = MOE_NT ? __builtin_nontemporal_load(wp + i) : wp[i];
            t.s = *(const uint32_t*)(ws + row * (K / 32) + s_lane);
        };

        // Activations first, then the ring prologue: the wave's 512-K slice for all 16 token
        // columns (the B operands of its 32 MFMAs per tile, held for the whole segment). The
        // order matters to the waitcnt pass: at loop entry the ring must be the newest loads
        // in flight, oldest slot first, exactly as in the steady state. Otherwise the loop-head
        // merge makes every trip wait as if slot 0 had just been issued (vmcnt(5)).
        u32x4 araw[NB][16];
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            const int64_t arow = GATE_UP ? (int64_t)(slot[b] / top_k) : (int64_t)slot[b];
            const u32x4* ap = (const u32x4*)(act + arow * K + moe_a_off(g, kw));
#pragma unroll
            for (int q = 0; q < 16; ++q) araw[b][q] = ap[q];
        }
        float cw[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) cw[b] = GATE_UP ? 0.f : a_weight[slot[b]];
        __builtin_amdgcn_sched_barrier(0);

        MoeTile ring[DEPTH];
#pragma unroll
        for (int d = 0; d < DEPTH - 1; ++d) issue(ring[d], d);
        __builtin_amdgcn_sched_barrier(0);

        uint32_t a[NB][64];
#pragma unroll
        for (int b = 0; b < NB; ++b)
#pragma unroll
            for (int q = 0; q < 16; ++q) moe_split_act(araw[b][q], &a[b][4 * q]);

        v4f32 acc[NB][TPW];
        // Unrolled by DEPTH so ring slots are registers; the tail steps of the last trip are
        // predicated off, not exited: an exit edge from mid-body makes the waitcnt pass drain
        // the ring at every loop head (vmcnt(1) before each issue, seen in the offline ISA).
        for (int s0 = 0; s0 < S; s0 += DEPTH) {
#pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                const int s = s0 + d;
                issue(ring[(d + DEPTH - 1) % DEPTH], s + DEPTH - 1);
                // Keep the loads above the decode: without this the machine scheduler sinks
                // them to the end of the block, behind this tile's decode + MFMA chain.
                __builtin_amdgcn_sched_barrier(0);
                if (s < S) {
                    const int j = d % TPW;  // == s % TPW because s0 % TPW == 0
                    if (j == 0) {
#pragma unroll
                        for (int b = 0; b < NB; ++b)
#pragma unroll
                            for (int jj = 0; jj < TPW; ++jj) acc[b][jj] = (v4f32){0.f, 0.f, 0.f, 0.f};
                    }
                    const MoeTile& t = ring[d];
                    MoeTab tab;
#pragma unroll
                    for (int q = 0; q < 16; ++q) {
                        if (q % 4 == 0) tab = moe_make_tab((t.s >> (8 * (q / 4))) & 0xffu);
                        uint32_t p[4];
                        moe_decode_word(t.w[q / 4][q % 4], tab, p);
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            acc[b][j] = moe_mfma(p[0], p[1], a[b][4 * q], a[b][4 * q + 1], acc[b][j]);
                            acc[b][j] = moe_mfma(p[2], p[3], a[b][4 * q + 2], a[b][4 * q + 3], acc[b][j]);
                        }
                    }
                    if (j == TPW - 1) {  // item epilogue: reduce K-slices across waves, store
                        const int item = it_lo + s / TPW;
                        const int n0 = item * NI;
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                        // LDS buffer parity over (item, group): consecutive epilogues alternate
                        // buffers, so one barrier per epilogue still suffices
                        const int buf = ((s / TPW) * NB + b) & 1;
                        if constexpr (GATE_UP) {
#pragma unroll
                            for (int jj = 0; jj < TPW; ++jj) red[buf][wave * TPW + jj][lane] = acc[b][jj];
                            __syncthreads();
                            if (wave == (item * NB + b) % WAVES && valid[b]) {  // rotate the epilogue over waves
#pragma unroll
                                for (int pt = 0; pt < PT; ++pt) {
                                    v4f32 gs = red[buf][pt][lane], us = red[buf][PT + pt][lane];
#pragma unroll
                                    for (int w = 1; w < WAVES; ++w) {
                                        gs += red[buf][w * TPW + pt][lane];
                                        us += red[buf][w * TPW + PT + pt][lane];
                                    }
                                    uint16_t h[4];
#pragma unroll
                                    for (int i = 0; i < 4; ++i)
                                        h[i] = moe_f2bf(gs[i] / (1.0f + __expf(-gs[i])) * us[i]);
                                    u32x2 v = {h[0] | ((uint32_t)h[1] << 16), h[2] | ((uint32_t)h[3] << 16)};
                                    *(u32x2*)(out + (int64_t)slot[b] * OUT_ROWS + n0 + 16 * pt + 4 * g) = v;
                                }
                            }
                        } else {
                            if (kw > 0) {
#pragma unroll
                                for (int jj = 0; jj < TPW; ++jj)
                                    red[buf][((kw - 1) * RW + rw) * TPW + jj][lane] = acc[b][jj];
                            }
                            __syncthreads();
                            if (kw == 0 && valid[b]) {
#pragma unroll
                                for (int jj = 0; jj < TPW; ++jj) {
                                    v4f32 v = acc[b][jj];
#pragma unroll
                                    for (int k2 = 1; k2 < KW; ++k2)
                                        v += red[buf][((k2 - 1) * RW + rw) * TPW + jj][lane];
                                    uint16_t h[4];
#pragma unroll
                                    for (int i = 0; i < 4; ++i) h[i] = moe_f2bf(v[i] * cw[b]);
                                    u32x2 o = {h[0] | ((uint32_t)h[1] << 16), h[2] | ((uint32_t)h[3] << 16)};
                                    *(u32x2*)(out + (int64_t)slot[b] * OUT_ROWS + n0 + 16 * (rw * TPW + jj) + 4 * g) = o;
                                }
                            }
                        }
                        }  // group b
                    }
                }
            }
        }
        // the next segment's first epilogue reuses LDS buffer 0; all reads of this segment's
        // last buffer happen before any wave passes this barrier
        __syncthreads();
    }
}


// Real per-rank shapes (Qwen3.5-397B-A17B, TP=4): hidden 4096, moe_intermediate 1024.
constexpr int MOE_HIDDEN = 4096, MOE_INTER = 1024;
#ifndef MOE_GU_PT
#define MOE_GU_PT 1  // gate_up: 16 * PT neurons (gate + up tiles) per item
#endif
#ifndef MOE_DN_RW
#define MOE_DN_RW 4  // down: row waves per program
#endif
#ifndef MOE_DN_TPW
#define MOE_DN_TPW 1  // down: row tiles per wave per item
#endif
#ifndef MOE_DEPTH
#define MOE_DEPTH 4  // register ring depth in tiles (DEPTH - 1 tiles of loads in flight)
#endif

#define MOE_GU_KERNEL moe_kernel<MOE_HIDDEN, MOE_INTER, MOE_HIDDEN / 512, 1, 2 * MOE_GU_PT, MOE_DEPTH, true>
#define MOE_DN_KERNEL moe_kernel<MOE_INTER, MOE_HIDDEN, MOE_INTER / 512, MOE_DN_RW, MOE_DN_TPW, MOE_DEPTH, false>

extern "C" int moe_hip_gate_up(const void* x, const void* wq, const void* ws, const void* sorted,
                               const void* unit, const void* n_units, void* inter, int top_k,
                               int grid, void* stream) {
    hipLaunchKernelGGL((MOE_GU_KERNEL), dim3(grid), dim3(64 * (MOE_HIDDEN / 512)), 0,
                       (hipStream_t)stream, (const uint16_t*)x, (const uint8_t*)wq,
                       (const uint8_t*)ws, (const int32_t*)sorted, (const int32_t*)unit,
                       (const int32_t*)n_units, (const float*)nullptr, (uint16_t*)inter, top_k);
    return (int)hipGetLastError();
}

extern "C" int moe_hip_down(const void* inter, const void* wq, const void* ws, const void* a_weight,
                            const void* sorted, const void* unit, const void* n_units, void* y,
                            int grid, void* stream) {
    hipLaunchKernelGGL((MOE_DN_KERNEL), dim3(grid), dim3(64 * (MOE_INTER / 512) * MOE_DN_RW), 0,
                       (hipStream_t)stream, (const uint16_t*)inter, (const uint8_t*)wq,
                       (const uint8_t*)ws, (const int32_t*)sorted, (const int32_t*)unit,
                       (const int32_t*)n_units, (const float*)a_weight, (uint16_t*)y, 0);
    return (int)hipGetLastError();
}

#ifndef MOE_DEPTH2
#define MOE_DEPTH2 MOE_DEPTH  // ring depth of the NB = 2 kernels (more VGPRs per tile go to activations)
#endif
#define MOE_GU_KERNEL2 moe_kernel<MOE_HIDDEN, MOE_INTER, MOE_HIDDEN / 512, 1, 2 * MOE_GU_PT, MOE_DEPTH2, true, 2>
#define MOE_DN_KERNEL2 moe_kernel<MOE_INTER, MOE_HIDDEN, MOE_INTER / 512, MOE_DN_RW, MOE_DN_TPW, MOE_DEPTH2, false, 2>

// NB = 2 (32-token units, SEED_MOE_HIP_BT32): same arguments as the NB = 1 launchers.
extern "C" int moe_hip_gate_up2(const void* x, const void* wq, const void* ws, const void* sorted,
                                const void* unit, const void* n_units, void* inter, int top_k,
                                int grid, void* stream) {
    hipLaunchKernelGGL((MOE_GU_KERNEL2), dim3(grid), dim3(64 * (MOE_HIDDEN / 512)), 0,
                       (hipStream_t)stream, (const uint16_t*)x, (const uint8_t*)wq,
                       (const uint8_t*)ws, (const int32_t*)sorted, (const int32_t*)unit,
                       (const int32_t*)n_units, (const float*)nullptr, (uint16_t*)inter, top_k);
    return (int)hipGetLastError();
}

extern "C" int moe_hip_down2(const void* inter, const void* wq, const void* ws, const void* a_weight,
                             const void* sorted, const void* unit, const void* n_units, void* y,
                             int grid, void* stream) {
    hipLaunchKernelGGL((MOE_DN_KERNEL2), dim3(grid), dim3(64 * (MOE_INTER / 512) * MOE_DN_RW), 0,
                       (hipStream_t)stream, (const uint16_t*)inter, (const uint8_t*)wq,
                       (const uint8_t*)ws, (const int32_t*)sorted, (const int32_t*)unit,
                       (const int32_t*)n_units, (const float*)a_weight, (uint16_t*)y, 0);
    return (int)hipGetLastError();
}

// Rows per item, for the host's static grid bound.
extern "C" int moe_hip_gate_up_item_rows() { return 16 * MOE_GU_PT; }
extern "C" int moe_hip_down_item_rows() { return 16 * MOE_DN_RW * MOE_DN_TPW; }

#endif  // MOE_HIP_HOST_TEST

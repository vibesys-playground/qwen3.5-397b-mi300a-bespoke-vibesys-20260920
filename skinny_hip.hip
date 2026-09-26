// Weight-streaming skinny bf16 GEMM for gfx942 (SEED_HIP_SKINNY_GEMM=1): out[M, N] = x[M, K] @ w[N, K]^T,
// M <= 64. Design, budget and routing: see skinny_hip.py.
//
// Wave step: 16 weight rows x 128 K (4 KB). Lane l = r + 16 g owns weight row r and loads
// four 16-byte pieces at K offsets 32 j + 8 g (j = 0..3). Each piece is two MFMA
// 16x16x16 A operands (its low and high 4 bf16); the B operand (activations, held in
// registers for the whole program) is loaded with the same K placement, so the MFMA's K
// order is a permutation of the true K order applied identically to both operands.
// MFMA D layout: lane (r, g) holds out[m = 16 mt + r][n0 + 4 g + i], i = 0..3.

#include <hip/hip_runtime.h>
#include <stdint.h>

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef short v4i16 __attribute__((ext_vector_type(4)));
typedef float v4f32 __attribute__((ext_vector_type(4)));

#define SK_VMCNT0 0x0F70  // s_waitcnt vmcnt(0), expcnt/lgkmcnt at max (gfx9 encoding)

__device__ inline v4f32 sk_mfma(u32x2 a, u32x2 b, v4f32 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        __builtin_bit_cast(v4i16, a), __builtin_bit_cast(v4i16, b), c, 0, 0, 0);
}

__device__ inline uint32_t sk_f2bf(float f) {  // round to nearest even; finite inputs
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

__device__ inline void sk_store(uint16_t* __restrict__ out, v4f32 v, int m, int n, int M, int N,
                                bool vec) {
    if (m >= M) return;
    uint16_t* p = out + (int64_t)m * N + n;
    if (vec && n + 3 < N) {
        u32x2 o = {sk_f2bf(v[0]) | (sk_f2bf(v[1]) << 16), sk_f2bf(v[2]) | (sk_f2bf(v[3]) << 16)};
        *(u32x2*)p = o;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            if (n + i < N) p[i] = (uint16_t)sk_f2bf(v[i]);
    }
}

// MT: 16-column M tiles (M <= 16 MT). KSW: 128-K steps per wave per 16-row tile (a wave's
// K slice is 128 KSW). KW: waves per program, all along K (program K slice 128 KSW KW;
// K / that = SPLIT programs along K, gridDim.y). DEPTH: register ring depth in steps.
// Program (bx, ks) owns 16-row tiles [bx * tpp, min(n_tiles, (bx + 1) * tpp)) of K split ks.
template <int MT, int KSW, int KW, int DEPTH>
__global__ __launch_bounds__(64 * KW) void sk_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, uint16_t* __restrict__ out,
    float* __restrict__ ws,      // SPLIT > 1: fp32 partials [SPLIT][n_tiles][MT][64] v4f32
    int* __restrict__ cnt,       // SPLIT > 1: arrival counters [n_tiles], zero between launches
    int M, int N, int K, int ldx, int ldw, int tpp) {
    static_assert(DEPTH % KSW == 0, "step-in-tile index must be a compile-time constant");
    static_assert(DEPTH >= 2, "ring");
    constexpr int KS = 128 * KSW;
    __shared__ v4f32 red[2][KW > 1 ? (KW - 1) * MT : 1][64];

    const int split = gridDim.y, ks = blockIdx.y;
    const int n_tiles = (N + 15) >> 4;
    const int t_lo = blockIdx.x * tpp;
    const int t_hi = min(n_tiles, t_lo + tpp);
    if (t_lo >= t_hi) return;
    const int S = (t_hi - t_lo) * KSW;

    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int lane = threadIdx.x & 63, r = lane & 15, g = lane >> 4;
    const int k0 = (ks * KW + wave) * KS + 8 * g;
    const bool vec = (N & 3) == 0;

    // Activations first (B operands for the program's whole life), then the ring prologue,
    // so at loop entry the ring is the newest loads in flight (see moe_hip.hip on why the
    // order matters to the waitcnt pass).
    u32x4 xr[KSW][4][MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int m = min(16 * mt + r, M - 1);  // columns >= M are computed, never stored
        const uint16_t* xp = x + (int64_t)m * ldx + k0;
#pragma unroll
        for (int kc = 0; kc < KSW; ++kc)
#pragma unroll
            for (int j = 0; j < 4; ++j) xr[kc][j][mt] = *(const u32x4*)(xp + 128 * kc + 32 * j);
    }
    __builtin_amdgcn_sched_barrier(0);

    auto issue = [&](u32x4 (&t)[4], int s) {
        s = s < S ? s : S - 1;  // past the end: reload the last step (a cache hit), no branch
        const int row = min(16 * (t_lo + s / KSW) + r, N - 1);
        const uint16_t* wp = w + (int64_t)row * ldw + k0 + 128 * (s % KSW);
#pragma unroll
        for (int j = 0; j < 4; ++j) t[j] = *(const u32x4*)(wp + 32 * j);
    };

    u32x4 ring[DEPTH][4];
#pragma unroll
    for (int d = 0; d < DEPTH - 1; ++d) issue(ring[d], d);
    __builtin_amdgcn_sched_barrier(0);

    v4f32 acc[MT];
    for (int s0 = 0; s0 < S; s0 += DEPTH) {
#pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            const int s = s0 + d;
            issue(ring[(d + DEPTH - 1) % DEPTH], s + DEPTH - 1);
            __builtin_amdgcn_sched_barrier(0);  // keep the loads above the MFMA chain
            if (s < S) {
                const int kc = d % KSW;  // == s % KSW since s0 % KSW == 0
                if (kc == 0) {
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) acc[mt] = (v4f32){0.f, 0.f, 0.f, 0.f};
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const u32x4 a = ring[d][j];
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const u32x4 b = xr[kc][j][mt];
                        acc[mt] = sk_mfma((u32x2){a.x, a.y}, (u32x2){b.x, b.y}, acc[mt]);
                        acc[mt] = sk_mfma((u32x2){a.z, a.w}, (u32x2){b.z, b.w}, acc[mt]);
                    }
                }
                if (kc == KSW - 1) {  // tile epilogue: reduce the KW K-slices, store
                    const int tile = t_lo + s / KSW;
                    const int buf = (s / KSW) & 1;
                    if constexpr (KW > 1) {
                        if (wave > 0) {
#pragma unroll
                            for (int mt = 0; mt < MT; ++mt) red[buf][(wave - 1) * MT + mt][lane] = acc[mt];
                        }
                        __syncthreads();
                    }
                    if (wave == 0) {
                        v4f32 v[MT];
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            v[mt] = acc[mt];
#pragma unroll
                            for (int w2 = 1; w2 < KW; ++w2) v[mt] += red[buf][(w2 - 1) * MT + mt][lane];
                        }
                        const int n = 16 * tile + 4 * g;
                        if (split == 1) {
#pragma unroll
                            for (int mt = 0; mt < MT; ++mt) sk_store(out, v[mt], 16 * mt + r, n, M, N, vec);
                        } else {
                            // Deterministic split-K: every split writes its fp32 partial; the
                            // last to arrive sums them in split order (the "last block
                            // reduces" pattern) and resets the counter for the next launch.
                            // Coherence without an L2 writeback: partials and the counter go
                            // through agent-scope (sc1) atomics, which MI300's per-XCD L2s keep
                            // coherent, and `s_waitcnt vmcnt(0)` orders the partial stores
                            // before the arrival (a release fence would add `buffer_wbl2`,
                            // which writes back the whole L2 and cost ~10 us per tile).
                            float* wp = ws + ((int64_t)(ks * n_tiles + tile) * MT) * 256 + 4 * lane;
#pragma unroll
                            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                                for (int i = 0; i < 4; ++i)
                                    __hip_atomic_store(wp + mt * 256 + i, v[mt][i], __ATOMIC_RELAXED,
                                                       __HIP_MEMORY_SCOPE_AGENT);
                            __atomic_signal_fence(__ATOMIC_SEQ_CST);
                            __builtin_amdgcn_s_waitcnt(SK_VMCNT0);
                            __atomic_signal_fence(__ATOMIC_SEQ_CST);
                            int old = 0;
                            if (lane == 0)
                                old = __hip_atomic_fetch_add(cnt + tile, 1, __ATOMIC_RELAXED,
                                                             __HIP_MEMORY_SCOPE_AGENT);
                            old = __builtin_amdgcn_readfirstlane(old);
                            if (old == split - 1) {
                                if (lane == 0)
                                    __hip_atomic_store(cnt + tile, 0, __ATOMIC_RELAXED,
                                                       __HIP_MEMORY_SCOPE_AGENT);
#pragma unroll
                                for (int mt = 0; mt < MT; ++mt) {
                                    v4f32 sum = {0.f, 0.f, 0.f, 0.f};
                                    for (int k2 = 0; k2 < split; ++k2) {
                                        float* rp = ws + ((int64_t)(k2 * n_tiles + tile) * MT + mt) * 256 + 4 * lane;
#pragma unroll
                                        for (int i = 0; i < 4; ++i)
                                            sum[i] += __hip_atomic_load(rp + i, __ATOMIC_RELAXED,
                                                                        __HIP_MEMORY_SCOPE_AGENT);
                                    }
                                    sk_store(out, sum, 16 * mt + r, n, M, N, vec);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Instantiation table. id = config index the host passes (skinny_hip.CONFIGS mirrors it).
#define SK_CONFIGS(X)                                                        \
    X(0, 1, 4, 8, 4) X(1, 1, 4, 4, 4) X(2, 1, 2, 8, 4) X(3, 1, 2, 4, 4)      \
    X(4, 1, 1, 8, 4) X(5, 1, 1, 4, 4) X(6, 1, 2, 2, 4) X(7, 1, 1, 2, 4)      \
    X(8, 2, 2, 8, 4) X(9, 2, 2, 4, 4) X(10, 2, 1, 8, 4) X(11, 2, 1, 4, 4)    \
    X(12, 2, 1, 16, 2) X(13, 2, 2, 16, 2) X(14, 2, 1, 2, 4)                  \
    X(15, 3, 1, 8, 4) X(16, 3, 1, 4, 4) X(17, 3, 1, 8, 2) X(18, 3, 1, 2, 4)  \
    X(19, 4, 1, 8, 4) X(20, 4, 1, 4, 4) X(21, 4, 1, 8, 2) X(22, 4, 1, 2, 4)  \
    X(23, 1, 4, 16, 4) X(24, 1, 2, 16, 4) X(25, 1, 1, 16, 4)

extern "C" int skinny_hip_gemm(int cfg, const void* x, const void* w, void* out, void* ws,
                               void* cnt, int M, int N, int K, int ldx, int ldw, int gx,
                               int split, int tpp, void* stream) {
    switch (cfg) {
#define SK_CASE(id, mt, ksw, kw, depth)                                                         \
    case id:                                                                                    \
        hipLaunchKernelGGL((sk_kernel<mt, ksw, kw, depth>), dim3(gx, split), dim3(64 * kw), 0, \
                           (hipStream_t)stream, (const uint16_t*)x, (const uint16_t*)w,         \
                           (uint16_t*)out, (float*)ws, (int*)cnt, M, N, K, ldx, ldw, tpp);      \
        break;
        SK_CONFIGS(SK_CASE)
#undef SK_CASE
        default:
            return -1;
    }
    return (int)hipGetLastError();
}

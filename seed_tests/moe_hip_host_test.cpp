// Host-side checks of moe_hip.hip's per-lane math, without a GPU (driven by test_moe_hip.py).
//
//   decode <out.bin>   every byte value x every e8m0 scale through moe_make_tab/moe_decode_word;
//                      writes uint32 [256 scales][256 bytes][4 pairs] for the numpy reference.
//   emulate <grid>...  both kernels at the real per-rank shapes, emulated lane by lane: the same
//                      helpers (decode, lane offsets, work split, tile rows), the documented
//                      MFMA 16x16x16 register layout, the same LDS reduction order. Compared
//                      against a float64 dequantize-then-matmul reference. Exit code 0 = pass.
//
// Build: clang++ -std=c++17 -O2 -DMOE_HIP_HOST_TEST -I.. moe_hip_host_test.cpp
// (ext_vector_type needs clang; any host clang works).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../moe_hip.hip"

static const float kFp4[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static int dump_decode(const char* path) {
    std::vector<uint32_t> out(256 * 256 * 4);
    for (int s = 0; s < 256; ++s) {
        const MoeTab tab = moe_make_tab((uint32_t)s);
        for (int b = 0; b < 256; ++b) {
            // four different bytes per word so every byte position sees every value
            uint32_t w = 0;
            for (int i = 0; i < 4; ++i) w |= (uint32_t)((b + 37 * i) & 0xff) << (8 * i);
            moe_decode_word(w, tab, &out[(s * 256 + b) * 4]);
        }
    }
    FILE* f = fopen(path, "wb");
    if (!f) return 2;
    fwrite(out.data(), 4, out.size(), f);
    fclose(f);
    return 0;
}

// ---- emulation ----------------------------------------------------------------------------

constexpr int H = 4096, I = 1024, TOP_K = 10, E = 6, T = 48;
constexpr int GU_PT = 1, GU_TPW = 2 * GU_PT, GU_KW = H / 512, GU_NI = 16 * GU_PT, GU_NT = I / GU_NI;
constexpr int DN_RW = 4, DN_TPW = 1, DN_KW = I / 512, DN_NI = 16 * DN_RW * DN_TPW, DN_NT = H / DN_NI;

static uint32_t rng_state = 12345;
static uint32_t rnd() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}

static double dq(const std::vector<uint8_t>& q, const std::vector<uint8_t>& sc, int row, int k, int K) {
    const uint8_t byte = q[(size_t)row * (K / 2) + k / 2];
    const int nib = (k & 1) ? byte >> 4 : byte & 15;
    const double v = (nib & 8 ? -1.0 : 1.0) * kFp4[nib & 7];
    return v * std::ldexp(1.0, (int)sc[(size_t)row * (K / 32) + k / 32] - 127);
}

// One wave's 16x16 MFMA tile over its 512-K slice: A from 64 lanes' decoded weights, B from
// 64 lanes' split activations, exactly as the kernel's registers hold them.
// acc[lane][i] = D[4 * (lane / 16) + i][lane % 16] (the 16x16x16 output layout).
static void wave_tile(const uint8_t* wq_tile, const uint8_t* ws_tile, int K, int kw,
                      const uint32_t a[64][64], float acc[64][4]) {
    static uint32_t A[64][64];  // [lane][mfma * 2 + half]
    for (int lane = 0; lane < 64; ++lane) {
        const int r = lane & 15, g = lane >> 4;
        const uint32_t* w = (const uint32_t*)(wq_tile + moe_w_off(r, g, kw, K));
        const uint32_t sdw = *(const uint32_t*)(ws_tile + moe_s_off(r, g, kw, K));
        MoeTab tab{};
        for (int q = 0; q < 16; ++q) {
            if (q % 4 == 0) tab = moe_make_tab((sdw >> (8 * (q / 4))) & 0xff);
            uint32_t p[4];
            moe_decode_word(w[q], tab, p);
            A[lane][4 * q + 0] = p[0];
            A[lane][4 * q + 1] = p[1];
            A[lane][4 * q + 2] = p[2];
            A[lane][4 * q + 3] = p[3];
        }
    }
    auto el = [](uint32_t pair0, uint32_t pair1, int jj) -> float {
        const uint32_t p = jj < 2 ? pair0 : pair1;
        return moe_bf2f((uint16_t)(jj & 1 ? p >> 16 : p & 0xffff));
    };
    for (int m = 0; m < 32; ++m) {
        float D[16][16] = {};
        for (int i = 0; i < 16; ++i)
            for (int c = 0; c < 16; ++c) {
                float sum = 0.f;
                for (int g = 0; g < 4; ++g)
                    for (int jj = 0; jj < 4; ++jj) {
                        const int la = i + 16 * g, lb = c + 16 * g;  // A: row + 16 kgrp, B: col + 16 kgrp
                        sum += el(A[la][2 * m], A[la][2 * m + 1], jj) * el(a[lb][2 * m], a[lb][2 * m + 1], jj);
                    }
                D[i][c] = sum;
            }
        for (int lane = 0; lane < 64; ++lane)
            for (int i = 0; i < 4; ++i) acc[lane][i] += D[4 * (lane >> 4) + i][lane & 15];
    }
}

struct Units {
    std::vector<int> sorted, unit;
    int n_units = 0;
};

// Same contract as mxfp4_gemv._bw_prep_kernel: live assignments grouped by local expert,
// each expert's run cut into <= 16-token units (expert, first position, count).
static Units make_units(const std::vector<int>& a_expert, const std::vector<uint16_t>& a_weight) {
    Units u;
    std::vector<std::vector<int>> by(E);
    for (int a = 0; a < (int)a_expert.size(); ++a)
        if (a_expert[a] >= 0 && a_expert[a] < E && a_weight[a] != 0) by[a_expert[a]].push_back(a);
    for (int e = 0; e < E; ++e) {
        const int start = (int)u.sorted.size();
        for (int a : by[e]) u.sorted.push_back(a);
        for (int j = 0; j * 16 < (int)by[e].size(); ++j) {
            u.unit.push_back(e);
            u.unit.push_back(start + 16 * j);
            u.unit.push_back(std::min(16, (int)by[e].size() - 16 * j));
            ++u.n_units;
        }
    }
    return u;
}

// Emulates moe_kernel<...> for every program of a `grid`-program launch.
template <bool GATE_UP>
static void emulate(int grid, const Units& un, const uint16_t* act, const std::vector<uint8_t>& wq,
                    const std::vector<uint8_t>& ws, const std::vector<uint16_t>& a_weight,
                    std::vector<uint16_t>& out) {
    constexpr int K = GATE_UP ? H : I, OUT_ROWS = GATE_UP ? I : H, W_ROWS = GATE_UP ? 2 * I : H;
    constexpr int KW = GATE_UP ? GU_KW : DN_KW, RW = GATE_UP ? 1 : DN_RW;
    constexpr int TPW = GATE_UP ? GU_TPW : DN_TPW, PT = TPW / 2, NI = GATE_UP ? GU_NI : DN_NI;
    constexpr int NT = OUT_ROWS / NI, WAVES = KW * RW;
    static uint32_t a[WAVES][64][64];
    static float acc[WAVES][TPW][64][4];
    for (int b = 0; b < grid; ++b) {
        for (int seg = 0;; ++seg) {
            int u, it_lo, it_hi;
            if (!moe_segment(b, grid, un.n_units, NT, seg, &u, &it_lo, &it_hi)) break;
            const int e = un.unit[3 * u], p0 = un.unit[3 * u + 1], cnt = un.unit[3 * u + 2];
            for (int wave = 0; wave < WAVES; ++wave) {
                const int kw = wave % KW;
                for (int lane = 0; lane < 64; ++lane) {
                    const int r = lane & 15, g = lane >> 4;
                    const int slot = un.sorted[p0 + (r < cnt ? r : 0)];
                    const int64_t arow = GATE_UP ? slot / TOP_K : slot;
                    const u32x4* ap = (const u32x4*)(act + arow * K + moe_a_off(g, kw));
                    for (int q = 0; q < 16; ++q) {
                        u32x4 d;
                        memcpy(&d, ap + q, 16);
                        moe_split_act(d, &a[wave][lane][4 * q]);
                    }
                }
            }
            for (int item = it_lo; item < it_hi; ++item) {
                const int n0 = item * NI;
                for (int wave = 0; wave < WAVES; ++wave) {
                    const int kw = wave % KW, rw = wave / KW;
                    for (int j = 0; j < TPW; ++j) {
                        memset(acc[wave][j], 0, sizeof(acc[wave][j]));
                        const int64_t row = (int64_t)e * W_ROWS + moe_tile_row(GATE_UP, OUT_ROWS, PT, TPW, rw, n0, j);
                        wave_tile(&wq[row * (K / 2)], &ws[row * (K / 32)], K, kw, a[wave], acc[wave][j]);
                    }
                }
                for (int lane = 0; lane < 64; ++lane) {
                    const int r = lane & 15, g = lane >> 4;
                    if (r >= cnt) continue;
                    const int slot = un.sorted[p0 + r];
                    if (GATE_UP) {
                        for (int pt = 0; pt < PT; ++pt)
                            for (int i = 0; i < 4; ++i) {
                                float gs = acc[0][pt][lane][i], us = acc[0][PT + pt][lane][i];
                                for (int w = 1; w < WAVES; ++w) gs += acc[w][pt][lane][i], us += acc[w][PT + pt][lane][i];
                                out[(size_t)slot * OUT_ROWS + n0 + 16 * pt + 4 * g + i] =
                                    moe_f2bf(gs / (1.0f + expf(-gs)) * us);
                            }
                    } else {
                        const float cw = moe_bf2f(a_weight[slot]);  // bf16-valued, as the kernel's fp32 input here
                        for (int rw = 0; rw < RW; ++rw)
                            for (int j = 0; j < TPW; ++j)
                                for (int i = 0; i < 4; ++i) {
                                    float v = acc[rw * KW][j][lane][i];
                                    for (int k2 = 1; k2 < KW; ++k2) v += acc[rw * KW + k2][j][lane][i];
                                    out[(size_t)slot * OUT_ROWS + n0 + 16 * (rw * TPW + j) + 4 * g + i] = moe_f2bf(v * cw);
                                }
                    }
                }
            }
        }
    }
}

static int run_emulation(const std::vector<int>& grids) {
    // No live local assignment (n_units 0): every program exits, no division by n_units.
    for (int grid : grids)
        for (int b = 0; b < grid; ++b) {
            int u, it_lo, it_hi;
            if (moe_segment(b, grid, 0, 64, 0, &u, &it_lo, &it_hi)) {
                printf("grid %d: program %d got a segment with n_units 0 -> FAIL\n", grid, b);
                return 1;
            }
        }
    const int A = T * TOP_K;
    std::vector<uint8_t> gq((size_t)E * 2 * I * H / 2), gs((size_t)E * 2 * I * H / 32);
    std::vector<uint8_t> dq_((size_t)E * H * I / 2), ds((size_t)E * H * I / 32);
    for (auto& v : gq) v = rnd() & 0xff;
    for (auto& v : dq_) v = rnd() & 0xff;
    for (auto& v : gs) v = 118 + rnd() % 5;  // near 2^-7, as seed_tests.random_experts
    for (auto& v : ds) v = 118 + rnd() % 5;
    std::vector<uint16_t> x((size_t)T * H);
    for (auto& v : x) v = moe_f2bf(((int)(rnd() % 2001) - 1000) / 1000.0f);
    // Routing: a skewed pile-up so expert 0 gets > 16 tokens (two units), some rows dropped
    // (other ranks' experts = -1 here, zero weight rows), the rest spread.
    std::vector<int> a_expert(A);
    std::vector<uint16_t> a_weight(A);
    for (int a = 0; a < A; ++a) {
        const uint32_t z = rnd() % 100;
        a_expert[a] = z < 8 ? 0 : z < 70 ? -1 : (int)(1 + rnd() % (E - 1));
        a_weight[a] = (rnd() % 20 == 0) ? 0 : moe_f2bf(0.05f + (rnd() % 100) / 400.0f);
    }
    const Units un = make_units(a_expert, a_weight);
    printf("units %d, live assignments %zu\n", un.n_units, un.sorted.size());

    // float64 reference: inter rounded to bf16 like the kernel, then down and routing weight
    std::vector<double> ref_y((size_t)A * H, 0.0);
    std::vector<uint16_t> ref_inter((size_t)A * I);
    std::vector<float> wg((size_t)2 * I * H), wd((size_t)H * I);
    for (int e = 0; e < E; ++e) {
        for (int n = 0; n < 2 * I; ++n)
            for (int k = 0; k < H; ++k) wg[(size_t)n * H + k] = (float)dq(gq, gs, e * 2 * I + n, k, H);
        for (int n = 0; n < H; ++n)
            for (int k = 0; k < I; ++k) wd[(size_t)n * I + k] = (float)dq(dq_, ds, e * H + n, k, I);
        for (int a : un.sorted) {
            if (a_expert[a] != e) continue;
            const int t = a / TOP_K;
            for (int n = 0; n < I; ++n) {
                double gg = 0, uu = 0;
                for (int k = 0; k < H; ++k) {
                    const double xv = moe_bf2f(x[(size_t)t * H + k]);
                    gg += wg[(size_t)n * H + k] * xv;
                    uu += wg[(size_t)(I + n) * H + k] * xv;
                }
                ref_inter[(size_t)a * I + n] = moe_f2bf((float)(gg / (1.0 + std::exp(-gg)) * uu));
            }
            for (int n = 0; n < H; ++n) {
                double y = 0;
                for (int k = 0; k < I; ++k) y += wd[(size_t)n * I + k] * (double)moe_bf2f(ref_inter[(size_t)a * I + k]);
                ref_y[(size_t)a * H + n] = y * moe_bf2f(a_weight[a]);
            }
        }
    }

    int fails = 0;
    for (int grid : grids) {
        std::vector<uint16_t> inter((size_t)A * I, 0xffff), y((size_t)A * H, 0xffff);  // NaN fill
        emulate<true>(grid, un, x.data(), gq, gs, a_weight, inter);
        emulate<false>(grid, un, inter.data(), dq_, ds, a_weight, y);
        double max_i = 0, scale_i = 0, max_y = 0, scale_y = 0;
        size_t unwritten = 0;
        for (int a : un.sorted) {
            for (int n = 0; n < I; ++n) {
                const uint16_t v = inter[(size_t)a * I + n];
                if (v == 0xffff) ++unwritten;
                max_i = std::max(max_i, (double)std::fabs(moe_bf2f(v) - moe_bf2f(ref_inter[(size_t)a * I + n])));
                scale_i = std::max(scale_i, (double)std::fabs(moe_bf2f(ref_inter[(size_t)a * I + n])));
            }
            for (int n = 0; n < H; ++n) {
                const uint16_t v = y[(size_t)a * H + n];
                if (v == 0xffff) ++unwritten;
                max_y = std::max(max_y, std::fabs(moe_bf2f(v) - ref_y[(size_t)a * H + n]));
                scale_y = std::max(scale_y, std::fabs(ref_y[(size_t)a * H + n]));
            }
        }
        // rows of dropped assignments must be untouched
        size_t stray = 0;
        std::vector<char> live(A, 0);
        for (int a : un.sorted) live[a] = 1;
        for (int a = 0; a < A; ++a)
            if (!live[a]) {
                for (int n = 0; n < I; ++n) stray += inter[(size_t)a * I + n] != 0xffff;
                for (int n = 0; n < H; ++n) stray += y[(size_t)a * H + n] != 0xffff;
            }
        const double ri = max_i / scale_i, ry = max_y / scale_y;
        const bool ok = unwritten == 0 && stray == 0 && ri < 2e-2 && ry < 2e-2;
        printf("grid %4d: inter max rel err %.3e, y max rel err %.3e, unwritten %zu, stray %zu -> %s\n",
               grid, ri, ry, unwritten, stray, ok ? "ok" : "FAIL");
        fails += !ok;
    }
    return fails ? 1 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && !strcmp(argv[1], "decode")) return dump_decode(argv[2]);
    if (argc >= 2 && !strcmp(argv[1], "emulate")) {
        std::vector<int> grids;
        for (int i = 2; i < argc; ++i) grids.push_back(atoi(argv[i]));
        if (grids.empty()) grids = {228};
        return run_emulation(grids);
    }
    fprintf(stderr, "usage: %s decode <out.bin> | emulate [grid...]\n", argv[0]);
    return 2;
}

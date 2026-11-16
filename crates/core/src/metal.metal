// Matrix-vector kernels over GGUF-encoded weights, read in place from the mapped model file.
// One simdgroup computes one output row; each lane handles 32-element chunks of that row.
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong offset;     // byte offset of row 0 in the weight buffer
    uint rows;
    uint cols;
    uint row_bytes;
};

constant uint ROWS_PER_TG = 4;

inline float h(device const uchar* p) { return float(*(device const half*)p); }

inline float chunk_f32(device const uchar* r, uint c, device const float* x) {
    device const float* w = (device const float*)r + c * 32;
    float s = 0;
    for (uint i = 0; i < 32; i++) s += w[i] * x[i];
    return s;
}

inline float chunk_f16(device const uchar* r, uint c, device const float* x) {
    device const half* w = (device const half*)r + c * 32;
    float s = 0;
    for (uint i = 0; i < 32; i++) s += float(w[i]) * x[i];
    return s;
}

inline float chunk_q4_0(device const uchar* r, uint c, device const float* x) {
    device const uchar* b = r + c * 18;
    float s = 0;
    for (uint j = 0; j < 16; j++) {
        uchar q = b[2 + j];
        s += float(int(q & 15) - 8) * x[j] + float(int(q >> 4) - 8) * x[j + 16];
    }
    return h(b) * s;
}

inline float chunk_q5_0(device const uchar* r, uint c, device const float* x) {
    device const uchar* b = r + c * 22;
    uint qh = uint(b[2]) | (uint(b[3]) << 8) | (uint(b[4]) << 16) | (uint(b[5]) << 24);
    float s = 0;
    for (uint j = 0; j < 16; j++) {
        uchar q = b[6 + j];
        int lo = int((q & 15) | (((qh >> j) << 4) & 0x10)) - 16;
        int hi = int((q >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
        s += float(lo) * x[j] + float(hi) * x[j + 16];
    }
    return h(b) * s;
}

inline float chunk_q8_0(device const uchar* r, uint c, device const float* x) {
    device const uchar* b = r + c * 34;
    device const char* q = (device const char*)(b + 2);
    float s = 0;
    for (uint i = 0; i < 32; i++) s += float(q[i]) * x[i];
    return h(b) * s;
}

inline void scale_min_k4(uint j, device const uchar* q, thread float& sc, thread float& m) {
    if (j < 4) {
        sc = float(q[j] & 63);
        m = float(q[j + 4] & 63);
    } else {
        sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// K-quants: chunk c is sub-block c % 8 of super-block c / 8.
inline float chunk_q4_k(device const uchar* r, uint c, device const float* x) {
    uint sub = c % 8;
    device const uchar* b = r + (c / 8) * 144;
    float sc, m;
    scale_min_k4(sub, b + 4, sc, m);
    device const uchar* q = b + 16 + 32 * (sub / 2);
    float s = 0, sx = 0;
    for (uint l = 0; l < 32; l++) {
        uint v = (sub & 1) ? (q[l] >> 4) : (q[l] & 15);
        s += float(v) * x[l];
        sx += x[l];
    }
    return h(b) * sc * s - h(b + 2) * m * sx;
}

inline float chunk_q5_k(device const uchar* r, uint c, device const float* x) {
    uint sub = c % 8;
    device const uchar* b = r + (c / 8) * 176;
    float sc, m;
    scale_min_k4(sub, b + 4, sc, m);
    device const uchar* qh = b + 16;
    device const uchar* q = b + 48 + 32 * (sub / 2);
    float s = 0, sx = 0;
    for (uint l = 0; l < 32; l++) {
        uint v = ((sub & 1) ? (q[l] >> 4) : (q[l] & 15)) + (((qh[l] >> sub) & 1) << 4);
        s += float(v) * x[l];
        sx += x[l];
    }
    return h(b) * sc * s - h(b + 2) * m * sx;
}

inline float chunk_q6_k(device const uchar* r, uint c, device const float* x) {
    uint j = c % 8, n = j / 4, k = j % 4;
    device const uchar* b = r + (c / 8) * 210;
    device const uchar* ql = b + 64 * n + 32 * (k % 2);
    device const uchar* qh = b + 128 + 32 * n;
    device const char* sc = (device const char*)(b + 192) + 8 * n + 2 * k;
    float s0 = 0, s1 = 0;
    for (uint l = 0; l < 32; l++) {
        uint lo = k < 2 ? (ql[l] & 15) : (ql[l] >> 4);
        int q = int(lo | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
        if (l < 16) s0 += float(q) * x[l]; else s1 += float(q) * x[l];
    }
    return h(b + 208) * (float(sc[0]) * s0 + float(sc[1]) * s1);
}

#define MATVEC(NAME, CHUNK)                                                                   \
kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],   \
                 device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],            \
                 uint tg [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                  \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    uint row = tg * ROWS_PER_TG + sg;                                                         \
    if (row >= p.rows) return;                                                                \
    device const uchar* r = w + p.offset + ulong(row) * p.row_bytes;                          \
    float acc = 0;                                                                            \
    for (uint c = lane; c < p.cols / 32; c += 32) acc += CHUNK(r, c, x + c * 32);             \
    acc = simd_sum(acc);                                                                      \
    if (lane == 0) y[row] = acc;                                                              \
}

MATVEC(matvec_f32, chunk_f32)
MATVEC(matvec_f16, chunk_f16)
MATVEC(matvec_q4_0, chunk_q4_0)
MATVEC(matvec_q5_0, chunk_q5_0)
MATVEC(matvec_q8_0, chunk_q8_0)
MATVEC(matvec_q4_k, chunk_q4_k)
MATVEC(matvec_q5_k, chunk_q5_k)
MATVEC(matvec_q6_k, chunk_q6_k)

// MoE decode without a CPU round trip: `MATVEC` for slot `tg.y` of the k chosen experts, whose
// matrix is expert ids[slot] of a stacked tensor (`expert_stride` bytes apart). Slot inputs and
// outputs are `x_stride` / `y_stride` floats apart (x_stride 0: every slot reads the same x).
struct IdParams {
    ulong offset;
    uint rows;
    uint cols;
    uint row_bytes;
    ulong expert_stride;
    uint x_stride;
    uint y_stride;
};

#define MATVEC_ID(NAME, CHUNK)                                                                \
kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],   \
                 device float* y [[buffer(2)]], constant IdParams& p [[buffer(3)]],          \
                 device const uint* ids [[buffer(4)]],                                        \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint sg [[simdgroup_index_in_threadgroup]],                                  \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    uint row = tg.x * ROWS_PER_TG + sg;                                                       \
    if (row >= p.rows) return;                                                                \
    uint slot = tg.y;                                                                         \
    device const uchar* r =                                                                   \
        w + p.offset + ulong(ids[slot]) * p.expert_stride + ulong(row) * p.row_bytes;         \
    device const float* xs = x + slot * p.x_stride;                                           \
    float acc = 0;                                                                            \
    for (uint c = lane; c < p.cols / 32; c += 32) acc += CHUNK(r, c, xs + c * 32);            \
    acc = simd_sum(acc);                                                                      \
    if (lane == 0) y[slot * p.y_stride + row] = acc;                                          \
}

MATVEC_ID(matvec_id_f32, chunk_f32)
MATVEC_ID(matvec_id_f16, chunk_f16)
MATVEC_ID(matvec_id_q4_0, chunk_q4_0)
MATVEC_ID(matvec_id_q5_0, chunk_q5_0)
MATVEC_ID(matvec_id_q8_0, chunk_q8_0)
MATVEC_ID(matvec_id_q4_k, chunk_q4_k)
MATVEC_ID(matvec_id_q5_k, chunk_q5_k)
MATVEC_ID(matvec_id_q6_k, chunk_q6_k)

// ---- Batched matmul for prompt processing: Y[n][rows] = X[n][cols] . W^T ----

inline void deq_f32(device const uchar* r, uint c, thread float* w) {
    device const float* s = (device const float*)r + c * 32;
    for (uint i = 0; i < 32; i++) w[i] = s[i];
}

inline void deq_f16(device const uchar* r, uint c, thread float* w) {
    device const half* s = (device const half*)r + c * 32;
    for (uint i = 0; i < 32; i++) w[i] = float(s[i]);
}

inline void deq_q4_0(device const uchar* r, uint c, thread float* w) {
    device const uchar* b = r + c * 18;
    float d = h(b);
    for (uint j = 0; j < 16; j++) {
        uchar q = b[2 + j];
        w[j] = d * float(int(q & 15) - 8);
        w[j + 16] = d * float(int(q >> 4) - 8);
    }
}

inline void deq_q5_0(device const uchar* r, uint c, thread float* w) {
    device const uchar* b = r + c * 22;
    float d = h(b);
    uint qh = uint(b[2]) | (uint(b[3]) << 8) | (uint(b[4]) << 16) | (uint(b[5]) << 24);
    for (uint j = 0; j < 16; j++) {
        uchar q = b[6 + j];
        w[j] = d * float(int((q & 15) | (((qh >> j) << 4) & 0x10)) - 16);
        w[j + 16] = d * float(int((q >> 4) | ((qh >> (j + 12)) & 0x10)) - 16);
    }
}

inline void deq_q8_0(device const uchar* r, uint c, thread float* w) {
    device const uchar* b = r + c * 34;
    device const char* q = (device const char*)(b + 2);
    float d = h(b);
    for (uint i = 0; i < 32; i++) w[i] = d * float(q[i]);
}

inline void deq_q4_k(device const uchar* r, uint c, thread float* w) {
    uint sub = c % 8;
    device const uchar* b = r + (c / 8) * 144;
    float sc, m;
    scale_min_k4(sub, b + 4, sc, m);
    float d = h(b) * sc, dm = h(b + 2) * m;
    device const uchar* q = b + 16 + 32 * (sub / 2);
    for (uint l = 0; l < 32; l++) w[l] = d * float((sub & 1) ? (q[l] >> 4) : (q[l] & 15)) - dm;
}

inline void deq_q5_k(device const uchar* r, uint c, thread float* w) {
    uint sub = c % 8;
    device const uchar* b = r + (c / 8) * 176;
    float sc, m;
    scale_min_k4(sub, b + 4, sc, m);
    float d = h(b) * sc, dm = h(b + 2) * m;
    device const uchar* qh = b + 16;
    device const uchar* q = b + 48 + 32 * (sub / 2);
    for (uint l = 0; l < 32; l++) {
        uint v = ((sub & 1) ? (q[l] >> 4) : (q[l] & 15)) + (((qh[l] >> sub) & 1) << 4);
        w[l] = d * float(v) - dm;
    }
}

inline void deq_q6_k(device const uchar* r, uint c, thread float* w) {
    uint j = c % 8, n = j / 4, k = j % 4;
    device const uchar* b = r + (c / 8) * 210;
    device const uchar* ql = b + 64 * n + 32 * (k % 2);
    device const uchar* qh = b + 128 + 32 * n;
    device const char* sc = (device const char*)(b + 192) + 8 * n + 2 * k;
    float d = h(b + 208);
    for (uint l = 0; l < 32; l++) {
        uint lo = k < 2 ? (ql[l] & 15) : (ql[l] >> 4);
        int q = int(lo | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
        w[l] = d * float(sc[l / 16]) * float(q);
    }
}

// Half of a 32-weight chunk (o = 0 or 16) as four float4, for the matmul's tile loads. The
// K-quant ones read 4 bytes at a time: one byte per load was a sixth of a prompt's matmul time.
inline uint4 bytes4(device const uchar* p) { return uint4(*(device const packed_uchar4*)p); }

inline void deq16_q4_k(device const uchar* r, uint c, uint o, thread float4* w) {
    uint sub = c % 8;
    device const uchar* b = r + (c / 8) * 144;
    float sc, m;
    scale_min_k4(sub, b + 4, sc, m);
    float d = h(b) * sc, dm = h(b + 2) * m;
    device const uchar* q = b + 16 + 32 * (sub / 2) + o;
    uint shift = 4 * (sub & 1);
    for (uint i = 0; i < 4; i++) w[i] = d * float4((bytes4(q + 4 * i) >> shift) & 15) - dm;
}

inline void deq16_q6_k(device const uchar* r, uint c, uint o, thread float4* w) {
    uint j = c % 8, n = j / 4, k = j % 4;
    device const uchar* b = r + (c / 8) * 210;
    device const uchar* ql = b + 64 * n + 32 * (k % 2) + o;
    device const uchar* qh = b + 128 + 32 * n + o;
    device const char* sc = (device const char*)(b + 192) + 8 * n + 2 * k;
    float d = h(b + 208) * float(sc[o / 16]);
    uint shift = 4 * (k / 2);
    for (uint i = 0; i < 4; i++) {
        uint4 lo = (bytes4(ql + 4 * i) >> shift) & 15;
        uint4 hi = (bytes4(qh + 4 * i) >> (2 * k)) & 3;
        w[i] = d * float4(int4(lo | (hi << 4)) - 32);
    }
}

#define DEQ16_VIA32(NAME, DEQ)                                                                \
inline void NAME(device const uchar* r, uint c, uint o, thread float4* w) {                   \
    float t[32];                                                                              \
    DEQ(r, c, t);                                                                             \
    for (uint i = 0; i < 4; i++) w[i] = float4(t[o + 4 * i], t[o + 4 * i + 1], t[o + 4 * i + 2], t[o + 4 * i + 3]); \
}
DEQ16_VIA32(deq16_f32, deq_f32)
DEQ16_VIA32(deq16_f16, deq_f16)
DEQ16_VIA32(deq16_q4_0, deq_q4_0)
DEQ16_VIA32(deq16_q5_0, deq_q5_0)
DEQ16_VIA32(deq16_q8_0, deq_q8_0)
DEQ16_VIA32(deq16_q5_k, deq_q5_k)

// Tiled matmul on the simdgroup 8x8 matrix units. A threadgroup (4 simdgroups) computes a
// tile of MM_R rows x MT tokens, MT a compile-time 8, 16, 32 or 64: a tile costs its weight
// decode whatever it holds, so a short prompt takes the smallest that fits and a long one 64
// (a 64-token tile for 8 tokens ran at a quarter of the 8-token one's speed). Per 64-column
// step every thread decodes 32 weights and reads MT / 2 activations into registers, stores
// them as 8x8 blocks in threadgroup memory ([row][k] and [token][k], half unless the batch is
// exact), then each simdgroup multiplies AB x BB blocks of it: 32 rows x half the tokens from
// 32 tokens up, 16 rows x all the tokens below. Everything a loop depends on is a constant:
// with run-time bounds the matrices are kept in memory.
constant uint MM_R = 64;

struct MatmulParams {
    ulong offset;
    uint rows;
    uint cols;
    uint row_bytes;
    uint tok0;        // first token of this dispatch
    uint n;           // tokens in x
};

#define MATMUL(NAME, DEQ, T, M, MT, AB, BB)                                                   \
kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],   \
                 device float* y [[buffer(2)]], constant MatmulParams& p [[buffer(3)]],      \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint tid [[thread_index_in_threadgroup]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]]) {                                \
    const uint TB = MT / 8, XU = MT / 8, SGR = MM_R / (8 * AB);                               \
    threadgroup T sa[(MM_R + MT) * 64];                                                       \
    threadgroup T* sb = sa + MM_R * 64;                                                       \
    threadgroup float* sm = (threadgroup float*)sa; /* an edge tile's output */               \
    uint r0 = tg.x * MM_R, t0 = p.tok0 + tg.y * MT;                                           \
    uint rb = (sg % SGR) * AB, tb = (sg / SGR) * BB; /* in blocks of 8 */                     \
    uint wr = tid / 2, wo = (tid % 2) * 16;                                                   \
    uint xt = tid * XU / 16, xo = 4 * (tid * XU % 16);                                        \
    bool row_ok = r0 + wr < p.rows, tok_ok = t0 + xt < p.n;                                   \
    device const uchar* wrow = w + p.offset + ulong(row_ok ? r0 + wr : 0) * p.row_bytes;      \
    device const float* xrow = x + ulong(tok_ok ? t0 + xt : 0) * p.cols + xo;                 \
    simdgroup_float8x8 acc[AB * BB];                                                          \
    for (uint i = 0; i < AB * BB; i++) acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); \
    for (uint c = 0; c < p.cols / 32; c += 2) {                                               \
        /* A step is two 32-column chunks; the last has one when cols is an odd multiple. */ \
        bool two = c + 1 < p.cols / 32;                                                       \
        float4 wv[8], xv[XU];                                                                 \
        DEQ(wrow, c, wo, wv);                                                                 \
        if (two) {                                                                            \
            DEQ(wrow, c + 1, wo, wv + 4);                                                     \
        } else {                                                                              \
            for (uint i = 4; i < 8; i++) wv[i] = float4(0.0f);                                \
        }                                                                                     \
        for (uint i = 0; i < XU; i++) {                                                       \
            bool in = two || xo + 4 * i < 32;                                                 \
            xv[i] = in ? float4(*(device const packed_float4*)(xrow + c * 32 + 4 * i))        \
                       : float4(0.0f);                                                        \
        }                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
        for (uint i = 0; i < 8; i++) {                                                        \
            uint k = (i / 4) * 32 + wo + (i % 4) * 4;                                         \
            *(threadgroup vec<T, 4>*)(sa + ((k / 8) * 8 + wr / 8) * 64 + (wr % 8) * 8 + k % 8) = \
                row_ok ? vec<T, 4>(wv[i]) : vec<T, 4>(T(0));                                  \
        }                                                                                     \
        for (uint i = 0; i < XU; i++) {                                                       \
            uint k = xo + 4 * i;                                                              \
            *(threadgroup vec<T, 4>*)(sb + ((k / 8) * TB + xt / 8) * 64 + (xt % 8) * 8 + k % 8) = \
                tok_ok ? vec<T, 4>(xv[i]) : vec<T, 4>(T(0));                                  \
        }                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
        for (uint k = 0; k < 8; k++) {                                                        \
            M a[AB], b[BB];                                                                   \
            simdgroup_barrier(mem_flags::mem_none);                                           \
            for (uint i = 0; i < AB; i++)                                                     \
                simdgroup_load(a[i], sa + (k * 8 + rb + i) * 64, 8, ulong2(0, 0), true);      \
            simdgroup_barrier(mem_flags::mem_none);                                           \
            for (uint j = 0; j < BB; j++) simdgroup_load(b[j], sb + (k * TB + tb + j) * 64, 8); \
            simdgroup_barrier(mem_flags::mem_none);                                           \
            for (uint j = 0; j < BB; j++)                                                     \
                for (uint i = 0; i < AB; i++)                                                 \
                    simdgroup_multiply_accumulate(acc[j * AB + i], b[j], a[i], acc[j * AB + i]); \
        }                                                                                     \
    }                                                                                         \
    if (r0 + MM_R <= p.rows && t0 + MT <= p.n) {                                              \
        device float* yp = y + ulong(t0 + 8 * tb) * p.rows + r0 + 8 * rb;                     \
        for (uint j = 0; j < BB; j++)                                                         \
            for (uint i = 0; i < AB; i++)                                                     \
                simdgroup_store(acc[j * AB + i], yp + ulong(8 * j) * p.rows + 8 * i, p.rows); \
        return;                                                                               \
    }                                                                                         \
    /* An edge tile: stage it as [token][row], then write y[token][row] within bounds. */     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
    for (uint j = 0; j < BB; j++)                                                             \
        for (uint i = 0; i < AB; i++)                                                         \
            simdgroup_store(acc[j * AB + i], sm + 8 * (tb + j) * MM_R + 8 * (rb + i), MM_R);  \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
    for (uint e = tid; e < MM_R * MT; e += 128) {                                             \
        uint tt = e / MM_R, rr = e % MM_R;                                                    \
        uint row = r0 + rr, t = t0 + tt;                                                      \
        if (row < p.rows && t < p.n) y[ulong(t) * p.rows + row] = sm[tt * MM_R + rr];         \
    }                                                                                         \
}

// matmul_<type>_t<tokens> with f32 tiles, matmul_h_<type>_t<tokens> with half ones.
#define MATMUL_TILES(NAME, DEQ, T, M)                                                         \
MATMUL(NAME##_t8, DEQ, T, M, 8, 2, 1)                                                         \
MATMUL(NAME##_t16, DEQ, T, M, 16, 2, 2)                                                       \
MATMUL(NAME##_t32, DEQ, T, M, 32, 4, 2)                                                       \
MATMUL(NAME##_t64, DEQ, T, M, 64, 4, 4)

MATMUL_TILES(matmul_f32, deq16_f32, float, simdgroup_float8x8)
MATMUL_TILES(matmul_f16, deq16_f16, float, simdgroup_float8x8)
MATMUL_TILES(matmul_q4_0, deq16_q4_0, float, simdgroup_float8x8)
MATMUL_TILES(matmul_q5_0, deq16_q5_0, float, simdgroup_float8x8)
MATMUL_TILES(matmul_q8_0, deq16_q8_0, float, simdgroup_float8x8)
MATMUL_TILES(matmul_q4_k, deq16_q4_k, float, simdgroup_float8x8)
MATMUL_TILES(matmul_q5_k, deq16_q5_k, float, simdgroup_float8x8)
MATMUL_TILES(matmul_q6_k, deq16_q6_k, float, simdgroup_float8x8)
MATMUL_TILES(matmul_h_q4_0, deq16_q4_0, half, simdgroup_half8x8)
MATMUL_TILES(matmul_h_q5_0, deq16_q5_0, half, simdgroup_half8x8)
MATMUL_TILES(matmul_h_q8_0, deq16_q8_0, half, simdgroup_half8x8)
MATMUL_TILES(matmul_h_q4_k, deq16_q4_k, half, simdgroup_half8x8)
MATMUL_TILES(matmul_h_q5_k, deq16_q5_k, half, simdgroup_half8x8)
MATMUL_TILES(matmul_h_q6_k, deq16_q6_k, half, simdgroup_half8x8)

// ---- Elementwise / normalization / attention kernels for running whole layers on the GPU ----

inline float tg_reduce_sum(float v, threadgroup float* red, uint lane, uint sg, uint n_sg) {
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < n_sg ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float tg_reduce_max(float v, threadgroup float* red, uint lane, uint sg, uint n_sg) {
    v = simd_max(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float t = lane < n_sg ? red[lane] : -INFINITY;
        t = simd_max(t);
        if (lane == 0) red[0] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

struct NormParams { uint n; float eps; };

// One threadgroup per row of `n`; `out` may alias `x`.
kernel void rmsnorm(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                    device const float* w [[buffer(2)]], constant NormParams& p [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                    uint tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    device const float* xr = x + row * p.n;
    device float* o = out + row * p.n;
    float s = 0;
    for (uint i = tid; i < p.n; i += tgs) s += xr[i] * xr[i];
    s = tg_reduce_sum(s, red, lane, sg, (tgs + 31) / 32);
    float scale = 1.0f / sqrt(s / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += tgs) o[i] = xr[i] * scale * w[i];
}

struct EltParams { uint n; float alpha; uint m; };

kernel void add(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) a[i] += b[i];
}

// a[i] += b[i % m]: adds a length-m vector (a bias) to each of n/m rows.
kernel void add_bcast(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                      constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) a[i] += b[i % p.m];
}

// MoE routing: dst row r = src row idx[r] (p.n = rows * dim, p.m = dim).
kernel void gather_rows(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                        device const uint* idx [[buffer(2)]], constant EltParams& p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = src[ulong(idx[i / p.m]) * p.m + i % p.m];
}

// MoE routing for one token: softmax over the router logits, the top k (ties go to the lower
// index), weights optionally renormalized over the k; with a shared expert, w[k] is its
// sigmoid gate. One simdgroup: lane l scans experts l, l + 32, ... (`taken` bit j = its j-th),
// and each of the k rounds picks the best remaining expert with simd reductions.
struct TopkParams { uint n_expert; uint k; uint norm; uint shared; };

kernel void moe_topk(device const float* logits [[buffer(0)]],
                     device const float* shared_gate [[buffer(1)]],
                     device uint* ids [[buffer(2)]], device float* w [[buffer(3)]],
                     constant TopkParams& p [[buffer(4)]],
                     uint lane [[thread_index_in_simdgroup]]) {
    float mx = -INFINITY;
    for (uint e = lane; e < p.n_expert; e += 32) mx = max(mx, logits[e]);
    mx = simd_max(mx);
    float sum = 0;
    for (uint e = lane; e < p.n_expert; e += 32) sum += exp(logits[e] - mx);
    sum = simd_sum(sum);
    ulong taken = 0;
    float chosen = 0;
    for (uint i = 0; i < p.k; i++) {
        float bv = -INFINITY;
        uint be = UINT_MAX;
        for (uint e = lane, j = 0; e < p.n_expert; e += 32, j++) {
            if (!((taken >> j) & 1) && logits[e] > bv) {
                bv = logits[e];
                be = e;
            }
        }
        float best = simd_max(bv);
        uint win = simd_min(bv == best ? be : UINT_MAX);
        if (win == be) taken |= 1ul << (win / 32);
        float wi = exp(best - mx) / sum;
        chosen += wi;
        if (lane == 0) {
            ids[i] = win;
            w[i] = wi;
        }
    }
    if (lane == 0) {
        if (p.norm) for (uint i = 0; i < p.k; i++) w[i] /= chosen;
        if (p.shared) w[p.k] = 1.0f / (1.0f + exp(-shared_gate[0]));
    }
}

// x += sum over r < m of w[r] * src row r (rows of n): the chosen experts' outputs, combined.
struct CombineParams { uint n; uint m; };

kernel void moe_combine(device float* x [[buffer(0)]], device const float* src [[buffer(1)]],
                        device const float* w [[buffer(2)]], constant CombineParams& p [[buffer(3)]],
                        uint j [[thread_position_in_grid]]) {
    if (j >= p.n) return;
    float s = 0;
    for (uint r = 0; r < p.m; r++) s += w[r] * src[r * p.n + j];
    x[j] += s;
}

// MoE combine: dst row idx[r] += w[r] * src row r. Indices within one call must be distinct.
kernel void scatter_axpy(device float* dst [[buffer(0)]], device const float* src [[buffer(1)]],
                         device const uint* idx [[buffer(2)]], device const float* w [[buffer(3)]],
                         constant EltParams& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        uint r = i / p.m;
        dst[ulong(idx[r]) * p.m + i % p.m] += w[r] * src[i];
    }
}

kernel void axpy(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                 constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) a[i] += p.alpha * b[i];
}

kernel void swiglu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                   constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        float v = g[i];
        g[i] = v / (1.0f + exp(-v)) * u[i];
    }
}

struct RopeParams { uint n_heads; uint head_dim; uint rope_dim; uint pos; float theta; uint neox; uint has_ff; };

// Grid: x over n_heads * rope_dim / 2 pairs, y over tokens (token t is at position pos + t).
kernel void rope(device float* x [[buffer(0)]], device const float* ff [[buffer(1)]],
                 constant RopeParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint half_dim = p.rope_dim / 2;
    if (gid.x >= p.n_heads * half_dim) return;
    uint h = gid.x / half_dim, i = gid.x % half_dim;
    float freq = precise::pow(p.theta, -2.0f * float(i) / float(p.rope_dim));
    if (p.has_ff) freq /= ff[i];
    float ang = float(p.pos + gid.y) * freq;
    float c = precise::cos(ang), s = precise::sin(ang);
    device float* hd = x + ulong(gid.y) * p.n_heads * p.head_dim + h * p.head_dim;
    uint a = p.neox ? i : 2 * i;
    uint b = p.neox ? i + half_dim : 2 * i + 1;
    float x0 = hd[a], x1 = hd[b];
    hd[a] = x0 * c - x1 * s;
    hd[b] = x0 * s + x1 * c;
}

// ---- KV cache: stored as f16 (default) or f32 (precise mode); T is the storage type ----

// dst[i] = src[i] for n elements, converting to the cache type.
kernel void store_kv_f32(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                         constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = src[i];
}

kernel void store_kv_f16(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]],
                         constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = half(src[i]);
}

struct AttnParams { uint n_head; uint n_kv; uint head_dim; uint n_pos; uint kv_dim; float scale; };

// Decode attention, split over the cached positions ("flash decoding"): threadgroup (h, c)
// handles positions c*ATTN_CHUNK.. of head h and writes its max, softmax sum and unnormalized
// output to `part` (head_dim + 2 floats per (head, chunk)); attention_merge combines the chunks.
constant uint ATTN_CHUNK = 256;

template <typename T>
inline void attention_part_impl(device const float* q, device const T* kc, device const T* vc,
                                device float* part, constant AttnParams& p,
                                threadgroup float* sc, threadgroup float* red, uint2 tg,
                                uint n_chunks, uint tid, uint tgs, uint lane, uint sg) {
    uint h = tg.x, c = tg.y, hd = p.head_dim, n_sg = (tgs + 31) / 32;
    uint start = c * ATTN_CHUNK, cnt = min(ATTN_CHUNK, p.n_pos - start);
    uint kv_off = (h / (p.n_head / p.n_kv)) * hd;
    device const float* qh = q + h * hd;
    float mx = -INFINITY;
    for (uint i = tid; i < cnt; i += tgs) {
        device const T* k = kc + ulong(start + i) * p.kv_dim + kv_off;
        float s = 0;
        for (uint d = 0; d < hd; d++) s += qh[d] * float(k[d]);
        s *= p.scale;
        sc[i] = s;
        mx = max(mx, s);
    }
    mx = tg_reduce_max(mx, red, lane, sg, n_sg);
    float sum = 0;
    for (uint i = tid; i < cnt; i += tgs) {
        float e = exp(sc[i] - mx);
        sc[i] = e;
        sum += e;
    }
    sum = tg_reduce_sum(sum, red, lane, sg, n_sg);
    device float* out = part + (ulong(h) * n_chunks + c) * (hd + 2);
    device const T* v = vc + ulong(start) * p.kv_dim + kv_off;
    for (uint d = tid; d < hd; d += tgs) {
        float o = 0;
        for (uint i = 0; i < cnt; i++) o += sc[i] * float(v[ulong(i) * p.kv_dim + d]);
        out[2 + d] = o;
    }
    if (tid == 0) {
        out[0] = mx;
        out[1] = sum;
    }
}

#define ATTENTION_PART(NAME, T)                                                               \
kernel void NAME(device const float* q [[buffer(0)]], device const T* kc [[buffer(1)]],      \
                 device const T* vc [[buffer(2)]], device float* part [[buffer(3)]],         \
                 constant AttnParams& p [[buffer(4)]],                                        \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint2 n_tg [[threadgroups_per_grid]],                                        \
                 uint tid [[thread_index_in_threadgroup]],                                    \
                 uint2 tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],\
                 uint sg [[simdgroup_index_in_threadgroup]]) {                                \
    threadgroup float sc[ATTN_CHUNK];                                                         \
    threadgroup float red[32];                                                                \
    attention_part_impl<T>(q, kc, vc, part, p, sc, red, tg, n_tg.y, tid, tgs.x, lane, sg);    \
}

ATTENTION_PART(attention_part_f32, float)
ATTENTION_PART(attention_part_f16, half)

struct AttnMergeParams { uint head_dim; uint n_chunks; };

// out[h] = sum_c exp(m_c - M) o_c / sum_c exp(m_c - M) l_c, over the chunks of head h.
kernel void attention_merge(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                            constant AttnMergeParams& p [[buffer(2)]],
                            uint h [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgs [[threads_per_threadgroup]]) {
    uint stride = p.head_dim + 2;
    device const float* ph = part + ulong(h) * p.n_chunks * stride;
    float m = -INFINITY;
    for (uint c = 0; c < p.n_chunks; c++) m = max(m, ph[c * stride]);
    float l = 0;
    for (uint c = 0; c < p.n_chunks; c++) l += ph[c * stride + 1] * exp(ph[c * stride] - m);
    for (uint d = tid; d < p.head_dim; d += tgs) {
        float o = 0;
        for (uint c = 0; c < p.n_chunks; c++) o += ph[c * stride + 2 + d] * exp(ph[c * stride] - m);
        out[h * p.head_dim + d] = o / l;
    }
}

struct BatchAttnParams { uint n_head; uint n_kv; uint head_dim; uint pos; uint kv_dim; float scale; uint n; };

// Causal attention for a chunk of queries on the matrix units: a threadgroup of one simdgroup
// takes head tg.x and the 8 queries from 8 * tg.y. Query t sits at position pos + t and sees
// cache rows 0..=pos + t. Per block of 32 keys: the 8 x 32 scores are four products over the
// head's dims, lane j then holds key j's column, so a row's max and sum are simd reductions
// (softmax is online: running max and sum, no scores buffer); the outputs are rescaled by a
// diagonal matrix and take P V as products. The caches hold a multiple of 8 rows, so a block
// of 8 keys is read whole; rows past a query's position get zero weight. head_dim must be a
// multiple of 8, at most 256.
template <typename T, typename M>
inline void attention_batch_impl(device const float* q, device const T* kc, device const T* vc,
                                 device float* out, constant BatchAttnParams& p,
                                 threadgroup T* sq, threadgroup float* so,
                                 threadgroup float* ss, threadgroup T* sp, threadgroup float* sd,
                                 uint2 tg, uint lane) {
    uint h = tg.x, t0 = tg.y * 8, hd = p.head_dim, nb = hd / 8;
    uint nq = min(8u, p.n - t0);
    uint n_pos = p.pos + t0 + nq;
    uint kv_off = (h / (p.n_head / p.n_kv)) * hd;
    ulong q_dim = ulong(p.n_head) * hd;
    for (uint e = lane; e < 8 * hd; e += 32) {
        uint r = e / hd;
        sq[e] = r < nq ? T(q[(t0 + r) * q_dim + h * hd + e % hd] * p.scale) : T(0);
    }
    sd[lane] = 0;
    sd[lane + 32] = 0;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 o[32];
    for (uint d = 0; d < nb; d++) o[d] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    float m[8], l[8];
    for (uint r = 0; r < 8; r++) {
        m[r] = -INFINITY;
        l[r] = 0;
    }
    for (uint base = 0; base < n_pos; base += 32) {
        uint blocks = min(4u, (n_pos - base + 7) / 8);
        for (uint pb = 0; pb < blocks; pb++) {
            simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8>(0.0f);
            device const T* kp = kc + ulong(base + 8 * pb) * p.kv_dim + kv_off;
            for (uint d = 0; d < nb; d++) {
                M mq, mk;
                simdgroup_load(mq, sq + 8 * d, hd);
                simdgroup_load(mk, kp + 8 * d, p.kv_dim, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sc, mq, mk, sc);
            }
            simdgroup_store(sc, ss + 8 * pb, 32);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint j = base + lane;
        float corr[8];
        for (uint r = 0; r < 8; r++) {
            bool seen = j <= p.pos + t0 + r && r < nq;
            float sv = seen ? ss[r * 32 + lane] : -INFINITY;
            float m_new = max(m[r], simd_max(sv));
            corr[r] = r < nq ? exp(m[r] - m_new) : 1.0f;
            float pj = seen ? exp(sv - m_new) : 0.0f;
            l[r] = l[r] * corr[r] + simd_sum(pj);
            m[r] = m_new;
            sp[r * 32 + lane] = T(pj);
        }
        if (lane < 8) sd[lane * 9] = corr[lane];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 md;
        simdgroup_load(md, sd, 8);
        for (uint d = 0; d < nb; d++) simdgroup_multiply(o[d], md, o[d]);
        for (uint pb = 0; pb < blocks; pb++) {
            M mp;
            simdgroup_load(mp, sp + 8 * pb, 32);
            device const T* vp = vc + ulong(base + 8 * pb) * p.kv_dim + kv_off;
            for (uint d = 0; d < nb; d++) {
                M mv;
                simdgroup_load(mv, vp + 8 * d, p.kv_dim);
                simdgroup_multiply_accumulate(o[d], mp, mv, o[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d = 0; d < nb; d++) simdgroup_store(o[d], so + 8 * d, hd);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < nq * hd; e += 32) {
        uint r = e / hd;
        out[(t0 + r) * q_dim + h * hd + e % hd] = so[e] / l[r];
    }
}

#define ATTENTION_BATCH(NAME, T, M)                                                           \
kernel void NAME(device const float* q [[buffer(0)]], device const T* kc [[buffer(1)]],      \
                 device const T* vc [[buffer(2)]], device float* out [[buffer(3)]],          \
                 constant BatchAttnParams& p [[buffer(4)]],                                   \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    threadgroup T sq[8 * 256];                                                                \
    threadgroup float so[8 * 256];                                                            \
    threadgroup float ss[8 * 32];                                                             \
    threadgroup T sp[8 * 32];                                                                 \
    threadgroup float sd[64];                                                                 \
    attention_batch_impl<T, M>(q, kc, vc, out, p, sq, so, ss, sp, sd, tg, lane);              \
}

ATTENTION_BATCH(attention_batch_f32, float, simdgroup_float8x8)
ATTENTION_BATCH(attention_batch_f16, half, simdgroup_half8x8)

// attention_batch_f16 for the usual head sizes, with the head size a compile-time constant and
// the cache staged in threadgroup memory. With run-time loop bounds the compiler kept the
// matrices in memory, which cost more than the products themselves (a 2048-token prompt on
// Llama-3.2-1B: 1011 -> 1199 tok/s from the constant alone); and reading 8 x 8 blocks straight
// from the cache, rows kv_dim apart, was most of the time before that. A threadgroup of
// AG_HEADS simdgroups takes KV head tg.x / subs, up to AG_HEADS of the query heads that share
// it (one per simdgroup) and the 8 queries from 8 * tg.y. Per block of 32 keys all threads
// copy K and V into threadgroup memory as 8 x 8 blocks, then every simdgroup runs
// attention_batch_impl's steps against that copy with its queries held in registers. `tm` is
// carved by hand, 32 KB being all a threadgroup gets:
//   halves: K blocks at 0, V blocks at 4096, queries [head][8][HD] at 8192;
//   per simdgroup from float 6144: scores [8][32], then P [8][32] halves, then the diagonal;
//   at the end the outputs [head][8][HD] as floats from 0.
constant uint AG_HEADS = 4;

template <uint HD>
inline void attention_batch_staged(device const float* q, device const half* kc,
                                   device const half* vc, device float* out,
                                   constant BatchAttnParams& p, threadgroup float* tm, uint2 tg,
                                   uint tid, uint sg, uint lane) {
    const uint nb = HD / 8;
    threadgroup half* sk = (threadgroup half*)tm;
    threadgroup half* sv = sk + 4096;
    threadgroup half* sq = sk + 8192 + sg * 1024;
    threadgroup float* ss = tm + 6144 + sg * 448;
    threadgroup half* sp = (threadgroup half*)(ss + 256);
    threadgroup float* sd = ss + 384;
    uint group = p.n_head / p.n_kv;
    uint subs = (group + AG_HEADS - 1) / AG_HEADS;
    uint kh = tg.x / subs, hg = (tg.x % subs) * AG_HEADS + sg;
    bool live = hg < group;
    uint h = kh * group + hg, t0 = tg.y * 8;
    uint nq = min(8u, p.n - t0);
    uint n_pos = p.pos + t0 + nq;
    uint kv_off = kh * HD;
    ulong q_dim = ulong(p.n_head) * HD;
    simdgroup_half8x8 mq[nb];
    simdgroup_float8x8 o[nb];
    float m[8], l[8];
    if (live) {
        for (uint e = lane; e < 8 * HD; e += 32) {
            uint r = e / HD;
            sq[e] = r < nq ? half(q[(t0 + r) * q_dim + h * HD + e % HD] * p.scale) : 0.0h;
        }
        sd[lane] = 0;
        sd[lane + 32] = 0;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = 0; d < nb; d++) {
            simdgroup_load(mq[d], sq + 8 * d, HD);
            o[d] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
        for (uint r = 0; r < 8; r++) {
            m[r] = -INFINITY;
            l[r] = 0;
        }
    }
    uint rows = (n_pos + 7) / 8 * 8;
    for (uint base = 0; base < n_pos; base += 32) {
        // K as [k][key] blocks and V as [key][dim] blocks, so both load without a transpose.
        for (uint e = tid; e < 8 * HD; e += 32 * AG_HEADS) {
            uint pos = e / (HD / 4), c = 4 * (e % (HD / 4));
            bool ok = base + pos < rows;
            ulong at = ulong(base + pos) * p.kv_dim + kv_off + c;
            uint blk = ((pos / 8) * nb + c / 8) * 64;
            half4 kq = ok ? half4(*(device const packed_half4*)(kc + at)) : half4(0.0h);
            threadgroup half* kb = sk + blk + (c % 8) * 8 + pos % 8;
            kb[0] = kq.x;
            kb[8] = kq.y;
            kb[16] = kq.z;
            kb[24] = kq.w;
            *(threadgroup half4*)(sv + blk + (pos % 8) * 8 + c % 8) =
                ok ? half4(*(device const packed_half4*)(vc + at)) : half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            uint blocks = min(4u, (n_pos - base + 7) / 8);
            for (uint pb = 0; pb < 4; pb++) {
                if (pb >= blocks) break;
                simdgroup_float8x8 sc = make_filled_simdgroup_matrix<float, 8>(0.0f);
                for (uint d = 0; d < nb; d++) {
                    simdgroup_half8x8 mk;
                    simdgroup_load(mk, sk + (pb * nb + d) * 64, 8);
                    simdgroup_multiply_accumulate(sc, mq[d], mk, sc);
                }
                simdgroup_store(sc, ss + 8 * pb, 32);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint j = base + lane;
            float corr[8];
            for (uint r = 0; r < 8; r++) {
                bool seen = j <= p.pos + t0 + r && r < nq;
                float s = seen ? ss[r * 32 + lane] : -INFINITY;
                float m_new = max(m[r], simd_max(s));
                corr[r] = r < nq ? exp(m[r] - m_new) : 1.0f;
                float pj = seen ? exp(s - m_new) : 0.0f;
                l[r] = l[r] * corr[r] + simd_sum(pj);
                m[r] = m_new;
                sp[r * 32 + lane] = half(pj);
            }
            if (lane < 8) sd[lane * 9] = corr[lane];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 md;
            simdgroup_load(md, sd, 8);
            for (uint d = 0; d < nb; d++) simdgroup_multiply(o[d], md, o[d]);
            for (uint pb = 0; pb < 4; pb++) {
                if (pb >= blocks) break;
                simdgroup_half8x8 mp;
                simdgroup_load(mp, sp + 8 * pb, 32);
                for (uint d = 0; d < nb; d++) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, sv + (pb * nb + d) * 64, 8);
                    simdgroup_multiply_accumulate(o[d], mp, mv, o[d]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) return;
    threadgroup float* so = tm + sg * 1024;
    for (uint d = 0; d < nb; d++) simdgroup_store(o[d], so + 8 * d, HD);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < nq * HD; e += 32) {
        uint r = e / HD;
        out[(t0 + r) * q_dim + h * HD + e % HD] = so[e] / l[r];
    }
}

#define ATTENTION_BATCH_STAGED(NAME, HD)                                                      \
kernel void NAME(device const float* q [[buffer(0)]], device const half* kc [[buffer(1)]],   \
                 device const half* vc [[buffer(2)]], device float* out [[buffer(3)]],       \
                 constant BatchAttnParams& p [[buffer(4)]],                                   \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint tid [[thread_index_in_threadgroup]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                  \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    threadgroup float tm[8192];                                                               \
    attention_batch_staged<HD>(q, kc, vc, out, p, tm, tg, tid, sg, lane);                     \
}

ATTENTION_BATCH_STAGED(attention_batch_staged_16, 16)
ATTENTION_BATCH_STAGED(attention_batch_staged_32, 32)
ATTENTION_BATCH_STAGED(attention_batch_staged_64, 64)
ATTENTION_BATCH_STAGED(attention_batch_staged_96, 96)
ATTENTION_BATCH_STAGED(attention_batch_staged_128, 128)

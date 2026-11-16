// Matrix-vector kernels over GGUF-encoded weights, read in place from the mapped model file.
// One simdgroup computes one output row; each lane handles 32-element chunks of that row.
#include <metal_stdlib>
using namespace metal;

struct Params {
    ulong offset;     // byte offset of row 0 in the weight buffer
    uint rows;
    uint cols;
    uint row_bytes;
    uint add;         // 0: y = W x, 1: y += W x (the residual stream), 2: y = W x + bias
};

inline float mv_base(uint add, device const float* y, device const float* bias, uint row) {
    return add == 1 ? y[row] : add == 2 ? bias[row] : 0.0f;
}

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

// Weights are read four bytes at a time: the number of loads, not the arithmetic, is what a
// thread waits on.
inline uint4 bytes4(device const uchar* p) { return uint4(*(device const packed_uchar4*)p); }
inline float sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }
inline uint word4(device const uchar* p) { return as_type<uint>(uchar4(*(device const packed_uchar4*)p)); }
inline float4 unpack4(uint w) { return float4(as_type<uchar4>(w)); }

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
                 device const float* bias [[buffer(4)]],                                     \
                 uint tg [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                  \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    uint row = tg * ROWS_PER_TG + sg;                                                         \
    if (row >= p.rows) return;                                                                \
    device const uchar* r = w + p.offset + ulong(row) * p.row_bytes;                          \
    float acc = 0;                                                                            \
    for (uint c = lane; c < p.cols / 32; c += 32) acc += CHUNK(r, c, x + c * 32);             \
    acc = simd_sum(acc);                                                                      \
    if (lane == 0) y[row] = acc + mv_base(p.add, y, bias, row);                                       \
}


// Quantized matvecs: a lane takes a unit of two rows: 64 columns of a K-quant, as the two
// 32-weight chunks that share their bytes, or one block of the 32-weight formats. It reads x once for both rows, and weights four bytes at a time. No arrays:
// values a loop indexes are kept in memory.
// Pair p of a Q4_K row is sub-blocks 2p' and 2p'+1 (p' = p % 4): the nibbles of 32 bytes.
inline float2 pair_q4_k(device const uchar* r0, device const uchar* r1, uint p,
                        device const float* x) {
    uint sub = (p % 4) * 2;
    device const uchar* b0 = r0 + (p / 4) * 144;
    device const uchar* b1 = r1 + (p / 4) * 144;
    float sa0, ma0, sa1, ma1, sb0, mb0, sb1, mb1;
    scale_min_k4(sub, b0 + 4, sa0, ma0);
    scale_min_k4(sub + 1, b0 + 4, sa1, ma1);
    scale_min_k4(sub, b1 + 4, sb0, mb0);
    scale_min_k4(sub + 1, b1 + 4, sb1, mb1);
    float2 sc0 = float2(sa0, sb0), sc1 = float2(sa1, sb1);
    float2 m0 = float2(ma0, mb0), m1 = float2(ma1, mb1);
    device const uchar* q0 = b0 + 16 + 16 * sub;
    device const uchar* q1 = b1 + 16 + 16 * sub;
    device const packed_float4* xv = (device const packed_float4*)(x + p * 64);
    float4 s00 = 0, s01 = 0, s10 = 0, s11 = 0, x0 = 0, x1 = 0;
    for (uint l = 0; l < 8; l++) {
        float4 a = xv[l], c = xv[8 + l];
        uint v0 = word4(q0 + 4 * l), v1 = word4(q1 + 4 * l);
        s00 += unpack4(v0 & 0x0F0F0F0Fu) * a;
        s01 += unpack4((v0 >> 4) & 0x0F0F0F0Fu) * c;
        s10 += unpack4(v1 & 0x0F0F0F0Fu) * a;
        s11 += unpack4((v1 >> 4) & 0x0F0F0F0Fu) * c;
        x0 += a;
        x1 += c;
    }
    float2 d = float2(h(b0), h(b1)), dm = float2(h(b0 + 2), h(b1 + 2));
    return d * (sc0 * float2(sum4(s00), sum4(s10)) + sc1 * float2(sum4(s01), sum4(s11))) -
           dm * (m0 * sum4(x0) + m1 * sum4(x1));
}

// Pair p of a Q6_K row is chunks k and k + 2 of a 128-weight group: the nibbles of 32 bytes
// of ql, with two of the four bit pairs of qh.
// The six bits of a weight are put together a word (four weights) at a time, and the -32 that
// centres them is taken off once per 16 weights, as 32 times the sum of x.
inline void half_q6_k(device const uchar* ql, device const uchar* qh, uint k, uint l,
                      float4 a, float4 c, thread float4& sa, thread float4& sb) {
    uint lo = word4(ql + 4 * l), hi = word4(qh + 4 * l) >> (2 * k);
    sa += unpack4((lo & 0x0F0F0F0Fu) | ((hi & 0x03030303u) << 4)) * a;
    sb += unpack4(((lo >> 4) & 0x0F0F0F0Fu) | (hi & 0x30303030u)) * c;
}
inline float2 pair_q6_k(device const uchar* r0, device const uchar* r1, uint p,
                        device const float* x) {
    uint n = (p % 4) / 2, k = p % 2;
    uint o = (p / 4) * 210;
    device const uchar* b0 = r0 + o;
    device const uchar* b1 = r1 + o;
    uint lq = 64 * n + 32 * k, hq = 128 + 32 * n;
    device const char* sc0 = (device const char*)(b0 + 192) + 8 * n + 2 * k;
    device const char* sc1 = (device const char*)(b1 + 192) + 8 * n + 2 * k;
    device const packed_float4* xv =
        (device const packed_float4*)(x + (p / 4) * 256 + n * 128 + k * 32);
    float4 a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
    float4 x0 = 0, x1 = 0, x2 = 0, x3 = 0;
    for (uint l = 0; l < 4; l++) {
        float4 a = xv[l], c = xv[16 + l];
        half_q6_k(b0 + lq, b0 + hq, k, l, a, c, a00, a02);
        half_q6_k(b1 + lq, b1 + hq, k, l, a, c, a10, a12);
        x0 += a;
        x2 += c;
    }
    for (uint l = 4; l < 8; l++) {
        float4 a = xv[l], c = xv[16 + l];
        half_q6_k(b0 + lq, b0 + hq, k, l, a, c, a01, a03);
        half_q6_k(b1 + lq, b1 + hq, k, l, a, c, a11, a13);
        x1 += a;
        x3 += c;
    }
    float4 xs = 32.0f * float4(sum4(x0), sum4(x1), sum4(x2), sum4(x3));
    float4 s0 = float4(sum4(a00), sum4(a01), sum4(a02), sum4(a03)) - xs;
    float4 s1 = float4(sum4(a10), sum4(a11), sum4(a12), sum4(a13)) - xs;
    float4 w0 = float4(float(sc0[0]), float(sc0[1]), float(sc0[4]), float(sc0[5]));
    float4 w1 = float4(float(sc1[0]), float(sc1[1]), float(sc1[4]), float(sc1[5]));
    return float2(h(b0 + 208) * dot(w0, s0), h(b1 + 208) * dot(w1, s1));
}

// The 32-weight block formats, a block of two rows per lane.
inline float2 two_q4_0(device const uchar* r0, device const uchar* r1, uint c,
                       device const float* x) {
    device const uchar* b0 = r0 + c * 18;
    device const uchar* b1 = r1 + c * 18;
    device const packed_float4* xv = (device const packed_float4*)(x + c * 32);
    float4 s0 = 0, s1 = 0, xs = 0;
    for (uint j = 0; j < 4; j++) {
        float4 a = xv[j], e = xv[4 + j];
        uint v0 = word4(b0 + 2 + 4 * j), v1 = word4(b1 + 2 + 4 * j);
        s0 += unpack4(v0 & 0x0F0F0F0Fu) * a + unpack4((v0 >> 4) & 0x0F0F0F0Fu) * e;
        s1 += unpack4(v1 & 0x0F0F0F0Fu) * a + unpack4((v1 >> 4) & 0x0F0F0F0Fu) * e;
        xs += a + e;
    }
    return float2(h(b0), h(b1)) * (float2(sum4(s0), sum4(s1)) - 8.0f * sum4(xs));
}

// Four bits of n, one to bit 4 of each byte: a weight's fifth bit, beside its nibble.
inline uint fifth4(uint n) { return (((n & 15u) * 0x00204081u) & 0x01010101u) << 4; }
inline float4 quad_q5_0(uint v, uint qh, uint j, float4 a, float4 e) {
    return unpack4((v & 0x0F0F0F0Fu) | fifth4(qh >> (4 * j))) * a +
           unpack4(((v >> 4) & 0x0F0F0F0Fu) | fifth4(qh >> (16 + 4 * j))) * e;
}
inline float2 two_q5_0(device const uchar* r0, device const uchar* r1, uint c,
                       device const float* x) {
    device const uchar* b0 = r0 + c * 22;
    device const uchar* b1 = r1 + c * 22;
    device const packed_float4* xv = (device const packed_float4*)(x + c * 32);
    uint h0 = word4(b0 + 2), h1 = word4(b1 + 2);
    float4 s0 = 0, s1 = 0, xs = 0;
    for (uint j = 0; j < 4; j++) {
        float4 a = xv[j], e = xv[4 + j];
        s0 += quad_q5_0(word4(b0 + 6 + 4 * j), h0, j, a, e);
        s1 += quad_q5_0(word4(b1 + 6 + 4 * j), h1, j, a, e);
        xs += a + e;
    }
    return float2(h(b0), h(b1)) * (float2(sum4(s0), sum4(s1)) - 16.0f * sum4(xs));
}

inline float2 two_q8_0(device const uchar* r0, device const uchar* r1, uint c,
                       device const float* x) {
    device const uchar* b0 = r0 + c * 34;
    device const uchar* b1 = r1 + c * 34;
    device const packed_float4* xv = (device const packed_float4*)(x + c * 32);
    float4 s0 = 0, s1 = 0;
    for (uint j = 0; j < 8; j++) {
        float4 a = xv[j];
        s0 += float4(char4(*(device const packed_char4*)(b0 + 2 + 4 * j))) * a;
        s1 += float4(char4(*(device const packed_char4*)(b1 + 2 + 4 * j))) * a;
    }
    return float2(h(b0), h(b1)) * float2(sum4(s0), sum4(s1));
}

#define MATVEC2(NAME, PAIR, U)                                                                   \
kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],   \
                 device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],            \
                 device const float* bias [[buffer(4)]],                                     \
                 uint tg [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                  \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    uint row = (tg * ROWS_PER_TG + sg) * 2;                                                   \
    if (row >= p.rows) return;                                                                \
    device const uchar* r = w + p.offset + ulong(row) * p.row_bytes;                          \
    /* An odd row count: the last simdgroup takes its row twice and drops one sum. */         \
    device const uchar* r1 = row + 1 < p.rows ? r + p.row_bytes : r;                          \
    float2 acc = 0;                                                                           \
    for (uint c = lane; c < p.cols / U; c += 32) acc += PAIR(r, r1, c, x);                   \
    float s0 = simd_sum(acc.x), s1 = simd_sum(acc.y);                                         \
    if (lane == 0) {                                                                          \
        y[row] = s0 + mv_base(p.add, y, bias, row);                                                    \
        if (row + 1 < p.rows) y[row + 1] = s1 + mv_base(p.add, y, bias, row + 1);                      \
    }                                                                                         \
}

// The same with four rows a simdgroup, for a matrix of very many rows (the output head):
// 8% faster there, where a lane's fixed costs are spread over only 64 columns, but slower on
// a layer's matrices, whose dispatch must wait for its last, longer threads.
#define MATVEC4(NAME, PAIR, U)                                                                   \
kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],   \
                 device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],            \
                 device const float* bias [[buffer(4)]],                                     \
                 uint tg [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                  \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    uint row = (tg * ROWS_PER_TG + sg) * 4;                                                   \
    if (row >= p.rows) return;                                                                \
    device const uchar* r = w + p.offset + ulong(row) * p.row_bytes;                          \
    /* Rows past the end repeat the last one; their sums are dropped. */                      \
    uint last = p.rows - 1 - row;                                                             \
    device const uchar* r1 = r + min(1u, last) * p.row_bytes;                                 \
    device const uchar* r2 = r + min(2u, last) * p.row_bytes;                                 \
    device const uchar* r3 = r + min(3u, last) * p.row_bytes;                                 \
    float2 acc = 0, acd = 0;                                                                  \
    for (uint c = lane; c < p.cols / U; c += 32) {                                            \
        acc += PAIR(r, r1, c, x);                                                             \
        acd += PAIR(r2, r3, c, x);                                                            \
    }                                                                                         \
    float s0 = simd_sum(acc.x), s1 = simd_sum(acc.y);                                         \
    float s2 = simd_sum(acd.x), s3 = simd_sum(acd.y);                                         \
    if (lane == 0) {                                                                          \
        y[row] = s0 + mv_base(p.add, y, bias, row);                                                    \
        if (last >= 1) y[row + 1] = s1 + mv_base(p.add, y, bias, row + 1);                             \
        if (last >= 2) y[row + 2] = s2 + mv_base(p.add, y, bias, row + 2);                             \
        if (last >= 3) y[row + 3] = s3 + mv_base(p.add, y, bias, row + 3);                             \
    }                                                                                         \
}

MATVEC4(matvec4_q4_k, pair_q4_k, 64)
MATVEC4(matvec4_q6_k, pair_q6_k, 64)
MATVEC4(matvec4_q8_0, two_q8_0, 32)

// One token's gate and up matvecs with the SwiGLU between them: the pair kernels' two rows
// are row `row` of each matrix, so y = silu(gate x) * (up x) costs one dispatch, not three.
struct SwigluParams {
    ulong gate;       // byte offsets of row 0 in their weight buffers
    ulong up;
    uint rows;
    uint cols;
    uint row_bytes;
};

#define MATVEC_SWIGLU(NAME, PAIR, U)                                                             \
kernel void NAME(device const uchar* wg [[buffer(0)]], device const uchar* wu [[buffer(1)]], \
                 device const float* x [[buffer(2)]], device float* y [[buffer(3)]],         \
                 constant SwigluParams& p [[buffer(4)]],                                      \
                 uint tg [[threadgroup_position_in_grid]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]],                                  \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    uint row = tg * ROWS_PER_TG + sg;                                                         \
    if (row >= p.rows) return;                                                                \
    device const uchar* rg = wg + p.gate + ulong(row) * p.row_bytes;                          \
    device const uchar* ru = wu + p.up + ulong(row) * p.row_bytes;                            \
    float2 acc = 0;                                                                           \
    for (uint c = lane; c < p.cols / U; c += 32) acc += PAIR(rg, ru, c, x);                   \
    float g = simd_sum(acc.x), u = simd_sum(acc.y);                                           \
    if (lane == 0) y[row] = g / (1.0f + exp(-g)) * u;                                         \
}

MATVEC_SWIGLU(matvec_swiglu_q4_k, pair_q4_k, 64)
MATVEC_SWIGLU(matvec_swiglu_q6_k, pair_q6_k, 64)
MATVEC_SWIGLU(matvec_swiglu_q4_0, two_q4_0, 32)
MATVEC_SWIGLU(matvec_swiglu_q5_0, two_q5_0, 32)
MATVEC_SWIGLU(matvec_swiglu_q8_0, two_q8_0, 32)

MATVEC(matvec_f32, chunk_f32)
MATVEC(matvec_f16, chunk_f16)
MATVEC2(matvec_q4_0, two_q4_0, 32)
MATVEC2(matvec_q5_0, two_q5_0, 32)
MATVEC2(matvec_q8_0, two_q8_0, 32)
MATVEC2(matvec_q4_k, pair_q4_k, 64)
MATVEC(matvec_q5_k, chunk_q5_k)
MATVEC2(matvec_q6_k, pair_q6_k, 64)

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

// Half of a 32-weight chunk (o = 0 or 16) as four float4, for the matmul's tile loads. The
// K-quant ones read 4 bytes at a time: one byte per load was a sixth of a prompt's matmul time.

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

inline void deq16_f32(device const uchar* r, uint c, uint o, thread float4* w) {
    device const packed_float4* s = (device const packed_float4*)(r + (c * 32 + o) * 4);
    for (uint i = 0; i < 4; i++) w[i] = s[i];
}

inline void deq16_f16(device const uchar* r, uint c, uint o, thread float4* w) {
    device const packed_half4* s = (device const packed_half4*)(r + (c * 32 + o) * 2);
    for (uint i = 0; i < 4; i++) w[i] = float4(half4(s[i]));
}

// A block's weights 0..16 are the low nibbles of its 16 bytes and 16..32 the high ones.
inline void deq16_q4_0(device const uchar* r, uint c, uint o, thread float4* w) {
    device const uchar* b = r + c * 18;
    float d = h(b);
    for (uint i = 0; i < 4; i++)
        w[i] = d * (unpack4((word4(b + 2 + 4 * i) >> (o / 4)) & 0x0F0F0F0Fu) - 8.0f);
}

inline void deq16_q5_0(device const uchar* r, uint c, uint o, thread float4* w) {
    device const uchar* b = r + c * 22;
    float d = h(b);
    uint qh = word4(b + 2) >> o;
    for (uint i = 0; i < 4; i++) {
        uint q = (word4(b + 6 + 4 * i) >> (o / 4)) & 0x0F0F0F0Fu;
        w[i] = d * (unpack4(q | fifth4(qh >> (4 * i))) - 16.0f);
    }
}

inline void deq16_q8_0(device const uchar* r, uint c, uint o, thread float4* w) {
    device const uchar* b = r + c * 34;
    float d = h(b);
    for (uint i = 0; i < 4; i++)
        w[i] = d * float4(char4(*(device const packed_char4*)(b + 2 + o + 4 * i)));
}

#define DEQ16_VIA32(NAME, DEQ)                                                                \
inline void NAME(device const uchar* r, uint c, uint o, thread float4* w) {                   \
    float t[32];                                                                              \
    DEQ(r, c, t);                                                                             \
    for (uint i = 0; i < 4; i++) w[i] = float4(t[o + 4 * i], t[o + 4 * i + 1], t[o + 4 * i + 2], t[o + 4 * i + 3]); \
}
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
    uint add;         // add into y instead of overwriting it
};

// One step of a tile: TWO (a literal) says whether it has both 32-column chunks or, as the
// last step of a matrix whose columns are an odd multiple of 32, only the first. A run-time
// test here cost 3% of a prompt.
#define MM_STEP(DEQ, T, M, AB, BB, TWO)                                                       \
    {                                                                                         \
        float4 wv[8], xv[XU];                                                                 \
        DEQ(wrow, c, wo, wv);                                                                 \
        if (TWO) {                                                                            \
            DEQ(wrow, c + 1, wo, wv + 4);                                                     \
        } else {                                                                              \
            for (uint i = 4; i < 8; i++) wv[i] = float4(0.0f);                                \
        }                                                                                     \
        for (uint i = 0; i < XU; i++) {                                                       \
            bool in = TWO || xo + 4 * i < 32;                                                 \
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
    }

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
    bool full = r0 + MM_R <= p.rows && t0 + MT <= p.n;                                        \
    device const uchar* wrow = w + p.offset + ulong(row_ok ? r0 + wr : 0) * p.row_bytes;      \
    device const float* xrow = x + ulong(tok_ok ? t0 + xt : 0) * p.cols + xo;                 \
    device float* yp = y + ulong(t0 + 8 * tb) * p.rows + r0 + 8 * rb;                         \
    simdgroup_float8x8 acc[AB * BB];                                                          \
    for (uint i = 0; i < AB * BB; i++) acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); \
    uint nc = p.cols / 32;                                                                    \
    for (uint c = 0; c + 1 < nc; c += 2) MM_STEP(DEQ, T, M, AB, BB, true)                     \
    if (nc & 1) {                                                                             \
        uint c = nc - 1;                                                                      \
        MM_STEP(DEQ, T, M, AB, BB, false)                                                     \
    }                                                                                         \
    if (full && p.add) {                                                                      \
        /* Adding into y. Seeding the sums with it instead put them in memory (3x slower). */ \
        simdgroup_float8x8 one = simdgroup_float8x8(1.0f); /* the identity */                 \
        for (uint j = 0; j < BB; j++)                                                         \
            for (uint i = 0; i < AB; i++) {                                                   \
                simdgroup_float8x8 old;                                                       \
                simdgroup_load(old, yp + ulong(8 * j) * p.rows + 8 * i, p.rows);              \
                simdgroup_multiply_accumulate(old, acc[j * AB + i], one, old);                \
                simdgroup_store(old, yp + ulong(8 * j) * p.rows + 8 * i, p.rows);             \
            }                                                                                 \
        return;                                                                               \
    }                                                                                         \
    if (full) {                                                                               \
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
        if (row < p.rows && t < p.n) {                                                        \
            ulong at = ulong(t) * p.rows + row;                                               \
            y[at] = p.add ? y[at] + sm[tt * MM_R + rr] : sm[tt * MM_R + rr];                  \
        }                                                                                     \
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

// The elementwise kernels over whole buffers take four values a thread (ELT4: the vector
// expression V while four remain, the scalar S for a tail).
#define ELT4(V, S)                                                                            \
    if (4 * i + 4 <= p.n) { V; }                                                              \
    else for (uint e = 4 * i; e < p.n; e++) { S; }
#define F4(P, I) (*(device packed_float4*)((P) + 4 * (I)))
#define CF4(P, I) (*(device const packed_float4*)((P) + 4 * (I)))

kernel void add(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ELT4(F4(a, i) = float4(F4(a, i)) + float4(CF4(b, i)), a[e] += b[e])
}

// a[i] += b[i % m]: adds a length-m vector (a bias) to each of n/m rows.
kernel void add_bcast(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                      constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) a[i] += b[i % p.m];
}

// MoE routing: dst row r = src row idx[r] (p.n = rows * dim, p.m = dim, a multiple of 4).
// Four values a thread.
kernel void gather_rows(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                        device const uint* idx [[buffer(2)]], constant EltParams& p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (4 * i < p.n) F4(dst, i) = CF4(src + ulong(idx[4 * i / p.m]) * p.m + 4 * i % p.m, 0);
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

// MoE combine over a chunk: row t of x += sum over j < k of w[t * k + j] * src row
// ids[t * k + j], the outputs of the experts token t chose. Four values a thread.
struct CombineRowsParams { uint n; uint dim; uint k; };

kernel void moe_combine_rows(device float* x [[buffer(0)]], device const float* src [[buffer(1)]],
                             device const uint* ids [[buffer(2)]], device const float* w [[buffer(3)]],
                             constant CombineRowsParams& p [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    if (4 * i >= p.n * p.dim) return;
    uint t = 4 * i / p.dim, d = 4 * i % p.dim;
    float4 s = float4(0.0f);
    for (uint j = 0; j < p.k; j++)
        s += w[t * p.k + j] * float4(CF4(src + ulong(ids[t * p.k + j]) * p.dim + d, 0));
    F4(x, i) = float4(F4(x, i)) + s;
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
    ELT4(F4(a, i) = float4(F4(a, i)) + p.alpha * float4(CF4(b, i)), a[e] += p.alpha * b[e])
}

kernel void swiglu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                   constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ELT4(float4 v = F4(g, i); F4(g, i) = v / (1.0f + exp(-v)) * float4(CF4(u, i)),
         g[e] = g[e] / (1.0f + exp(-g[e])) * u[e])
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

// rope, written to the cache instead of in place (rope_dim == head_dim, so every value moves).
#define ROPE_STORE(NAME, T)                                                                   \
kernel void NAME(device const float* x [[buffer(0)]], device const float* ff [[buffer(1)]],  \
                 constant RopeParams& p [[buffer(2)]], device T* dst [[buffer(3)]],          \
                 uint2 gid [[thread_position_in_grid]]) {                                     \
    uint half_dim = p.rope_dim / 2;                                                           \
    if (gid.x >= p.n_heads * half_dim) return;                                                \
    uint h = gid.x / half_dim, i = gid.x % half_dim;                                          \
    float freq = precise::pow(p.theta, -2.0f * float(i) / float(p.rope_dim));                 \
    if (p.has_ff) freq /= ff[i];                                                              \
    float ang = float(p.pos + gid.y) * freq;                                                  \
    float c = precise::cos(ang), s = precise::sin(ang);                                       \
    ulong at = ulong(gid.y) * p.n_heads * p.head_dim + h * p.head_dim;                        \
    uint a = p.neox ? i : 2 * i;                                                              \
    uint b = p.neox ? i + half_dim : 2 * i + 1;                                               \
    float x0 = x[at + a], x1 = x[at + b];                                                     \
    dst[at + a] = T(x0 * c - x1 * s);                                                         \
    dst[at + b] = T(x0 * s + x1 * c);                                                         \
}

ROPE_STORE(rope_store_f32, float)
ROPE_STORE(rope_store_f16, half)

// dst[i] = src[i] for n elements, converting to the cache type.
kernel void store_kv_f32(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                         constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ELT4(F4(dst, i) = CF4(src, i), dst[e] = src[e])
}

kernel void store_kv_f16(device const float* src [[buffer(0)]], device half* dst [[buffer(1)]],
                         constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    ELT4(*(device packed_half4*)(dst + 4 * i) = half4(float4(CF4(src, i))), dst[e] = half(src[e]))
}

struct AttnParams { uint n_head; uint n_kv; uint head_dim; uint n_pos; uint kv_dim; float scale; };

// Decode attention, split over the cached positions ("flash decoding"): threadgroup (h, c)
// handles positions c*ATTN_CHUNK.. of head h and writes its max, softmax sum and unnormalized
// output to `part` (head_dim + 2 floats per (head, chunk)); attention_merge combines the chunks.
constant uint ATTN_CHUNK = 256;

template <typename T>
inline void attention_part_impl(device const float* q, device const T* kc, device const T* vc,
                                device float* part, device float* res, constant AttnParams& p,
                                threadgroup float* sc, threadgroup float* pv,
                                threadgroup float* red, uint2 tg,
                                uint n_chunks, uint tid, uint tgs, uint lane, uint sg) {
    uint h = tg.x, c = tg.y, hd = p.head_dim, n_sg = (tgs + 31) / 32;
    uint start = c * ATTN_CHUNK, cnt = min(ATTN_CHUNK, p.n_pos - start);
    uint kv_off = (h / (p.n_head / p.n_kv)) * hd;
    device const float* qh = q + h * hd;
    float mx = -INFINITY;
    for (uint i = tid; i < cnt; i += tgs) {
        device const T* k = kc + ulong(start + i) * p.kv_dim + kv_off;
        float4 s4 = 0;
        for (uint d = 0; d + 4 <= hd; d += 4)
            s4 += float4(*(device const packed_float4*)(qh + d)) *
                  float4(*(device const vec<T, 4>*)(k + d));
        float s = sum4(s4);
        for (uint d = hd & ~3u; d < hd; d++) s += qh[d] * float(k[d]);
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
    // Four dims a thread, the positions dealt out over the threads that share them, then
    // one thread per four dims adds the shares up.
    uint groups = hd / 4, shares = tgs / groups;
    if (hd % 4 == 0 && shares >= 1 && tgs <= ATTN_CHUNK / 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint g = tid % groups, share = tid / groups;
        float4 o = 0;
        if (share < shares)
            for (uint i = share; i < cnt; i += shares)
                o += sc[i] * float4(*(device const vec<T, 4>*)(v + ulong(i) * p.kv_dim + 4 * g));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (share < shares) *(threadgroup float4*)(pv + 4 * tid) = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (share == 0) {
            for (uint j = 1; j < shares; j++) o += *(threadgroup float4*)(pv + 4 * (j * groups + g));
            /* The only chunk is the result; there is nothing to merge. */
            device float* to = n_chunks == 1 ? res + h * hd + 4 * g : out + 2 + 4 * g;
            *(device packed_float4*)to = n_chunks == 1 ? o / sum : o;
        }
    } else {
        for (uint d = tid; d < hd; d += tgs) {
            float o = 0;
            for (uint i = 0; i < cnt; i++) o += sc[i] * float(v[ulong(i) * p.kv_dim + d]);
            if (n_chunks == 1) res[h * hd + d] = o / sum; else out[2 + d] = o;
        }
    }
    if (tid == 0 && n_chunks > 1) {
        out[0] = mx;
        out[1] = sum;
    }
}

#define ATTENTION_PART(NAME, T)                                                               \
kernel void NAME(device const float* q [[buffer(0)]], device const T* kc [[buffer(1)]],      \
                 device const T* vc [[buffer(2)]], device float* part [[buffer(3)]],         \
                 constant AttnParams& p [[buffer(4)]], device float* res [[buffer(5)]],      \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint2 n_tg [[threadgroups_per_grid]],                                        \
                 uint tid [[thread_index_in_threadgroup]],                                    \
                 uint2 tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],\
                 uint sg [[simdgroup_index_in_threadgroup]]) {                                \
    threadgroup float sc[ATTN_CHUNK];                                                         \
    threadgroup float pv[ATTN_CHUNK * 2]; /* a float4 per thread */                           \
    threadgroup float red[32];                                                                \
    attention_part_impl<T>(q, kc, vc, part, res, p, sc, pv, red, tg, n_tg.y, tid, tgs.x, lane, sg);    \
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

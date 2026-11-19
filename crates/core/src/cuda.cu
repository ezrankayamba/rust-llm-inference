// CUDA port of metal.metal: matvec over GGUF-encoded weights plus the per-layer ops.
// Compiled at runtime with NVRTC. One warp computes one matvec output row, or one row of a
// matmul for a tile of TILE tokens; each lane takes one weight of every 32.
// No #include: NVRTC's builtin headers vary by toolkit version, so f16 is decoded by hand.

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long long ulong;

#define ROWS_PER_BLOCK 4
#define TILE 32

#define NEG_INF __uint_as_float(0xff800000u)

// Little-endian IEEE half at p (any alignment) to f32.
__device__ inline float h_bits(uint b) {
    uint sign = (b & 0x8000u) << 16, e = (b >> 10) & 31, m = b & 1023;
    if (e == 0) {
        float f = (float)m * 5.9604645e-8f;  // subnormal: m * 2^-24
        return sign ? -f : f;
    }
    uint bits = e == 31 ? (sign | 0x7f800000u | (m << 13)) : (sign | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(bits);
}

__device__ inline float h(const uchar* p) {
    return h_bits((uint)p[0] | ((uint)p[1] << 8));
}

__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__device__ inline float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Weight l (0..32) of chunk c (32 weights) of the row at r, per weight type. A warp calls these
// with l = its lane, so its 32 loads of a weight's bytes or of x are next to each other.
__device__ inline float wt_f32(const uchar* r, uint c, uint l) {
    return ((const float*)r)[c * 32 + l];
}

__device__ inline float wt_f16(const uchar* r, uint c, uint l) {
    return h(r + (c * 32 + l) * 2);
}

__device__ inline float wt_q4_0(const uchar* r, uint c, uint l) {
    const uchar* b = r + c * 18;
    uchar q = b[2 + (l & 15)];
    return h(b) * (float)((int)(l < 16 ? (q & 15) : (q >> 4)) - 8);
}

__device__ inline float wt_q5_0(const uchar* r, uint c, uint l) {
    const uchar* b = r + c * 22;
    uint qh = (uint)b[2] | ((uint)b[3] << 8) | ((uint)b[4] << 16) | ((uint)b[5] << 24);
    uchar q = b[6 + (l & 15)];
    uint v = (l < 16 ? (q & 15) : (q >> 4)) | (((qh >> l) & 1) << 4);
    return h(b) * (float)((int)v - 16);
}

__device__ inline float wt_q8_0(const uchar* r, uint c, uint l) {
    const uchar* b = r + c * 34;
    return h(b) * (float)((const signed char*)b)[2 + l];
}

__device__ inline void scale_min_k4(uint j, const uchar* q, float* sc, float* m) {
    if (j < 4) {
        *sc = (float)(q[j] & 63);
        *m = (float)(q[j + 4] & 63);
    } else {
        *sc = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

__device__ inline float wt_q4_k(const uchar* r, uint c, uint l) {
    uint sub = c % 8;
    const uchar* b = r + (c / 8) * 144;
    float sc, m;
    scale_min_k4(sub, b + 4, &sc, &m);
    uchar q = b[16 + 32 * (sub / 2) + l];
    uint v = (sub & 1) ? (q >> 4) : (q & 15);
    return h(b) * sc * (float)v - h(b + 2) * m;
}

__device__ inline float wt_q5_k(const uchar* r, uint c, uint l) {
    uint sub = c % 8;
    const uchar* b = r + (c / 8) * 176;
    float sc, m;
    scale_min_k4(sub, b + 4, &sc, &m);
    uchar q = b[48 + 32 * (sub / 2) + l];
    uint v = ((sub & 1) ? (q >> 4) : (q & 15)) + (((b[16 + l] >> sub) & 1) << 4);
    return h(b) * sc * (float)v - h(b + 2) * m;
}

__device__ inline float wt_q6_k(const uchar* r, uint c, uint l) {
    uint j = c % 8, n = j / 4, k = j % 4;
    const uchar* b = r + (c / 8) * 210;
    uchar ql = b[64 * n + 32 * (k % 2) + l];
    uchar qh = b[128 + 32 * n + l];
    float sc = (float)((const signed char*)b)[192 + 8 * n + 2 * k + l / 16];
    uint lo = k < 2 ? (ql & 15) : (ql >> 4);
    int q = (int)(lo | (((qh >> (2 * k)) & 3) << 4)) - 32;
    return h(b + 208) * sc * (float)q;
}

// y = W x, one grid row (y) per matrix of a stack: row e uses the matrix ids[e] * id_stride
// bytes after w_off, x at (e / x_div) * x_stride and y at e * y_stride. A plain matvec is one
// row with ids[0] = 0. MoE layers run every token's chosen experts as rows: x_div experts read
// one token's x.
#define MATVEC(NAME, WT)                                                                       \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const float* x, ulong x_off,      \
                                float* y, ulong y_off, uint rows, uint cols, uint row_bytes,   \
                                const uint* ids, ulong id_stride, uint x_stride,               \
                                uint x_div, uint y_stride) {                                   \
    /* A block of 128 threads takes 4 rows, a warp each. One of 256 takes a single row, its */ \
    /* 8 warps every 8th step of it: a small matrix (a MoE router) is all memory latency,   */ \
    /* and that leaves each warp fewer rounds of reads.                                     */ \
    __shared__ float red[8];                                                                   \
    uint warp = threadIdx.x / 32, lane = threadIdx.x % 32;                                     \
    bool wide = blockDim.x > ROWS_PER_BLOCK * 32;                                              \
    uint row = wide ? blockIdx.x : blockIdx.x * ROWS_PER_BLOCK + warp;                         \
    uint c0 = wide ? warp : 0, step = wide ? 8 : 1, steps = cols / 32;                         \
    if (row >= rows) return;                                                                   \
    uint e = blockIdx.y;                                                                       \
    const uchar* r = w + w_off + (ulong)ids[e] * id_stride + (ulong)row * row_bytes;           \
    const float* xs = x + x_off + (ulong)(e / x_div) * x_stride + lane;                        \
    float acc = 0;                                                                             \
    /* 8 steps' reads are issued before the first is used: used at once, each makes the   */   \
    /* warp wait out the memory latency.                                                   */   \
    for (uint c = c0; c < steps; c += 8 * step) {                                              \
        float wv[8], xv[8];                                                                    \
        _Pragma("unroll")                                                                      \
        for (uint j = 0; j < 8; j++) {                                                         \
            uint cj = min(c + j * step, steps - 1);                                            \
            wv[j] = WT(r, cj, lane);                                                           \
            xv[j] = xs[cj * 32];                                                               \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint j = 0; j < 8; j++)                                                           \
            if (c + j * step < steps) acc += wv[j] * xv[j];                                    \
    }                                                                                          \
    acc = warp_sum(acc);                                                                       \
    if (wide) {                                                                                \
        if (lane == 0) red[warp] = acc;                                                        \
        __syncthreads();                                                                       \
        acc = 0;                                                                               \
        for (uint i = 0; i < 8; i++) acc += red[i];                                            \
        if (warp) return;                                                                      \
    }                                                                                          \
    if (lane == 0) y[y_off + (ulong)e * y_stride + row] = acc;                                 \
}

// The tensor-core compile (RLI_MMA, see MATMUL_MMA) builds only its own kernels.
#ifndef RLI_MMA
MATVEC(matvec_f32, wt_f32)
MATVEC(matvec_f16, wt_f16)
#endif

// Two bytes at p (even address) as a little-endian u16.
__device__ inline uint u16(const uchar* p) {
    return *(const unsigned short*)p;
}

// The quantized matvecs take 256 weights per step (a K-quant super-block, or 8 blocks of 32):
// a lane's share of the dot product of the weights at b with their 256 values of x. Each lane
// decodes 8 weights from a few bytes and their scales once, where the wt_* functions above
// decode the scales for every weight.
__device__ inline float sb_q4_0(const uchar* b, const float* x, uint lane) {
    uint blk = lane / 4, i = 4 * (lane % 4);
    b += 18 * blk;
    const uchar* q = b + 2 + i;
    const float* x0 = x + 32 * blk + i;
    float s = 0;
    for (uint k = 0; k < 4; k++)
        s += (float)((int)(q[k] & 15) - 8) * x0[k] + (float)((int)(q[k] >> 4) - 8) * x0[k + 16];
    return h(b) * s;
}

__device__ inline float sb_q5_0(const uchar* b, const float* x, uint lane) {
    uint blk = lane / 4, i = 4 * (lane % 4);
    b += 22 * blk;
    uint qh = (uint)b[2] | ((uint)b[3] << 8) | ((uint)b[4] << 16) | ((uint)b[5] << 24);
    const uchar* q = b + 6 + i;
    const float* x0 = x + 32 * blk + i;
    float s = 0;
    for (uint k = 0; k < 4; k++) {
        int lo = (int)((q[k] & 15) | (((qh >> (i + k)) & 1) << 4)) - 16;
        int hi = (int)((q[k] >> 4) | (((qh >> (i + k + 16)) & 1) << 4)) - 16;
        s += (float)lo * x0[k] + (float)hi * x0[k + 16];
    }
    return h(b) * s;
}

__device__ inline float sb_q8_0(const uchar* b, const float* x, uint lane) {
    uint blk = lane / 4, i = 8 * (lane % 4);
    b += 34 * blk;
    const signed char* q = (const signed char*)b + 2 + i;
    const float* xs = x + 32 * blk + i;
    float s = 0;
    for (uint k = 0; k < 8; k++) s += (float)q[k] * xs[k];
    return h(b) * s;
}

__device__ inline float sb_q4_k(const uchar* b, const float* x, uint lane) {
    uint p = lane / 8, i = 4 * (lane % 8);  // nibble pair p: sub-blocks 2p (low) and 2p + 1
    float sc0, m0, sc1, m1;
    scale_min_k4(2 * p, b + 4, &sc0, &m0);
    scale_min_k4(2 * p + 1, b + 4, &sc1, &m1);
    const uchar* q = b + 16 + 4 * lane;
    const float* x0 = x + 64 * p + i;
    const float* x1 = x0 + 32;
    float s0 = 0, s1 = 0, t0 = 0, t1 = 0;
    for (uint k = 0; k < 4; k++) {
        s0 += (float)(q[k] & 15) * x0[k];
        s1 += (float)(q[k] >> 4) * x1[k];
        t0 += x0[k];
        t1 += x1[k];
    }
    return h(b) * (sc0 * s0 + sc1 * s1) - h(b + 2) * (m0 * t0 + m1 * t1);
}

__device__ inline float sb_q5_k(const uchar* b, const float* x, uint lane) {
    uint p = lane / 8, i = 4 * (lane % 8);
    float sc0, m0, sc1, m1;
    scale_min_k4(2 * p, b + 4, &sc0, &m0);
    scale_min_k4(2 * p + 1, b + 4, &sc1, &m1);
    const uchar* qh = b + 16 + i;
    const uchar* q = b + 48 + 4 * lane;
    const float* x0 = x + 64 * p + i;
    const float* x1 = x0 + 32;
    float s0 = 0, s1 = 0, t0 = 0, t1 = 0;
    for (uint k = 0; k < 4; k++) {
        s0 += (float)((q[k] & 15) | (((qh[k] >> (2 * p)) & 1) << 4)) * x0[k];
        s1 += (float)((q[k] >> 4) | (((qh[k] >> (2 * p + 1)) & 1) << 4)) * x1[k];
        t0 += x0[k];
        t1 += x1[k];
    }
    return h(b) * (sc0 * s0 + sc1 * s1) - h(b + 2) * (m0 * t0 + m1 * t1);
}

__device__ inline float sb_q6_k(const uchar* b, const float* x, uint lane) {
    uint n = lane / 16, j = 2 * (lane % 16);  // half n, positions j and j + 1 of its 32
    uint ql0 = u16(b + 64 * n + j), ql1 = u16(b + 64 * n + 32 + j), qh = u16(b + 128 + 32 * n + j);
    const signed char* sc = (const signed char*)b + 192 + 8 * n + j / 16;
    const float* xs = x + 128 * n + j;
    float s = 0;
    for (uint k = 0; k < 4; k++) {  // quarter k of the half: 32 weights
        uint ql = k % 2 ? ql1 : ql0, sh = k < 2 ? 0 : 4;
        int q0 = (int)(((ql >> sh) & 15) | (((qh >> (2 * k)) & 3) << 4)) - 32;
        int q1 = (int)(((ql >> (8 + sh)) & 15) | (((qh >> (8 + 2 * k)) & 3) << 4)) - 32;
        s += (float)sc[2 * k] * ((float)q0 * xs[32 * k] + (float)q1 * xs[32 * k + 1]);
    }
    return h(b + 208) * s;
}

// SB_BYTES is the size of 256 weights. Columns past the last whole 256 (a 32-weight format
// can have some) go one weight per lane, as in MATVEC.
#define MATVEC_SB(NAME, SB, SB_BYTES, WT)                                                      \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const float* x, ulong x_off,      \
                                float* y, ulong y_off, uint rows, uint cols, uint row_bytes,   \
                                const uint* ids, ulong id_stride, uint x_stride,               \
                                uint x_div, uint y_stride) {                                   \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32;                                                              \
    if (row >= rows) return;                                                                   \
    uint e = blockIdx.y;                                                                       \
    const uchar* r = w + w_off + (ulong)ids[e] * id_stride + (ulong)row * row_bytes;           \
    const float* xs = x + x_off + (ulong)(e / x_div) * x_stride;                               \
    float acc = 0;                                                                             \
    for (uint c = 0; c < cols / 256; c++) acc += SB(r + c * SB_BYTES, xs + c * 256, lane);     \
    for (uint c = cols / 256 * 8; c < cols / 32; c++) acc += WT(r, c, lane) * xs[c * 32 + lane]; \
    acc = warp_sum(acc);                                                                       \
    if (lane == 0) y[y_off + (ulong)e * y_stride + row] = acc;                                 \
}

#ifndef RLI_MMA
MATVEC_SB(matvec_q4_0, sb_q4_0, 8 * 18, wt_q4_0)
MATVEC_SB(matvec_q5_0, sb_q5_0, 8 * 22, wt_q5_0)
MATVEC_SB(matvec_q8_0, sb_q8_0, 8 * 34, wt_q8_0)
MATVEC_SB(matvec_q4_k, sb_q4_k, 144, wt_q4_k)
MATVEC_SB(matvec_q5_k, sb_q5_k, 176, wt_q5_k)
MATVEC_SB(matvec_q6_k, sb_q6_k, 210, wt_q6_k)
#endif

// Y = X W^T for n_tok tokens: x is n_tok rows of cols, y is n_tok rows of `rows`. Grid: x over
// groups of ROWS_PER_BLOCK weight rows, y over tiles of TILE tokens. A warp decodes each weight
// of its row once and multiplies it into every token of its tile. A tile's tokens past the last
// one repeat it and are not written, which keeps the loops' bounds constant.
#define MATMUL(NAME, WT)                                                                       \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const float* x, ulong x_off,      \
                                float* y, ulong y_off, uint rows, uint cols, uint row_bytes,   \
                                uint n_tok) {                                                  \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32;                                                              \
    uint t0 = blockIdx.y * TILE;                                                               \
    if (row >= rows) return;                                                                   \
    uint last = n_tok - 1 - t0;                                                                \
    const uchar* r = w + w_off + (ulong)row * row_bytes;                                       \
    const float* xt[TILE];                                                                     \
    float acc[TILE];                                                                           \
    for (uint t = 0; t < TILE; t++) {                                                          \
        xt[t] = x + x_off + (ulong)(t0 + (t < last ? t : last)) * cols + lane;                 \
        acc[t] = 0;                                                                            \
    }                                                                                          \
    for (uint c = 0; c < cols / 32; c++) {                                                     \
        float wv = WT(r, c, lane);                                                             \
        for (uint t = 0; t < TILE; t++) acc[t] += wv * xt[t][c * 32];                          \
    }                                                                                          \
    for (uint t = 0; t < TILE; t++) {                                                          \
        float a = warp_sum(acc[t]);                                                            \
        if (lane == 0 && t <= last) y[y_off + (ulong)(t0 + t) * rows + row] = a;               \
    }                                                                                          \
}

#ifndef RLI_MMA
MATMUL(matmul_f32, wt_f32)
MATMUL(matmul_f16, wt_f16)
MATMUL(matmul_q4_0, wt_q4_0)
MATMUL(matmul_q5_0, wt_q5_0)
MATMUL(matmul_q8_0, wt_q8_0)
MATMUL(matmul_q4_k, wt_q4_k)
MATMUL(matmul_q5_k, wt_q5_k)
MATMUL(matmul_q6_k, wt_q6_k)
#endif

// ---- f16 GEMM for long prompts: weights dequantized to f16 once per matmul ----
//
// The int8 tensor-core matmul spends most of its time scaling each 32-column product by its
// two block scales. Weights and activations as f16 need none of that: a matmul over 64 tokens
// or more dequantizes its matrix into scratch (`dequant_*`) and multiplies it by the
// activations (`to_f16`) on f16 tensor cores (`gemm_f16`, in the tensor-core compile).

__device__ inline unsigned short f2h(float f) {
    unsigned short r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}

#ifndef RLI_MMA
extern "C" __global__ void to_f16(const float* x, ulong x_off, unsigned short* out, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2h(x[x_off + i]);
}

// to_f16, and zeros in up to three results of the matmuls that will add into them (n0, n1
// and n2 values; 0 for none): a launch instead of four.
extern "C" __global__ void to_f16_zero(const float* x, ulong x_off, unsigned short* out, uint n,
                                       float* z0, ulong o0, uint n0, float* z1, ulong o1, uint n1,
                                       float* z2, ulong o2, uint n2) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2h(x[x_off + i]);
    if (i < n0) z0[o0 + i] = 0;
    if (i < n1) z1[o1 + i] = 0;
    if (i < n2) z2[o2 + i] = 0;
}

// A warp per 32 weights of a row.
#define DEQUANT(NAME, WT)                                                                      \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, unsigned short* out, uint rows, uint cols, \
                                uint row_bytes) {                                              \
    uint lane = threadIdx.x % 32;                                                              \
    ulong chunk = (ulong)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;                    \
    uint row = (uint)(chunk / (cols / 32)), c = (uint)(chunk % (cols / 32));                   \
    if (row >= rows) return;                                                                   \
    float v = WT(w + w_off + (ulong)row * row_bytes, c, lane);                                 \
    out[(ulong)row * cols + c * 32 + lane] = f2h(v);                                           \
}

DEQUANT(dequant_f32, wt_f32)
DEQUANT(dequant_f16, wt_f16)
DEQUANT(dequant_q4_0, wt_q4_0)
DEQUANT(dequant_q5_0, wt_q5_0)
DEQUANT(dequant_q8_0, wt_q8_0)
DEQUANT(dequant_q5_k, wt_q5_k)
#endif

// ---- Int8 matmul: activations quantized to int8, 4 products per dp4a ----
//
// Y = X W^T for n_tok tokens over activations quantized by quant_x. A block of 8 warps takes 32
// weight rows (a lane each) and MQ_T tokens (MQ_T / 8 per warp). Per 256 columns it copies that
// tile to shared memory, the weights decoded to int8 with a scale and an offset per 16 values,
// then every thread multiplies its row into its warp's tokens. Weights are stored transposed
// with a pad (word * MQ_PAD + row) so a warp's 32 rows read 32 different banks.
#define MQ_T 64
#define MQ_TPW (MQ_T / 8)
#define MQ_PAD 33

// c plus the dot product of the four int8 in a with those in b: one instruction on compute
// capability 6.1 and up, which this file is compiled for. No headers are included here, so the
// intrinsic is spelled out; a host build of this file supplies its own.
#ifndef RLI_HOST_BUILD
__device__ inline int dp4a(int a, int b, int c) {
    int r;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
#endif

// Four bytes at p (even address) as a little-endian u32.
__device__ inline uint u32e(const uchar* p) {
    return u16(p) | (u16(p + 2) << 16);
}

// The 32 values a warp holds, one per lane, as block `blk` of quantized activations:
// q = round(v / d) with d = max|v| / 127, as int8 in xq; d in xd; and d * (the sum of q) for
// each half of the block in xs, which the weight formats with an offset need. The kernels
// that produce a matvec's input call it on their way out, which saves `quant_x`'s launch.
__device__ inline void quant_block(float v, ulong blk, signed char* xq, float* xd, float* xs) {
    uint lane = threadIdx.x % 32;
    float mx = warp_max(fabsf(v));
    float inv = mx > 0 ? 127.0f / mx : 0.0f;
    int q = (int)rintf(v * inv);
    xq[blk * 32 + lane] = (signed char)q;
    float lo = warp_sum(lane < 16 ? (float)q : 0.0f), hi = warp_sum(lane < 16 ? 0.0f : (float)q);
    if (lane == 0) {
        float d = mx / 127.0f;
        xd[blk] = d;
        xs[2 * blk] = d * lo;
        xs[2 * blk + 1] = d * hi;
    }
}

// One 32-value block of one token per warp.
extern "C" __global__ void quant_x(const float* x, ulong x_off, signed char* xq, float* xd,
                                   float* xs, uint n_blocks) {
    uint blk = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (blk >= n_blocks) return;
    quant_block(x[x_off + (ulong)blk * 32 + threadIdx.x % 32], blk, xq, xd, xs);
}

// Tile loaders: a lane's share of the 256 weights at b, for tile row r. They fill wq (64 words
// of 4 int8), and per 16 weights ws and wm, so that weight = ws * q - wm.
__device__ inline void mq_load_q4_k(const uchar* b, uint lane, uint r, uint* wq, float* ws,
                                    float* wm) {
    uint q = u32e(b + 16 + 4 * lane);  // low nibbles: sub-block 2p; high: 2p + 1
    uint p = lane / 8, i = lane % 8;
    wq[(16 * p + i) * MQ_PAD + r] = q & 0x0F0F0F0Fu;
    wq[(16 * p + 8 + i) * MQ_PAD + r] = (q >> 4) & 0x0F0F0F0Fu;
    if (lane < 16) {
        float sc, m;
        scale_min_k4(lane / 2, b + 4, &sc, &m);
        ws[lane * MQ_PAD + r] = h(b) * sc;
        wm[lane * MQ_PAD + r] = h(b + 2) * m;
    }
}

__device__ inline void mq_load_q6_k(const uchar* b, uint lane, uint r, uint* wq, float* ws,
                                    float* wm) {
    uint k = lane / 8, l = 4 * (lane % 8);  // quarter k of a half, values l..l + 3 of its 32
    for (uint n = 0; n < 2; n++) {
        uint ql = u32e(b + 64 * n + 32 * (k % 2) + l), qh = u32e(b + 128 + 32 * n + l);
        uint v = ((ql >> (k < 2 ? 0 : 4)) & 0x0F0F0F0Fu) | (((qh >> (2 * k)) & 0x03030303u) << 4);
        // Each byte is 0..63; subtract 32 from each without borrowing across bytes.
        wq[(32 * n + lane) * MQ_PAD + r] = ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
    }
    if (lane < 16) {
        ws[lane * MQ_PAD + r] = h(b + 208) * (float)((const signed char*)b)[192 + lane];
        wm[lane * MQ_PAD + r] = 0;
    }
}

#define MATMUL_Q(NAME, LOAD, SB_BYTES)                                                         \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const uint* xq, const float* xd,  \
                                const float* xs, float* y, ulong y_off, uint rows, uint cols,  \
                                uint row_bytes, uint n_tok) {                                  \
    __shared__ uint wq[64 * MQ_PAD];                                                           \
    __shared__ float ws[16 * MQ_PAD];                                                          \
    __shared__ float wm[16 * MQ_PAD];                                                          \
    __shared__ uint tq[MQ_T * 64];                                                             \
    __shared__ float td[MQ_T * 8];                                                             \
    __shared__ float ts[MQ_T * 16];                                                            \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint row0 = blockIdx.x * 32, t0 = blockIdx.y * MQ_T;                                       \
    uint last = n_tok - 1 - t0;                                                                \
    float acc[MQ_TPW];                                                                         \
    for (uint j = 0; j < MQ_TPW; j++) acc[j] = 0;                                              \
    for (uint k = 0; k < cols / 256; k++) {                                                    \
        for (uint r = warp; r < 32; r += 8) {                                                  \
            uint row = row0 + r < rows ? row0 + r : rows - 1;                                  \
            LOAD(w + w_off + (ulong)row * row_bytes + (ulong)k * SB_BYTES, lane, r, wq, ws, wm); \
        }                                                                                      \
        for (uint t = warp; t < MQ_T; t += 8) {                                                \
            ulong tok = t0 + (t < last ? t : last);                                            \
            const uint* src = xq + tok * (cols / 4) + k * 64;                                  \
            tq[t * 64 + lane] = src[lane];                                                     \
            tq[t * 64 + 32 + lane] = src[32 + lane];                                           \
            if (lane < 8) td[t * 8 + lane] = xd[tok * (cols / 32) + k * 8 + lane];             \
            if (lane < 16) ts[t * 16 + lane] = xs[tok * (cols / 16) + k * 16 + lane];          \
        }                                                                                      \
        __syncthreads();                                                                       \
        if (warp * MQ_TPW <= last) {                                                           \
            for (uint g = 0; g < 16; g++) {                                                    \
                const uint* wg = wq + 4 * g * MQ_PAD + lane;                                   \
                int w0 = wg[0], w1 = wg[MQ_PAD], w2 = wg[2 * MQ_PAD], w3 = wg[3 * MQ_PAD];     \
                float sc = ws[g * MQ_PAD + lane], mn = wm[g * MQ_PAD + lane];                  \
                for (uint j = 0; j < MQ_TPW; j++) {                                            \
                    uint t = warp * MQ_TPW + j;                                                \
                    const int* xw = (const int*)tq + t * 64 + 4 * g;                           \
                    int si = dp4a(w0, xw[0], 0);                                             \
                    si = dp4a(w1, xw[1], si);                                                \
                    si = dp4a(w2, xw[2], si);                                                \
                    si = dp4a(w3, xw[3], si);                                                \
                    acc[j] += sc * td[t * 8 + g / 2] * (float)si - mn * ts[t * 16 + g];        \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
    for (uint j = 0; j < MQ_TPW; j++) {                                                        \
        uint t = warp * MQ_TPW + j;                                                            \
        if (t <= last && row0 + lane < rows) y[y_off + (ulong)(t0 + t) * rows + row0 + lane] = acc[j]; \
    }                                                                                          \
}

#ifndef RLI_MMA
MATMUL_Q(mmq_q4_k, mq_load_q4_k, 144)
MATMUL_Q(mmq_q6_k, mq_load_q6_k, 210)
#endif

// The n (token, expert) pairs of a chunk grouped by expert: perm lists the pairs expert by
// expert, and expert e's are perm[start[e]] up to perm[start[e + 1]]. One block.
extern "C" __global__ void moe_group(const uint* ids, uint* perm, uint* start, uint n,
                                     uint n_expert) {
    __shared__ uint cnt[2049];
    uint t = threadIdx.x;
    for (uint e = t; e <= n_expert; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (uint i = t; i < n; i += blockDim.x) atomicAdd(&cnt[ids[i] + 1], 1u);
    __syncthreads();
    if (t == 0)
        for (uint e = 1; e <= n_expert; e++) cnt[e] += cnt[e - 1];
    __syncthreads();
    for (uint e = t; e <= n_expert; e += blockDim.x) start[e] = cnt[e];
    __syncthreads();
    for (uint i = t; i < n; i += blockDim.x) perm[atomicAdd(&cnt[ids[i]], 1u)] = i;
}

// MATMUL_Q for stacked experts over the pairs moe_group listed: block (x, e) takes 32 rows of
// expert e (matrices `stride` bytes apart) and multiplies them into every pair that chose e,
// MQI_T at a time, so an expert's weights are decoded once per chunk and not once per pair.
// Pair p reads activation row p / x_div and writes row p of y. An expert has few pairs, so
// they are dealt out to the warps in turn: 16 pairs are two per warp, not two warps' worth.
#ifndef MQI_T
#define MQI_T 64
#endif
#define MQI_TPW (MQI_T / 8)
#define MATMUL_Q_ID(NAME, LOAD, SB_BYTES)                                                      \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, ulong stride, const uint* perm,   \
                                const uint* start, const uint* xq, const float* xd,            \
                                const float* xs, uint x_div, float* y, ulong y_off, uint rows, \
                                uint cols, uint row_bytes) {                                   \
    __shared__ uint wq[64 * MQ_PAD];                                                           \
    __shared__ float ws[16 * MQ_PAD];                                                          \
    __shared__ float wm[16 * MQ_PAD];                                                          \
    __shared__ uint tq[MQI_T * 64];                                                             \
    __shared__ float td[MQI_T * 8];                                                             \
    __shared__ float ts[MQI_T * 16];                                                            \
    __shared__ uint pr[MQI_T];                                                                 \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint row0 = blockIdx.x * 32, p1 = start[blockIdx.y + 1];                                   \
    w += w_off + blockIdx.y * stride;                                                          \
    for (uint p0 = start[blockIdx.y]; p0 < p1; p0 += MQI_T) {                                  \
        uint last = (p1 - p0 < MQI_T ? p1 - p0 : MQI_T) - 1;                                    \
        if (threadIdx.x <= last) pr[threadIdx.x] = perm[p0 + threadIdx.x];                     \
        __syncthreads();                                                                       \
        float acc[MQI_TPW];                                                                    \
        for (uint j = 0; j < MQI_TPW; j++) acc[j] = 0;                                         \
        for (uint k = 0; k < cols / 256; k++) {                                                \
            for (uint r = warp; r < 32; r += 8) {                                              \
                uint row = row0 + r < rows ? row0 + r : rows - 1;                              \
                LOAD(w + (ulong)row * row_bytes + (ulong)k * SB_BYTES, lane, r, wq, ws, wm);   \
            }                                                                                  \
            for (uint t = warp; t <= last; t += 8) {                                           \
                ulong tok = pr[t] / x_div;                                                     \
                const uint* src = xq + tok * (cols / 4) + k * 64;                              \
                tq[t * 64 + lane] = src[lane];                                                 \
                tq[t * 64 + 32 + lane] = src[32 + lane];                                       \
                if (lane < 8) td[t * 8 + lane] = xd[tok * (cols / 32) + k * 8 + lane];         \
                if (lane < 16) ts[t * 16 + lane] = xs[tok * (cols / 16) + k * 16 + lane];      \
            }                                                                                  \
            __syncthreads();                                                                   \
            if (warp <= last) {                                                                \
                for (uint g = 0; g < 16; g++) {                                                \
                    const uint* wg = wq + 4 * g * MQ_PAD + lane;                               \
                    int w0 = wg[0], w1 = wg[MQ_PAD], w2 = wg[2 * MQ_PAD], w3 = wg[3 * MQ_PAD]; \
                    float sc = ws[g * MQ_PAD + lane], mn = wm[g * MQ_PAD + lane];              \
                    for (uint j = 0; j < MQI_TPW; j++) {                                       \
                        uint t = j * 8 + warp;                                                 \
                        if (t > last) break;                                                   \
                        const int* xw = (const int*)tq + t * 64 + 4 * g;                       \
                        int si = dp4a(w0, xw[0], 0);                                           \
                        si = dp4a(w1, xw[1], si);                                              \
                        si = dp4a(w2, xw[2], si);                                              \
                        si = dp4a(w3, xw[3], si);                                              \
                        acc[j] += sc * td[t * 8 + g / 2] * (float)si - mn * ts[t * 16 + g];    \
                    }                                                                          \
                }                                                                              \
            }                                                                                  \
            __syncthreads();                                                                   \
        }                                                                                      \
        for (uint j = 0; j < MQI_TPW; j++) {                                                   \
            uint t = j * 8 + warp;                                                             \
            if (t <= last && row0 + lane < rows) y[y_off + (ulong)pr[t] * rows + row0 + lane] = acc[j]; \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
}

#ifndef RLI_MMA
MATMUL_Q_ID(mmq_q4_k_id, mq_load_q4_k, 144)
MATMUL_Q_ID(mmq_q6_k_id, mq_load_q6_k, 210)
#endif

// ---- Int8 matvec: one token (or one matrix of a stack per grid row) over quant_x's output ----
//
// A warp takes one weight row, several super-blocks a step, and each lane a whole sub-block:
// it decodes that sub-block's scale, chains its weights through dp4a into one integer and
// converts once. `left` is the number of super-blocks left in the row.

// 4 weights of quarter k of a half Q6_K super-block, from their low nibbles (in ql, already
// the right 4 bytes) and high bit pairs, as int8.
__device__ inline uint word_q6_k(uint ql, uint qh, uint k) {
    uint v = ((ql >> (k < 2 ? 0 : 4)) & 0x0F0F0F0Fu) | (((qh >> (2 * k)) & 0x03030303u) << 4);
    // Each byte is 0..63; subtract 32 from each without borrowing across bytes.
    return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
}

// 16 bytes at a 16-byte boundary, as one read.
struct __attribute__((aligned(16))) U128 {
    uint v[4];
};

// Q4_K, 4 super-blocks a step, in four 16-byte reads a lane (the row must lie at a multiple of
// 16 bytes): the super-block's scales, 16 bytes of nibbles and the 16 activations under each
// half of them. Lane 8u + j of super-block u decodes sub-block j's scale; its 16 bytes are
// weights 16 (j % 2).. of sub-blocks j & ~1 (low nibbles) and j | 1 (high), so it also needs
// the scale its neighbour decoded.
__device__ inline float mvq_q4_k_step(const uchar* r, const int* xq, const float* xd,
                                      const float* xs, uint left, uint lane) {
    uint u = lane / 8, j = lane % 8, hf = j % 2, l = j % 4;
    float f = 0, mins = 0;
    int s0 = 0, s1 = 0;
    if (u < left) {
        const uchar* b = r + u * 144;
        U128 hd = *(const U128*)b;
        U128 q = *(const U128*)(b + 16 + 16 * j);
        U128 x0 = *(const U128*)(xq + 64 * u + 16 * (j / 2) + 4 * hf);
        U128 x1 = *(const U128*)(xq + 64 * u + 16 * (j / 2) + 8 + 4 * hf);
        _Pragma("unroll")
        for (uint k = 0; k < 4; k++) {
            s0 = dp4a(q.v[k] & 0x0F0F0F0Fu, x0.v[k], s0);
            s1 = dp4a((q.v[k] >> 4) & 0x0F0F0F0Fu, x1.v[k], s1);
        }
        // Scale bytes l, 4 + l and 8 + l of the 12 that follow d and dmin.
        uint b0 = (hd.v[1] >> 8 * l) & 255, b1 = (hd.v[2] >> 8 * l) & 255;
        uint b2 = (hd.v[3] >> 8 * l) & 255;
        uint sc = j < 4 ? b0 & 63 : (b2 & 15) | ((b0 >> 6) << 4);
        uint mn = j < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4);
        f = h_bits(hd.v[0] & 0xFFFF) * (float)sc * xd[8 * u + j];
        mins = h_bits(hd.v[0] >> 16) * (float)mn * (xs[16 * u + 2 * j] + xs[16 * u + 2 * j + 1]);
    }
    float fo = __shfl_xor_sync(0xffffffff, f, 1);
    return (hf ? fo : f) * (float)s0 + (hf ? f : fo) * (float)s1 - mins;
}

// mvq_q4_k_step for 8 super-blocks, both halves read before either is used: with few rows in
// flight (a token's matvecs) a warp's time is the waits for its reads, and this halves them
// (decode on an A100: Llama-3.2-1B +4.5%, Qwen3-30B-A3B +3%, Mistral-7B +1.9%). Over the 128
// (token, expert) rows of a 16-token MoE chunk it was 2% slower: mvq_q4_k_many keeps 4 a step.
__device__ inline float mvq_q4_k_step8(const uchar* r, const int* xq, const float* xd,
                                      const float* xs, uint left, uint lane) {
    uint u = lane / 8, j = lane % 8, hf = j % 2, l = j % 4;
    float f[2] = {0, 0}, mins[2] = {0, 0};
    int s0[2] = {0, 0}, s1[2] = {0, 0};
    U128 hd[2], q[2], x0[2], x1[2];
    float d[2], sa[2], sb[2];
    // Both halves of the step's 8 super-blocks are read before either is used.
    #pragma unroll
    for (uint t = 0; t < 2; t++) {
        uint v = u + 4 * t;
        if (v >= left) continue;
        const uchar* b = r + v * 144;
        hd[t] = *(const U128*)b;
        q[t] = *(const U128*)(b + 16 + 16 * j);
        x0[t] = *(const U128*)(xq + 64 * v + 16 * (j / 2) + 4 * hf);
        x1[t] = *(const U128*)(xq + 64 * v + 16 * (j / 2) + 8 + 4 * hf);
        d[t] = xd[8 * v + j];
        sa[t] = xs[16 * v + 2 * j];
        sb[t] = xs[16 * v + 2 * j + 1];
    }
    #pragma unroll
    for (uint t = 0; t < 2; t++) {
        if (u + 4 * t >= left) continue;
        #pragma unroll
        for (uint k = 0; k < 4; k++) {
            s0[t] = dp4a(q[t].v[k] & 0x0F0F0F0Fu, x0[t].v[k], s0[t]);
            s1[t] = dp4a((q[t].v[k] >> 4) & 0x0F0F0F0Fu, x1[t].v[k], s1[t]);
        }
        uint b0 = (hd[t].v[1] >> 8 * l) & 255, b1 = (hd[t].v[2] >> 8 * l) & 255;
        uint b2 = (hd[t].v[3] >> 8 * l) & 255;
        uint sc = j < 4 ? b0 & 63 : (b2 & 15) | ((b0 >> 6) << 4);
        uint mn = j < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4);
        f[t] = h_bits(hd[t].v[0] & 0xFFFF) * (float)sc * d[t];
        mins[t] = h_bits(hd[t].v[0] >> 16) * (float)mn * (sa[t] + sb[t]);
    }
    float fo0 = __shfl_xor_sync(0xffffffff, f[0], 1), fo1 = __shfl_xor_sync(0xffffffff, f[1], 1);
    return (hf ? fo0 : f[0]) * (float)s0[0] + (hf ? f[0] : fo0) * (float)s1[0] - mins[0] +
           (hf ? fo1 : f[1]) * (float)s0[1] + (hf ? f[1] : fo1) * (float)s1[1] - mins[1];
}

struct __attribute__((aligned(16))) UL2 {
    ulong v[2];
};

// The 24 bytes at p, at any address, from the two 16-byte words around it: two reads where
// byte or 2-byte loads would be a dozen. Reads up to 14 bytes past them, which the weight
// arena's slack allows for.
__device__ inline void ld24(const uchar* p, ulong* out) {
    uint o = (uint)((ulong)p & 15), sh = 8 * (o & 7);
    const UL2* w = (const UL2*)(p - o);
    UL2 lo = w[0], hi = w[1];
    ulong a = o < 8 ? lo.v[0] : lo.v[1], b = o < 8 ? lo.v[1] : hi.v[0];
    ulong c = o < 8 ? hi.v[0] : hi.v[1], d = o < 8 ? hi.v[1] : 0;
    if (sh) {
        a = (a >> sh) | (b << (64 - sh));
        b = (b >> sh) | (c << (64 - sh));
        c = (c >> sh) | (d << (64 - sh));
    }
    out[0] = a;
    out[1] = b;
    out[2] = c;
}

// Q6_K, 4 super-blocks a step. Lane 8u + j of super-block u takes 16 bytes of low bits (half
// n = j / 4 of the block, its bytes 32 kk + 16 hf on): their low nibbles are 16 weights of
// quarter kk and their high nibbles 16 of quarter kk + 2, 64 weights on.
__device__ inline float mvq_q6_k_step(const uchar* r, const int* xq, const float* xd,
                                      const float* xs, uint left, uint lane) {
    uint u = lane / 8, j = lane % 8;
    if (u >= left) return 0;
    uint n = j / 4, kk = j / 2 % 2, hf = j % 2;
    const uchar* b = r + u * 210;
    ulong ql[3], qh[3], sc[3];
    ld24(b + 64 * n + 32 * kk + 16 * hf, ql);
    ld24(b + 128 + 32 * n + 16 * hf, qh);
    ld24(b + 192, sc);
    uint at = 128 * n + 32 * kk + 16 * hf;  // the first weight's index in the super-block
    U128 xa = *(const U128*)(xq + 64 * u + at / 4);
    U128 xb = *(const U128*)(xq + 64 * u + at / 4 + 16);
    int sa = 0, sb = 0;
    _Pragma("unroll")
    for (uint t = 0; t < 4; t++) {
        uint lw = (uint)(ql[t / 2] >> 32 * (t % 2)), hw = (uint)(qh[t / 2] >> 32 * (t % 2));
        sa = dp4a(word_q6_k(lw, hw, kk), xa.v[t], sa);
        sb = dp4a(word_q6_k(lw, hw, kk + 2), xb.v[t], sb);
    }
    // The scales of the two groups of 16: bytes at / 16 and at / 16 + 4 of 16, then d.
    uint ia = at / 16;
    float fa = (float)(signed char)(sc[ia / 8] >> 8 * (ia % 8));
    float fb = (float)(signed char)(sc[ia / 8] >> 8 * (ia % 8 + 4));
    const float* dx = xd + 8 * u + at / 32;
    return h_bits((uint)sc[2] & 0xFFFF) * (fa * dx[0] * (float)sa + fb * dx[2] * (float)sb);
}

// Scale and minimum of sub-block i, from the 12 scale bytes in words 1..3 of a Q4_K
// super-block's first 16 bytes.
__device__ inline void q4_k_scale(const U128& hd, uint i, float* sc, float* mn) {
    uint l = i % 4;
    uint b0 = (hd.v[1] >> 8 * l) & 255, b1 = (hd.v[2] >> 8 * l) & 255;
    uint b2 = (hd.v[3] >> 8 * l) & 255;
    *sc = h_bits(hd.v[0] & 0xFFFF) * (float)(i < 4 ? b0 & 63 : (b2 & 15) | ((b0 >> 6) << 4));
    *mn = h_bits(hd.v[0] >> 16) * (float)(i < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4));
}

#ifndef RLI_MMA

// Q4_K written out as f16, in 16-byte reads and stores (the row must lie at a multiple of 16
// bytes; `dequant_q4_k_any` below takes any). A warp takes 4 super-blocks of a row; lane
// 8u + j reads 16 bytes of super-block u, which hold weights 16 (j % 2).. of sub-blocks
// j & ~1 (low nibbles) and j | 1 (high).
extern "C" __global__ void dequant_q4_k(const uchar* w, ulong w_off, unsigned short* out,
                                        uint rows, uint cols, uint row_bytes) {
    uint lane = threadIdx.x % 32, u = lane / 8, j = lane % 8;
    uint steps = (cols / 256 + 3) / 4;
    ulong chunk = (ulong)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    uint row = (uint)(chunk / steps), sb = 4 * (uint)(chunk % steps) + u;
    if (row >= rows || sb >= cols / 256) return;
    const uchar* b = w + w_off + (ulong)row * row_bytes + (ulong)sb * 144;
    U128 hd = *(const U128*)b;
    U128 q = *(const U128*)(b + 16 + 16 * j);
    float sa, ma, sc, mc;
    q4_k_scale(hd, j & ~1u, &sa, &ma);
    q4_k_scale(hd, j | 1u, &sc, &mc);
    U128 lo[2], hi[2];
    #pragma unroll
    for (uint k = 0; k < 8; k++) {
        // Two weights a word: bytes 2k and 2k + 1 of the 16.
        uint by = (q.v[k / 2] >> 16 * (k % 2)) & 0xFFFF;
        uint a0 = f2h(sa * (float)(by & 15) - ma), a1 = f2h(sa * (float)((by >> 8) & 15) - ma);
        uint c0 = f2h(sc * (float)((by >> 4) & 15) - mc), c1 = f2h(sc * (float)(by >> 12) - mc);
        lo[k / 4].v[k % 4] = a0 | (a1 << 16);
        hi[k / 4].v[k % 4] = c0 | (c1 << 16);
    }
    U128* o = (U128*)(out + (ulong)row * cols + sb * 256 + 32 * (j & ~1u) + 16 * (j % 2));
    o[0] = lo[0];
    o[1] = lo[1];
    o[4] = hi[0];
    o[5] = hi[1];
}
DEQUANT(dequant_q4_k_any, wt_q4_k)

// Q6_K written out as f16, a warp per super-block: a lane takes 8 weights (8 bytes of low
// bits and of high bits through `ld24`, as blocks of 210 bytes are aligned to nothing) and
// stores them as one 16-byte word. A thread per weight took 5 byte loads and a 2-byte store
// each, 280 us for a 4096x4096 matrix on an A100.
extern "C" __global__ void dequant_q6_k(const uchar* w, ulong w_off, unsigned short* out,
                                        uint rows, uint cols, uint row_bytes) {
    uint lane = threadIdx.x % 32, nsb = cols / 256;
    ulong chunk = (ulong)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    uint row = (uint)(chunk / nsb), sb = (uint)(chunk % nsb);
    if (row >= rows) return;
    const uchar* b = w + w_off + (ulong)row * row_bytes + (ulong)sb * 210;
    // As wt_q6_k for weights 8 lane..: half n, quarter k of it, bytes m0.. of the quarter.
    uint n = lane / 16, k = (lane % 16) / 4, m0 = 8 * (lane % 4);
    ulong ql[3], qh[3];
    ld24(b + 64 * n + 32 * (k % 2) + m0, ql);
    ld24(b + 128 + 32 * n + m0, qh);
    float sc = (float)((const signed char*)b)[192 + 8 * n + 2 * k + m0 / 16];
    float d = h(b + 208);
    U128 o;
    #pragma unroll
    for (uint t = 0; t < 8; t++) {
        uint lb = (uint)(ql[0] >> 8 * t) & 255, hb = (uint)(qh[0] >> 8 * t) & 255;
        uint lo = k < 2 ? (lb & 15) : (lb >> 4);
        int q = (int)(lo | (((hb >> (2 * k)) & 3) << 4)) - 32;
        uint v = f2h(d * sc * (float)q);
        o.v[t / 2] = t % 2 ? o.v[t / 2] | (v << 16) : v;
    }
    *(U128*)(out + (ulong)row * cols + sb * 256 + 8 * lane) = o;
}
#endif

// Grid rows, ids and strides as in MATVEC; x_stride counts values of x. With `add`, the result
// is added to y. The *_pair kernels take a grid two deep, whose second layer runs a matrix (or
// stack) of the same shape z_stride bytes after the first over the same x, into y2: a token's
// gate and up in one launch. They are separate kernels because the three parameters alone
// made the plain ones 2-5% slower.
#define MVQ_PAIR_ARGS , float* y2, ulong y2_off, ulong z_stride
#define MVQ_PAIR_PICK                                                                          \
    if (blockIdx.z) {                                                                          \
        w_off += z_stride;                                                                     \
        y = y2;                                                                                \
        y_off = y2_off;                                                                        \
    }
#define MATVEC_Q(NAME, STEP, SB_BYTES, NSB, ARGS, PICK)                                        \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const int* xq, const float* xd,   \
                                const float* xs, float* y, ulong y_off, uint rows, uint cols,  \
                                uint row_bytes, const uint* ids, ulong id_stride,              \
                                uint x_stride, uint x_div, uint y_stride, uint add ARGS) {     \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32;                                                              \
    if (row >= rows) return;                                                                   \
    uint e = blockIdx.y, n_sb = cols / 256;                                                    \
    PICK                                                                                       \
    const uchar* r = w + w_off + (ulong)ids[e] * id_stride + (ulong)row * row_bytes;           \
    ulong xo = (ulong)(e / x_div) * x_stride;                                                  \
    const int* q = xq + xo / 4;                                                                \
    const float* d = xd + xo / 32;                                                             \
    const float* s = xs + xo / 16;                                                             \
    float acc = 0;                                                                             \
    for (uint c = 0; c < n_sb; c += NSB)                                                       \
        acc += STEP(r + c * SB_BYTES, q + 64 * c, d + 8 * c, s + 16 * c, n_sb - c, lane);      \
    acc = warp_sum(acc);                                                                       \
    float* yr = y + y_off + (ulong)e * y_stride + row;                                         \
    if (lane == 0) *yr = add ? *yr + acc : acc;                                                \
}

#ifndef RLI_MMA
MATVEC_Q(mvq_q4_k, mvq_q4_k_step8, 144, 8, , )
MATVEC_Q(mvq_q4_k_many, mvq_q4_k_step, 144, 4, , )
MATVEC_Q(mvq_q6_k, mvq_q6_k_step, 210, 4, , )
MATVEC_Q(mvq_q4_k_pair, mvq_q4_k_step8, 144, 8, MVQ_PAIR_ARGS, MVQ_PAIR_PICK)
MATVEC_Q(mvq_q6_k_pair, mvq_q6_k_step, 210, 4, MVQ_PAIR_ARGS, MVQ_PAIR_PICK)

// mvq_q4_k for a few tokens of one matrix: a warp takes a weight row and up to MVM_P tokens
// a pass, so the row's weights and scales are read and decoded once for all of them, where
// a grid row per token read them once each. Token t reads row t of the activations and
// writes (with `add`, adds into) row t of y.
// A kernel per MVM_P, since one that holds 4 tokens' activations runs 2 tokens 6% slower
// than one that holds 2 (Mistral-7B, 2-token chunks: 225 -> 240 tok/s).
#define MVQ_MULTI(NAME, MVM_P)                                                                 \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const int* xq,                    \
                                          const float* xd, const float* xs, float* y,          \
                                          ulong y_off, uint rows, uint cols, uint row_bytes,   \
                                          uint n_tok, uint add) {                              \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32, n_sb = cols / 256;                                           \
    if (row >= rows) return;                                                                   \
    const uchar* r = w + w_off + (ulong)row * row_bytes;                                       \
    uint u = lane / 8, j = lane % 8, hf = j % 2, l = j % 4;                                    \
    for (uint p0 = 0; p0 < n_tok; p0 += MVM_P) {                                               \
        uint np = n_tok - p0 < MVM_P ? n_tok - p0 : MVM_P;                                     \
        float acc[MVM_P];                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < MVM_P; p++) acc[p] = 0;                                           \
        for (uint c = 0; c < n_sb; c += 4) {                                                   \
            float f[MVM_P], mins[MVM_P];                                                       \
            int s0[MVM_P], s1[MVM_P];                                                          \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < MVM_P; p++) {                                                 \
                f[p] = mins[p] = 0;                                                            \
                s0[p] = s1[p] = 0;                                                             \
            }                                                                                  \
            if (c + u < n_sb) {                                                                \
                const uchar* b = r + (c + u) * 144;                                            \
                U128 hd = *(const U128*)b;                                                     \
                U128 q = *(const U128*)(b + 16 + 16 * j);                                      \
                U128 x0[MVM_P], x1[MVM_P];                                                     \
                float d[MVM_P], sa[MVM_P], sb[MVM_P];                                          \
                _Pragma("unroll")                                                              \
                for (uint p = 0; p < MVM_P; p++) {                                             \
                    if (p >= np) continue;                                                     \
                    ulong xo = (ulong)(p0 + p) * cols + 256 * (c + u);                         \
                    x0[p] = *(const U128*)(xq + xo / 4 + 16 * (j / 2) + 4 * hf);               \
                    x1[p] = *(const U128*)(xq + xo / 4 + 16 * (j / 2) + 8 + 4 * hf);           \
                    d[p] = xd[xo / 32 + j];                                                    \
                    sa[p] = xs[xo / 16 + 2 * j];                                               \
                    sb[p] = xs[xo / 16 + 2 * j + 1];                                           \
                }                                                                              \
                uint b0 = (hd.v[1] >> 8 * l) & 255, b1 = (hd.v[2] >> 8 * l) & 255;             \
                uint b2 = (hd.v[3] >> 8 * l) & 255;                                            \
                uint sc = j < 4 ? b0 & 63 : (b2 & 15) | ((b0 >> 6) << 4);                      \
                uint mn = j < 4 ? b1 & 63 : (b2 >> 4) | ((b1 >> 6) << 4);                      \
                float fs = h_bits(hd.v[0] & 0xFFFF) * (float)sc;                               \
                float fm = h_bits(hd.v[0] >> 16) * (float)mn;                                  \
                _Pragma("unroll")                                                              \
                for (uint p = 0; p < MVM_P; p++) {                                             \
                    if (p >= np) continue;                                                     \
                    _Pragma("unroll")                                                          \
                    for (uint k = 0; k < 4; k++) {                                             \
                        s0[p] = dp4a(q.v[k] & 0x0F0F0F0Fu, x0[p].v[k], s0[p]);                 \
                        s1[p] = dp4a((q.v[k] >> 4) & 0x0F0F0F0Fu, x1[p].v[k], s1[p]);          \
                    }                                                                          \
                    f[p] = fs * d[p];                                                          \
                    mins[p] = fm * (sa[p] + sb[p]);                                            \
                }                                                                              \
            }                                                                                  \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < MVM_P; p++) {                                                 \
                if (p >= np) continue;                                                         \
                float fo = __shfl_xor_sync(0xffffffff, f[p], 1);                               \
                float fa = hf ? fo : f[p], fb = hf ? f[p] : fo;                                \
                acc[p] += fa * (float)s0[p] + fb * (float)s1[p] - mins[p];                     \
            }                                                                                  \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < MVM_P; p++) {                                                     \
            if (p >= np) continue;                                                             \
            float a = warp_sum(acc[p]);                                                        \
            float* yr = y + y_off + (ulong)(p0 + p) * rows + row;                              \
            if (lane == 0) *yr = add ? *yr + a : a;                                            \
        }                                                                                      \
    }                                                                                          \
}
MVQ_MULTI(mvq_q4_k_multi2, 2)
MVQ_MULTI(mvq_q4_k_multi3, 3)
MVQ_MULTI(mvq_q4_k_multi4, 4)
// The same for Q6_K: mvq_q6_k_step's weights, decoded once for the pass's tokens.
#define MVQ6_MULTI(NAME, MVM_P)                                                                \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const int* xq,                    \
                                const float* xd, const float* xs, float* y,                    \
                                ulong y_off, uint rows, uint cols, uint row_bytes,             \
                                uint n_tok, uint add) {                                        \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32, n_sb = cols / 256;                                           \
    if (row >= rows) return;                                                                   \
    const uchar* r = w + w_off + (ulong)row * row_bytes;                                       \
    uint u = lane / 8, j = lane % 8, n = j / 4, kk = j / 2 % 2, hf = j % 2;                    \
    uint at = 128 * n + 32 * kk + 16 * hf, ia = at / 16;                                       \
    for (uint p0 = 0; p0 < n_tok; p0 += MVM_P) {                                               \
        uint np = n_tok - p0 < MVM_P ? n_tok - p0 : MVM_P;                                     \
        float acc[MVM_P];                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < MVM_P; p++) acc[p] = 0;                                           \
        for (uint c = 0; c + u < n_sb; c += 4) {                                               \
            const uchar* b = r + (c + u) * 210;                                                \
            ulong ql[3], qh[3], sc[3];                                                         \
            ld24(b + 64 * n + 32 * kk + 16 * hf, ql);                                          \
            ld24(b + 128 + 32 * n + 16 * hf, qh);                                              \
            ld24(b + 192, sc);                                                                 \
            U128 xa[MVM_P], xb[MVM_P];                                                         \
            float da[MVM_P], db[MVM_P];                                                        \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < MVM_P; p++) {                                                 \
                if (p >= np) continue;                                                         \
                ulong xo = (ulong)(p0 + p) * cols + 256 * (c + u) + at;                        \
                xa[p] = *(const U128*)(xq + xo / 4);                                           \
                xb[p] = *(const U128*)(xq + xo / 4 + 16);                                      \
                da[p] = xd[xo / 32];                                                           \
                db[p] = xd[xo / 32 + 2];                                                       \
            }                                                                                  \
            uint wa[4], wb[4];                                                                 \
            _Pragma("unroll")                                                                  \
            for (uint t = 0; t < 4; t++) {                                                     \
                uint lw = (uint)(ql[t / 2] >> 32 * (t % 2));                                   \
                uint hw = (uint)(qh[t / 2] >> 32 * (t % 2));                                   \
                wa[t] = word_q6_k(lw, hw, kk);                                                 \
                wb[t] = word_q6_k(lw, hw, kk + 2);                                             \
            }                                                                                  \
            float d = h_bits((uint)sc[2] & 0xFFFF);                                            \
            float fa = d * (float)(signed char)(sc[ia / 8] >> 8 * (ia % 8));                   \
            float fb = d * (float)(signed char)(sc[ia / 8] >> 8 * (ia % 8 + 4));               \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < MVM_P; p++) {                                                 \
                if (p >= np) continue;                                                         \
                int sa = 0, sb = 0;                                                            \
                _Pragma("unroll")                                                              \
                for (uint t = 0; t < 4; t++) {                                                 \
                    sa = dp4a(wa[t], xa[p].v[t], sa);                                          \
                    sb = dp4a(wb[t], xb[p].v[t], sb);                                          \
                }                                                                              \
                acc[p] += fa * da[p] * (float)sa + fb * db[p] * (float)sb;                     \
            }                                                                                  \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < MVM_P; p++) {                                                     \
            if (p >= np) continue;                                                             \
            float a = warp_sum(acc[p]);                                                        \
            float* yr = y + y_off + (ulong)(p0 + p) * rows + row;                              \
            if (lane == 0) *yr = add ? *yr + a : a;                                            \
        }                                                                                      \
    }                                                                                          \
}
MVQ6_MULTI(mvq_q6_k_multi2, 2)
MVQ6_MULTI(mvq_q6_k_multi4, 4)
#endif

#ifdef RLI_MMA
// The same product on int8 tensor cores (compute capability 8.0 and up; this part is compiled
// separately, for that and with a cap on registers). One instruction multiplies 16 weight rows
// by 8 tokens over 16 or 32 columns for a whole warp: lane 4i + j gives the 4 weights of rows
// i and i + 8 at columns 4j..4j+3 (and 16 + 4j.. for 32 columns) and the same columns of token
// i, and receives the sums of rows i and i + 8 with tokens 2j and 2j + 1.
//
// A block takes MA_R rows and MA_T tokens, 128 columns at a time. Warp w owns 16 rows
// (w % MA_RG), keeps their weights and scales in registers, and runs them against every
// (MA_NW / MA_RG)-th octet of tokens. Rows and tokens lie MA_S words apart in shared memory so
// a warp's loads fall in different banks. On an A100 the multiplies are a small part of the
// time: what set the shape was reading the tiles (larger is better) against registers per
// thread, which limit how many warps a multiprocessor holds (smaller is better). Every block
// stages its tokens again, so the tile is tall: 64 rows by 128 tokens read four times the
// weights' bytes in tokens on a 64-token chunk.
//
// T is the tokens of a block: a chunk of up to 16 or 32 has kernels of that size, since a
// block stages its whole tile whatever the chunk holds. At 16 the Q4_K lanes read their
// operands from the matrix themselves (DIRECT; a warp shares its rows with no other, so
// shared memory was a detour, 109 us against 91 for 14336x4096): no gain at 32 tokens, and
// 20% slower at 64.
//
// With `parts` the grid is that many times as deep: a block takes one part of the column
// steps and adds its sums into y, which the caller has set. Few rows are few blocks, and
// most of the GPU idle (a 4096x4096 matrix by 64 tokens: 179 us in 32 blocks, 56 in 512).
// `parts` 0 stores the sums of all the columns.
#define MA_R 128
#define MA_NW 8
#define MA_S 36
#define MA_RG (MA_R / 16)
#define MA_NO(T) ((T) / 8 / (MA_NW / MA_RG))

// c += a b^T over 16 columns or 32.
__device__ inline void mma16_s8(int a0, int a1, int b, int* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b));
}

__device__ inline void mma32_s8(int a0, int a1, int a2, int a3, int b0, int b1, int* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Tile loading reads everything it needs from global memory into registers before it stores
// anything: a warp waits for a read only when it uses the result, so the reads of all its rows
// are in flight together. With a store after each read they took one memory latency per row.
#define MA_ROWP(R) (w + sb_off + (ulong)(row0 + (R) < rows ? row0 + (R) : rows - 1) * row_bytes)

// K32: one scale per 32 weights (Q4_K, with an offset wm), so a step covers 32 columns and a
// row's half super-block takes 16 lanes; otherwise 16 columns and 32 lanes (Q6_K, no offset).
// Weights are ws * q - wm.
#define MATMUL_MMA(NAME, SB_BYTES, K32, T)                                                           \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const uint* xq, const float* xd,  \
                                const float* xs, float* y, ulong y_off, uint rows, uint cols,  \
                                uint row_bytes, uint n_tok, uint parts) {                      \
    const uint NG = K32 ? 4 : 8, NP = MA_R / MA_NW / (K32 ? 2 : 1), NT = T / MA_NW;         \
    uint acc_on = parts, ntt = (n_tok + T - 1) / T, part = blockIdx.y / ntt;             \
    if (!parts) parts = 1;                                                                     \
    const bool DIRECT = K32 && T <= 16;                                                        \
    __shared__ uint wq[DIRECT ? 1 : MA_R * MA_S];                                              \
    __shared__ uint tq[T * MA_S];                                                           \
    __shared__ float ws[DIRECT ? 1 : MA_R * (K32 ? 4 : 8)];                                    \
    __shared__ float wm[K32 && !DIRECT ? MA_R * 4 : 1];                                        \
    __shared__ float td[T * 4];                                                             \
    __shared__ float ts[K32 ? T * 4 : 1];                                                   \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint gi = lane / 4, gj = lane % 4;                                                         \
    uint row0 = blockIdx.x * MA_R, t0 = blockIdx.y % ntt * T;                               \
    uint last = n_tok - 1 - t0;                                                                \
    uint ra = 16 * (warp % MA_RG) + gi, ow = warp / MA_RG;                                     \
    float acc[4 * MA_NO(T)];                                                                      \
    _Pragma("unroll")                                                                          \
    for (uint j = 0; j < 4 * MA_NO(T); j++) acc[j] = 0;                                           \
    for (uint kh = part * (cols / 128) / parts; kh < (part + 1) * (cols / 128) / parts; kh++) {\
        ulong sb_off = w_off + (ulong)(kh / 2) * SB_BYTES;                                     \
        uint hf = kh % 2;                                                                      \
        uint v0[16], v1[16], w2[8], w3[8];                                                     \
        int a[8][4];                                                                           \
        float sa[8], sb[8], ma[8], mb[8];                                                      \
        if (DIRECT) {                                                                          \
            /* A warp's rows are its own: each lane reads its operands from the matrix. */     \
            const uchar* rb[2] = {MA_ROWP(ra), MA_ROWP(ra + 8)};                               \
            _Pragma("unroll")                                                                  \
            for (uint i = 0; i < 8; i++)                                                       \
                v0[i] = u32e(rb[i % 2] + 16 + 64 * hf + 32 * (i / 4) + 16 * (i / 2 % 2) + 4 * gj);\
            _Pragma("unroll")                                                                  \
            for (uint i = 0; i < 8; i++) v1[i] = u32e(rb[i % 2] + 4 * (i / 2));                \
            _Pragma("unroll")                                                                  \
            for (uint i = 0; i < 8; i++) {                                                     \
                /* Low nibbles are sub-block 2 (i / 4), high the next; columns 16 (i / 2 % 2).. */\
                a[2 * (i / 4)][i % 2 + 2 * (i / 2 % 2)] = v0[i] & 0x0F0F0F0Fu;                 \
                a[2 * (i / 4) + 1][i % 2 + 2 * (i / 2 % 2)] = (v0[i] >> 4) & 0x0F0F0F0Fu;      \
            }                                                                                  \
            _Pragma("unroll")                                                                  \
            for (uint i = 0; i < 8; i++) {                                                     \
                /* Sub-block 4 hf + l of row i % 2: bytes l, 4 + l and 8 + l of the 12. */     \
                uint r = i % 2, l = i / 2;                                                     \
                uint b0 = (v1[2 + r] >> 8 * l) & 255, b1 = (v1[4 + r] >> 8 * l) & 255;         \
                uint b2 = (v1[6 + r] >> 8 * l) & 255;                                          \
                uint sc = hf ? (b2 & 0xF) | ((b0 >> 6) << 4) : b0 & 63;                        \
                uint mn = hf ? (b2 >> 4) | ((b1 >> 6) << 4) : b1 & 63;                         \
                (r ? sb : sa)[l] = h_bits(v1[r] & 0xFFFF) * (float)sc;                         \
                (r ? mb : ma)[l] = h_bits(v1[r] >> 16) * (float)mn;                            \
            }                                                                                  \
        } else if (K32) {                                                                      \
            uint l = lane % 16, rl = 2 * warp + lane / 16;                                     \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < NP; p++)                                                      \
                v0[p] = u32e(MA_ROWP(rl + p * 2 * MA_NW) + 16 + 64 * hf + 4 * l);              \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < NP; p++) {                                                    \
                uint* wr = wq + (rl + p * 2 * MA_NW) * MA_S + 16 * (l / 8) + l % 8;            \
                wr[0] = v0[p] & 0x0F0F0F0Fu; /* low nibbles: sub-block 2 (l / 8) */            \
                wr[8] = (v0[p] >> 4) & 0x0F0F0F0Fu;                                            \
            }                                                                                  \
            if (l < 4) {                                                                       \
                _Pragma("unroll")                                                              \
                for (uint p = 0; p < NP; p++) {                                                \
                    const uchar* b = MA_ROWP(rl + p * 2 * MA_NW);                              \
                    v0[p] = u32e(b);                                                           \
                    v1[p] = u32e(b + 4);                                                       \
                    w2[p] = u32e(b + 8);                                                       \
                    w3[p] = u32e(b + 12);                                                      \
                }                                                                              \
                _Pragma("unroll")                                                              \
                for (uint p = 0; p < NP; p++) {                                                \
                    /* Sub-block 4 hf + l: bytes l, 4 + l and 8 + l of the 12 scale bytes. */  \
                    uint b0 = (v1[p] >> 8 * l) & 255, b1 = (w2[p] >> 8 * l) & 255;             \
                    uint b2 = (w3[p] >> 8 * l) & 255;                                          \
                    uint sc = hf ? (b2 & 0xF) | ((b0 >> 6) << 4) : b0 & 63;                    \
                    uint mn = hf ? (b2 >> 4) | ((b1 >> 6) << 4) : b1 & 63;                     \
                    uint r = rl + p * 2 * MA_NW;                                               \
                    ws[r * 4 + l] = h_bits(v0[p] & 0xFFFF) * (float)sc;                        \
                    wm[K32 ? r * 4 + l : 0] = h_bits(v0[p] >> 16) * (float)mn;                 \
                }                                                                              \
            }                                                                                  \
        } else {                                                                               \
            uint k = lane / 8, i = 4 * (lane % 8);                                             \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < NP; p++) {                                                    \
                const uchar* b = MA_ROWP(warp + p * MA_NW);                                    \
                v0[p] = u32e(b + 64 * hf + 32 * (k % 2) + i);                                  \
                v1[p] = u32e(b + 128 + 32 * hf + i);                                           \
            }                                                                                  \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < NP; p++)                                                      \
                wq[(warp + p * MA_NW) * MA_S + lane] = word_q6_k(v0[p], v1[p], k);          \
            if (lane < 8) {                                                                    \
                _Pragma("unroll")                                                              \
                for (uint p = 0; p < NP; p++) {                                                \
                    const uchar* b = MA_ROWP(warp + p * MA_NW);                                \
                    v0[p] = u16(b + 208);                                                      \
                    v1[p] = (uint)(int)((const signed char*)b)[192 + 8 * hf + lane];           \
                }                                                                              \
                _Pragma("unroll")                                                              \
                for (uint p = 0; p < NP; p++)                                                  \
                    ws[(warp + p * MA_NW) * 8 + lane] = h_bits(v0[p]) * (float)(int)v1[p];     \
            }                                                                                  \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < NT; p++) {                                                        \
            uint t = warp + p * MA_NW;                                                         \
            v0[p] = xq[(ulong)(t0 + (t < last ? t : last)) * (cols / 4) + kh * 32 + lane];     \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < NT; p++) tq[(warp + p * MA_NW) * MA_S + lane] = v0[p];            \
        if (lane < 4) {                                                                        \
            float f0[16], f1[16], f2[16];                                                      \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < NT; p++) {                                                    \
                uint t = warp + p * MA_NW;                                                     \
                ulong tok = t0 + (t < last ? t : last);                                        \
                f0[p] = xd[tok * (cols / 32) + kh * 4 + lane];                                 \
                f1[p] = K32 ? xs[tok * (cols / 16) + kh * 8 + 2 * lane] : 0;                   \
                f2[p] = K32 ? xs[tok * (cols / 16) + kh * 8 + 2 * lane + 1] : 0;               \
            }                                                                                  \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < NT; p++) {                                                    \
                uint t = warp + p * MA_NW;                                                     \
                td[t * 4 + lane] = f0[p];                                                      \
                if (K32) ts[t * 4 + lane] = f1[p] + f2[p];                                     \
            }                                                                                  \
        }                                                                                      \
        __syncthreads();                                                                       \
        {                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint g = 0; g < (DIRECT ? 0 : NG); g++) {                                         \
            const uint* ap = wq + ra * MA_S + (K32 ? 8 : 4) * g + gj;                          \
            a[g][0] = ap[0];                                                                   \
            a[g][1] = ap[8 * MA_S];                                                            \
            a[g][2] = K32 ? ap[4] : 0;                                                         \
            a[g][3] = K32 ? ap[8 * MA_S + 4] : 0;                                              \
            sa[g] = ws[ra * NG + g];                                                           \
            sb[g] = ws[(ra + 8) * NG + g];                                                     \
            ma[g] = K32 ? wm[ra * NG + g] : 0;                                                 \
            mb[g] = K32 ? wm[(ra + 8) * NG + g] : 0;                                           \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint o = 0; o < MA_NO(T); o++) {                                                     \
            uint tb = (o * (MA_NW / MA_RG) + ow) * 8, ta = tb + 2 * gj;                        \
            if (tb > last) continue;                                                           \
            _Pragma("unroll")                                                                  \
            for (uint g = 0; g < NG; g++) {                                                    \
                const uint* bp = tq + (tb + gi) * MA_S + (K32 ? 8 : 4) * g + gj;               \
                uint di = K32 ? g : g / 2;                                                     \
                float d0 = td[ta * 4 + di], d1 = td[(ta + 1) * 4 + di];                        \
                int c[4] = {0, 0, 0, 0};                                                       \
                if (K32) mma32_s8(a[g][0], a[g][1], a[g][2], a[g][3], bp[0], bp[4], c);        \
                else mma16_s8(a[g][0], a[g][1], bp[0], c);                                     \
                acc[4 * o] += sa[g] * d0 * (float)c[0];                                        \
                acc[4 * o + 1] += sa[g] * d1 * (float)c[1];                                    \
                acc[4 * o + 2] += sb[g] * d0 * (float)c[2];                                    \
                acc[4 * o + 3] += sb[g] * d1 * (float)c[3];                                    \
                if (K32) {                                                                     \
                    float s0 = ts[ta * 4 + g], s1 = ts[(ta + 1) * 4 + g];                      \
                    acc[4 * o] -= ma[g] * s0;                                                  \
                    acc[4 * o + 1] -= ma[g] * s1;                                              \
                    acc[4 * o + 2] -= mb[g] * s0;                                              \
                    acc[4 * o + 3] -= mb[g] * s1;                                              \
                }                                                                              \
            }                                                                                  \
        }                                                                                      \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
    _Pragma("unroll")                                                                          \
    for (uint j = 0; j < 4 * MA_NO(T); j++) {                                                     \
        uint t = (j / 4 * (MA_NW / MA_RG) + ow) * 8 + 2 * gj + j % 2;                          \
        uint row = row0 + ra + 8 * (j / 2 % 2);                                                \
        if (t > last || row >= rows) continue;                                                 \
        if (acc_on) atomicAdd(y + y_off + (ulong)(t0 + t) * rows + row, acc[j]);               \
        else y[y_off + (ulong)(t0 + t) * rows + row] = acc[j];                                 \
    }                                                                                          \
}

MATMUL_MMA(mma_q4_k, 144, 1, 64)
MATMUL_MMA(mma_q6_k, 210, 0, 64)
MATMUL_MMA(mma_q4_k_t32, 144, 1, 32)
MATMUL_MMA(mma_q6_k_t32, 210, 0, 32)
MATMUL_MMA(mma_q4_k_t16, 144, 1, 16)
MATMUL_MMA(mma_q6_k_t16, 210, 0, 16)

// c += a b^T over 16 columns of f16: lane 4i + j gives columns 2j, 2j + 1 and 8 + 2j, 9 + 2j
// of rows i and i + 8 and of token i, and receives the sums of rows i and i + 8 with tokens
// 2j and 2j + 1.
__device__ inline void mma16_f16(const uint* a, uint b0, uint b1, float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// y = W x for n_tok tokens, W and x in f16 (rows of cols, a multiple of GE_K). A block of NW
// warps takes R rows and T tokens, GE_K columns at a time through shared memory; a warp owns
// 16 rows, holds their weights in registers and runs them against 8 octets of tokens. Rows
// and tokens lie GE_S words apart so a warp's loads fall in different banks. Tiles are copied
// 16 bytes at a time, every read issued before the first store (see MATMUL_MMA): thread i
// takes words 4 (i % 8).. of rows and tokens i / 8 + 4 NW p.
//
// More tokens per block means fewer reads of the weights, but a matrix of few rows then has
// too few blocks to occupy the GPU, which costs more: the caller picks. Blocks of 32 rows or
// 256 tokens were slower on an A100, and so are those of 128 rows unless about as many run
// as the GPU holds (docs/benchmarks.md). For that the columns can be split too: the grid is
// `parts` times as deep in tokens, block y takes token tile y % ntt and part y / ntt of the
// column steps, and each adds its sums into y (`acc`), which the caller has set. That is a
// kernel of its own (ACC): an atomic add costs these blocks nothing measurable, but the
// choices made at run time made the other kernels 4-8% slower. It also takes a range of the
// rows: `rows` of them from `base`, of a y with `ldy` rows a token.
#define GE_K 64
#define GE_S 36
struct __attribute__((aligned(16))) V128 {
    uint v[4];
};
#define GEMM_F16(NAME, R, T, NW, RG, ACC)                                                         \
extern "C" __global__ void NAME(const uint* wh, const uint* xh, float* y, ulong y_off,         \
                                uint rows, uint cols, uint n_tok, uint parts, uint base,       \
                                uint ldy, uint acc) {                                          \
    const uint NRG = R / (16 * RG), NWG = NW / NRG, NO = T / 8 / NWG;                          \
    if (!ACC) parts = 1, base = 0, ldy = rows, acc = 0;                                        \
    __shared__ uint wt[R * GE_S];                                                              \
    __shared__ uint xt[T * GE_S];                                                              \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint gi = lane / 4, gj = lane % 4;                                                         \
    uint ntt = (n_tok + T - 1) / T, part = ACC ? blockIdx.y / ntt : 0, steps = cols / GE_K;    \
    uint row0 = blockIdx.x * R, t0 = (ACC ? blockIdx.y % ntt : blockIdx.y) * T;                \
    uint last = n_tok - 1 - t0, kw = cols / 2;                                                 \
    uint ra = 16 * RG * (warp % NRG) + gi, ow = warp / NRG;                                    \
    uint tr = threadIdx.x / 8, tw = 4 * (threadIdx.x % 8);                                     \
    float c[RG][T / 8 / (NW / (R / (16 * RG)))][4];                                            \
    _Pragma("unroll")                                                                          \
    for (uint g = 0; g < RG; g++)                                                              \
        _Pragma("unroll")                                                                      \
        for (uint o = 0; o < NO; o++) c[g][o][0] = c[g][o][1] = c[g][o][2] = c[g][o][3] = 0;   \
    for (uint ks = part * steps / parts; ks < (part + 1) * steps / parts; ks++) {              \
        V128 v[8];                                                                             \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < R / (4 * NW); p++) {                                              \
            uint r = row0 + tr + 4 * NW * p;                                                   \
            r = base + (r < rows ? r : rows - 1);                                              \
            v[p] = *(const V128*)(wh + (ulong)r * kw + ks * (GE_K / 2) + tw);                  \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < R / (4 * NW); p++)                                                \
            *(V128*)(wt + (tr + 4 * NW * p) * GE_S + tw) = v[p];                               \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < T / (4 * NW); p++) {                                              \
            uint t = tr + 4 * NW * p;                                                          \
            v[p] = *(const V128*)(xh + (ulong)(t0 + (t < last ? t : last)) * kw + ks * (GE_K / 2) + tw); \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < T / (4 * NW); p++)                                                \
            *(V128*)(xt + (tr + 4 * NW * p) * GE_S + tw) = v[p];                               \
        __syncthreads();                                                                       \
        uint a[RG][GE_K / 16][4];                                                              \
        _Pragma("unroll")                                                                      \
        for (uint g = 0; g < RG; g++)                                                          \
            _Pragma("unroll")                                                                  \
            for (uint k = 0; k < GE_K / 16; k++) {                                             \
                const uint* ap = wt + (ra + 16 * g) * GE_S + 8 * k + gj;                       \
                a[g][k][0] = ap[0];                                                            \
                a[g][k][1] = ap[8 * GE_S];                                                     \
                a[g][k][2] = ap[4];                                                            \
                a[g][k][3] = ap[8 * GE_S + 4];                                                 \
            }                                                                                  \
        _Pragma("unroll")                                                                      \
        for (uint o = 0; o < NO; o++) {                                                        \
            uint tb = (NWG * o + ow) * 8;                                                      \
            if (tb > last) continue;                                                           \
            _Pragma("unroll")                                                                  \
            for (uint k = 0; k < GE_K / 16; k++) {                                             \
                const uint* bp = xt + (tb + gi) * GE_S + 8 * k + gj;                           \
                uint b0 = bp[0], b1 = bp[4];                                                   \
                _Pragma("unroll")                                                              \
                for (uint g = 0; g < RG; g++) mma16_f16(a[g][k], b0, b1, c[g][o]);             \
            }                                                                                  \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
    _Pragma("unroll")                                                                          \
    for (uint g = 0; g < RG; g++)                                                              \
        _Pragma("unroll")                                                                      \
        for (uint j = 0; j < 4 * NO; j++) {                                                    \
            uint t = (NWG * (j / 4) + ow) * 8 + 2 * gj + j % 2;                                \
            uint row = row0 + ra + 16 * g + 8 * (j / 2 % 2);                                   \
            if (t > last || row >= rows) continue;                                             \
            float* out = y + y_off + (ulong)(t0 + t) * ldy + base + row;                       \
            if (acc)                                                                           \
                atomicAdd(out, c[g][j / 4][j % 4]);                                            \
            else                                                                               \
                *out = c[g][j / 4][j % 4];                                                     \
        }                                                                                      \
}

GEMM_F16(gemm_f16_64x64, 64, 64, 8, 1, 0)
// Four warps of four row groups each, so a read of the token tile feeds four products: 14-20%
// faster on an A100 than eight warps of one group, 5-7% than four of two.
GEMM_F16(gemm_f16_64x128, 64, 128, 4, 4, 0)
// A warp of 64 rows by 64 tokens: where the column parts fill the GPU (`gemm_parts`).
GEMM_F16(gemm_f16_128x128, 128, 128, 4, 4, 1)

// Eight bytes at any address, as two aligned reads (so up to 8 bytes past the end are read;
// the weights end in slack for that).
__device__ inline ulong ld8(const uchar* p) {
    const ulong* a = (const ulong*)((ulong)p & ~7ull);
    uint sh = 8 * (uint)((ulong)p & 7);
    ulong lo = a[0], hi = a[1];
    return sh ? (lo >> sh) | (hi << (64 - sh)) : lo;
}

// The decoders of GEMM_ID. A step is 64 columns: quarter j of the super-block at b, of which
// a thread decodes 16 weights of one row, columns 8 dp.. (lo) and 32 + 8 dp.. (hi), as f16.
// *_hdr reads what a super-block's four steps share, *_load a step's bytes (`first` in the
// block's first step; `raw` is kept from step to step), *_dec decodes.
//
// Q4_K: the quarter is 32 bytes, the low nibbles one sub-block and the high ones the next.
// The row must lie at a multiple of 16 bytes.
__device__ inline void gq4_k_hdr(const uchar* b, ulong* hdr) {
    hdr[0] = ((const ulong*)b)[0];
    hdr[1] = ((const ulong*)b)[1];
    // d and dmin as f32, converted here and not in each of the super-block's four steps.
    hdr[2] = __float_as_uint(h_bits((uint)hdr[0] & 0xFFFF)) |
             ((ulong)__float_as_uint(h_bits((uint)(hdr[0] >> 16) & 0xFFFF)) << 32);
}
__device__ inline void gq4_k_load(const uchar* b, uint j, bool first, uint dp, ulong* raw) {
    raw[0] = *(const ulong*)(b + 16 + 32 * j + 8 * dp);
}
// The scales and offsets of sub-blocks 2j (sa, ma) and 2j + 1 (sc, mc), from gq4_k_hdr's words.
__device__ inline void gq4_k_scales(const ulong* hdr, uint j, float* sa, float* ma, float* sc,
                                    float* mc) {
    float d = __uint_as_float((uint)hdr[2]), dmin = __uint_as_float((uint)(hdr[2] >> 32));
    // 6 bits each, the first four sub-blocks' in a byte's low bits, the rest split between
    // the top two bits of those bytes and a nibble (q4_k_scale).
    uint v1 = (uint)(hdr[0] >> 32), v2 = (uint)hdr[1], v3 = (uint)(hdr[1] >> 32);
    uint b0 = (v1 >> 16 * (j % 2)) & 0xFFFF, b1 = (v2 >> 16 * (j % 2)) & 0xFFFF;
    uint b2 = (v3 >> 16 * (j % 2)) & 0xFFFF;
    uint sl = j < 2 ? b0 : (b2 & 0x0F0F) | ((b0 >> 2) & 0x3030);
    uint ml = j < 2 ? b1 : ((b2 >> 4) & 0x0F0F) | ((b1 >> 2) & 0x3030);
    *sa = d * (float)(sl & 63), *sc = d * (float)((sl >> 8) & 63);
    *ma = dmin * (float)(ml & 63), *mc = dmin * (float)((ml >> 8) & 63);
}
__device__ inline void gq4_k_dec(const ulong* hdr, const ulong* raw, uint j, uint dp, V128& lo,
                                 V128& hi) {
    float sa, ma, sc, mc;
    gq4_k_scales(hdr, j, &sa, &ma, &sc, &mc);
    #pragma unroll
    for (uint k = 0; k < 4; k++) {
        uint by = (uint)(raw[0] >> 16 * k) & 0xFFFF;
        uint a0 = f2h(sa * (float)(by & 15) - ma);
        uint a1 = f2h(sa * (float)((by >> 8) & 15) - ma);
        uint c0 = f2h(sc * (float)((by >> 4) & 15) - mc);
        uint c1 = f2h(sc * (float)(by >> 12) - mc);
        lo.v[k] = a0 | (a1 << 16);
        hi.v[k] = c0 | (c1 << 16);
    }
}

// Q6_K: half j / 2 of the super-block has 64 bytes of low nibbles and 32 of high bit pairs;
// quarter j takes the nibbles' low (j even) or high halves, and a scale per 16 weights.
__device__ inline void gq6_k_hdr(const uchar* b, ulong* hdr) {
    hdr[0] = ld8(b + 192);
    hdr[1] = ld8(b + 200);
    hdr[2] = ld8(b + 208);
}
__device__ inline void gq6_k_load(const uchar* b, uint j, bool first, uint dp, ulong* raw) {
    // Quarters 2n and 2n + 1 are the two nibbles of the same bytes: read once for both.
    if (j % 2 && !first) return;
    uint n = j / 2;
    raw[0] = ld8(b + 64 * n + 8 * dp);
    raw[1] = ld8(b + 64 * n + 32 + 8 * dp);
    raw[2] = ld8(b + 128 + 32 * n + 8 * dp);
}
__device__ inline void gq6_k_dec(const ulong* hdr, const ulong* raw, uint j, uint dp, V128& lo,
                                 V128& hi) {
    uint sh = 4 * (j % 2), is = 8 * (j / 2) + dp / 2 + sh;
    float d = h_bits((uint)hdr[2] & 0xFFFF);
    float fa = d * (float)(signed char)(hdr[is / 8] >> 8 * (is % 8));
    float fb = d * (float)(signed char)(hdr[(is + 2) / 8] >> 8 * ((is + 2) % 8));
    ulong qa = raw[0] >> sh, qb = raw[1] >> sh, ha = raw[2] >> sh, hb = raw[2] >> (sh + 2);
    #pragma unroll
    for (uint k = 0; k < 4; k++) {
        uint a = (uint)(qa >> 16 * k), c = (uint)(qb >> 16 * k);
        uint e = (uint)(ha >> 16 * k), g = (uint)(hb >> 16 * k);
        int a0 = (a & 15) | ((e & 3) << 4), a1 = ((a >> 8) & 15) | (((e >> 8) & 3) << 4);
        int c0 = (c & 15) | ((g & 3) << 4), c1 = ((c >> 8) & 15) | (((g >> 8) & 3) << 4);
        lo.v[k] = f2h(fa * (float)(a0 - 32)) | ((uint)f2h(fa * (float)(a1 - 32)) << 16);
        hi.v[k] = f2h(fb * (float)(c0 - 32)) | ((uint)f2h(fb * (float)(c1 - 32)) << 16);
    }
}

// GEMM_F16 for stacked experts over the pairs moe_group listed, the weights decoded to f16 on
// the way into shared memory: block (x, e) takes 64 rows of expert e (matrices `stride` bytes
// apart) and the pairs that chose e, 64 at a time. Pair p reads row p / x_div of xh and writes
// row p of y. Thread i decodes for row i / 4, into words 4 (i % 4).. of each half of the step.
// As in GEMM_F16, a step's reads are all issued before its first store.
// T is the tokens a block takes at a time, 64 or 128, and it has 4 NW warps: each takes 16
// rows by T / NW tokens, and the first 8 decode. 128 tokens on 8 warps had a warp hold twice
// the sums and tiles, and ran Q4_K 6-13% slower than on 16 (Q6_K, though, 40% faster).
#define GEMM_ID(NAME, SB_BYTES, HDR, LOAD, DEC, DENSE, ARGS, NTOK, PARTS, T, NW)               \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, ulong stride, const uint* perm,   \
                                const uint* start, const uint* xh, uint x_div, float* y,       \
                                ulong y_off, uint rows, uint cols, uint row_bytes ARGS) {      \
    uint ntt = DENSE ? ((NTOK) + T - 1) / T : 1, part = DENSE ? blockIdx.y / ntt : 0;          \
    uint ey = DENSE ? 0 : blockIdx.y, pb = DENSE ? blockIdx.y % ntt * T : start[ey];           \
    uint kb = part * (cols / 64) / (PARTS), ke = (part + 1) * (cols / 64) / (PARTS);           \
    __shared__ uint wt[64 * GE_S];                                                             \
    __shared__ uint xt[T * GE_S];                                                              \
    __shared__ uint pr[T];                                                                     \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint gi = lane / 4, gj = lane % 4;                                                         \
    uint row0 = blockIdx.x * 64, kw = cols / 2;                                                \
    uint p1 = DENSE ? ((NTOK) < pb + T ? (NTOK) : pb + T) : start[ey + 1];                     \
    uint ra = 16 * (warp % 4) + gi, ow = warp / 4;                                             \
    uint tr = threadIdx.x / 8, tw = 4 * (threadIdx.x % 8);                                     \
    uint dr = threadIdx.x / 4, dp = threadIdx.x % 4;                                           \
    bool dec = threadIdx.x < 256;                                                              \
    const uchar* wr = w + w_off + ey * stride +                                                \
                      (ulong)(row0 + dr < rows ? row0 + dr : rows - 1) * row_bytes;            \
    for (uint p0 = pb; p0 < p1; p0 += T) {                                                     \
        uint last = (p1 - p0 < T ? p1 - p0 : T) - 1;                                           \
        if (threadIdx.x <= last) pr[threadIdx.x] = DENSE ? p0 + threadIdx.x : perm[p0 + threadIdx.x];\
        __syncthreads();                                                                       \
        ulong xr[T / (16 * NW)];                                                               \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < T / (16 * NW); p++)                                               \
            xr[p] = tr + 16 * NW * p <= last ? pr[tr + 16 * NW * p] / x_div : 0;               \
        float c[T / (8 * NW)][4];                                                              \
        _Pragma("unroll")                                                                      \
        for (uint o = 0; o < T / (8 * NW); o++) c[o][0] = c[o][1] = c[o][2] = c[o][3] = 0;     \
        ulong hdr[3], raw[3];                                                                  \
        for (uint ks = kb; ks < ke; ks++) {                                                    \
            uint j = ks % 4;                                                                   \
            const uchar* b = wr + (ulong)(ks / 4) * SB_BYTES;                                  \
            if (dec) {                                                                         \
                if (j == 0 || ks == kb) HDR(b, hdr);                                           \
                LOAD(b, j, ks == kb, dp, raw);                                                 \
            }                                                                                  \
            V128 v[T / (16 * NW)];                                                             \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < T / (16 * NW); p++)                                           \
                if (tr + 16 * NW * p <= last)                                                  \
                    v[p] = *(const V128*)(xh + xr[p] * kw + ks * 32 + tw);                     \
            if (dec) {                                                                         \
                V128 lo, hi;                                                                   \
                DEC(hdr, raw, j, dp, lo, hi);                                                  \
                *(V128*)(wt + dr * GE_S + 4 * dp) = lo;                                        \
                *(V128*)(wt + dr * GE_S + 16 + 4 * dp) = hi;                                   \
            }                                                                                  \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < T / (16 * NW); p++)                                           \
                if (tr + 16 * NW * p <= last)                                                  \
                    *(V128*)(xt + (tr + 16 * NW * p) * GE_S + tw) = v[p];                      \
            __syncthreads();                                                                   \
            uint a[4][4];                                                                      \
            _Pragma("unroll")                                                                  \
            for (uint k = 0; k < 4; k++) {                                                     \
                const uint* ap = wt + ra * GE_S + 8 * k + gj;                                  \
                a[k][0] = ap[0];                                                               \
                a[k][1] = ap[8 * GE_S];                                                        \
                a[k][2] = ap[4];                                                               \
                a[k][3] = ap[8 * GE_S + 4];                                                    \
            }                                                                                  \
            _Pragma("unroll")                                                                  \
            for (uint o = 0; o < T / (8 * NW); o++) {                                          \
                uint tb = (NW * o + ow) * 8;                                                   \
                if (tb > last) continue;                                                       \
                _Pragma("unroll")                                                              \
                for (uint k = 0; k < 4; k++) {                                                 \
                    const uint* bp = xt + (tb + gi) * GE_S + 8 * k + gj;                       \
                    mma16_f16(a[k], bp[0], bp[4], c[o]);                                       \
                }                                                                              \
            }                                                                                  \
            __syncthreads();                                                                   \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint j = 0; j < T / (2 * NW); j++) {                                              \
            uint t = (NW * (j / 4) + ow) * 8 + 2 * gj + j % 2;                                 \
            uint row = row0 + ra + 8 * (j / 2 % 2);                                            \
            if (t > last || row >= rows) continue;                                             \
            float* out = y + y_off + (ulong)pr[t] * rows + row;                                \
            if (DENSE) atomicAdd(out, c[j / 4][j % 4]);                                        \
            else *out = c[j / 4][j % 4];                                                       \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
}

GEMM_ID(gemm_q4_k_id, 144, gq4_k_hdr, gq4_k_load, gq4_k_dec, 0, , 0, 1, 64, 2)
GEMM_ID(gemm_q6_k_id, 210, gq6_k_hdr, gq6_k_load, gq6_k_dec, 0, , 0, 1, 64, 2)
#define GEMM_DENSE_ARGS , uint n_tok, uint parts
GEMM_ID(gemm_q4_k_dense, 144, gq4_k_hdr, gq4_k_load, gq4_k_dec, 1, GEMM_DENSE_ARGS, n_tok, parts, 64, 2)
GEMM_ID(gemm_q4_k_dense128, 144, gq4_k_hdr, gq4_k_load, gq4_k_dec, 1, GEMM_DENSE_ARGS, n_tok, parts, 128, 4)
GEMM_ID(gemm_q6_k_dense, 210, gq6_k_hdr, gq6_k_load, gq6_k_dec, 1, GEMM_DENSE_ARGS, n_tok, parts, 64, 2)
GEMM_ID(gemm_q6_k_dense128, 210, gq6_k_hdr, gq6_k_load, gq6_k_dec, 1, GEMM_DENSE_ARGS, n_tok, parts, 128, 2)

// GEMM_ID for few tokens a tile: a block takes 128 rows by 32 tokens, and a lane decodes the
// weights of its own two rows that its multiplies take straight from the matrix. Only the
// tokens go through shared memory: a warp shares its rows with no other, and staging them
// cost more than decoding them. A step's 64 columns are dealt so that lane gj's 16 are those
// of one 8-byte word of the row, 8gj.. and 32 + 8gj.. (four 2-byte reads per row, with the
// columns in order, were 6% slower): multiply k takes columns 8gj + 4k + 2half.. as its
// columns 8half + 2gj.., and the tokens are staged in that order.
//
// *_load reads a step's bytes of one row (kept from step to step in `raw`; `first` in the
// block's first step), *_dec decodes them into rows h of the four multiplies' fragments.
//
// Q4_K: the low nibbles are sub-block 2j, the high ones 2j + 1. Rows must lie at a multiple
// of 16 bytes.
__device__ inline void dq4_k_load(const uchar* b, uint j, bool first, uint gj, ulong* raw) {
    raw[0] = *(const ulong*)(b + 16 + 32 * j + 8 * gj);
}
__device__ inline void dq4_k_dec(const ulong* hdr, const ulong* raw, uint j, uint gj, uint h,
                                 uint a[4][4]) {
    float sa, ma, sc, mc;
    gq4_k_scales(hdr, j, &sa, &ma, &sc, &mc);
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint q = (uint)(raw[0] >> 16 * i) & 0xFFFF;
        uint a0 = f2h(sa * (float)(q & 15) - ma);
        uint a1 = f2h(sa * (float)((q >> 8) & 15) - ma);
        uint c0 = f2h(sc * (float)((q >> 4) & 15) - mc);
        uint c1 = f2h(sc * (float)(q >> 12) - mc);
        // Multiply i / 2 of each sub-block; rows h, its columns 2gj.. then 8 + 2gj...
        a[i / 2][h + 2 * (i % 2)] = a0 | (a1 << 16);
        a[2 + i / 2][h + 2 * (i % 2)] = c0 | (c1 << 16);
    }
}
// Q6_K: 8 bytes of low nibbles for each half of the lane's columns and 8 of high bits for
// both, read once for the two steps that share them; the lane's columns share a scale.
__device__ inline void dq6_k_load(const uchar* b, uint j, bool first, uint gj, ulong* raw) {
    if (j % 2 && !first) return;
    raw[0] = ld8(b + 64 * (j / 2) + 8 * gj);
    raw[1] = ld8(b + 64 * (j / 2) + 32 + 8 * gj);
    raw[2] = ld8(b + 128 + 32 * (j / 2) + 8 * gj);
}
__device__ inline void dq6_k_dec(const ulong* hdr, const ulong* raw, uint j, uint gj, uint h,
                                 uint a[4][4]) {
    uint sh = 4 * (j % 2), ia = 8 * (j / 2) + sh + gj / 2, ic = ia + 2;
    float d = h_bits((uint)hdr[2] & 0xFFFF);
    float fa = d * (float)(signed char)(hdr[ia / 8] >> 8 * (ia % 8));
    float fc = d * (float)(signed char)(hdr[ic / 8] >> 8 * (ic % 8));
    #pragma unroll
    for (uint i = 0; i < 4; i++) {
        uint la = (uint)(raw[0] >> 16 * i) >> sh, lc = (uint)(raw[1] >> 16 * i) >> sh;
        uint e = (uint)(raw[2] >> 16 * i) >> sh, g = e >> 2;
        int a0 = (la & 15) | ((e & 3) << 4), a1 = ((la >> 8) & 15) | (((e >> 8) & 3) << 4);
        int c0 = (lc & 15) | ((g & 3) << 4), c1 = ((lc >> 8) & 15) | (((g >> 8) & 3) << 4);
        a[i / 2][h + 2 * (i % 2)] =
            f2h(fa * (float)(a0 - 32)) | ((uint)f2h(fa * (float)(a1 - 32)) << 16);
        a[2 + i / 2][h + 2 * (i % 2)] =
            f2h(fc * (float)(c0 - 32)) | ((uint)f2h(fc * (float)(c1 - 32)) << 16);
    }
}

// With ID, block (x, e) takes 128 rows of expert e and the pairs that chose it, 32 at a
// time, and stores its results; without, 32 of a chunk's tokens and a part of the columns,
// adding into y.
#define GEMM_DIRECT(NAME, SB_BYTES, HDR, LOAD, DEC, ID, ARGS, NTOK, PARTS)                     \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, ulong stride, const uint* perm,   \
                                const uint* start, const uint* xh, uint x_div, float* y,       \
                                ulong y_off, uint rows, uint cols, uint row_bytes ARGS) {      \
    __shared__ uint xt[32 * GE_S];                                                             \
    __shared__ uint pr[32];                                                                    \
    uint ntt = ID ? 1 : ((NTOK) + 31) / 32, part = ID ? 0 : blockIdx.y / ntt;                  \
    uint ey = ID ? blockIdx.y : 0, pb = ID ? start[ey] : blockIdx.y % ntt * 32;                \
    uint p1 = ID ? start[ey + 1] : ((NTOK) < pb + 32 ? (NTOK) : pb + 32);                      \
    uint kb = part * (cols / 64) / (PARTS), ke = (part + 1) * (cols / 64) / (PARTS);           \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32, gi = lane / 4, gj = lane % 4;       \
    uint row0 = blockIdx.x * 128, ra = 16 * warp + gi, kw = cols / 2;                          \
    uint tr = threadIdx.x / 8, tw = 4 * (threadIdx.x % 8);                                     \
    const uchar* wr[2];                                                                        \
    _Pragma("unroll")                                                                          \
    for (uint h = 0; h < 2; h++) {                                                             \
        uint r = row0 + ra + 8 * h;                                                            \
        wr[h] = w + w_off + ey * stride + (ulong)(r < rows ? r : rows - 1) * row_bytes;        \
    }                                                                                          \
    for (uint p0 = pb; p0 < p1; p0 += 32) {                                                    \
        uint last = (p1 - p0 < 32 ? p1 - p0 : 32) - 1;                                         \
        if (ID) {                                                                              \
            if (threadIdx.x <= last) pr[threadIdx.x] = perm[p0 + threadIdx.x];                 \
            __syncthreads();                                                                   \
        }                                                                                      \
        uint tl = tr <= last ? tr : last;                                                      \
        ulong xr = ID ? pr[tl] / x_div : p0 + tl;                                              \
        const V128* xp = (const V128*)(xh + xr * kw + tw);                                     \
        float c[4][4];                                                                         \
        _Pragma("unroll")                                                                      \
        for (uint o = 0; o < 4; o++) c[o][0] = c[o][1] = c[o][2] = c[o][3] = 0;                \
        ulong hdr[2][3], raw[2][3];                                                            \
        for (uint ks = kb; ks < ke; ks++) {                                                    \
            uint j = ks % 4;                                                                   \
            _Pragma("unroll")                                                                  \
            for (uint h = 0; h < 2; h++) {                                                     \
                const uchar* b = wr[h] + (ulong)(ks / 4) * SB_BYTES;                           \
                if (j == 0 || ks == kb) HDR(b, hdr[h]);                                        \
                LOAD(b, j, ks == kb, gj, raw[h]);                                              \
            }                                                                                  \
            V128 v = xp[ks * 8];                                                               \
            _Pragma("unroll")                                                                  \
            for (uint i = 0; i < 4; i++)                                                       \
                xt[tr * GE_S + 16 * (tw / 16) + 4 * i + tw / 4 % 4] = v.v[i];                  \
            uint a[4][4];                                                                      \
            _Pragma("unroll")                                                                  \
            for (uint h = 0; h < 2; h++) DEC(hdr[h], raw[h], j, gj, h, a);                     \
            __syncthreads();                                                                   \
            _Pragma("unroll")                                                                  \
            for (uint o = 0; o < 4; o++) {                                                     \
                if (8 * o > last) continue;                                                    \
                _Pragma("unroll")                                                              \
                for (uint k = 0; k < 4; k++) {                                                 \
                    const uint* bp = xt + (8 * o + gi) * GE_S + 8 * k + gj;                    \
                    mma16_f16(a[k], bp[0], bp[4], c[o]);                                       \
                }                                                                              \
            }                                                                                  \
            __syncthreads();                                                                   \
        }                                                                                      \
        _Pragma("unroll")                                                                      \
        for (uint j = 0; j < 16; j++) {                                                        \
            uint t = 8 * (j / 4) + 2 * gj + j % 2, row = row0 + ra + 8 * (j / 2 % 2);          \
            if (t > last || row >= rows) continue;                                             \
            float* out = y + y_off + (ulong)(ID ? pr[t] : p0 + t) * rows + row;                \
            if (ID) *out = c[j / 4][j % 4];                                                    \
            else atomicAdd(out, c[j / 4][j % 4]);                                              \
        }                                                                                      \
        if (ID) __syncthreads();                                                               \
    }                                                                                          \
}

GEMM_DIRECT(gemm_q4_k_direct, 144, gq4_k_hdr, dq4_k_load, dq4_k_dec, 0, GEMM_DENSE_ARGS, n_tok, parts)
GEMM_DIRECT(gemm_q6_k_direct, 210, gq6_k_hdr, dq6_k_load, dq6_k_dec, 0, GEMM_DENSE_ARGS, n_tok, parts)
GEMM_DIRECT(gemm_q4_k_direct_id, 144, gq4_k_hdr, dq4_k_load, dq4_k_dec, 1, , 0, 1)
GEMM_DIRECT(gemm_q6_k_direct_id, 210, gq6_k_hdr, dq6_k_load, dq6_k_dec, 1, , 0, 1)

// attn_scores and attn_pv (below, in the other compile) on f16 tensor cores, with f32 sums:
// the same grids and tiles of 32 query rows by 32 positions, staged as f16. head_dim a
// multiple of 32, at most 128. Not under `precise`.
//
// Scores: a warp takes 16 query rows by 8 positions; Q is the product's rows, K its columns.
#define AT_S 68  // words between staged rows of head_dim / 2: a warp's reads in 32 banks
extern "C" __global__ void attn_scores_h(const float* q, const float* kc, float* scores,
                                         uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                         uint kv_dim, uint max_ctx, float scale, uint tok0,
                                         uint n_tok) {
    __shared__ uint kt[32 * AT_S];
    __shared__ uint qt[32 * AT_S];
    uint tid = threadIdx.x, g_n = n_head / n_kv, tpt = 32 / g_n;
    uint base = blockIdx.y * 32, t0 = blockIdx.z * tpt;
    uint t_end = min(t0 + tpt, n_tok);
    if (base >= n_pos + tok0 + t_end - 1) return;
    uint n4 = head_dim / 4;
    float4 kr[4], qr[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint i = tid + j * blockDim.x, r = i / n4, d = 4 * (i % n4), t = t0 + r / g_n;
        kr[j] = qr[j] = make_float4(0, 0, 0, 0);
        if (i >= 32 * n4) continue;
        if (base + r < n_pos + tok0 + t_end - 1)
            kr[j] = *(const float4*)(kc + (ulong)(base + r) * kv_dim + blockIdx.x * head_dim + d);
        if (t < n_tok)
            qr[j] = *(const float4*)(
                q + ((ulong)(tok0 + t) * n_head + blockIdx.x * g_n + r % g_n) * head_dim + d);
    }
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint i = tid + j * blockDim.x, r = i / n4, w = 2 * (i % n4);
        if (i >= 32 * n4) continue;
        kt[r * AT_S + w] = f2h(kr[j].x) | ((uint)f2h(kr[j].y) << 16);
        kt[r * AT_S + w + 1] = f2h(kr[j].z) | ((uint)f2h(kr[j].w) << 16);
        qt[r * AT_S + w] = f2h(qr[j].x) | ((uint)f2h(qr[j].y) << 16);
        qt[r * AT_S + w + 1] = f2h(qr[j].z) | ((uint)f2h(qr[j].w) << 16);
    }
    __syncthreads();
    uint lane = tid % 32, warp = tid / 32, gi = lane / 4, gj = lane % 4;
    uint rg = warp % 2, oc = warp / 2;
    float c[4] = {0, 0, 0, 0};
    for (uint k = 0; k < head_dim / 16; k++) {
        const uint* ap = qt + (16 * rg + gi) * AT_S + 8 * k + gj;
        const uint* bp = kt + (8 * oc + gi) * AT_S + 8 * k + gj;
        uint a[4] = {ap[0], ap[8 * AT_S], ap[4], ap[8 * AT_S + 4]};
        mma16_f16(a, bp[0], bp[4], c);
    }
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        uint r = 16 * rg + gi + 8 * (j / 2), p = 8 * oc + 2 * gj + j % 2, t = t0 + r / g_n;
        if (t < n_tok && base + p < n_pos + tok0 + t)
            scores[((ulong)t * n_head + blockIdx.x * g_n + r % g_n) * max_ctx + base + p] =
                c[j] * scale;
    }
}

// The weighted sums of V: 256 threads a block; a warp takes 16 query rows by head_dim / 4
// values, the scores its product's rows and V, staged with a row per value, its columns.
// With `half` the result goes to oh as f16 too, for the output projection.
#define AV_S 20  // words between staged rows of 16
#define PVH_R 64  // query rows a block
#define PVH_READ(BASE)                                                                         \
    _Pragma("unroll")                                                                          \
    for (uint j = 0; j < 4; j++) {                                                             \
        uint i = tid + j * blockDim.x, p = i / n4;                                             \
        vr[j] = i < 32 * n4 && (BASE) + p < limit                                              \
            ? *(const float4*)(vc + (ulong)((BASE) + p) * kv_dim + blockIdx.x * head_dim       \
                               + 4 * (i % n4))                                                 \
            : make_float4(0, 0, 0, 0);                                                         \
    }                                                                                          \
    _Pragma("unroll")                                                                          \
    for (uint j = 0; j < PVH_R / 16; j++) {                                                    \
        uint i = tid + j * blockDim.x, r = i / 16, t = t0 + r / g_n;                           \
        const float* sc =                                                                      \
            scores + ((ulong)t * n_head + blockIdx.x * g_n + r % g_n) * max_ctx + (BASE);      \
        _Pragma("unroll")                                                                      \
        for (uint e = 0; e < 2; e++) {                                                         \
            uint p = 2 * (i % 16) + e;                                                         \
            sr[j][e] = t < n_tok && (BASE) + p < n_pos + tok0 + t ? sc[p] : 0.0f;              \
        }                                                                                      \
    }
extern "C" __global__ void attn_pv_h(const float* scores, const float* vc, float* out,
                                     uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                     uint kv_dim, uint max_ctx, uint tok0, uint n_tok,
                                     unsigned short* oh, uint half) {
    __shared__ uint vt[128 * AV_S];
    __shared__ uint ps[PVH_R * AV_S];
    uint tid = threadIdx.x, g_n = n_head / n_kv, tpt = PVH_R / g_n, t0 = blockIdx.y * tpt;
    uint t_end = min(t0 + tpt, n_tok), limit = n_pos + tok0 + t_end - 1;
    uint n4 = head_dim / 4, no = head_dim * PVH_R / 1024;
    uint lane = tid % 32, warp = tid / 32, gi = lane / 4, gj = lane % 4;
    uint rg = warp % (PVH_R / 16), o0 = (warp / (PVH_R / 16)) * no;
    float c[PVH_R / 8][4];
    #pragma unroll
    for (uint o = 0; o < PVH_R / 8; o++) c[o][0] = c[o][1] = c[o][2] = c[o][3] = 0;
    float4 vr[4];
    float sr[PVH_R / 16][2];
    PVH_READ(0)
    for (uint base = 0; base < limit; base += 32) {
        unsigned short* vh = (unsigned short*)vt;
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uint i = tid + j * blockDim.x, p = i / n4, d = 4 * (i % n4);
            if (i >= 32 * n4) continue;
            vh[d * 2 * AV_S + p] = f2h(vr[j].x);
            vh[(d + 1) * 2 * AV_S + p] = f2h(vr[j].y);
            vh[(d + 2) * 2 * AV_S + p] = f2h(vr[j].z);
            vh[(d + 3) * 2 * AV_S + p] = f2h(vr[j].w);
        }
        #pragma unroll
        for (uint j = 0; j < PVH_R / 16; j++) {
            uint i = tid + j * blockDim.x;
            ps[i / 16 * AV_S + i % 16] = f2h(sr[j][0]) | ((uint)f2h(sr[j][1]) << 16);
        }
        // The next step's tiles, read while this one is multiplied.
        if (base + 32 < limit) {
            PVH_READ(base + 32)
        }
        __syncthreads();
        #pragma unroll
        for (uint kh = 0; kh < 2; kh++) {
            const uint* ap = ps + (16 * rg + gi) * AV_S + 8 * kh + gj;
            uint a[4] = {ap[0], ap[8 * AV_S], ap[4], ap[8 * AV_S + 4]};
            #pragma unroll
            for (uint o = 0; o < PVH_R / 8; o++) {
                if (o >= no) continue;
                const uint* bp = vt + (8 * (o0 + o) + gi) * AV_S + 8 * kh + gj;
                mma16_f16(a, bp[0], bp[4], c[o]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (uint j = 0; j < PVH_R / 2; j++) {
        uint o = j / 4, r = 16 * rg + gi + 8 * (j / 2 % 2), t = t0 + r / g_n;
        if (o >= no || t >= n_tok) continue;
        uint d = 8 * (o0 + o) + 2 * gj + j % 2;
        ulong at = ((ulong)(tok0 + t) * n_head + blockIdx.x * g_n + r % g_n) * head_dim + d;
        out[at] = c[o][j % 4];
        if (half) oh[at] = f2h(c[o][j % 4]);
    }
}

#else  // the rest of the file is not part of the tensor-core compile

// Block-wide reductions for blockDim.x <= 1024 (a multiple of 32).
__device__ inline float block_sum(float v, float* red) {
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32, n_warps = blockDim.x / 32;
    v = warp_sum(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = threadIdx.x < n_warps ? red[threadIdx.x] : 0.0f;
    if (warp == 0) t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

__device__ inline float block_max(float v, float* red) {
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32, n_warps = blockDim.x / 32;
    v = warp_max(v);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = threadIdx.x < n_warps ? red[threadIdx.x] : NEG_INF;
    if (warp == 0) t = warp_max(t);
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    float r = red[0];
    __syncthreads();
    return r;
}

// One block per row of n; in-place when x_off == out_off on the same buffer.
// With `quant` 1 (n a multiple of 32), the result is quantized into xq, xd, xs as well; with
// 2, written as f16 at xq, for the tensor-core matmuls (a launch of `to_f16` less).
extern "C" __global__ void rmsnorm(const float* x, ulong x_off, float* out, ulong out_off,
                                   const float* w, uint n, float eps, signed char* xq,
                                   float* xd, float* xs, uint quant) {
    __shared__ float red[32];
    const float* xr = x + x_off + (ulong)blockIdx.x * n;
    float* o = out + out_off + (ulong)blockIdx.x * n;
    float s = 0;
    for (uint i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, red);
    float scale = rsqrtf(s / (float)n + eps);
    for (uint i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i] * scale * w[i];
        o[i] = v;
        if (quant == 1) quant_block(v, ((ulong)blockIdx.x * n + i) / 32, xq, xd, xs);
        if (quant == 2) ((unsigned short*)xq)[(ulong)blockIdx.x * n + i] = f2h(v);
    }
}

extern "C" __global__ void axpy(float* a, ulong a_off, const float* b, ulong b_off, uint n, float alpha) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[a_off + i] += alpha * b[b_off + i];
}

// Zeroes rows base.. (`rows` of them) of the n tokens at a, `ldy` values each.
extern "C" __global__ void zero_rows(float* a, ulong a_off, uint ldy, uint base, uint rows, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * n) a[a_off + (ulong)(i / rows) * ldy + base + i % rows] = 0;
}

// a (n values, rows of m) += b (m values) on every row.
extern "C" __global__ void add_bias(float* a, ulong a_off, const float* b, uint n, uint m) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[a_off + i] += b[i % m];
}

// `quant` as for rmsnorm.
extern "C" __global__ void swiglu(float* g, ulong g_off, const float* u, ulong u_off, uint n,
                                  signed char* xq, float* xd, float* xs, uint quant) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[g_off + i];
        v = v / (1.0f + expf(-v)) * u[u_off + i];
        g[g_off + i] = v;
        if (quant == 1) quant_block(v, i / 32, xq, xd, xs);
        if (quant == 2) ((unsigned short*)xq)[i] = f2h(v);
    }
}

// x holds n_heads heads of consecutive tokens, heads_per_tok each, the first at position pos;
// x2 likewise (the keys, after the queries).
extern "C" __global__ void rope(float* x, ulong x_off, const float* ff, uint has_ff, uint n_heads,
                                uint head_dim, uint rope_dim, uint pos, float theta, uint neox,
                                uint heads_per_tok, float* x2, ulong x2_off, uint n_heads2,
                                uint heads_per_tok2) {
    uint half_dim = rope_dim / 2;
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_heads * half_dim) {
        gid -= n_heads * half_dim;
        if (gid >= n_heads2 * half_dim) return;
        x = x2;
        x_off = x2_off;
        heads_per_tok = heads_per_tok2;
    }
    uint hh = gid / half_dim, i = gid % half_dim;
    float freq = powf(theta, -2.0f * (float)i / (float)rope_dim);
    if (has_ff) freq /= ff[i];
    float s, c;
    sincosf((float)(pos + hh / heads_per_tok) * freq, &s, &c);
    float* hd = x + x_off + (ulong)hh * head_dim;
    uint a = neox ? i : 2 * i;
    uint b = neox ? i + half_dim : 2 * i + 1;
    float x0 = hd[a], x1 = hd[b];
    hd[a] = x0 * c - x1 * s;
    hd[b] = x0 * s + x1 * c;
}

// rmsnorm over each head (weights wq for x's heads, wk for x2's), then rope: Qwen3 normalizes
// its queries and keys, and this saves a token two launches a layer. One block per head, a
// thread per value: head_dim threads, a multiple of 32.
extern "C" __global__ void norm_rope(float* x, ulong x_off, const float* wq, const float* ff,
                                     uint has_ff, uint n_heads, uint head_dim, uint rope_dim,
                                     uint pos, float theta, uint neox, uint heads_per_tok,
                                     float* x2, ulong x2_off, const float* wk,
                                     uint heads_per_tok2, float eps) {
    __shared__ float red[32];
    __shared__ float val[1024];
    uint hh = blockIdx.x, i = threadIdx.x, half_dim = rope_dim / 2;
    const float* wn = wq;
    if (hh >= n_heads) {
        hh -= n_heads;
        x = x2;
        x_off = x2_off;
        heads_per_tok = heads_per_tok2;
        wn = wk;
    }
    float* hd = x + x_off + (ulong)hh * head_dim;
    float v = hd[i], wv = wn[i];
    float s = block_sum(v * v, red);
    v = v * rsqrtf(s / (float)head_dim + eps) * wv;
    val[i] = v;
    __syncthreads();
    if (i >= rope_dim) {
        hd[i] = v;
        return;
    }
    // This value's pair, its other half, and whether this is the first of the two.
    uint j = neox ? i % half_dim : i / 2;
    bool first = neox ? i < half_dim : !(i & 1);
    uint other = neox ? (first ? i + half_dim : i - half_dim) : i ^ 1;
    float freq = powf(theta, -2.0f * (float)j / (float)rope_dim);
    if (has_ff) freq /= ff[j];
    float sn, c;
    sincosf((float)(pos + hh / heads_per_tok) * freq, &sn, &c);
    float o = val[other];
    hd[i] = first ? v * c - o * sn : o * sn + v * c;
}

__device__ inline uint warp_min_u(uint v) {
    for (int o = 16; o > 0; o >>= 1) {
        uint u = __shfl_xor_sync(0xffffffff, v, o);
        v = u < v ? u : v;
    }
    return v;
}

// One warp per token: softmax over its n_expert router logits, then the k largest as its ids[]
// and w[] (renormalized to sum to 1 when norm is set). With a shared expert, a token has k + 1
// weights, the last its sigmoid gate. Takes up to 2048 experts (64 per lane).
extern "C" __global__ void moe_topk(const float* logits, const float* shared_gate, uint* ids,
                                    float* w, uint n_expert, uint k, uint norm, uint shared) {
    // The logits and the chosen weights are kept in shared memory: every pass over device
    // memory would wait out its latency once per read.
    __shared__ float lg[2048];
    __shared__ float wk[2048];
    uint lane = threadIdx.x, tok = blockIdx.x;
    logits += (ulong)tok * n_expert;
    shared_gate += tok;
    ids += (ulong)tok * k;
    w += (ulong)tok * (k + shared);
    float mx = NEG_INF;
    for (uint e = lane; e < n_expert; e += 32) {
        lg[e] = logits[e];
        mx = fmaxf(mx, lg[e]);
    }
    mx = warp_max(mx);
    float sum = 0;
    for (uint e = lane; e < n_expert; e += 32) sum += expf(lg[e] - mx);
    sum = warp_sum(sum);
    ulong taken = 0;
    float chosen = 0;
    for (uint i = 0; i < k; i++) {
        float bv = NEG_INF;
        uint be = 0xffffffffu;
        for (uint e = lane, j = 0; e < n_expert; e += 32, j++) {
            if (!((taken >> j) & 1) && lg[e] > bv) {
                bv = lg[e];
                be = e;
            }
        }
        float best = warp_max(bv);
        uint win = warp_min_u(bv == best ? be : 0xffffffffu);
        if (win == be && win < n_expert) taken |= 1ull << (win / 32);
        float wi = expf(best - mx) / sum;
        chosen += wi;
        if (lane == 0) {
            ids[i] = win < n_expert ? win : 0;  // no winner (NaN logits): stay inside the stack
            wk[i] = wi;
        }
    }
    __syncwarp();
    for (uint i = lane; i < k; i += 32) w[i] = norm ? wk[i] / chosen : wk[i];
    if (lane == 0 && shared) w[k] = 1.0f / (1.0f + expf(-shared_gate[0]));
}

// Row t of x (n_tok rows of n) += sum over r < m of w[t * w_stride + r] * src row t * m + r:
// each token's chosen experts' outputs, combined.
extern "C" __global__ void moe_combine(float* x, ulong x_off, const float* src, ulong src_off,
                                       const float* w, ulong w_off, uint w_stride, uint n,
                                       uint m, uint n_tok) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * n_tok) return;
    uint t = i / n, j = i % n;
    const float* wt = w + w_off + (ulong)t * w_stride;
    const float* rows = src + src_off + (ulong)t * m * n + j;
    float s = 0;
    for (uint r = 0; r < m; r++) s += wt[r] * rows[(ulong)r * n];
    x[x_off + i] += s;
}

// One block per query head (x) and token (y): token tok0 + y of the batch attends to the first
// n_pos + tok0 + y cached positions. `scores` is one row of max_ctx per block. Needs
// head_dim <= blockDim.x <= 1024. With `quant` (head_dim a multiple of 32), the result is
// quantized into xq, xd, xs as well.
extern "C" __global__ void attention(const float* q, const float* kc, const float* vc, float* out,
                                     float* scores, uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                     uint kv_dim, uint max_ctx, float scale, uint tok0,
                                     signed char* xq, float* xd, float* xs, uint quant) {
    __shared__ float red[32];
    __shared__ float part[1024];
    uint hh = blockIdx.x;
    uint tok = tok0 + blockIdx.y;
    n_pos += tok;
    uint kv_off = (hh / (n_head / n_kv)) * head_dim;
    const float* qh = q + ((ulong)tok * n_head + hh) * head_dim;
    float* oh = out + ((ulong)tok * n_head + hh) * head_dim;
    float* sc = scores + ((ulong)blockIdx.y * n_head + hh) * max_ctx;
    // Scores: a warp per position, its lanes across the head's values, so K is read in order.
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32, n_warps = blockDim.x / 32;
    float mx = NEG_INF;
    for (uint t = warp; t < n_pos; t += n_warps) {
        const float* k = kc + (ulong)t * kv_dim + kv_off;
        float s = 0;
        for (uint d = lane; d < head_dim; d += 32) s += qh[d] * k[d];
        s = warp_sum(s) * scale;
        if (lane == 0) sc[t] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0;
    for (uint t = threadIdx.x; t < n_pos; t += blockDim.x) {
        float e = expf(sc[t] - mx);
        sc[t] = e;
        sum += e;
    }
    sum = block_sum(sum, red);  // its __syncthreads also orders the sc[] writes above
    // The weighted sum of V: `parts` threads per output value, each over every parts-th
    // position, so no thread walks the whole context.
    uint parts = blockDim.x / head_dim;
    uint d = threadIdx.x % head_dim, pt = threadIdx.x / head_dim;
    if (pt < parts) {
        float o = 0;
        for (uint t = pt; t < n_pos; t += parts) o += sc[t] * vc[(ulong)t * kv_dim + kv_off + d];
        part[pt * head_dim + d] = o;
    }
    __syncthreads();
    if (pt == 0) {
        float o = 0;
        for (uint p = 0; p < parts; p++) o += part[p * head_dim + d];
        o /= sum;
        oh[d] = o;
        if (quant) quant_block(o, (((ulong)tok * n_head + hh) * head_dim + d) / 32, xq, xd, xs);
    }
}

// Decode attention, in two launches so a long context is read by many blocks at once.
// `attention_part`: one block of AP_T threads per KV head (x) and 32 cached positions (y).
// The up to AP_G query heads sharing the KV head are scored together, so K and V are read
// once. Each block's softmax is relative to its own maximum; it writes that, the sum and the
// unnormalized weighted V per query head to `parts` (head_dim + 2 floats per head and block),
// which `attention_merge` combines. Needs head_dim <= AP_T and a multiple of 4. Under about
// 128 positions the second launch costs more than `attention` loses by reading in turn.
#define AP_T 256
#define AP_G 8   // AP_T / 32: a warp per query head in the softmax
#define AP_B 8   // reads issued together
#define AP_V 16  // values of V a thread reads at the start
#define AP_M 32  // values of the parts a merging thread reads at the start
#define AP_PARTS 2048

extern "C" __global__ void attention_part(const float* q, const float* kc, const float* vc,
                                          float* parts, uint n_head, uint n_kv, uint head_dim,
                                          uint n_pos, uint kv_dim, float scale) {
    __shared__ float qs[AP_G * AP_T];
    __shared__ float sc[AP_G * AP_T];
    __shared__ float ev[AP_G * 32];
    __shared__ float ms[AP_G * 2];
    uint tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    uint g_n = n_head / n_kv, h0 = blockIdx.x * g_n;
    uint base = blockIdx.y * 32, cnt = min(32u, n_pos - base);
    ulong kv_off = (ulong)base * kv_dim + blockIdx.x * head_dim;
    // The block's reads of K and V are all issued here, before anything waits on one: a read
    // that is used at once makes the thread wait out the memory latency, and K, the scores'
    // softmax and V were three such waits in a row. For K, 8 threads per position (the lane)
    // take every 8th group of 4 values of its row (head_dim <= AP_T: at most AP_B groups).
    // For V, `np` threads per output value take every np-th position.
    uint n4 = head_dim / 4;
    uint np = AP_T / head_dim, d = tid % head_dim, pt = tid / head_dim;
    const float4* k = (const float4*)(kc + kv_off + (ulong)min(lane, cnt - 1) * kv_dim);
    float4 kk[AP_B];
    for (uint j = 0; j < AP_B; j++)
        if (warp + 8 * j < n4) kk[j] = k[warp + 8 * j];
    float vv[AP_V];
    for (uint j = 0; j < AP_V; j++)
        if (pt + j * np < cnt) vv[j] = vc[kv_off + (ulong)(pt + j * np) * kv_dim + d];
    for (uint i = tid; i < g_n * head_dim; i += AP_T) qs[i] = q[(ulong)h0 * head_dim + i];
    __syncthreads();
    // Scores, against four query heads at a time.
    for (uint g0 = 0; g0 < g_n; g0 += 4) {
        float s[4] = {0, 0, 0, 0};
        if (lane < cnt) {
            for (uint j = 0; j < AP_B && warp + 8 * j < n4; j++) {
                for (uint i = 0; i < 4 && g0 + i < g_n; i++) {
                    const float* qg = qs + (g0 + i) * head_dim + 4 * (warp + 8 * j);
                    s[i] += qg[0] * kk[j].x + qg[1] * kk[j].y + qg[2] * kk[j].z
                        + qg[3] * kk[j].w;
                }
            }
        }
        for (uint i = 0; i < 4; i++) sc[(g0 + i) % AP_G * AP_T + tid] = s[i];
    }
    __syncthreads();
    // Softmax within the block: warp g takes query head g, its lanes the positions.
    if (warp < g_n) {
        float v = 0;
        for (uint j = 0; j < 8; j++) v += sc[warp * AP_T + 32 * j + lane];
        v = lane < cnt ? v * scale : NEG_INF;
        float mx = warp_max(v);
        float e = lane < cnt ? expf(v - mx) : 0.0f;
        ev[warp * 32 + lane] = e;
        float sum = warp_sum(e);
        if (lane == 0) {
            ms[2 * warp] = mx;
            ms[2 * warp + 1] = sum;
        }
    }
    __syncthreads();
    // The weighted sum of V.
    for (uint g0 = 0; g0 < g_n; g0 += 4) {
        float o[4] = {0, 0, 0, 0};
        if (pt < np) {
            for (uint j = 0; j < AP_V && pt + j * np < cnt; j++)
                for (uint i = 0; i < 4; i++) o[i] += ev[(g0 + i) % AP_G * 32 + pt + j * np] * vv[j];
            for (uint p0 = pt + AP_V * np; p0 < cnt; p0 += np * AP_B) {
                float v[AP_B];
                for (uint j = 0; j < AP_B; j++)
                    v[j] = vc[kv_off + (ulong)min(p0 + j * np, cnt - 1) * kv_dim + d];
                for (uint j = 0; j < AP_B && p0 + j * np < cnt; j++)
                    for (uint i = 0; i < 4; i++)
                        o[i] += ev[(g0 + i) % AP_G * 32 + p0 + j * np] * v[j];
            }
        }
        for (uint i = 0; i < 4; i++) sc[(g0 + i) % AP_G * AP_T + tid] = o[i];
    }
    __syncthreads();
    if (pt != 0) return;
    for (uint g = 0; g < g_n; g++) {
        float o = 0;
        for (uint p = 0; p < np; p++) o += sc[g * AP_T + p * head_dim + d];
        float* part = parts + ((ulong)(h0 + g) * gridDim.y + blockIdx.y) * (head_dim + 2);
        part[d] = o;
        if (d == 0) {
            part[head_dim] = ms[2 * g];
            part[head_dim + 1] = ms[2 * g + 1];
        }
    }
}

// One block of AP_T threads per query head: the head's `n_parts` (at most AP_PARTS) partial
// results rescaled to their common maximum, added and normalized. `quant` as in `attention`.
extern "C" __global__ void attention_merge(const float* parts, float* out, uint head_dim,
                                           uint n_parts, signed char* xq, float* xd, float* xs,
                                           uint quant) {
    __shared__ float red[32];
    __shared__ float wf[AP_PARTS];
    __shared__ float pa[AP_T];
    uint tid = threadIdx.x, stride = head_dim + 2;
    const float* part = parts + (ulong)blockIdx.x * n_parts * stride;
    // As in `attention_part`, a thread's first reads are issued before it waits on any.
    uint np = AP_T / head_dim, d = tid % head_dim, pt = tid / head_dim;
    uint c1 = min(tid, n_parts - 1);
    float m0 = part[c1 * stride + head_dim], s0 = part[c1 * stride + head_dim + 1];
    float vv[AP_M];
    for (uint j = 0; j < AP_M; j++)
        if (pt + j * np < n_parts) vv[j] = part[(pt + j * np) * stride + d];
    float mx = tid < n_parts ? m0 : NEG_INF;
    for (uint c = tid + AP_T; c < n_parts; c += AP_T)
        mx = fmaxf(mx, part[c * stride + head_dim]);
    mx = block_max(mx, red);
    float sum = 0;
    if (tid < n_parts) {
        float f = expf(m0 - mx);
        wf[tid] = f;
        sum = s0 * f;
    }
    for (uint c = tid + AP_T; c < n_parts; c += AP_T) {
        float f = expf(part[c * stride + head_dim] - mx);
        wf[c] = f;
        sum += part[c * stride + head_dim + 1] * f;
    }
    sum = block_sum(sum, red);  // its __syncthreads also orders the wf[] writes
    // `np` threads per output value, each over every np-th part.
    float o = 0;
    if (pt < np) {
        for (uint j = 0; j < AP_M && pt + j * np < n_parts; j++) o += vv[j] * wf[pt + j * np];
        for (uint c0 = pt + AP_M * np; c0 < n_parts; c0 += np * AP_B) {
            float v[AP_B];
            for (uint j = 0; j < AP_B; j++)
                v[j] = part[min(c0 + j * np, n_parts - 1) * stride + d];
            for (uint j = 0; j < AP_B && c0 + j * np < n_parts; j++) o += v[j] * wf[c0 + j * np];
        }
    }
    pa[tid] = o;
    __syncthreads();
    if (pt != 0) return;
    o = 0;
    for (uint p = 0; p < np; p++) o += pa[p * head_dim + d];
    o /= sum;
    out[(ulong)blockIdx.x * head_dim + d] = o;
    if (quant) quant_block(o, ((ulong)blockIdx.x * head_dim + d) / 32, xq, xd, xs);
}

// Prompt attention as tiled products, in three launches, so that a tile of K or V is read
// once for all the queries it meets: `attention` reads every cached row again for each query
// head and token. A query row is a (token, query head) pair; AB tokens' worth of the heads
// sharing a KV head make a tile of 32 rows, so heads per KV head must divide 32. Needs
// head_dim <= 128 and a multiple of 4. `scores` holds a row of max_ctx per head and token of
// the launch; token tok0 + t attends to the first n_pos + tok0 + t positions.

// Scores: one block of 256 threads per KV head (x), 32 cached positions (y) and tile of query
// rows (z). Thread (rows 4 rg.., position p) takes four scores.
extern "C" __global__ void attn_scores(const float* q, const float* kc, float* scores,
                                       uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                       uint kv_dim, uint max_ctx, float scale, uint tok0,
                                       uint n_tok) {
    __shared__ float ks[32 * 129];  // rows padded by one, so a warp's reads fall in 32 banks
    __shared__ float qs[32 * 128];
    uint tid = threadIdx.x, g_n = n_head / n_kv, tpt = 32 / g_n;
    uint base = blockIdx.y * 32, t0 = blockIdx.z * tpt;
    uint t_end = min(t0 + tpt, n_tok);
    if (base >= n_pos + tok0 + t_end - 1) return;  // past what the tile's last token sees
    // Both tiles in 16-byte reads (at most 4 each a thread), all issued before the first store.
    uint n4 = head_dim / 4;
    float4 kr[4], qr[4];
    _Pragma("unroll")
    for (uint j = 0; j < 4; j++) {
        uint i = tid + j * blockDim.x, r = i / n4, d = 4 * (i % n4), t = t0 + r / g_n;
        kr[j] = qr[j] = make_float4(0, 0, 0, 0);
        if (i >= 32 * n4) continue;
        if (base + r < n_pos + tok0 + t_end - 1)
            kr[j] = *(const float4*)(kc + (ulong)(base + r) * kv_dim + blockIdx.x * head_dim + d);
        if (t < n_tok)
            qr[j] = *(const float4*)(
                q + ((ulong)(tok0 + t) * n_head + blockIdx.x * g_n + r % g_n) * head_dim + d);
    }
    _Pragma("unroll")
    for (uint j = 0; j < 4; j++) {
        uint i = tid + j * blockDim.x, r = i / n4, d = 4 * (i % n4);
        if (i >= 32 * n4) continue;
        float* kd = ks + r * (head_dim + 1) + d;
        kd[0] = kr[j].x, kd[1] = kr[j].y, kd[2] = kr[j].z, kd[3] = kr[j].w;
        *(float4*)(qs + r * head_dim + d) = qr[j];
    }
    __syncthreads();
    uint p = tid % 32, rg = tid / 32;
    const float* k = ks + p * (head_dim + 1);
    const float* q4 = qs + 4 * rg * head_dim;
    float s[4] = {0, 0, 0, 0};
    for (uint d = 0; d < head_dim; d++) {
        float kd = k[d];
        _Pragma("unroll")
        for (uint j = 0; j < 4; j++) s[j] += q4[j * head_dim + d] * kd;
    }
    for (uint j = 0; j < 4; j++) {
        uint r = 4 * rg + j, t = t0 + r / g_n;
        if (t < n_tok && base + p < n_pos + tok0 + t)
            scores[((ulong)t * n_head + blockIdx.x * g_n + r % g_n) * max_ctx + base + p] =
                s[j] * scale;
    }
}

// Softmax of each row of scores in place: one block per query head (x) and token (y).
extern "C" __global__ void attn_softmax(float* scores, uint n_head, uint n_pos, uint max_ctx,
                                        uint tok0) {
    __shared__ float red[32];
    float* sc = scores + ((ulong)blockIdx.y * n_head + blockIdx.x) * max_ctx;
    uint n = n_pos + tok0 + blockIdx.y;
    float mx = NEG_INF;
    for (uint t = threadIdx.x; t < n; t += blockDim.x) mx = fmaxf(mx, sc[t]);
    mx = block_max(mx, red);
    float sum = 0;
    for (uint t = threadIdx.x; t < n; t += blockDim.x) sum += expf(sc[t] - mx);
    sum = block_sum(sum, red);
    for (uint t = threadIdx.x; t < n; t += blockDim.x) sc[t] = expf(sc[t] - mx) / sum;
}

// The weighted sums of V: one block of 2 head_dim threads per KV head (x) and tile of query
// rows (y), walking the context 32 positions at a time. Thread (rows 4 rg.., values 4 dq..)
// keeps 16 sums. A step's tiles are read into registers while the step before is multiplied,
// so no step waits on memory: a thread takes 4 16-byte words of V and up to PV_S scores.
#define PV_S 8
#define PV_READ(BASE)                                                                          \
    _Pragma("unroll")                                                                          \
    for (uint j = 0; j < 4; j++) {                                                             \
        uint i = tid + j * blockDim.x, p = i / n4;                                             \
        vr[j] = (BASE) + p < limit                                                             \
            ? *(const float4*)(vc + (ulong)((BASE) + p) * kv_dim + blockIdx.x * head_dim       \
                               + 4 * (i % n4))                                                 \
            : make_float4(0, 0, 0, 0);                                                         \
    }                                                                                          \
    _Pragma("unroll")                                                                          \
    for (uint j = 0; j < PV_S; j++) {                                                          \
        uint i = tid + j * blockDim.x, r = i / 32, p = i % 32, t = t0 + r / g_n;               \
        sr[j] = i < 32 * 32 && t < n_tok && (BASE) + p < n_pos + tok0 + t                      \
            ? scores[((ulong)t * n_head + blockIdx.x * g_n + r % g_n) * max_ctx + (BASE) + p]  \
            : 0.0f;                                                                            \
    }
extern "C" __global__ void attn_pv(const float* scores, const float* vc, float* out,
                                   uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                   uint kv_dim, uint max_ctx, uint tok0, uint n_tok) {
    __shared__ float vs[32 * 128];
    __shared__ float ps[32 * 32];
    uint tid = threadIdx.x, g_n = n_head / n_kv, tpt = 32 / g_n, t0 = blockIdx.y * tpt;
    uint t_end = min(t0 + tpt, n_tok), limit = n_pos + tok0 + t_end - 1;
    uint n4 = head_dim / 4, dq = tid % n4, rg = tid / n4;
    float o[16];
    for (uint i = 0; i < 16; i++) o[i] = 0;
    float4 vr[4];
    float sr[PV_S];
    PV_READ(0)
    for (uint base = 0; base < limit; base += 32) {
        _Pragma("unroll")
        for (uint j = 0; j < 4; j++) *(float4*)(vs + 4 * (tid + j * blockDim.x)) = vr[j];
        _Pragma("unroll")
        for (uint j = 0; j < PV_S; j++)
            if (tid + j * blockDim.x < 32 * 32) ps[tid + j * blockDim.x] = sr[j];
        for (uint i = tid + PV_S * blockDim.x; i < 32 * 32; i += blockDim.x) {
            uint r = i / 32, p = i % 32, t = t0 + r / g_n;
            ps[i] = t < n_tok && base + p < n_pos + tok0 + t
                ? scores[((ulong)t * n_head + blockIdx.x * g_n + r % g_n) * max_ctx + base + p]
                : 0.0f;
        }
        if (base + 32 < limit) {
            PV_READ(base + 32)
        }
        __syncthreads();
        for (uint p = 0; p < 32; p++) {
            const float* v = vs + p * head_dim + 4 * dq;
            _Pragma("unroll")
            for (uint j = 0; j < 4; j++) {
                float pr = ps[(4 * rg + j) * 32 + p];
                _Pragma("unroll")
                for (uint i = 0; i < 4; i++) o[4 * j + i] += pr * v[i];
            }
        }
        __syncthreads();
    }
    for (uint j = 0; j < 4; j++) {
        uint r = 4 * rg + j, t = t0 + r / g_n;
        if (t >= n_tok) continue;
        float* oh = out + ((ulong)(tok0 + t) * n_head + blockIdx.x * g_n + r % g_n) * head_dim;
        for (uint i = 0; i < 4; i++) oh[4 * dq + i] = o[4 * j + i];
    }
}
#endif  // RLI_MMA

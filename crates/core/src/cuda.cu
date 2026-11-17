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
DEQUANT(dequant_q6_k, wt_q6_k)
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
MATVEC_Q(mvq_q4_k, mvq_q4_k_step, 144, 4, , )
MATVEC_Q(mvq_q6_k, mvq_q6_k_step, 210, 4, , )
MATVEC_Q(mvq_q4_k_pair, mvq_q4_k_step, 144, 4, MVQ_PAIR_ARGS, MVQ_PAIR_PICK)
MATVEC_Q(mvq_q6_k_pair, mvq_q6_k_step, 210, 4, MVQ_PAIR_ARGS, MVQ_PAIR_PICK)
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
// thread, which limit how many warps a multiprocessor holds (smaller is better).
#define MA_R 64
#define MA_T 128
#define MA_NW 8
#define MA_S 36
#define MA_RG (MA_R / 16)
#define MA_NO (MA_T / 8 / (MA_NW / MA_RG))

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
#define MATMUL_MMA(NAME, SB_BYTES, K32)                                                        \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const uint* xq, const float* xd,  \
                                const float* xs, float* y, ulong y_off, uint rows, uint cols,  \
                                uint row_bytes, uint n_tok) {                                  \
    const uint NG = K32 ? 4 : 8, NP = MA_R / MA_NW / (K32 ? 2 : 1), NT = MA_T / MA_NW;         \
    __shared__ uint wq[MA_R * MA_S];                                                           \
    __shared__ uint tq[MA_T * MA_S];                                                           \
    __shared__ float ws[MA_R * (K32 ? 4 : 8)];                                                 \
    __shared__ float wm[K32 ? MA_R * 4 : 1];                                                   \
    __shared__ float td[MA_T * 4];                                                             \
    __shared__ float ts[K32 ? MA_T * 4 : 1];                                                   \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint gi = lane / 4, gj = lane % 4;                                                         \
    uint row0 = blockIdx.x * MA_R, t0 = blockIdx.y * MA_T;                                     \
    uint last = n_tok - 1 - t0;                                                                \
    uint ra = 16 * (warp % MA_RG) + gi, ow = warp / MA_RG;                                     \
    float acc[4 * MA_NO];                                                                      \
    _Pragma("unroll")                                                                          \
    for (uint j = 0; j < 4 * MA_NO; j++) acc[j] = 0;                                           \
    for (uint kh = 0; kh < cols / 128; kh++) {                                                 \
        ulong sb_off = w_off + (ulong)(kh / 2) * SB_BYTES;                                     \
        uint hf = kh % 2;                                                                      \
        uint v0[16], v1[16], v2[16], v3[16];                                                   \
        if (K32) {                                                                             \
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
                    v2[p] = u32e(b + 8);                                                       \
                    v3[p] = u32e(b + 12);                                                      \
                }                                                                              \
                _Pragma("unroll")                                                              \
                for (uint p = 0; p < NP; p++) {                                                \
                    /* Sub-block 4 hf + l: bytes l, 4 + l and 8 + l of the 12 scale bytes. */  \
                    uint b0 = (v1[p] >> 8 * l) & 255, b1 = (v2[p] >> 8 * l) & 255;             \
                    uint b2 = (v3[p] >> 8 * l) & 255;                                          \
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
        int a[8][4];                                                                           \
        float sa[8], sb[8], ma[8], mb[8];                                                      \
        _Pragma("unroll")                                                                      \
        for (uint g = 0; g < NG; g++) {                                                        \
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
        for (uint o = 0; o < MA_NO; o++) {                                                     \
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
    for (uint j = 0; j < 4 * MA_NO; j++) {                                                     \
        uint t = (j / 4 * (MA_NW / MA_RG) + ow) * 8 + 2 * gj + j % 2;                          \
        uint row = row0 + ra + 8 * (j / 2 % 2);                                                \
        if (t <= last && row < rows) y[y_off + (ulong)(t0 + t) * rows + row] = acc[j];         \
    }                                                                                          \
}

MATMUL_MMA(mma_q4_k, 144, 1)
MATMUL_MMA(mma_q6_k, 210, 0)

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
// too few blocks to occupy the GPU, which costs more: the caller picks. Blocks of 128 rows,
// 32 rows or 256 tokens were all slower on an A100 (docs/benchmarks.md).
#define GE_K 64
#define GE_S 36
struct __attribute__((aligned(16))) V128 {
    uint v[4];
};
#define GEMM_F16(NAME, R, T, NW, RG)                                                           \
extern "C" __global__ void NAME(const uint* wh, const uint* xh, float* y, ulong y_off,         \
                                uint rows, uint cols, uint n_tok) {                            \
    const uint NRG = R / (16 * RG), NWG = NW / NRG, NO = T / 8 / NWG;                          \
    __shared__ uint wt[R * GE_S];                                                              \
    __shared__ uint xt[T * GE_S];                                                              \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint gi = lane / 4, gj = lane % 4;                                                         \
    uint row0 = blockIdx.x * R, t0 = blockIdx.y * T;                                           \
    uint last = n_tok - 1 - t0, kw = cols / 2;                                                 \
    uint ra = 16 * RG * (warp % NRG) + gi, ow = warp / NRG;                                    \
    uint tr = threadIdx.x / 8, tw = 4 * (threadIdx.x % 8);                                     \
    float c[RG][T / 8 / (NW / (R / (16 * RG)))][4];                                            \
    _Pragma("unroll")                                                                          \
    for (uint g = 0; g < RG; g++)                                                              \
        _Pragma("unroll")                                                                      \
        for (uint o = 0; o < NO; o++) c[g][o][0] = c[g][o][1] = c[g][o][2] = c[g][o][3] = 0;   \
    for (uint ks = 0; ks < cols / GE_K; ks++) {                                                \
        V128 v[8];                                                                             \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < R / (4 * NW); p++) {                                              \
            uint r = row0 + tr + 4 * NW * p;                                                   \
            v[p] = *(const V128*)(wh + (ulong)(r < rows ? r : rows - 1) * kw + ks * (GE_K / 2) + tw); \
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
            if (t <= last && row < rows)                                                       \
                y[y_off + (ulong)(t0 + t) * rows + row] = c[g][j / 4][j % 4];                  \
        }                                                                                      \
}

GEMM_F16(gemm_f16_64x64, 64, 64, 8, 1)
// Four warps of two row groups each: 8-12% faster on an A100 than eight warps of one.
GEMM_F16(gemm_f16_64x128, 64, 128, 4, 2)

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
// *_hdr reads what a super-block's four steps share, *_load a step's bytes, *_dec decodes.
//
// Q4_K: the quarter is 32 bytes, the low nibbles one sub-block and the high ones the next.
// The row must lie at a multiple of 16 bytes.
__device__ inline void gq4_k_hdr(const uchar* b, ulong* hdr) {
    hdr[0] = ((const ulong*)b)[0];
    hdr[1] = ((const ulong*)b)[1];
}
__device__ inline void gq4_k_load(const uchar* b, uint j, uint dp, ulong* raw) {
    raw[0] = *(const ulong*)(b + 16 + 32 * j + 8 * dp);
}
__device__ inline void gq4_k_dec(const ulong* hdr, const ulong* raw, uint j, uint dp, V128& lo,
                                 V128& hi) {
    U128 hd;
    hd.v[0] = (uint)hdr[0];
    hd.v[1] = (uint)(hdr[0] >> 32);
    hd.v[2] = (uint)hdr[1];
    hd.v[3] = (uint)(hdr[1] >> 32);
    float sa, ma, sc, mc;
    q4_k_scale(hd, 2 * j, &sa, &ma);
    q4_k_scale(hd, 2 * j + 1, &sc, &mc);
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
__device__ inline void gq6_k_load(const uchar* b, uint j, uint dp, ulong* raw) {
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
#define GEMM_ID(NAME, SB_BYTES, HDR, LOAD, DEC)                                                \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, ulong stride, const uint* perm,   \
                                const uint* start, const uint* xh, uint x_div, float* y,       \
                                ulong y_off, uint rows, uint cols, uint row_bytes) {           \
    __shared__ uint wt[64 * GE_S];                                                             \
    __shared__ uint xt[64 * GE_S];                                                             \
    __shared__ uint pr[64];                                                                    \
    uint lane = threadIdx.x % 32, warp = threadIdx.x / 32;                                     \
    uint gi = lane / 4, gj = lane % 4;                                                         \
    uint row0 = blockIdx.x * 64, p1 = start[blockIdx.y + 1], kw = cols / 2;                    \
    uint ra = 16 * (warp % 4) + gi, ow = warp / 4;                                             \
    uint tr = threadIdx.x / 8, tw = 4 * (threadIdx.x % 8);                                     \
    uint dr = threadIdx.x / 4, dp = threadIdx.x % 4;                                           \
    const uchar* wr = w + w_off + blockIdx.y * stride +                                        \
                      (ulong)(row0 + dr < rows ? row0 + dr : rows - 1) * row_bytes;            \
    for (uint p0 = start[blockIdx.y]; p0 < p1; p0 += 64) {                                     \
        uint last = (p1 - p0 < 64 ? p1 - p0 : 64) - 1;                                         \
        if (threadIdx.x <= last) pr[threadIdx.x] = perm[p0 + threadIdx.x];                     \
        __syncthreads();                                                                       \
        ulong xr[2];                                                                           \
        _Pragma("unroll")                                                                      \
        for (uint p = 0; p < 2; p++) xr[p] = tr + 32 * p <= last ? pr[tr + 32 * p] / x_div : 0; \
        float c[4][4];                                                                         \
        _Pragma("unroll")                                                                      \
        for (uint o = 0; o < 4; o++) c[o][0] = c[o][1] = c[o][2] = c[o][3] = 0;                \
        ulong hdr[3], raw[3];                                                                  \
        for (uint ks = 0; ks < cols / 64; ks++) {                                              \
            uint j = ks % 4;                                                                   \
            const uchar* b = wr + (ulong)(ks / 4) * SB_BYTES;                                  \
            if (j == 0) HDR(b, hdr);                                                           \
            LOAD(b, j, dp, raw);                                                               \
            V128 v[2];                                                                         \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < 2; p++)                                                       \
                if (tr + 32 * p <= last) v[p] = *(const V128*)(xh + xr[p] * kw + ks * 32 + tw); \
            V128 lo, hi;                                                                       \
            DEC(hdr, raw, j, dp, lo, hi);                                                      \
            *(V128*)(wt + dr * GE_S + 4 * dp) = lo;                                            \
            *(V128*)(wt + dr * GE_S + 16 + 4 * dp) = hi;                                       \
            _Pragma("unroll")                                                                  \
            for (uint p = 0; p < 2; p++)                                                       \
                if (tr + 32 * p <= last) *(V128*)(xt + (tr + 32 * p) * GE_S + tw) = v[p];      \
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
            for (uint o = 0; o < 4; o++) {                                                     \
                uint tb = (2 * o + ow) * 8;                                                    \
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
        for (uint j = 0; j < 16; j++) {                                                        \
            uint t = (2 * (j / 4) + ow) * 8 + 2 * gj + j % 2;                                  \
            uint row = row0 + ra + 8 * (j / 2 % 2);                                            \
            if (t <= last && row < rows) y[y_off + (ulong)pr[t] * rows + row] = c[j / 4][j % 4]; \
        }                                                                                      \
        __syncthreads();                                                                       \
    }                                                                                          \
}

GEMM_ID(gemm_q4_k_id, 144, gq4_k_hdr, gq4_k_load, gq4_k_dec)
GEMM_ID(gemm_q6_k_id, 210, gq6_k_hdr, gq6_k_load, gq6_k_dec)

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
// With `quant` (n a multiple of 32), the result is quantized into xq, xd, xs as well.
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
        if (quant) quant_block(v, ((ulong)blockIdx.x * n + i) / 32, xq, xd, xs);
    }
}

extern "C" __global__ void axpy(float* a, ulong a_off, const float* b, ulong b_off, uint n, float alpha) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[a_off + i] += alpha * b[b_off + i];
}

// a (n values, rows of m) += b (m values) on every row.
extern "C" __global__ void add_bias(float* a, ulong a_off, const float* b, uint n, uint m) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[a_off + i] += b[i % m];
}

// With `quant` (n a multiple of 32), the result is quantized into xq, xd, xs as well.
extern "C" __global__ void swiglu(float* g, ulong g_off, const float* u, ulong u_off, uint n,
                                  signed char* xq, float* xd, float* xs, uint quant) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[g_off + i];
        v = v / (1.0f + expf(-v)) * u[u_off + i];
        g[g_off + i] = v;
        if (quant) quant_block(v, i / 32, xq, xd, xs);
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
    for (uint i = tid; i < g_n * head_dim; i += AP_T) qs[i] = q[(ulong)h0 * head_dim + i];
    __syncthreads();
    // Scores: 8 threads per position (the lane), each taking every 8th group of 4 values of
    // its K row against four query heads. All of a batch's reads are issued before the first
    // is used: a read that is used at once makes the thread wait out the memory latency.
    uint n4 = head_dim / 4;
    for (uint g0 = 0; g0 < g_n; g0 += 4) {
        float s[4] = {0, 0, 0, 0};
        if (lane < cnt) {
            const float4* k = (const float4*)(kc + kv_off + (ulong)lane * kv_dim);
            for (uint j0 = warp; j0 < n4; j0 += 8 * AP_B) {
                float4 kk[AP_B];
                for (uint j = 0; j < AP_B; j++) kk[j] = k[min(j0 + 8 * j, n4 - 1)];
                for (uint j = 0; j < AP_B && j0 + 8 * j < n4; j++) {
                    for (uint i = 0; i < 4 && g0 + i < g_n; i++) {
                        const float* qg = qs + (g0 + i) * head_dim + 4 * (j0 + 8 * j);
                        s[i] += qg[0] * kk[j].x + qg[1] * kk[j].y + qg[2] * kk[j].z
                            + qg[3] * kk[j].w;
                    }
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
    // The weighted sum of V: `np` threads per output value, each over every np-th position.
    uint np = AP_T / head_dim, d = tid % head_dim, pt = tid / head_dim;
    for (uint g0 = 0; g0 < g_n; g0 += 4) {
        float o[4] = {0, 0, 0, 0};
        if (pt < np) {
            for (uint p0 = pt; p0 < cnt; p0 += np * AP_B) {
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
    float mx = NEG_INF;
    for (uint c = tid; c < n_parts; c += AP_T) mx = fmaxf(mx, part[c * stride + head_dim]);
    mx = block_max(mx, red);
    float sum = 0;
    for (uint c = tid; c < n_parts; c += AP_T) {
        float f = expf(part[c * stride + head_dim] - mx);
        wf[c] = f;
        sum += part[c * stride + head_dim + 1] * f;
    }
    sum = block_sum(sum, red);  // its __syncthreads also orders the wf[] writes
    // `np` threads per output value, each over every np-th part.
    uint np = AP_T / head_dim, d = tid % head_dim, pt = tid / head_dim;
    float o = 0;
    if (pt < np) {
        for (uint c0 = pt; c0 < n_parts; c0 += np * AP_B) {
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
    for (uint i = tid; i < 32 * head_dim; i += blockDim.x) {
        uint p = i / head_dim, d = i % head_dim;
        ks[p * (head_dim + 1) + d] = base + p < n_pos + tok0 + t_end - 1
            ? kc[(ulong)(base + p) * kv_dim + blockIdx.x * head_dim + d] : 0.0f;
    }
    for (uint i = tid; i < 32 * head_dim; i += blockDim.x) {
        uint r = i / head_dim, d = i % head_dim, t = t0 + r / g_n;
        qs[i] = t < n_tok
            ? q[((ulong)(tok0 + t) * n_head + blockIdx.x * g_n + r % g_n) * head_dim + d] : 0.0f;
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
// keeps 16 sums.
extern "C" __global__ void attn_pv(const float* scores, const float* vc, float* out,
                                   uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                   uint kv_dim, uint max_ctx, uint tok0, uint n_tok) {
    __shared__ float vs[32 * 128];
    __shared__ float ps[32 * 32];
    uint tid = threadIdx.x, g_n = n_head / n_kv, tpt = 32 / g_n, t0 = blockIdx.y * tpt;
    uint t_end = min(t0 + tpt, n_tok), limit = n_pos + tok0 + t_end - 1;
    uint dq = tid % (head_dim / 4), rg = tid / (head_dim / 4);
    float o[16];
    for (uint i = 0; i < 16; i++) o[i] = 0;
    for (uint base = 0; base < limit; base += 32) {
        for (uint i = tid; i < 32 * head_dim; i += blockDim.x) {
            uint p = i / head_dim, d = i % head_dim;
            vs[i] = base + p < limit
                ? vc[(ulong)(base + p) * kv_dim + blockIdx.x * head_dim + d] : 0.0f;
        }
        for (uint i = tid; i < 32 * 32; i += blockDim.x) {
            uint r = i / 32, p = i % 32, t = t0 + r / g_n;
            ps[i] = t < n_tok && base + p < n_pos + tok0 + t
                ? scores[((ulong)t * n_head + blockIdx.x * g_n + r % g_n) * max_ctx + base + p]
                : 0.0f;
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

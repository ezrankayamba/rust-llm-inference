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
__device__ inline float h(const uchar* p) {
    uint b = (uint)p[0] | ((uint)p[1] << 8);
    uint sign = (b & 0x8000u) << 16, e = (b >> 10) & 31, m = b & 1023;
    if (e == 0) {
        float f = (float)m * 5.9604645e-8f;  // subnormal: m * 2^-24
        return sign ? -f : f;
    }
    uint bits = e == 31 ? (sign | 0x7f800000u | (m << 13)) : (sign | ((e + 112) << 23) | (m << 13));
    return __uint_as_float(bits);
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
// bytes after w_off, x at e * x_stride and y at e * y_stride. A plain matvec is one row with
// ids[0] = 0. MoE layers run their chosen experts as rows.
#define MATVEC(NAME, WT)                                                                       \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const float* x, ulong x_off,      \
                                float* y, ulong y_off, uint rows, uint cols, uint row_bytes,   \
                                const uint* ids, ulong id_stride, uint x_stride,               \
                                uint y_stride) {                                               \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32;                                                              \
    if (row >= rows) return;                                                                   \
    uint e = blockIdx.y;                                                                       \
    const uchar* r = w + w_off + (ulong)ids[e] * id_stride + (ulong)row * row_bytes;           \
    const float* xs = x + x_off + (ulong)e * x_stride + lane;                                  \
    float acc = 0;                                                                             \
    for (uint c = 0; c < cols / 32; c++) acc += WT(r, c, lane) * xs[c * 32];                   \
    acc = warp_sum(acc);                                                                       \
    if (lane == 0) y[y_off + (ulong)e * y_stride + row] = acc;                                 \
}

MATVEC(matvec_f32, wt_f32)
MATVEC(matvec_f16, wt_f16)

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
                                uint y_stride) {                                               \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32;                                                              \
    if (row >= rows) return;                                                                   \
    uint e = blockIdx.y;                                                                       \
    const uchar* r = w + w_off + (ulong)ids[e] * id_stride + (ulong)row * row_bytes;           \
    const float* xs = x + x_off + (ulong)e * x_stride;                                         \
    float acc = 0;                                                                             \
    for (uint c = 0; c < cols / 256; c++) acc += SB(r + c * SB_BYTES, xs + c * 256, lane);     \
    for (uint c = cols / 256 * 8; c < cols / 32; c++) acc += WT(r, c, lane) * xs[c * 32 + lane]; \
    acc = warp_sum(acc);                                                                       \
    if (lane == 0) y[y_off + (ulong)e * y_stride + row] = acc;                                 \
}

MATVEC_SB(matvec_q4_0, sb_q4_0, 8 * 18, wt_q4_0)
MATVEC_SB(matvec_q5_0, sb_q5_0, 8 * 22, wt_q5_0)
MATVEC_SB(matvec_q8_0, sb_q8_0, 8 * 34, wt_q8_0)
MATVEC_SB(matvec_q4_k, sb_q4_k, 144, wt_q4_k)
MATVEC_SB(matvec_q5_k, sb_q5_k, 176, wt_q5_k)
MATVEC_SB(matvec_q6_k, sb_q6_k, 210, wt_q6_k)

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

MATMUL(matmul_f32, wt_f32)
MATMUL(matmul_f16, wt_f16)
MATMUL(matmul_q4_0, wt_q4_0)
MATMUL(matmul_q5_0, wt_q5_0)
MATMUL(matmul_q8_0, wt_q8_0)
MATMUL(matmul_q4_k, wt_q4_k)
MATMUL(matmul_q5_k, wt_q5_k)
MATMUL(matmul_q6_k, wt_q6_k)

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
extern "C" __global__ void rmsnorm(const float* x, ulong x_off, float* out, ulong out_off,
                                   const float* w, uint n, float eps) {
    __shared__ float red[32];
    const float* xr = x + x_off + (ulong)blockIdx.x * n;
    float* o = out + out_off + (ulong)blockIdx.x * n;
    float s = 0;
    for (uint i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, red);
    float scale = rsqrtf(s / (float)n + eps);
    for (uint i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * scale * w[i];
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

extern "C" __global__ void swiglu(float* g, ulong g_off, const float* u, ulong u_off, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[g_off + i];
        g[g_off + i] = v / (1.0f + expf(-v)) * u[u_off + i];
    }
}

// x holds n_heads heads of consecutive tokens, heads_per_tok each, the first at position pos.
extern "C" __global__ void rope(float* x, ulong x_off, const float* ff, uint has_ff, uint n_heads,
                                uint head_dim, uint rope_dim, uint pos, float theta, uint neox,
                                uint heads_per_tok) {
    uint half_dim = rope_dim / 2;
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_heads * half_dim) return;
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

__device__ inline uint warp_min_u(uint v) {
    for (int o = 16; o > 0; o >>= 1) {
        uint u = __shfl_xor_sync(0xffffffff, v, o);
        v = u < v ? u : v;
    }
    return v;
}

// One warp: softmax over the n_expert router logits, then the k largest as ids[] and w[]
// (renormalized to sum to 1 when norm is set). With a shared expert, w[k] is its sigmoid gate.
// Takes up to 2048 experts (64 per lane).
extern "C" __global__ void moe_topk(const float* logits, const float* shared_gate, uint* ids,
                                    float* w, uint n_expert, uint k, uint norm, uint shared) {
    uint lane = threadIdx.x;
    float mx = NEG_INF;
    for (uint e = lane; e < n_expert; e += 32) mx = fmaxf(mx, logits[e]);
    mx = warp_max(mx);
    float sum = 0;
    for (uint e = lane; e < n_expert; e += 32) sum += expf(logits[e] - mx);
    sum = warp_sum(sum);
    ulong taken = 0;
    float chosen = 0;
    for (uint i = 0; i < k; i++) {
        float bv = NEG_INF;
        uint be = 0xffffffffu;
        for (uint e = lane, j = 0; e < n_expert; e += 32, j++) {
            if (!((taken >> j) & 1) && logits[e] > bv) {
                bv = logits[e];
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
            w[i] = wi;
        }
    }
    if (lane == 0) {
        if (norm) for (uint i = 0; i < k; i++) w[i] /= chosen;
        if (shared) w[k] = 1.0f / (1.0f + expf(-shared_gate[0]));
    }
}

// x += sum over r < m of w[r] * src row r (rows of n): the chosen experts' outputs, combined.
extern "C" __global__ void moe_combine(float* x, ulong x_off, const float* src, const float* w,
                                       uint n, uint m) {
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float s = 0;
    for (uint r = 0; r < m; r++) s += w[r] * src[(ulong)r * n + j];
    x[x_off + j] += s;
}

// One block per query head (x) and token (y): token tok0 + y of the batch attends to the first
// n_pos + tok0 + y cached positions. `scores` is one row of max_ctx per block. Needs
// head_dim <= blockDim.x <= 1024.
extern "C" __global__ void attention(const float* q, const float* kc, const float* vc, float* out,
                                     float* scores, uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                     uint kv_dim, uint max_ctx, float scale, uint tok0) {
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
        oh[d] = o / sum;
    }
}

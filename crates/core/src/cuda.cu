// CUDA port of metal.metal: matvec over GGUF-encoded weights plus the per-layer ops.
// Compiled at runtime with NVRTC. One warp computes one matvec output row.
// UNVERIFIED: written without access to an NVIDIA GPU; see crates/core/src/cuda.rs.
// No #include: NVRTC's builtin headers vary by toolkit version, so f16 is decoded by hand.

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long long ulong;

#define ROWS_PER_BLOCK 4

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

__device__ inline float chunk_f32(const uchar* r, uint c, const float* x) {
    const float* w = (const float*)r + c * 32;
    float s = 0;
    for (uint i = 0; i < 32; i++) s += w[i] * x[i];
    return s;
}

__device__ inline float chunk_f16(const uchar* r, uint c, const float* x) {
    const uchar* w = r + c * 64;
    float s = 0;
    for (uint i = 0; i < 32; i++) s += h(w + 2 * i) * x[i];
    return s;
}

__device__ inline float chunk_q4_0(const uchar* r, uint c, const float* x) {
    const uchar* b = r + c * 18;
    float s = 0;
    for (uint j = 0; j < 16; j++) {
        uchar q = b[2 + j];
        s += (float)((int)(q & 15) - 8) * x[j] + (float)((int)(q >> 4) - 8) * x[j + 16];
    }
    return h(b) * s;
}

__device__ inline float chunk_q5_0(const uchar* r, uint c, const float* x) {
    const uchar* b = r + c * 22;
    uint qh = (uint)b[2] | ((uint)b[3] << 8) | ((uint)b[4] << 16) | ((uint)b[5] << 24);
    float s = 0;
    for (uint j = 0; j < 16; j++) {
        uchar q = b[6 + j];
        int lo = (int)((q & 15) | (((qh >> j) << 4) & 0x10)) - 16;
        int hi = (int)((q >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
        s += (float)lo * x[j] + (float)hi * x[j + 16];
    }
    return h(b) * s;
}

__device__ inline float chunk_q8_0(const uchar* r, uint c, const float* x) {
    const uchar* b = r + c * 34;
    const signed char* q = (const signed char*)(b + 2);
    float s = 0;
    for (uint i = 0; i < 32; i++) s += (float)q[i] * x[i];
    return h(b) * s;
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

__device__ inline float chunk_q4_k(const uchar* r, uint c, const float* x) {
    uint sub = c % 8;
    const uchar* b = r + (c / 8) * 144;
    float sc, m;
    scale_min_k4(sub, b + 4, &sc, &m);
    const uchar* q = b + 16 + 32 * (sub / 2);
    float s = 0, sx = 0;
    for (uint l = 0; l < 32; l++) {
        uint v = (sub & 1) ? (q[l] >> 4) : (q[l] & 15);
        s += (float)v * x[l];
        sx += x[l];
    }
    return h(b) * sc * s - h(b + 2) * m * sx;
}

__device__ inline float chunk_q5_k(const uchar* r, uint c, const float* x) {
    uint sub = c % 8;
    const uchar* b = r + (c / 8) * 176;
    float sc, m;
    scale_min_k4(sub, b + 4, &sc, &m);
    const uchar* qh = b + 16;
    const uchar* q = b + 48 + 32 * (sub / 2);
    float s = 0, sx = 0;
    for (uint l = 0; l < 32; l++) {
        uint v = ((sub & 1) ? (q[l] >> 4) : (q[l] & 15)) + (((qh[l] >> sub) & 1) << 4);
        s += (float)v * x[l];
        sx += x[l];
    }
    return h(b) * sc * s - h(b + 2) * m * sx;
}

__device__ inline float chunk_q6_k(const uchar* r, uint c, const float* x) {
    uint j = c % 8, n = j / 4, k = j % 4;
    const uchar* b = r + (c / 8) * 210;
    const uchar* ql = b + 64 * n + 32 * (k % 2);
    const uchar* qh = b + 128 + 32 * n;
    const signed char* sc = (const signed char*)(b + 192) + 8 * n + 2 * k;
    float s0 = 0, s1 = 0;
    for (uint l = 0; l < 32; l++) {
        uint lo = k < 2 ? (ql[l] & 15) : (ql[l] >> 4);
        int q = (int)(lo | (((qh[l] >> (2 * k)) & 3) << 4)) - 32;
        if (l < 16) s0 += (float)q * x[l]; else s1 += (float)q * x[l];
    }
    return h(b + 208) * ((float)sc[0] * s0 + (float)sc[1] * s1);
}

#define MATVEC(NAME, CHUNK)                                                                    \
extern "C" __global__ void NAME(const uchar* w, ulong w_off, const float* x, ulong x_off,      \
                                float* y, ulong y_off, uint rows, uint cols, uint row_bytes) { \
    uint row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;                                 \
    uint lane = threadIdx.x % 32;                                                              \
    if (row >= rows) return;                                                                   \
    const uchar* r = w + w_off + (ulong)row * row_bytes;                                       \
    const float* xs = x + x_off;                                                               \
    float acc = 0;                                                                             \
    for (uint c = lane; c < cols / 32; c += 32) acc += CHUNK(r, c, xs + c * 32);               \
    acc = warp_sum(acc);                                                                       \
    if (lane == 0) y[y_off + row] = acc;                                                       \
}

MATVEC(matvec_f32, chunk_f32)
MATVEC(matvec_f16, chunk_f16)
MATVEC(matvec_q4_0, chunk_q4_0)
MATVEC(matvec_q5_0, chunk_q5_0)
MATVEC(matvec_q8_0, chunk_q8_0)
MATVEC(matvec_q4_k, chunk_q4_k)
MATVEC(matvec_q5_k, chunk_q5_k)
MATVEC(matvec_q6_k, chunk_q6_k)

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

extern "C" __global__ void swiglu(float* g, ulong g_off, const float* u, ulong u_off, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[g_off + i];
        g[g_off + i] = v / (1.0f + expf(-v)) * u[u_off + i];
    }
}

extern "C" __global__ void rope(float* x, ulong x_off, const float* ff, uint has_ff, uint n_heads,
                                uint head_dim, uint rope_dim, uint pos, float theta, uint neox) {
    uint half_dim = rope_dim / 2;
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_heads * half_dim) return;
    uint hh = gid / half_dim, i = gid % half_dim;
    float freq = powf(theta, -2.0f * (float)i / (float)rope_dim);
    if (has_ff) freq /= ff[i];
    float s, c;
    sincosf((float)pos * freq, &s, &c);
    float* hd = x + x_off + hh * head_dim;
    uint a = neox ? i : 2 * i;
    uint b = neox ? i + half_dim : 2 * i + 1;
    float x0 = hd[a], x1 = hd[b];
    hd[a] = x0 * c - x1 * s;
    hd[b] = x0 * s + x1 * c;
}

// One block per query head; `scores` is n_head * max_ctx scratch.
extern "C" __global__ void attention(const float* q, const float* kc, const float* vc, float* out,
                                     float* scores, uint n_head, uint n_kv, uint head_dim, uint n_pos,
                                     uint kv_dim, uint max_ctx, float scale) {
    __shared__ float red[32];
    uint hh = blockIdx.x;
    uint kv_off = (hh / (n_head / n_kv)) * head_dim;
    const float* qh = q + hh * head_dim;
    float* sc = scores + (ulong)hh * max_ctx;
    float mx = NEG_INF;
    for (uint t = threadIdx.x; t < n_pos; t += blockDim.x) {
        const float* k = kc + (ulong)t * kv_dim + kv_off;
        float s = 0;
        for (uint d = 0; d < head_dim; d++) s += qh[d] * k[d];
        s *= scale;
        sc[t] = s;
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
    for (uint d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float o = 0;
        for (uint t = 0; t < n_pos; t++) o += sc[t] * vc[(ulong)t * kv_dim + kv_off + d];
        out[hh * head_dim + d] = o / sum;
    }
}

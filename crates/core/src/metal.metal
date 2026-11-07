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

struct EltParams { uint n; float alpha; };

kernel void add(device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                constant EltParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) a[i] += b[i];
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

kernel void rope(device float* x [[buffer(0)]], device const float* ff [[buffer(1)]],
                 constant RopeParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    uint half_dim = p.rope_dim / 2;
    if (gid >= p.n_heads * half_dim) return;
    uint h = gid / half_dim, i = gid % half_dim;
    float freq = precise::pow(p.theta, -2.0f * float(i) / float(p.rope_dim));
    if (p.has_ff) freq /= ff[i];
    float ang = float(p.pos) * freq;
    float c = precise::cos(ang), s = precise::sin(ang);
    device float* hd = x + h * p.head_dim;
    uint a = p.neox ? i : 2 * i;
    uint b = p.neox ? i + half_dim : 2 * i + 1;
    float x0 = hd[a], x1 = hd[b];
    hd[a] = x0 * c - x1 * s;
    hd[b] = x0 * s + x1 * c;
}

struct AttnParams { uint n_head; uint n_kv; uint head_dim; uint n_pos; uint kv_dim; uint max_ctx; float scale; };

// One threadgroup per query head; `scores` is n_head * max_ctx scratch.
kernel void attention(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]],
                      device const float* vc [[buffer(2)]], device float* out [[buffer(3)]],
                      device float* scores [[buffer(4)]], constant AttnParams& p [[buffer(5)]],
                      uint h [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
                      uint tgs [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    uint n_sg = (tgs + 31) / 32;
    uint kv_off = (h / (p.n_head / p.n_kv)) * p.head_dim;
    device const float* qh = q + h * p.head_dim;
    device float* sc = scores + h * p.max_ctx;
    float mx = -INFINITY;
    for (uint t = tid; t < p.n_pos; t += tgs) {
        device const float* k = kc + t * p.kv_dim + kv_off;
        float s = 0;
        for (uint d = 0; d < p.head_dim; d++) s += qh[d] * k[d];
        s *= p.scale;
        sc[t] = s;
        mx = max(mx, s);
    }
    mx = tg_reduce_max(mx, red, lane, sg, n_sg);
    float sum = 0;
    for (uint t = tid; t < p.n_pos; t += tgs) {
        float e = exp(sc[t] - mx);
        sc[t] = e;
        sum += e;
    }
    sum = tg_reduce_sum(sum, red, lane, sg, n_sg);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint d = tid; d < p.head_dim; d += tgs) {
        float o = 0;
        for (uint t = 0; t < p.n_pos; t++) o += sc[t] * vc[t * p.kv_dim + kv_off + d];
        out[h * p.head_dim + d] = o / sum;
    }
}

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

// Tiled matmul on the simdgroup 8x8 matrix units. A threadgroup (4 simdgroups) computes a
// MM_R x MM_T tile of rows x tokens; per 32-column step, simdgroups 0-1 dequantize the weight
// tile into threadgroup memory while 2-3 stage the activations, then each simdgroup multiplies
// a 32-row x 16-token sub-tile. All f32, so results track the matvec path closely.
constant uint MM_R = 64;
constant uint MM_T = 32;

struct MatmulParams {
    ulong offset;
    uint rows;
    uint cols;
    uint row_bytes;
    uint n;           // tokens
};

#define MATMUL(NAME, DEQ)                                                                     \
kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],   \
                 device float* y [[buffer(2)]], constant MatmulParams& p [[buffer(3)]],      \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint tid [[thread_index_in_threadgroup]],                                    \
                 uint sg [[simdgroup_index_in_threadgroup]]) {                                \
    threadgroup float sa[MM_R * 32]; /* weights [row][k]; reused for the output tile */       \
    threadgroup float sb[MM_T * 32]; /* activations [token][k] */                             \
    uint r0 = tg.x * MM_R, t0 = tg.y * MM_T;                                                  \
    uint rb = (sg % 2) * 32, tb = (sg / 2) * 16;                                              \
    simdgroup_float8x8 acc[8];                                                                \
    for (uint i = 0; i < 8; i++) acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);       \
    for (uint c = 0; c < p.cols / 32; c++) {                                                  \
        if (tid < MM_R) {                                                                     \
            uint row = r0 + tid;                                                              \
            float wv[32];                                                                     \
            if (row < p.rows) {                                                               \
                DEQ(w + p.offset + ulong(row) * p.row_bytes, c, wv);                          \
            } else {                                                                          \
                for (uint i = 0; i < 32; i++) wv[i] = 0.0f;                                   \
            }                                                                                 \
            for (uint i = 0; i < 32; i++) sa[tid * 32 + i] = wv[i];                           \
        } else {                                                                              \
            uint j = tid - MM_R, tt = j / 2, half_k = (j % 2) * 16;                           \
            uint t = t0 + tt;                                                                 \
            device const float* xs = x + ulong(t) * p.cols + c * 32 + half_k;                 \
            for (uint i = 0; i < 16; i++) sb[tt * 32 + half_k + i] = t < p.n ? xs[i] : 0.0f;  \
        }                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
        for (uint k = 0; k < 32; k += 8) {                                                    \
            simdgroup_float8x8 a[4], b[2];                                                    \
            for (uint i = 0; i < 4; i++) simdgroup_load(a[i], sa + (rb + 8 * i) * 32 + k, 32);\
            for (uint j = 0; j < 2; j++)                                                      \
                simdgroup_load(b[j], sb + (tb + 8 * j) * 32 + k, 32, ulong2(0, 0), true);     \
            for (uint i = 0; i < 4; i++)                                                      \
                for (uint j = 0; j < 2; j++)                                                  \
                    simdgroup_multiply_accumulate(acc[i * 2 + j], a[i], b[j], acc[i * 2 + j]);\
        }                                                                                     \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
    }                                                                                         \
    /* Stage the 64 x 32 tile as [row][token], then write y[token][row] with bounds. */       \
    for (uint i = 0; i < 4; i++)                                                              \
        for (uint j = 0; j < 2; j++)                                                          \
            simdgroup_store(acc[i * 2 + j], sa + (rb + 8 * i) * MM_T + tb + 8 * j, MM_T);     \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                          \
    for (uint e = tid; e < MM_R * MM_T; e += 128) {                                           \
        uint tt = e / MM_R, rr = e % MM_R;                                                    \
        uint row = r0 + rr, t = t0 + tt;                                                      \
        if (row < p.rows && t < p.n) y[ulong(t) * p.rows + row] = sa[rr * MM_T + tt];         \
    }                                                                                         \
}

MATMUL(matmul_f32, deq_f32)
MATMUL(matmul_f16, deq_f16)
MATMUL(matmul_q4_0, deq_q4_0)
MATMUL(matmul_q5_0, deq_q5_0)
MATMUL(matmul_q8_0, deq_q8_0)
MATMUL(matmul_q4_k, deq_q4_k)
MATMUL(matmul_q5_k, deq_q5_k)
MATMUL(matmul_q6_k, deq_q6_k)

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

struct BatchAttnParams { uint n_head; uint n_kv; uint head_dim; uint pos; uint kv_dim; float scale; };

// Causal attention for a chunk of queries: threadgroup (head h, token t), one simdgroup each.
// Query t sits at position pos + t and sees cache rows 0..=pos + t. Softmax is computed online
// (running max and sum over blocks of 32 keys), so no scores buffer is needed. Lane l owns
// output dims l, l + 32, ... (head_dim must be a multiple of 32, at most 256).
template <typename T>
inline void attention_batch_impl(device const float* q, device const T* kc, device const T* vc,
                                 device float* out, constant BatchAttnParams& p, uint2 tg,
                                 uint lane) {
    uint h = tg.x, t = tg.y, hd = p.head_dim, dpl = hd / 32;
    uint n_pos = p.pos + t + 1;
    uint kv_off = (h / (p.n_head / p.n_kv)) * hd;
    ulong q_dim = ulong(p.n_head) * hd;
    device const float* qh = q + t * q_dim + h * hd;
    float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0;
    for (uint base = 0; base < n_pos; base += 32) {
        uint j = base + lane;
        float s = -INFINITY;
        if (j < n_pos) {
            device const T* k = kc + ulong(j) * p.kv_dim + kv_off;
            float d = 0;
            for (uint i = 0; i < hd; i++) d += qh[i] * float(k[i]);
            s = d * p.scale;
        }
        float m_new = max(m, simd_max(s));
        float corr = exp(m - m_new);
        float pj = j < n_pos ? exp(s - m_new) : 0.0f;
        l = l * corr + simd_sum(pj);
        for (uint d = 0; d < dpl; d++) o[d] *= corr;
        uint cnt = min(32u, n_pos - base);
        for (uint jj = 0; jj < cnt; jj++) {
            float pw = simd_shuffle(pj, jj);
            device const T* v = vc + ulong(base + jj) * p.kv_dim + kv_off;
            for (uint d = 0; d < dpl; d++) o[d] += pw * float(v[lane + 32 * d]);
        }
        m = m_new;
    }
    device float* oh = out + t * q_dim + h * hd;
    for (uint d = 0; d < dpl; d++) oh[lane + 32 * d] = o[d] / l;
}

#define ATTENTION_BATCH(NAME, T)                                                              \
kernel void NAME(device const float* q [[buffer(0)]], device const T* kc [[buffer(1)]],      \
                 device const T* vc [[buffer(2)]], device float* out [[buffer(3)]],          \
                 constant BatchAttnParams& p [[buffer(4)]],                                   \
                 uint2 tg [[threadgroup_position_in_grid]],                                   \
                 uint lane [[thread_index_in_simdgroup]]) {                                   \
    attention_batch_impl<T>(q, kc, vc, out, p, tg, lane);                                     \
}

ATTENTION_BATCH(attention_batch_f32, float)
ATTENTION_BATCH(attention_batch_f16, half)

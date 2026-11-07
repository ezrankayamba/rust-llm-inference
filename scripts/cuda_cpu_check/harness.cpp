// Host-side harness for the CUDA kernels' per-type decoders (see check.py). Runs the
// `chunk_*` functions of crates/core/src/cuda.cu serially on the CPU; warp/block code is stubbed.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define __device__
#define __global__
#define __shared__ static
struct Dim3 { unsigned x, y, z; };
Dim3 threadIdx, blockIdx, blockDim;
float __shfl_xor_sync(unsigned, float v, int) { return v; }
void __syncthreads() {}
float __uint_as_float(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
float rsqrtf(float x) { return 1.0f / sqrtf(x); }
void sincosf(float a, float* s, float* c) { *s = sinf(a); *c = cosf(a); }
#include "../../crates/core/src/cuda.cu"
// usage: harness <type> <rows> <cols> <row_bytes> <w.bin> <x.f32> <y.f32>
int main(int argc, char** argv) {
    const char* t = argv[1]; uint rows = atoi(argv[2]), cols = atoi(argv[3]), rb = atoi(argv[4]);
    auto slurp = [](const char* p) { FILE* f = fopen(p, "rb"); std::vector<uchar> v; int c; while ((c = fgetc(f)) != EOF) v.push_back(c); fclose(f); return v; };
    auto w = slurp(argv[5]); auto xb = slurp(argv[6]);
    const float* x = (const float*)xb.data();
    float (*fn)(const uchar*, uint, const float*) =
        !strcmp(t,"F32")?chunk_f32: !strcmp(t,"F16")?chunk_f16: !strcmp(t,"Q4_0")?chunk_q4_0:
        !strcmp(t,"Q5_0")?chunk_q5_0: !strcmp(t,"Q8_0")?chunk_q8_0: !strcmp(t,"Q4_K")?chunk_q4_k:
        !strcmp(t,"Q5_K")?chunk_q5_k: chunk_q6_k;
    std::vector<float> y(rows);
    for (uint r = 0; r < rows; r++) { float a = 0; for (uint c = 0; c < cols / 32; c++) a += fn(w.data() + (size_t)r * rb, c, x + c * 32); y[r] = a; }
    FILE* o = fopen(argv[7], "wb"); fwrite(y.data(), 4, rows, o); fclose(o);
}

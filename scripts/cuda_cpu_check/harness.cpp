// Host-side harness for the CUDA kernels' per-type decoders (see check.py). Runs the
// `wt_*` and `sb_*` functions of crates/core/src/cuda.cu serially on the CPU; warp/block code is
// stubbed.
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
unsigned __shfl_xor_sync(unsigned, unsigned v, int) { return v; }
void __syncthreads() {}
float __uint_as_float(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
float rsqrtf(float x) { return 1.0f / sqrtf(x); }
void sincosf(float a, float* s, float* c) { *s = sinf(a); *c = cosf(a); }
#define ulong cu_ulong  // glibc already has a ulong, of another type
#include "../../crates/core/src/cuda.cu"
// usage: harness <type> <rows> <cols> <row_bytes> <w.bin> <x.f32> <y.f32> <wt|sb>
int main(int argc, char** argv) {
    const char* t = argv[1]; uint rows = atoi(argv[2]), cols = atoi(argv[3]), rb = atoi(argv[4]);
    auto slurp = [](const char* p) { FILE* f = fopen(p, "rb"); std::vector<uchar> v; int c; while ((c = fgetc(f)) != EOF) v.push_back(c); fclose(f); return v; };
    auto w = slurp(argv[5]); auto xb = slurp(argv[6]);
    const float* x = (const float*)xb.data();
    float (*wt)(const uchar*, uint, uint) =
        !strcmp(t,"F32")?wt_f32: !strcmp(t,"F16")?wt_f16: !strcmp(t,"Q4_0")?wt_q4_0:
        !strcmp(t,"Q5_0")?wt_q5_0: !strcmp(t,"Q8_0")?wt_q8_0: !strcmp(t,"Q4_K")?wt_q4_k:
        !strcmp(t,"Q5_K")?wt_q5_k: wt_q6_k;
    // The matvec kernels of the quantized types take 256 weights per step instead.
    float (*sb)(const uchar*, const float*, uint) =
        !strcmp(t,"Q4_0")?sb_q4_0: !strcmp(t,"Q5_0")?sb_q5_0: !strcmp(t,"Q8_0")?sb_q8_0:
        !strcmp(t,"Q4_K")?sb_q4_k: !strcmp(t,"Q5_K")?sb_q5_k: !strcmp(t,"Q6_K")?sb_q6_k: nullptr;
    bool wide = !strcmp(argv[8], "sb");
    if (wide && !sb) return 2;
    std::vector<float> y(rows);
    for (uint r = 0; r < rows; r++) {
        const uchar* row = w.data() + (size_t)r * rb;
        float a = 0;
        if (wide) {
            for (uint c = 0; c < cols / 256; c++)
                for (uint l = 0; l < 32; l++) a += sb(row + (size_t)c * (rb * 8 / (cols / 32)), x + c * 256, l);
            for (uint c = cols / 256 * 8; c < cols / 32; c++)
                for (uint l = 0; l < 32; l++) a += wt(row, c, l) * x[c * 32 + l];
        } else {
            for (uint c = 0; c < cols / 32; c++)
                for (uint l = 0; l < 32; l++) a += wt(row, c, l) * x[c * 32 + l];
        }
        y[r] = a;
    }
    FILE* o = fopen(argv[7], "wb"); fwrite(y.data(), 4, rows, o); fclose(o);
}

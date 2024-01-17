#include <arm_neon.h>

// Inspired from:
// See: https://github.com/alivanz/go-simd

void VmulqF32(float32x4_t* r, float32x4_t* v0, float32x4_t* v1) { *r = vmulq_f32(*v0, *v1); }
void VaddvqF32(float32_t* r, float32x4_t* v0) { *r = vaddvq_f32(*v0); }

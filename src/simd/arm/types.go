package arm

/*
#include <arm_neon.h>
*/
import "C"

// Inspired from:
// See: https://github.com/alivanz/go-simd

// typedef float float32_t;
type Float32 = C.float32_t

// typedef __attribute__((neon_vector_type(4))) float32_t float32x4_t;
type Float32X4 = C.float32x4_t

// typedef __attribute__((neon_vector_type(8))) int8_t int8x8_t;
type Int8X8 = C.int8x8_t

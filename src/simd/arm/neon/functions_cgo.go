package neon

/*
#include <arm_neon.h>
*/
import "C"

// Inspired from:
// See: https://github.com/alivanz/go-simd

type int8x8 = C.int8x8_t

func vmulS8_cgo(r, v0, v1 *int8x8) {
	*r = C.vmul_s8(*v0, *v1)
}

package neon

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/simd/arm"
)

// Inspired from:
// See: https://github.com/alivanz/go-simd

/*
#include <arm_neon.h>
*/
import "C"

// Floating-point Multiply (vector). This instruction multiplies corresponding floating-point values in the vectors in the two source SIMD&FP registers, places the result in a vector, and writes the vector to the destination SIMD&FP register.
//
//go:linkname VmulqF32 VmulqF32
//go:noescape
func VmulqF32(r *arm.Float32X4, v0 *arm.Float32X4, v1 *arm.Float32X4)

// Floating-point add across vector
//
//go:linkname VaddvqF32 VaddvqF32
//go:noescape
func VaddvqF32(r *arm.Float32, v0 *arm.Float32X4)

// BFloat16 dot product

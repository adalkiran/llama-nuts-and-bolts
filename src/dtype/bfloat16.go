package dtype

import (
	"encoding/binary"
	"math"
	"strconv"
)

//See: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
//See: https://en.wikipedia.org/wiki/Floating-point_arithmetic
//See: https://cloud.google.com/tpu/docs/bfloat16

type BFloat16 uint16

func (bf BFloat16) Bits() uint16 {
	return uint16(bf)
}

func (bf BFloat16) Float32() float32 {
	return math.Float32frombits(uint32(bf) << 16)
}

func (bf BFloat16) Float64() float64 {
	return float64(bf.Float32())
}

func (bf *BFloat16) String() string {
	return strconv.FormatFloat(bf.Float64(), 'f', -1, 32)
}

func BFloat16fromFloat32(f32 float32) BFloat16 {
	return BFloat16(math.Float32bits(f32) >> 16)
}

func BFloat16frombits(b16 uint16) BFloat16 {
	return BFloat16(b16)
}

func ReadBFloat16LittleEndian(b []byte) BFloat16 {
	return BFloat16(binary.LittleEndian.Uint16(b))
}

func ReadBFloat16BigEndian(b []byte) BFloat16 {
	return BFloat16(binary.BigEndian.Uint16(b))
}

func WriteBFloat16LittleEndian(b []byte, v BFloat16) {
	binary.LittleEndian.PutUint16(b, v.Bits())
}

func WriteBFloat16BigEndian(b []byte, v BFloat16) {
	binary.BigEndian.PutUint16(b, v.Bits())
}

func BFloat16bitsToFloat32(b16 uint16) float32 {
	return math.Float32frombits(uint32(b16) << 16)
}

func Float32ToBFloat16bits(f32 float32) uint16 {
	return uint16(math.Float32bits(f32) >> 16)
}

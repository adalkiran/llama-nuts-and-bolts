package ml

import (
	"fmt"
	"math"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

var (
	TABLE_SILU [1 << 16]float32
)

// See: https://tutorialedge.net/golang/the-go-init-function/
func init() {
	for i := 0; i < (1 << 16); i++ {
		ii := dtype.BFloat16frombits(uint16(i)).Float64()
		TABLE_SILU[i] = _silu(ii)
	}
}

func _silu(x float64) float32 {
	// See: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
	return float32(x / (1.0 + math.Exp(float64(-x))))
}

func Silu(input *Tensor) (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
	inputItemSize := input.DataType.ItemSize

	dst := NewEmptyTensor(input.Size, input.DataType)
	writeOffset := 0
	for readOffset := 0; readOffset < input.GetBytesCount(); readOffset += inputItemSize {
		item := input.GetItemByOffset(readOffset)
		switch input.DataType {
		case DT_BF16:
			item := item.(dtype.BFloat16)
			resultItem := dtype.BFloat16fromFloat32(TABLE_SILU[item.Bits()])
			dst.SetItemByOffset(writeOffset, resultItem)
		case DT_F32:
			item := item.(float32)
			resultItem := TABLE_SILU[dtype.BFloat16fromFloat32(item).Bits()]
			dst.SetItemByOffset(writeOffset, resultItem)
		default:
			return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
		}
		writeOffset += inputItemSize
	}
	return dst, nil
}

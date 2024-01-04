package ml

import (
	"fmt"
	"math"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

func ARange(start int, end int, step int, dataType DataType) (*Tensor, error) {
	if start >= end {
		return nil, fmt.Errorf("start value %d must be less than end value %d in ARange", start, end)
	}
	result := NewEmptyTensor([]int{int(math.Ceil(float64(end-start) / float64(step)))}, dataType)
	i := 0
	for val := start; val < end; val += step {
		var valConv any
		switch result.DataType {
		case DT_BF16:
			valConv = dtype.BFloat16fromFloat32(float32(val))
		case DT_F32:
			valConv = float32(val)
		default:
			return nil, fmt.Errorf("unsupported tensor datatype %s", result.DataType)
		}
		if err := result.SetItem([]int{i}, valConv); err != nil {
			return nil, err
		}
		i++
	}
	return result, nil
}

func Outer(vec1 *Tensor, vec2 *Tensor) (*Tensor, error) {
	if err := processErrors(
		checkIsVector(vec1),
		checkIsVector(vec2),
		checkSameDataType(vec1, vec2),
	); err != nil {
		return nil, err
	}
	itemSize := vec1.DataType.ItemSize()
	result := NewEmptyTensor([]int{vec1.Size[0], vec2.Size[0]}, vec1.DataType)
	for i := 0; i < vec1.Size[0]; i++ {
		rowVal := vec1.GetItemByOffset(i * itemSize)
		switch result.DataType {
		case DT_BF16:
			row := rowVal.(dtype.BFloat16)
			for j := 0; j < vec2.Size[0]; j++ {
				col := vec2.GetItemByOffset(j * itemSize).(dtype.BFloat16)
				val := dtype.BFloat16fromFloat32(row.Float32() * col.Float32())
				if err := result.SetItem([]int{i, j}, val); err != nil {
					return nil, err
				}
			}
		default:
			return nil, fmt.Errorf("unsupported tensor datatype %s", result.DataType)
		}
	}
	return result, nil
}

func Full(size []int, dataType DataType, fillValue any) (*Tensor, error) {
	result := NewEmptyTensor(size, dataType)
	err := result.Apply(func(val any) any {
		return fillValue
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func Zeros(size []int, dataType DataType) (*Tensor, error) {
	var val any
	switch dataType {
	case DT_BF16:
		val = dtype.BFloat16fromFloat32(0)
	case DT_F32:
		val = float32(0)
	default:
		return nil, fmt.Errorf("unsupported tensor datatype %s", dataType)
	}
	return Full(size, dataType, val)
}

func Ones(size []int, dataType DataType) (*Tensor, error) {
	var val any
	switch dataType {
	case DT_BF16:
		val = dtype.BFloat16fromFloat32(1)
	case DT_F32:
		val = float32(1)
	default:
		return nil, fmt.Errorf("unsupported tensor datatype %s", dataType)
	}
	return Full(size, dataType, val)
}

func ZerosLike(input *Tensor) (*Tensor, error) {
	return Zeros(input.Size, input.DataType)
}

func OnesLike(input *Tensor) (*Tensor, error) {
	return Ones(input.Size, input.DataType)
}

func Polar(abs *Tensor, angle *Tensor) (*Tensor, error) {
	// See: (For formula) https://pytorch.org/docs/stable/generated/torch.polar.html
	if err := processErrors(
		checkSameShape(abs, angle),
		checkSameDataType(abs, angle),
		// Currently only 2D matrices are supported
		checkIsMatrix(abs),
		checkIsMatrix(angle),
	); err != nil {
		return nil, err
	}

	dst := NewEmptyTensor(abs.Size, DT_COMPLEX)
	for i := 0; i < dst.Size[0]; i++ {
		for j := 0; j < dst.Size[1]; j++ {
			absItem, err := abs.GetItem([]int{i, j})
			if err != nil {
				return nil, err
			}
			angleItem, err := angle.GetItem([]int{i, j})
			if err != nil {
				return nil, err
			}
			var absItemConv float64
			var angleItemConv float64
			switch abs.DataType {
			case DT_BF16:
				absItemConv = absItem.(dtype.BFloat16).Float64()
				angleItemConv = angleItem.(dtype.BFloat16).Float64()
			case DT_F32:
				absItemConv = float64(absItem.(float32))
				angleItemConv = float64(angleItem.(float32))
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", abs.DataType)
			}

			realPart := absItemConv * math.Cos(angleItemConv)
			imagPart := absItemConv * math.Sin(angleItemConv)
			resultItem := complex64(complex(realPart, imagPart))
			if err := dst.SetItem([]int{i, j}, resultItem); err != nil {
				return nil, err
			}
		}
	}
	return dst, nil
}

func Fwd_Get_Rows(embedding *Tensor, tokens *Tensor) (*Tensor, error) {
	if err := processErrors(
		checkIsMatrix(embedding),
		checkIsVector(tokens),
	); err != nil {
		return nil, err
	}

	if tokens.DataType != DT_UINT16 {
		return nil, fmt.Errorf("tensor is not in data type %s: \"%s\" is %s", DT_UINT16, tokens.Name, tokens.DataType)
	}

	sequenceLength := tokens.Size[0]
	embeddingDim := embedding.Size[1]
	dst := NewEmptyTensor([]int{sequenceLength, embeddingDim}, embedding.DataType)

	rowCount := tokens.GetElementCount()

	for rowIdx := 0; rowIdx < rowCount; rowIdx++ {
		rowVal, err := tokens.GetItem([]int{rowIdx})
		if err != nil {
			return nil, err
		}
		row := int(rowVal.(uint16))
		readOffsetStart := embedding.calculateByteOffset([]int{row, 0})
		readOffsetEnd := embedding.calculateByteOffset([]int{row + 1, 0})
		rowBytes := embedding.RawData[readOffsetStart:readOffsetEnd]
		writeOffsetStart := dst.calculateByteOffset([]int{rowIdx, 0})
		copy(dst.RawData[writeOffsetStart:], rowBytes)
	}
	return dst, nil
}

func TriangularUpper(input *Tensor, diagonal int) (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.triu.html

	rowCount := input.Size[0]
	colCount := input.Size[1]

	dst := NewEmptyTensor(input.Size, input.DataType)
	for i := 0; i < rowCount; i++ {
		for j := 0; j < colCount; j++ {
			if j-i >= diagonal {
				loc := []int{i, j}
				val, _ := input.GetItem(loc)
				if err := dst.SetItem(loc, val); err != nil {
					return nil, err
				}
			}
		}
	}

	return dst, nil
}

func Pow(input *Tensor, power float64) (*Tensor, error) {
	dstDataType := input.DataType
	switch input.DataType {
	case DT_BF16:
		dstDataType = DT_F32
	}
	inputItemSize := input.DataType.ItemSize()

	dst := NewEmptyTensor(input.Size, dstDataType)
	writeOffset := 0
	for readOffset := 0; readOffset < input.GetBytesCount(); readOffset += inputItemSize {
		item := input.GetItemByOffset(readOffset)
		switch input.DataType {
		case DT_BF16:
			item := item.(dtype.BFloat16)
			resultItem := float32(math.Pow(item.Float64(), power))
			dst.SetItemByOffset(writeOffset, resultItem)
		case DT_F32:
			item := item.(float32)
			resultItem := float32(math.Pow(float64(item), power))
			dst.SetItemByOffset(writeOffset, resultItem)
		default:
			return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
		}
		writeOffset += dstDataType.ItemSize()
	}
	return dst, nil
}

func Mean(input *Tensor, dim int, keepdim bool) (*Tensor, error) {
	if dim != -1 && dim != len(input.Size) {
		return nil, fmt.Errorf("function Mean currently supports only dim=-1 or dim=(last dimension of input)")
	}
	dstSize := make([]int, len(input.Size))
	copy(dstSize, input.Size)

	if keepdim {
		dstSize[len(dstSize)-1] = 1
	} else {
		dstSize = dstSize[:len(dstSize)-1]
	}
	itemSize := input.DataType.ItemSize()
	dst := NewEmptyTensor(dstSize, input.DataType)
	inputLastSize := input.Size[len(input.Size)-1]
	inputStride := inputLastSize * itemSize

	fmt.Println()
	dstOffset := 0
	for readGroupOffset := 0; readGroupOffset < input.GetBytesCount(); readGroupOffset += inputStride {
		groupSum := float32(0)
		for groupItemIdx := 0; groupItemIdx < inputLastSize; groupItemIdx++ {
			var itemF32 float32
			item := input.GetItemByOffset(readGroupOffset + groupItemIdx*itemSize)
			switch input.DataType {
			case DT_BF16:
				itemF32 = item.(dtype.BFloat16).Float32()
			case DT_F32:
				itemF32 = item.(float32)
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
			}
			groupSum += itemF32
		}
		groupMeanF32 := groupSum / float32(inputLastSize)
		var groupMean any
		switch input.DataType {
		case DT_BF16:
			groupMean = dtype.BFloat16fromFloat32(groupMeanF32)
		case DT_F32:
			groupMean = groupMeanF32
		}

		if err := dst.SetItemByOffset(dstOffset, groupMean); err != nil {
			return nil, err
		}
		dstOffset += itemSize
	}
	return dst, nil
}

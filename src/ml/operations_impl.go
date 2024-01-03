package ml

import (
	"fmt"
	"math"

	"github.com/x448/float16"
)

func ARange(start int, end int, step int, dataType DataType) (*Tensor, error) {
	if start >= end {
		return nil, fmt.Errorf("start value %d must be less than end value %d in ARange", start, end)
	}
	result := NewEmptyTensor([]int{int(float32(end-start) / float32(step))}, dataType)
	for i := start; i < end; i += step {
		result.SetItem([]int{i / step}, float16.Fromfloat32(float32(i)))
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
			row := rowVal.(float16.Float16)
			for j := 0; j < vec2.Size[0]; j++ {
				col := vec2.GetItemByOffset(j * itemSize).(float16.Float16)
				val := float16.Fromfloat32(row.Float32() * col.Float32())
				result.SetItem([]int{i, j}, val)
			}
		default:
			return nil, fmt.Errorf("unsupported tensor datatype %s", result.DataType)
		}
	}
	return result, nil
}

func Full(size []int, fillValue any) *Tensor {
	result := NewEmptyTensor(size, DT_BF16)
	result.Apply(func(val any) any {
		return fillValue
	})
	return result
}

func Zeros(size []int) *Tensor {
	return Full(size, float16.Fromfloat32(0))
}

func Ones(size []int) *Tensor {
	return Full(size, float16.Fromfloat32(1))
}

func OnesLike(input *Tensor) *Tensor {
	return Ones(input.Size)
}

func Polar(abs *Tensor, angle *Tensor) (*Tensor, error) {
	// See: (For formula) https://pytorch.org/docs/stable/generated/torch.polar.html
	if err := processErrors(
		checkSameShape(abs, angle),
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
			absItemConv := float64(absItem.(float16.Float16).Float32())
			angleItemConv := float64(angleItem.(float16.Float16).Float32())
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

func TriangularUpper(input *Tensor, diagonal int) *Tensor {
	// See: https://pytorch.org/docs/stable/generated/torch.triu.html

	rowCount := input.Size[0]
	colCount := input.Size[1]

	dst := NewEmptyTensor(input.Size, input.DataType)
	for i := 0; i < rowCount; i++ {
		for j := i; j < colCount; j++ {
			if j-i >= diagonal {
				loc := []int{i, j}
				val, _ := input.GetItem(loc)
				dst.SetItem(loc, val)
			}
		}
	}

	return dst
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
			item := item.(float16.Float16)
			resultItem := float32(math.Pow(float64(item.Float32()), power))
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
				itemF32 = item.(float16.Float16).Float32()
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
			groupMean = float16.Fromfloat32(groupMeanF32)
		case DT_F32:
			groupMean = groupMeanF32
		}

		if err := dst.SetItemByOffset(dstOffset, groupMean); err != nil {
			return nil, err
		}
		dstOffset += itemSize
	}

	fmt.Printf("%s\n", dst.String())
	return dst, nil
}

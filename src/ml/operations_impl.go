package ml

import (
	"fmt"

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
		row := vec1.GetItemByOffset(i * itemSize)
		for j := 0; j < vec2.Size[0]; j++ {
			col := vec2.GetItemByOffset(j * itemSize)
			val := float16.Fromfloat32(row.Float32() * col.Float32())
			result.SetItem([]int{i, j}, val)
		}
	}
	return result, nil
}

func Full(size []int, fillValue float16.Float16) *Tensor {
	result := NewEmptyTensor(size, DT_BF16)
	result.Apply(func(val float16.Float16) float16.Float16 {
		return fillValue
	})
	return result
}

func Zeros(size []int) *Tensor {
	return Full(size, 0)
}

func Ones(size []int) *Tensor {
	return Full(size, 1)
}

func OnesLike(input *Tensor) *Tensor {
	return Ones(input.Size)
}

func Polar(abs *Tensor, angle *Tensor) *Tensor {
	return abs
}

func Fwd_Get_Rows(embedding *Tensor, tokens *Tensor) (*Tensor, error) {
	if err := processErrors(
		checkIsMatrix(embedding),
		checkIsVector(tokens),
		checkSameDataType(embedding, tokens),
	); err != nil {
		return nil, err
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
		row := int(uint16(rowVal))
		readOffsetStart := embedding.calculateByteOffset([]int{row, 0})
		readOffsetEnd := embedding.calculateByteOffset([]int{row + 1, 0})
		rowBytes := embedding.RawData[readOffsetStart:readOffsetEnd]
		writeOffsetStart := dst.calculateByteOffset([]int{rowIdx, 0})
		copy(dst.RawData[writeOffsetStart:], rowBytes)
	}
	return dst, nil
}

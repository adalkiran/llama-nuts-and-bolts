package ml

import (
	"fmt"
	"math"
	"reflect"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

func ARange(start int, end int, step int, dataType DataType) (*Tensor, error) {
	if start >= end {
		return nil, fmt.Errorf("start value %d must be less than end value %d in ARange", start, end)
	}
	result := NewEmptyTensor([]int{int(math.Ceil(float64(end-start) / float64(step)))}, dataType)
	i := 0
	for val := start; val < end; val += step {
		if err := result.SetItem_FromFloat32([]int{i}, float32(val)); err != nil {
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
	itemSize := vec1.DataType.ItemSize
	result := NewEmptyTensor([]int{vec1.Size[0], vec2.Size[0]}, vec1.DataType)
	for i := 0; i < vec1.Size[0]; i++ {
		rowValF32, err := vec1.GetItemByOffset_AsFloat32(i * itemSize)
		if err != nil {
			return nil, err
		}
		for j := 0; j < vec2.Size[0]; j++ {
			colValF32, err := vec2.GetItemByOffset_AsFloat32(j * itemSize)
			if err != nil {
				return nil, err
			}
			valF32 := rowValF32 * colValF32
			if err := result.SetItem_FromFloat32([]int{i, j}, valF32); err != nil {
				return nil, err
			}
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

	absItemSize := abs.DataType.ItemSize
	dstItemSize := dst.DataType.ItemSize

	writeOffset := 0
	for readOffset := 0; readOffset < abs.GetBytesCount(); readOffset += absItemSize {
		absItemF32, err := abs.GetItemByOffset_AsFloat32(readOffset)
		if err != nil {
			return nil, err
		}
		angleItemF32, err := angle.GetItemByOffset_AsFloat32(readOffset)
		if err != nil {
			return nil, err
		}

		absItemF64 := float64(absItemF32)
		angleItemF64 := float64(angleItemF32)

		realPart := absItemF64 * math.Cos(angleItemF64)
		imagPart := absItemF64 * math.Sin(angleItemF64)
		resultItem := complex64(complex(realPart, imagPart))
		if err := dst.SetItemByOffset(writeOffset, resultItem); err != nil {
			return nil, err
		}
		writeOffset += dstItemSize
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

	if tokens.DataType != DT_INT32 {
		return nil, fmt.Errorf("tensor is not in data type %s: \"%s\" is %s", DT_INT32, tokens.Name, tokens.DataType)
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
		row := int(rowVal.(int32))
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
	inputItemSize := input.DataType.ItemSize
	dstItemSize := dstDataType.ItemSize

	dst := NewEmptyTensor(input.Size, dstDataType)
	writeOffset := 0
	for readOffset := 0; readOffset < input.GetBytesCount(); readOffset += inputItemSize {
		itemF32, err := input.GetItemByOffset_AsFloat32(readOffset)
		if err != nil {
			return nil, err
		}
		dst.SetItemByOffset_FromFloat32(writeOffset, float32(math.Pow(float64(itemF32), power)))
		writeOffset += dstItemSize
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
	itemSize := input.DataType.ItemSize
	dst := NewEmptyTensor(dstSize, input.DataType)
	inputLastSize := input.Size[len(input.Size)-1]
	inputStride := inputLastSize * itemSize

	dstOffset := 0
	for readGroupOffset := 0; readGroupOffset < input.GetBytesCount(); readGroupOffset += inputStride {
		groupSum := float32(0)
		for groupItemIdx := 0; groupItemIdx < inputLastSize; groupItemIdx++ {
			itemF32, err := input.GetItemByOffset_AsFloat32(readGroupOffset + groupItemIdx*itemSize)
			if err != nil {
				return nil, err
			}
			groupSum += itemF32
		}
		groupMeanF32 := groupSum / float32(inputLastSize)
		if err := dst.SetItemByOffset_FromFloat32(dstOffset, groupMeanF32); err != nil {
			return nil, err
		}
		dstOffset += itemSize
	}
	return dst, nil
}

func AddScalar(input *Tensor, scalar any) (*Tensor, error) {
	if input.DataType.FuncSet == nil {
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}
	if !input.DataType.FuncSet.IsCompatible(scalar) {
		return nil, fmt.Errorf("expected scalar argument type is %s, got %v (%v)", input.DataType.Name, scalar, reflect.TypeOf(scalar))
	}
	scalarF32 := input.DataType.FuncSet.ToFloat32(scalar)

	dst := DuplicateTensor(input)
	if err := dst.Apply_AsFloat32(func(val float32) float32 {
		return val + scalarF32
	}); err != nil {
		return nil, err
	}
	return dst, nil
}

func DivToScalar(input *Tensor, scalar any) (*Tensor, error) {
	if input.DataType.FuncSet == nil {
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}
	if !input.DataType.FuncSet.IsCompatible(scalar) {
		return nil, fmt.Errorf("expected scalar argument type is %s, got %v (%v)", input.DataType.Name, scalar, reflect.TypeOf(scalar))
	}
	scalarF32 := input.DataType.FuncSet.ToFloat32(scalar)

	dst := DuplicateTensor(input)
	if err := dst.Apply_AsFloat32(func(val float32) float32 {
		return val / scalarF32
	}); err != nil {
		return nil, err
	}
	return dst, nil
}

func RSqrt(input *Tensor) (*Tensor, error) {
	// See: (For formula) https://pytorch.org/docs/stable/generated/torch.rsqrt.html
	// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
	if input.DataType.FuncSet == nil {
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}

	dst := DuplicateTensor(input)
	if err := dst.Apply_AsFloat32(func(val float32) float32 {
		return float32(float64(1) / math.Sqrt(float64(val)))
	}); err != nil {
		return nil, err
	}
	return dst, nil
}

func Add(input *Tensor, other *Tensor) (*Tensor, error) {
	refTensor, expandingTensor, err := CheckBroadcastable(input, other, true)
	if err != nil {
		return nil, err
	}
	dst := NewEmptyTensor(refTensor.Size, refTensor.DataType)
	if refTensor.DataType != DT_COMPLEX {
		for iterator := IterateOverTwo(refTensor, expandingTensor, 0); iterator.HasNext(); {
			loc1, loc2 := iterator.Next()
			if err != nil {
				return nil, err
			}

			val1F32, err := refTensor.GetItem_AsFloat32(loc1)
			if err != nil {
				return nil, err
			}

			val2F32, err := expandingTensor.GetItem_AsFloat32(loc2)
			if err != nil {
				return nil, err
			}

			resultValF32 := val1F32 + val2F32

			if err := dst.SetItem_FromFloat32(loc1, resultValF32); err != nil {
				return nil, err
			}
		}
	} else {
		if expandingTensor.DataType != DT_COMPLEX {
			return nil, fmt.Errorf("unsupported tensor datatypes %s and %s", refTensor.DataType, expandingTensor.DataType)
		}
		for iterator := IterateOverTwo(refTensor, expandingTensor, 0); iterator.HasNext(); {
			loc1, loc2 := iterator.Next()
			if err != nil {
				return nil, err
			}
			val1, err := refTensor.GetItem(loc1)
			if err != nil {
				return nil, err
			}
			val2, err := expandingTensor.GetItem(loc2)
			if err != nil {
				return nil, err
			}

			val1Complex64 := val1.(complex64)
			val2Complex64 := val2.(complex64)

			resultValComplex64 := val1Complex64 + val2Complex64

			if err := dst.SetItem(loc1, resultValComplex64); err != nil {
				return nil, err
			}
		}
	}
	return dst, nil
}

func MultiplyElementwise(input *Tensor, other *Tensor) (*Tensor, error) {
	refTensor, expandingTensor, err := CheckBroadcastable(input, other, true)
	if err != nil {
		return nil, err
	}
	dst := NewEmptyTensor(refTensor.Size, refTensor.DataType)
	if refTensor.DataType != DT_COMPLEX {
		for iterator := IterateOverTwo(refTensor, expandingTensor, 0); iterator.HasNext(); {
			loc1, loc2 := iterator.Next()
			if err != nil {
				return nil, err
			}

			val1F32, err := refTensor.GetItem_AsFloat32(loc1)
			if err != nil {
				return nil, err
			}

			val2F32, err := expandingTensor.GetItem_AsFloat32(loc2)
			if err != nil {
				return nil, err
			}

			resultValF32 := val1F32 * val2F32

			if err := dst.SetItem_FromFloat32(loc1, resultValF32); err != nil {
				return nil, err
			}
		}
	} else {
		if expandingTensor.DataType != DT_COMPLEX {
			return nil, fmt.Errorf("unsupported tensor datatypes %s and %s", refTensor.DataType, expandingTensor.DataType)
		}
		for iterator := IterateOverTwo(refTensor, expandingTensor, 0); iterator.HasNext(); {
			loc1, loc2 := iterator.Next()
			if err != nil {
				return nil, err
			}
			val1, err := refTensor.GetItem(loc1)
			if err != nil {
				return nil, err
			}
			val2, err := expandingTensor.GetItem(loc2)
			if err != nil {
				return nil, err
			}

			val1Complex64 := val1.(complex64)
			val2Complex64 := val2.(complex64)

			resultValComplex64 := val1Complex64 * val2Complex64

			if err := dst.SetItem(loc1, resultValComplex64); err != nil {
				return nil, err
			}
		}
	}
	return dst, nil
}

func LinearTransformation(input *Tensor, weights *Tensor) (*Tensor, error) {
	if err := checkSameDataType(input, weights); err != nil {
		return nil, err
	}
	colsSize := input.Size[1]
	// Linear unit weights size: [out_features, in_features]
	weightsInputSize := weights.Size[1]

	if colsSize != weightsInputSize {
		return nil, fmt.Errorf("columns size %d of input tensor (%v) should be equal with %d input features count of weights tensor (%v)", colsSize, input.Size, weightsInputSize, weights.Size)
	}

	switch input.DataType {
	case DT_BF16:
		return linearTransformation_BF16(input, weights)
	case DT_F32:
		return linearTransformation_F32(input, weights)
	default:
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}
}

func MatMul(input *Tensor, other *Tensor) (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.matmul.html
	if err := checkSameDataType(input, other); err != nil {
		return nil, err
	}

	inputColsSize := input.Size[len(input.Size)-1]

	otherRowsSize := other.Size[len(other.Size)-2]

	if inputColsSize != otherRowsSize {
		return nil, fmt.Errorf("columns size %d of input tensor (%v) should be equal with rows size %d of other tensor (%v)", inputColsSize, input.Size, otherRowsSize, other.Size)
	}

	inputSizeFirstPart := input.Size[0 : len(input.Size)-2]
	othertSizeFirstPart := other.Size[0 : len(other.Size)-2]

	if !reflect.DeepEqual(inputSizeFirstPart, othertSizeFirstPart) {
		return nil, fmt.Errorf("first parts of dimensions are not compatible;  %v of input tensor (%v) and %v of other tensor (%v)", inputSizeFirstPart, input.Size, othertSizeFirstPart, other.Size)
	}

	switch input.DataType {
	case DT_BF16:
		return matMul_BF16(input, other)
	default:
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}
}

func Softmax(input *Tensor, dim int) (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
	if dim != len(input.Size)-1 {
		return nil, fmt.Errorf("currenlty Softmax supports only last dimension of input tensor as dim argument")
	}
	dst := NewEmptyTensorLike(input, true)
	inputSizeFirstPart := input.Size[0 : len(input.Size)-1]
	inputItemSize := input.DataType.ItemSize
	blockSize := input.Size[dim] * inputItemSize
	for iteratorFirstPart := IterateOverSize(inputSizeFirstPart, 0); iteratorFirstPart.HasNext(); {
		locFirstPart := append(iteratorFirstPart.Next(), 0)
		startOffset := input.calculateByteOffset(locFirstPart)
		endOffset := startOffset + blockSize

		rowExpSum := float64(0)
		for offset := startOffset; offset < endOffset; offset += inputItemSize {
			itemF32, err := input.GetItemByOffset_AsFloat32(offset)
			if err != nil {
				return nil, err
			}
			rowExpSum += math.Exp(float64(itemF32))
		}

		for offset := startOffset; offset < endOffset; offset += inputItemSize {
			itemF32, err := input.GetItemByOffset_AsFloat32(offset)
			if err != nil {
				return nil, err
			}
			dstValF32 := float32(math.Exp(float64(itemF32)) / rowExpSum)
			dst.SetItemByOffset_FromFloat32(offset, dstValF32)
		}
	}
	return dst, nil
}

func Argmax(input *Tensor, dim int) (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.argmax.html
	if dim != len(input.Size)-1 {
		return nil, fmt.Errorf("currenlty Argmax supports only last dimension of input tensor as dim argument")
	}
	dstSize := make([]int, len(input.Size)-1)
	copy(dstSize, input.Size[0:len(input.Size)-1])
	dst := NewEmptyTensor(dstSize, DT_INT32)
	inputItemSize := input.DataType.ItemSize
	blockSize := input.Size[dim] * inputItemSize
	for iteratorDst := IterateOver(dst, 0); iteratorDst.HasNext(); {
		locDst := iteratorDst.Next()
		locFirstPart := append(locDst, 0)
		readOffsetStart := input.calculateByteOffset(locFirstPart)
		readOffsetEnd := readOffsetStart + blockSize

		maxValue := float32(-math.MaxFloat32)
		maxIdx := -1
		idx := -1
		for offset := readOffsetStart; offset < readOffsetEnd; offset += inputItemSize {
			idx++
			itemF32, err := input.GetItemByOffset_AsFloat32(offset)
			if err != nil {
				return nil, err
			}
			if maxValue < itemF32 {
				maxValue = itemF32
				maxIdx = idx
			}
		}
		if err := dst.SetItem(locDst, int32(maxIdx)); err != nil {
			return nil, err
		}
	}
	return dst, nil
}

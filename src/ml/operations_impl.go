package ml

import (
	"fmt"
	"math"
	"reflect"
	"time"

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
	inputItemSize := input.DataType.ItemSize()

	dst := NewEmptyTensor(input.Size, dstDataType)
	writeOffset := 0
	for readOffset := 0; readOffset < input.GetBytesCount(); readOffset += inputItemSize {
		item := input.GetItemByOffset(readOffset)
		switch input.DataType {
		case DT_BF16:
			item := item.(dtype.BFloat16)
			resultItem := float32(math.Pow(item.Float64(), power))
			if err := dst.SetItemByOffset(writeOffset, resultItem); err != nil {
				return nil, err
			}
		case DT_F32:
			item := item.(float32)
			resultItem := float32(math.Pow(float64(item), power))
			if err := dst.SetItemByOffset(writeOffset, resultItem); err != nil {
				return nil, err
			}
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

func AddScalar(input *Tensor, scalar any) (*Tensor, error) {
	switch input.DataType {
	case DT_BF16:
		if _, ok := scalar.(dtype.BFloat16); !ok {
			return nil, fmt.Errorf("expected scalar argument type is %s, got %v (%v)", "BFloat16", scalar, reflect.TypeOf(scalar))
		}
	case DT_F32:
		if _, ok := scalar.(float32); !ok {
			return nil, fmt.Errorf("expected scalar argument type is %s, got %v (%v)", "float32", scalar, reflect.TypeOf(scalar))
		}
	default:
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}
	dst := DuplicateTensor(input)
	if err := dst.Apply(func(val any) any {
		switch scalar := scalar.(type) {
		case dtype.BFloat16:
			val := val.(dtype.BFloat16)
			return val + scalar
		case float32:
			val := val.(float32)
			return val + scalar
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return dst, nil
}

func DivToScalar(input *Tensor, scalar any) (*Tensor, error) {
	switch input.DataType {
	case DT_BF16:
		if _, ok := scalar.(dtype.BFloat16); !ok {
			if _, ok := scalar.(float32); !ok {
				return nil, fmt.Errorf("expected scalar argument type is %s, got %v (%v)", "BFloat16 or float32", scalar, reflect.TypeOf(scalar))
			}
		}
	case DT_F32:
		if _, ok := scalar.(float32); !ok {
			return nil, fmt.Errorf("expected scalar argument type is %s, got %v (%v)", "float32", scalar, reflect.TypeOf(scalar))
		}
	default:
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}
	dst := DuplicateTensor(input)
	var scalarF32 float32
	switch scalar := scalar.(type) {
	case dtype.BFloat16:
		scalarF32 = scalar.Float32()
	case float32:
		scalarF32 = scalar
	default:
		return nil, fmt.Errorf("incompatible tensor datatype %s and scalar type %v", input.DataType, reflect.TypeOf(scalar))
	}
	if err := dst.Apply(func(val any) any {
		switch val := val.(type) {
		case dtype.BFloat16:
			valF32 := val.Float32()
			return dtype.BFloat16fromFloat32(valF32 / scalarF32)
		case float32:
			return val / scalarF32
		}
		return nil
	}); err != nil {
		return nil, err
	}
	return dst, nil
}

func RSqrt(input *Tensor) (*Tensor, error) {
	// See: (For formula) https://pytorch.org/docs/stable/generated/torch.rsqrt.html
	// Returns a new tensor with the reciprocal of the square-root of each of the elements of input.
	switch input.DataType {
	case DT_BF16: // Do nothing
	case DT_F32: // Do nothing
	default:
		return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
	}

	dst := DuplicateTensor(input)
	if err := dst.Apply(func(val any) any {
		switch val := val.(type) {
		case dtype.BFloat16:
			return dtype.BFloat16fromFloat32(float32(float64(1) / math.Sqrt(val.Float64())))
		case float32:
			return float32(float64(1) / math.Sqrt(float64(val)))
		}
		return nil
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
			val1, err := refTensor.GetItem(loc1)
			if err != nil {
				return nil, err
			}
			val2, err := expandingTensor.GetItem(loc2)
			if err != nil {
				return nil, err
			}

			var val1F32 float32
			var val2F32 float32

			switch refTensor.DataType {
			case DT_BF16:
				val1F32 = float32(val1.(dtype.BFloat16).Float32())
			case DT_F32:
				val1F32 = val1.(float32)
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", refTensor.DataType)
			}

			switch expandingTensor.DataType {
			case DT_BF16:
				val2F32 = float32(val2.(dtype.BFloat16).Float32())
			case DT_F32:
				val2F32 = val2.(float32)
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", expandingTensor.DataType)
			}
			resultValF32 := val1F32 + val2F32

			var resultVal any
			switch dst.DataType {
			case DT_BF16:
				resultVal = dtype.BFloat16fromFloat32(resultValF32)
			case DT_F32:
				resultVal = resultValF32
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", dst.DataType)
			}

			if err := dst.SetItem(loc1, resultVal); err != nil {
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
			val1, err := refTensor.GetItem(loc1)
			if err != nil {
				return nil, err
			}
			val2, err := expandingTensor.GetItem(loc2)
			if err != nil {
				return nil, err
			}

			var val1F32 float32
			var val2F32 float32

			switch refTensor.DataType {
			case DT_BF16:
				val1F32 = float32(val1.(dtype.BFloat16).Float32())
			case DT_F32:
				val1F32 = val1.(float32)
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", refTensor.DataType)
			}

			switch expandingTensor.DataType {
			case DT_BF16:
				val2F32 = float32(val2.(dtype.BFloat16).Float32())
			case DT_F32:
				val2F32 = val2.(float32)
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", expandingTensor.DataType)
			}
			resultValF32 := val1F32 * val2F32

			var resultVal any
			switch dst.DataType {
			case DT_BF16:
				resultVal = dtype.BFloat16fromFloat32(resultValF32)
			case DT_F32:
				resultVal = resultValF32
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", dst.DataType)
			}

			if err := dst.SetItem(loc1, resultVal); err != nil {
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

func linearTransformation_BF16(input *Tensor, weights *Tensor) (*Tensor, error) {
	rowsSize := input.Size[0]

	// Linear unit weights size: [out_features, in_features]
	weightsOutputSize := weights.Size[0]
	weightsInputSize := weights.Size[1]

	dstF32 := NewEmptyTensor([]int{rowsSize, weightsOutputSize}, DT_F32)

	inputItemSize := input.DataType.ItemSize()
	dstItemSize := dstF32.DataType.ItemSize()

	for rowIdx := 0; rowIdx < rowsSize; rowIdx++ {
		inputRowOffset := input.calculateByteOffset([]int{rowIdx, 0})
		dstRowOffset := dstF32.calculateByteOffset([]int{rowIdx, 0})
		for wOutIdx := 0; wOutIdx < weightsOutputSize; wOutIdx++ {
			weightsWOutOffset := weights.calculateByteOffset([]int{wOutIdx, 0})

			// location: {rowIdx, wOutIdx}
			dstItemOffset := dstRowOffset + wOutIdx*dstItemSize
			valDstF32 := dstF32.GetItemByOffset_F32(dstItemOffset)

			for wInIdx := 0; wInIdx < weightsInputSize; wInIdx++ {
				// Goal in Python manner: dst[rowIdx][wOutIdx] += input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]

				// Getting input[rowIdx][wInIdx]
				// location: {rowIdx, wInIdx}
				val1F32 := input.GetItemByOffset_BF16(inputRowOffset + wInIdx*inputItemSize).Float32()

				// Getting weights[wOutIdx][wInIdx]
				// location: {wOutIdx, wInIdx}
				val2F32 := weights.GetItemByOffset_BF16(weightsWOutOffset + wInIdx*inputItemSize).Float32()

				//Calculating: input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]
				multiplicationValF32 := val1F32 * val2F32

				//Calculating:  dst[rowIdx][wOutIdx] += multiplicationValF32
				valDstF32 += multiplicationValF32
			}

			// location: {rowIdx, wOutIdx}
			if err := dstF32.SetItemByOffset_F32(dstItemOffset, valDstF32); err != nil {
				return nil, err
			}
		}
	}
	return dstF32.ToBFloat16()
}

func linearTransformation_F32(input *Tensor, weights *Tensor) (*Tensor, error) {
	rowsSize := input.Size[0]

	// Linear unit weights size: [out_features, in_features]
	weightsOutputSize := weights.Size[0]
	weightsInputSize := weights.Size[1]

	dstF32 := NewEmptyTensor([]int{rowsSize, weightsOutputSize}, DT_F32)

	inputItemSize := input.DataType.ItemSize()
	dstItemSize := dstF32.DataType.ItemSize()

	for rowIdx := 0; rowIdx < rowsSize; rowIdx++ {
		inputRowOffset := input.calculateByteOffset([]int{rowIdx, 0})
		dstRowOffset := dstF32.calculateByteOffset([]int{rowIdx, 0})
		for wOutIdx := 0; wOutIdx < weightsOutputSize; wOutIdx++ {
			weightsWOutOffset := weights.calculateByteOffset([]int{wOutIdx, 0})

			// location: {rowIdx, wOutIdx}
			valDstF32 := dstF32.GetItemByOffset_F32(dstRowOffset + wOutIdx*dstItemSize)

			for wInIdx := 0; wInIdx < weightsInputSize; wInIdx++ {
				// Goal in Python manner: dst[rowIdx][wOutIdx] += input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]

				// Getting input[rowIdx][wInIdx], location: {rowIdx, wInIdx}
				val1F32 := input.GetItemByOffset_F32(inputRowOffset + wInIdx*inputItemSize)
				// Getting weights[wOutIdx][wInIdx], location: {wOutIdx, wInIdx}
				val2F32 := weights.GetItemByOffset_F32(weightsWOutOffset + wInIdx*inputItemSize)
				//Calculating: input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]
				multiplicationValF32 := val1F32 * val2F32
				//Calculating:  dst[rowIdx][wOutIdx] += multiplicationValF32
				valDstF32 += multiplicationValF32
			}

			// location: {rowIdx, wOutIdx}
			if err := dstF32.SetItemByOffset_F32(dstRowOffset+wOutIdx*dstItemSize, valDstF32); err != nil {
				return nil, err
			}
		}
	}
	return dstF32, nil
}

func LinearTransformation(input *Tensor, weights *Tensor) (*Tensor, error) {
	if err := checkSameDataType(input, weights); err != nil {
		return nil, err
	}
	startTime := time.Now()

	defer func() {
		endTime := time.Now()
		duration := endTime.Sub(startTime)
		fmt.Printf("Function LinearTransformation with %v and %v took %v\n", input.Size, weights.Size, duration)
	}()

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

func matMul_BF16(input *Tensor, other *Tensor) (*Tensor, error) {
	inputRowsSize := input.Size[len(input.Size)-2]
	inputColsSize := input.Size[len(input.Size)-1]

	otherColsSize := other.Size[len(other.Size)-1]

	inputSizeFirstPart := input.Size[0 : len(input.Size)-2]
	dstF32 := NewEmptyTensor(append(append([]int{}, inputSizeFirstPart...), inputRowsSize, otherColsSize), DT_F32)

	inputItemSize := input.DataType.ItemSize()
	dstItemSize := dstF32.DataType.ItemSize()

	for iteratorFirstPart := IterateOverSize(inputSizeFirstPart, 0); iteratorFirstPart.HasNext(); {
		locFirstPart := iteratorFirstPart.Next()
		for rowIdx := 0; rowIdx < inputRowsSize; rowIdx++ {
			// location: {locFirstPart, rowIdx, 0}
			rowLocStart := append(append([]int{}, locFirstPart...), []int{rowIdx, 0}...)
			inputRowOffset := input.calculateByteOffset(rowLocStart)
			dstRowOffset := dstF32.calculateByteOffset(rowLocStart)

			for otherColIdx := 0; otherColIdx < otherColsSize; otherColIdx++ {
				// location: {locFirstPart, rowIdx, otherColIdx}
				dstItemOffset := dstRowOffset + otherColIdx*dstItemSize
				valDstF32 := dstF32.GetItemByOffset_F32(dstItemOffset)

				for inputColIdx := 0; inputColIdx < inputColsSize; inputColIdx++ {
					// Goal in Python manner: dst[rowIdx][otherColIdx] += input[rowIdx][inputColIdx] * other[inputColIdx][otherColIdx]

					// location: {locFirstPart, inputColIdx, 0}
					otherInputColOffset := other.calculateByteOffset(append(append([]int{}, locFirstPart...), []int{inputColIdx, 0}...))

					// Getting input[rowIdx][inputColIdx], location: {locFirstPart, rowIdx, inputColIdx}
					val1F32 := input.GetItemByOffset_BF16(inputRowOffset + inputColIdx*inputItemSize).Float32()

					// Getting other[inputColIdx][otherColIdx], location: {locFirstPart, inputColIdx, otherColIdx}
					val2F32 := other.GetItemByOffset_BF16(otherInputColOffset + otherColIdx*inputItemSize).Float32()

					//Calculating: input[rowIdx][inputColIdx] * other[inputColIdx][otherColIdx]
					multiplicationValF32 := val1F32 * val2F32

					//Calculating:  dst[rowIdx][otherColIdx] += multiplicationValF32
					valDstF32 += multiplicationValF32
				}

				// location: {locFirstPart, rowIdx, otherColIdx}
				if err := dstF32.SetItemByOffset_F32(dstItemOffset, valDstF32); err != nil {
					return nil, err
				}
			}
		}
	}
	return dstF32.ToBFloat16()
}

func matMul_F32(input *Tensor, other *Tensor) (*Tensor, error) {
	inputRowsSize := input.Size[len(input.Size)-2]
	inputColsSize := input.Size[len(input.Size)-1]

	otherColsSize := other.Size[len(other.Size)-1]

	inputSizeFirstPart := input.Size[0 : len(input.Size)-2]
	dstF32 := NewEmptyTensor(append(append([]int{}, inputSizeFirstPart...), inputRowsSize, otherColsSize), DT_F32)

	inputItemSize := input.DataType.ItemSize()
	dstItemSize := dstF32.DataType.ItemSize()

	for iteratorFirstPart := IterateOverSize(inputSizeFirstPart, 0); iteratorFirstPart.HasNext(); {
		locFirstPart := iteratorFirstPart.Next()
		for rowIdx := 0; rowIdx < inputRowsSize; rowIdx++ {
			// location: {locFirstPart, rowIdx, 0}
			rowLocStart := append(append([]int{}, locFirstPart...), []int{rowIdx, 0}...)
			inputRowOffset := input.calculateByteOffset(rowLocStart)
			dstRowOffset := dstF32.calculateByteOffset(rowLocStart)

			for otherColIdx := 0; otherColIdx < otherColsSize; otherColIdx++ {
				// location: {locFirstPart, rowIdx, otherColIdx}
				dstItemOffset := dstRowOffset + otherColIdx*dstItemSize
				valDstF32 := dstF32.GetItemByOffset_F32(dstItemOffset)

				for inputColIdx := 0; inputColIdx < inputColsSize; inputColIdx++ {
					// Goal in Python manner: dst[rowIdx][otherColIdx] += input[rowIdx][inputColIdx] * other[inputColIdx][otherColIdx]

					// location: {locFirstPart, inputColIdx, 0}
					inputColOffset := input.calculateByteOffset(append(append([]int{}, locFirstPart...), []int{inputColIdx, 0}...))

					// Getting input[rowIdx][inputColIdx], location: {locFirstPart, rowIdx, inputColIdx}
					val1F32 := input.GetItemByOffset_F32(inputRowOffset + inputColIdx*inputItemSize)

					// Getting other[inputColIdx][otherColIdx], location: {locFirstPart, inputColIdx, otherColIdx}
					val2F32 := other.GetItemByOffset_F32(inputColOffset + inputColIdx*inputItemSize)

					//Calculating: input[rowIdx][inputColIdx] * other[inputColIdx][otherColIdx]
					multiplicationValF32 := val1F32 * val2F32

					//Calculating:  dst[rowIdx][otherColIdx] += multiplicationValF32
					valDstF32 += multiplicationValF32
				}

				// location: {locFirstPart, rowIdx, otherColIdx}
				if err := dstF32.SetItemByOffset_F32(dstItemOffset, valDstF32); err != nil {
					return nil, err
				}
			}
		}
	}
	return dstF32, nil
}

func MatMul(input *Tensor, other *Tensor) (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.matmul.html
	if err := checkSameDataType(input, other); err != nil {
		return nil, err
	}
	startTime := time.Now()

	defer func() {
		endTime := time.Now()
		duration := endTime.Sub(startTime)
		fmt.Printf("Function MatMul with %v and %v took %v\n", input.Size, other.Size, duration)
	}()

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
	case DT_F32:
		return matMul_F32(input, other)
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
	inputItemSize := input.DataType.ItemSize()
	blockSize := input.Size[dim] * inputItemSize
	for iteratorFirstPart := IterateOverSize(inputSizeFirstPart, 0); iteratorFirstPart.HasNext(); {
		locFirstPart := append(iteratorFirstPart.Next(), 0)
		startOffset := input.calculateByteOffset(locFirstPart)
		endOffset := startOffset + blockSize

		rowExpSum := float64(0)
		for offset := startOffset; offset < endOffset; offset += inputItemSize {
			item := input.GetItemByOffset(offset)
			switch item := item.(type) {
			case dtype.BFloat16:
				rowExpSum += math.Exp(item.Float64())
			case float32:
				rowExpSum += math.Exp(float64(item))
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
			}
		}

		for offset := startOffset; offset < endOffset; offset += inputItemSize {
			item := input.GetItemByOffset(offset)
			switch item := item.(type) {
			case dtype.BFloat16:
				dstVal := dtype.BFloat16fromFloat32(float32(math.Exp(item.Float64()) / rowExpSum))
				dst.SetItemByOffset(offset, dstVal)
			case float32:
				dstVal := float32(math.Exp(float64(item)) / rowExpSum)
				dst.SetItemByOffset(offset, dstVal)
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
			}
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
	inputItemSize := input.DataType.ItemSize()
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
			item := input.GetItemByOffset(offset)
			var itemF32 float32
			switch item := item.(type) {
			case dtype.BFloat16:
				itemF32 = item.Float32()
			case float32:
				itemF32 = item
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
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

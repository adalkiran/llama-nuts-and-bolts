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

func LinearTransformation(input *Tensor, weights *Tensor) (*Tensor, error) {
	rowsSize := input.Size[0]
	colsSize := input.Size[1]

	// Linear unit weights size: [out_features, in_features]
	weightsOutputSize := weights.Size[0]
	weightsInputSize := weights.Size[1]

	if colsSize != weightsInputSize {
		return nil, fmt.Errorf("columns size %d of input tensor (%v) should be equal with %d input features count of weights tensor (%v)", colsSize, input.Size, weightsInputSize, weights.Size)
	}

	dst := NewEmptyTensor([]int{rowsSize, weightsOutputSize}, input.DataType)
	for rowIdx := 0; rowIdx < rowsSize; rowIdx++ {
		for wOutIdx := 0; wOutIdx < weightsOutputSize; wOutIdx++ {
			var valDstF32 float32

			valDst, err := dst.GetItem([]int{rowIdx, wOutIdx})
			if err != nil {
				return nil, err
			}

			switch dst.DataType {
			case DT_BF16:
				valDstF32 = float32(valDst.(dtype.BFloat16).Float32())
			case DT_F32:
				valDstF32 = valDst.(float32)
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", dst.DataType)
			}

			for wInIdx := 0; wInIdx < weightsInputSize; wInIdx++ {
				// Goal in Python manner: dst[rowIdx][wOutIdx] += input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]

				// Getting input[rowIdx][wInIdx]
				val1, err := input.GetItem([]int{rowIdx, wInIdx})
				if err != nil {
					return nil, err
				}
				// Getting weights[wOutIdx][wInIdx]
				val2, err := weights.GetItem([]int{wOutIdx, wInIdx})
				if err != nil {
					return nil, err
				}

				var val1F32 float32
				var val2F32 float32

				switch input.DataType {
				case DT_BF16:
					val1F32 = float32(val1.(dtype.BFloat16).Float32())
				case DT_F32:
					val1F32 = val1.(float32)
				default:
					return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
				}

				switch weights.DataType {
				case DT_BF16:
					val2F32 = float32(val2.(dtype.BFloat16).Float32())
				case DT_F32:
					val2F32 = val2.(float32)
				default:
					return nil, fmt.Errorf("unsupported tensor datatype %s", weights.DataType)
				}

				//Calculating: input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]
				multiplicationValF32 := val1F32 * val2F32

				//Calculating:  dst[rowIdx][wOutIdx] += multiplicationValF32
				valDstF32 += multiplicationValF32
			}

			switch dst.DataType {
			case DT_BF16:
				valDst = dtype.BFloat16fromFloat32(valDstF32)
			case DT_F32:
				valDst = valDstF32
			default:
				return nil, fmt.Errorf("unsupported tensor datatype %s", dst.DataType)
			}

			if err := dst.SetItem([]int{rowIdx, wOutIdx}, valDst); err != nil {
				return nil, err
			}
		}
	}

	return dst, nil
}

func MatMul(input *Tensor, other *Tensor) (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.matmul.html
	inputRowsSize := input.Size[len(input.Size)-2]
	inputColsSize := input.Size[len(input.Size)-1]

	otherRowsSize := other.Size[len(other.Size)-2]
	otherColsSize := other.Size[len(other.Size)-1]

	if inputColsSize != otherRowsSize {
		return nil, fmt.Errorf("columns size %d of input tensor (%v) should be equal with rows size %d of other tensor (%v)", inputColsSize, input.Size, otherRowsSize, other.Size)
	}

	inputSizeFirstPart := input.Size[0 : len(input.Size)-2]
	othertSizeFirstPart := other.Size[0 : len(other.Size)-2]

	if !reflect.DeepEqual(inputSizeFirstPart, othertSizeFirstPart) {
		return nil, fmt.Errorf("first parts of dimensions are not compatible;  %v of input tensor (%v) and %v of other tensor (%v)", inputSizeFirstPart, input.Size, othertSizeFirstPart, other.Size)
	}

	dst := NewEmptyTensor(append(append([]int{}, inputSizeFirstPart...), inputRowsSize, otherColsSize), input.DataType)

	for iteratorFirstPart := IterateOverSize(inputSizeFirstPart, 0); iteratorFirstPart.HasNext(); {
		locFirstPart := iteratorFirstPart.Next()
		for inputRowIdx := 0; inputRowIdx < inputRowsSize; inputRowIdx++ {
			for otherColIdx := 0; otherColIdx < otherColsSize; otherColIdx++ {
				var valDstF32 float32
				valDst, err := dst.GetItem(append(append([]int{}, locFirstPart...), []int{inputRowIdx, otherColIdx}...))
				if err != nil {
					return nil, err
				}

				switch dst.DataType {
				case DT_BF16:
					valDstF32 = float32(valDst.(dtype.BFloat16).Float32())
				case DT_F32:
					valDstF32 = valDst.(float32)
				default:
					return nil, fmt.Errorf("unsupported tensor datatype %s", dst.DataType)
				}

				for inputColIdx := 0; inputColIdx < inputColsSize; inputColIdx++ {
					// Goal in Python manner: dst[inputRowIdx][otherColIdx] += input[inputRowIdx][inputColIdx] * other[inputColIdx][otherColIdx]

					// Getting input[inputRowIdx][inputColIdx]
					val1, err := input.GetItem(append(append([]int{}, locFirstPart...), []int{inputRowIdx, inputColIdx}...))
					if err != nil {
						return nil, err
					}
					// Getting other[inputColIdx][otherColIdx]
					val2, err := other.GetItem(append(append([]int{}, locFirstPart...), []int{inputColIdx, otherColIdx}...))
					if err != nil {
						return nil, err
					}

					var val1F32 float32
					var val2F32 float32

					switch input.DataType {
					case DT_BF16:
						val1F32 = float32(val1.(dtype.BFloat16).Float32())
					case DT_F32:
						val1F32 = val1.(float32)
					default:
						return nil, fmt.Errorf("unsupported tensor datatype %s", input.DataType)
					}

					switch other.DataType {
					case DT_BF16:
						val2F32 = float32(val2.(dtype.BFloat16).Float32())
					case DT_F32:
						val2F32 = val2.(float32)
					default:
						return nil, fmt.Errorf("unsupported tensor datatype %s", other.DataType)
					}

					//Calculating: input[inputRowIdx][inputColIdx] * other[inputColIdx][otherColIdx]
					multiplicationValF32 := val1F32 * val2F32

					//Calculating:  dst[inputRowIdx][otherColIdx] += multiplicationValF32
					valDstF32 += multiplicationValF32
				}

				switch dst.DataType {
				case DT_BF16:
					valDst = dtype.BFloat16fromFloat32(valDstF32)
				case DT_F32:
					valDst = valDstF32
				default:
					return nil, fmt.Errorf("unsupported tensor datatype %s", dst.DataType)
				}

				if err := dst.SetItem(append(append([]int{}, locFirstPart...), []int{inputRowIdx, otherColIdx}...), valDst); err != nil {
					return nil, err
				}
			}
		}
	}
	return dst, nil
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

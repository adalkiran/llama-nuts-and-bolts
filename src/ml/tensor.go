package ml

import (
	"fmt"
	"strings"
	"unsafe"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

type Tensor struct {
	Name     string
	Size     []int
	Stride   []int
	DataType DataType
	RawData  []byte

	ByteStride []int
}

func NewTensor(name string, size []int, stride []int, dataType DataType, RawData []byte) *Tensor {
	return &Tensor{
		Name:       name,
		Size:       size,
		Stride:     stride,
		DataType:   dataType,
		RawData:    RawData,
		ByteStride: calculateByteStride(size, dataType),
	}
}

func NewEmptyTensorEx(name string, size []int, dataType DataType, allocateRawData bool) *Tensor {
	result := &Tensor{
		Name:       name,
		Size:       size,
		DataType:   dataType,
		ByteStride: calculateByteStride(size, dataType),
	}
	if allocateRawData {
		result.RawData = make([]byte, result.GetBytesCount())
	}
	return result
}

func NewEmptyTensor(size []int, dataType DataType) *Tensor {
	return NewEmptyTensorEx("", size, dataType, true)
}

func NewEmptyTensorLike(input *Tensor, allocateRawData bool) *Tensor {
	return NewEmptyTensorEx(input.Name, input.Size, input.DataType, allocateRawData)
}

func DuplicateTensor(input *Tensor) *Tensor {
	dst := NewEmptyTensorLike(input, true)
	copy(dst.RawData, input.RawData)
	return dst
}

func (t *Tensor) GetElementCount() int {
	result := 1
	for _, shapeItem := range t.Size {
		result = result * shapeItem
	}
	return result
}

func (t *Tensor) GetBytesCount() int {
	return t.GetElementCount() * t.DataType.ItemSize
}

func (t *Tensor) String() string {
	shorten := true
	if len(t.Size) == 1 {
		shorten = false
	}
	return fmt.Sprintf("[Tensor \"%s\"](%s, shape=%v, dtype=%s)", t.Name, t.dimensionToString([]int{}, shorten), t.Size, t.DataType.Name)
}

func (t *Tensor) StringLong() string {
	return fmt.Sprintf("[Tensor \"%s\"](%s, shape=%v, dtype=%s)", t.Name, t.dimensionToString([]int{}, false), t.Size, t.DataType.Name)
}

func (t *Tensor) dimensionToString(loc []int, shorten bool) string {
	currentDimension := len(loc)
	if currentDimension < len(t.Size) {
		currentDimLength := t.Size[currentDimension]
		var sb strings.Builder
		sb.WriteString("[")
		lineLen := 0
		if currentDimLength < 6 || !shorten {
			for i := 0; i < currentDimLength; i++ {
				if i > 0 {
					sb.WriteString(", ")
				}
				dimStr := t.dimensionToString(append(loc, i), shorten)
				if sb.Len()+len(dimStr)-lineLen >= 80 {
					sb.WriteString("\n")
					lineLen = sb.Len()
				}
				sb.WriteString(dimStr)
			}
		} else {
			for i := 0; i < 3; i++ {
				if i > 0 {
					sb.WriteString(", ")
				}
				dimStr := t.dimensionToString(append(loc, i), shorten)
				if sb.Len()+len(dimStr)-lineLen >= 80 {
					sb.WriteString("\n")
					lineLen = sb.Len()
				}
				sb.WriteString(dimStr)
			}
			if currentDimension < 1 {
				sb.WriteString(",\n...,\n")
				lineLen = sb.Len()
			} else {
				sb.WriteString(", ..., ")
			}
			for i := currentDimLength - 3; i < currentDimLength; i++ {
				if i > currentDimLength-3 {
					sb.WriteString(", ")
				}
				dimStr := t.dimensionToString(append(loc, i), shorten)
				if i > currentDimLength-3 && sb.Len()+len(dimStr)-lineLen >= 80 {
					sb.WriteString("\n")
					lineLen = sb.Len()
				}
				sb.WriteString(dimStr)
			}
		}
		sb.WriteString("]")
		return sb.String()
	}
	item, err := t.GetItem(loc)
	if err != nil {
		return "err"
	}
	if t.DataType.FuncSet == nil {
		return fmt.Sprintf("%v", item)
	} else {
		return t.DataType.FuncSet.ToString(item)
	}
}

func (t *Tensor) IsVector() bool {
	return len(t.Size) == 1
}

func (t *Tensor) IsMatrix() bool {
	return len(t.Size) == 2
}

func (t *Tensor) GetItem(loc []int) (any, error) {
	if len(t.Size) == 0 {
		// A check existing in Pytorch
		return 0, fmt.Errorf("invalid index of a 0-dim tensor. Use `tensor.item()`")
	}
	if len(loc) != len(t.Size) {
		return 0, fmt.Errorf("dimensions are not compatible: tensor is %dD, loc is %dD", len(t.Size), len(loc))
	}
	offset := t.calculateByteOffset(loc)
	return t.GetItemByOffset(offset), nil
}

func (t *Tensor) SetItem(loc []int, val any) error {
	if len(loc) != len(t.Size) {
		return fmt.Errorf("dimensions are not compatible: tensor is %dD, loc is %dD", len(t.Size), len(loc))
	}
	offset := t.calculateByteOffset(loc)
	return t.SetItemByOffset(offset, val)
}

func (t *Tensor) GetItemByOffset(offset int) any {
	if t.DataType.FuncSet == nil {
		return fmt.Errorf("unsupported tensor datatype %s", t.DataType)
	}
	return t.DataType.FuncSet.ReadItem(unsafe.Pointer(&t.RawData[offset]))
}

func (t *Tensor) GetItemByOffset_AsFloat32(offset int) (float32, error) {
	if t.DataType.FuncSet == nil {
		return 0, fmt.Errorf("unsupported tensor datatype %s", t.DataType)
	}
	return t.DataType.FuncSet.ReadItem_AsFloat32(unsafe.Pointer(&t.RawData[offset])), nil
}

func (t *Tensor) SetItemByOffset(offset int, val any) error {
	if t.DataType.FuncSet == nil {
		return fmt.Errorf("unsupported tensor datatype %s", t.DataType)
	}
	return t.DataType.FuncSet.WriteItem(unsafe.Pointer(&t.RawData[offset]), val)
}

func (t *Tensor) SetItemByOffset_FromFloat32(offset int, val float32) error {
	if t.DataType.FuncSet == nil {
		return fmt.Errorf("unsupported tensor datatype %s", t.DataType)
	}
	t.DataType.FuncSet.WriteItem_FromFloat32(unsafe.Pointer(&t.RawData[offset]), val)
	return nil
}

func (t *Tensor) GetItem_AsFloat32(loc []int) (float32, error) {
	offset := t.calculateByteOffset(loc)
	return t.GetItemByOffset_AsFloat32(offset)
}

func (t *Tensor) SetItem_FromFloat32(loc []int, val float32) error {
	offset := t.calculateByteOffset(loc)
	return t.SetItemByOffset_FromFloat32(offset, val)
}

func (t *Tensor) Item() any {
	return t.GetItemByOffset(0)
}

func (t *Tensor) Apply(fn func(val any) any) error {
	for offset := 0; offset < len(t.RawData); offset += t.DataType.ItemSize {
		val := t.GetItemByOffset(offset)
		val = fn(val)
		if val == nil {
			return fmt.Errorf("nil cannot be assigned as a tensor item but Apply function returned nil")
		}
		if err := t.SetItemByOffset(offset, val); err != nil {
			return err
		}
	}
	return nil
}

func (t *Tensor) Apply_AsFloat32(fn func(val float32) float32) error {
	for offset := 0; offset < len(t.RawData); offset += t.DataType.ItemSize {
		val, err := t.GetItemByOffset_AsFloat32(offset)
		if err != nil {
			return err
		}
		val = fn(val)
		if err := t.SetItemByOffset_FromFloat32(offset, val); err != nil {
			return err
		}
	}
	return nil
}

func (t *Tensor) calculateByteOffset(loc []int) int {
	offset := 0
	for i := 0; i < len(loc); i++ {
		offset += loc[i] * t.ByteStride[i]
	}
	return offset
}

func calculateByteStride(size []int, dataType DataType) []int {
	if len(size) == 0 {
		return []int{1}
	}
	result := make([]int, len(size))

	result[len(size)-1] = dataType.ItemSize
	for i := len(size) - 2; i >= 0; i-- {
		result[i] = result[i+1] * size[i+1]
	}
	return result
}

func (t *Tensor) Slice(locStart []int, locEnd []int) (*Tensor, error) {
	if len(locStart) != len(locEnd) {
		return nil, fmt.Errorf("locStart %d and locEnd %d don't have same dimensions", len(locStart), len(locEnd))
	}
	if len(locStart) == 0 || len(locStart) > len(t.Size) {
		return nil, fmt.Errorf("locStart %d and tensor \"%s\" %d don't have compatible dimensions", len(locStart), t.Name, len(t.Size))
	}
	dstSize := make([]int, len(t.Size))
	for dimension := 0; dimension < len(locStart); dimension++ {
		if locStart[dimension] < 0 || locStart[dimension] > t.Size[dimension] ||
			locEnd[dimension] < 0 || locEnd[dimension] > t.Size[dimension] ||
			locEnd[dimension]-locStart[dimension] < 0 {
			return nil, fmt.Errorf("incompatible locStart, locEnd values and tensor")
		}
		dstSize[dimension] = locEnd[dimension] - locStart[dimension]
	}
	for dimension := len(locStart); dimension < len(t.Size); dimension++ {
		dstSize[dimension] = t.Size[dimension]
	}
	for dimension := 0; dimension < len(dstSize)-1; dimension++ {
		if dstSize[dimension] == 0 {
			dstSize = dstSize[1:]
			dimension--
		} else {
			break
		}
	}

	dst := NewEmptyTensor(dstSize, t.DataType)

	err := sliceTensorDimension(t, dst, locStart, locEnd, nil, 0)
	if err != nil {
		return nil, err
	}
	return dst, nil
}

func sliceTensorDimension(t *Tensor, dst *Tensor, locStart []int, locEnd []int, dstLocStart []int, currentDimension int) error {
	if dstLocStart == nil {
		dstLocStart = make([]int, len(dst.Size))
	}
	if currentDimension < len(locStart)-1 {
		currentDimStart := locStart[currentDimension]
		currentDimEnd := locEnd[currentDimension]
		if currentDimension < len(locEnd) && currentDimStart == currentDimEnd {
			currentDimEnd++
		}

		locStartLocal := make([]int, len(locStart))
		copy(locStartLocal, locStart)
		locEndLocal := make([]int, len(locEnd))
		copy(locEndLocal, locEnd)

		for i := currentDimStart; i < currentDimEnd; i++ {
			locStartLocal[currentDimension] = i
			locEndLocal[currentDimension] = i
			dstLocStart[currentDimension] = i - currentDimStart
			sliceTensorDimension(t, dst, locStartLocal, locEndLocal, dstLocStart, currentDimension+1)
		}
	} else {
		readLocStart := locStart
		readLocEnd := locEnd

		for dimension := currentDimension + 1; dimension < len(dst.Size); dimension++ {
			readLocStart = append(readLocStart, 0)
			readLocEnd = append(readLocEnd, 0) // Offset of next item
		}

		readOffsetStart := t.calculateByteOffset(readLocStart)
		readOffsetEnd := t.calculateByteOffset(readLocEnd)
		writeOffsetStart := dst.calculateByteOffset(dstLocStart)

		if bytesCount := copy(dst.RawData[writeOffsetStart:], t.RawData[readOffsetStart:readOffsetEnd]); bytesCount != readOffsetEnd-readOffsetStart {
			return fmt.Errorf("error while copying bytes in sliceTensorDimension, expected: %d, actual: %d", readOffsetEnd-readOffsetStart, bytesCount)
		}
	}
	return nil
}

func (t *Tensor) SetSlice(locStart []int, locEnd []int, val *Tensor) error {
	if t.DataType != val.DataType {
		return fmt.Errorf("incompatible tensor data types: %v and %v", t.DataType, val.DataType)
	}
	if len(locStart) != len(locEnd) {
		return fmt.Errorf("locStart %d and locEnd %d don't have same dimensions", len(locStart), len(locEnd))
	}
	if len(locStart) == 0 || len(locStart) > len(t.Size) {
		return fmt.Errorf("locStart %d and tensor \"%s\" %d don't have compatible dimensions", len(locStart), t.Name, len(t.Size))
	}
	if len(val.Size) > len(t.Size) {
		return fmt.Errorf("tensor  \"%s\" %d and tensor \"%s\" %d don't have compatible dimensions", val.Name, len(val.Size), t.Name, len(t.Size))
	}
	for dstDimension := 0; dstDimension < len(locStart); dstDimension++ {
		if locEnd[dstDimension] > t.Size[dstDimension] {
			return fmt.Errorf("tensor  \"%s\" %v and locEnd=%v are not compatible", t.Name, t.Size, locEnd)
		}
	}

	if err := setSliceTensorDimension(val, t, nil, locStart, 0); err != nil {
		return err
	}
	return nil
}

func setSliceTensorDimension(val *Tensor, dst *Tensor, valLocStart []int, dstLocStart []int, currentDimension int) error {
	writeLocStart := dstLocStart

	for dimension := currentDimension + 1; dimension < len(dst.Size); dimension++ {
		writeLocStart = append(writeLocStart, 0)
	}

	readOffsetStart := 0
	readOffsetEnd := len(val.RawData)
	writeOffsetStart := dst.calculateByteOffset(dstLocStart)
	if bytesCount := copy(dst.RawData[writeOffsetStart:], val.RawData[readOffsetStart:readOffsetEnd]); bytesCount != readOffsetEnd-readOffsetStart {
		return fmt.Errorf("error while copying bytes in sliceTensorDimension, expected: %d, actual: %d", readOffsetEnd-readOffsetStart, bytesCount)
	}
	return nil
}

func (t *Tensor) Reshape(newSize []int) (*Tensor, error) {
	newElementCount := 1
	for _, shapeItem := range newSize {
		newElementCount = newElementCount * shapeItem
	}
	if newElementCount != t.GetElementCount() {
		return nil, fmt.Errorf("shape %v is invalid for input of element count %d", newSize, t.GetElementCount())
	}
	dst := NewEmptyTensorEx(t.Name, newSize, t.DataType, false)
	dst.RawData = t.RawData
	return dst, nil
}

func CheckBroadcastableOnce(size1 []int, size2 []int) bool {
	// See: https://pytorch.org/docs/stable/notes/broadcasting.html
	// Aim is to find if can "size2" be expanded to adapt "size1"

	for dimension := max(len(size1), len(size2)) - 1; dimension >= 0; dimension-- {
		if dimension >= len(size1) {
			size1 = append([]int{1}, size1...)
		}
		if dimension >= len(size2) {
			size2 = append([]int{1}, size2...)
		}
	}
	for dimension := 0; dimension < len(size1); dimension++ {
		if size1[dimension]%size2[dimension] != 0 {
			return false
		}
	}
	return true
}

func CheckBroadcastable(t1 *Tensor, t2 *Tensor, isCommutative bool) (refTensor *Tensor, expandingTensor *Tensor, err error) {
	// See: https://pytorch.org/docs/stable/notes/broadcasting.html
	if CheckBroadcastableOnce(t1.Size, t2.Size) {
		return t1, t2, nil
	}
	if isCommutative && CheckBroadcastableOnce(t2.Size, t1.Size) {
		return t2, t1, nil
	}
	return nil, nil, fmt.Errorf("two tensor shapes cannot be broadcasted: %v and %v", t1.Size, t2.Size)
}

func (t *Tensor) ToBFloat16() (*Tensor, error) {
	if t.DataType == DT_BF16 {
		return t, nil
	}
	inputItemSize := t.DataType.ItemSize

	switch t.DataType {
	case DT_F32:
		dst := NewEmptyTensorEx(t.Name, t.Size, DT_BF16, true)
		dstItemSize := dst.DataType.ItemSize

		tPtr := unsafe.Pointer(&t.RawData[0])
		dstPtr := unsafe.Pointer(&dst.RawData[0])

		writeOffset := 0
		for readOffset := 0; readOffset < t.GetBytesCount(); readOffset += inputItemSize {
			resultItemBits := dtype.Float32ToBFloat16bits(*(*float32)(unsafe.Add(tPtr, readOffset)))
			*(*uint16)(unsafe.Add(dstPtr, writeOffset)) = resultItemBits
			writeOffset += dstItemSize
		}
		return dst, nil
	default:
		return nil, fmt.Errorf("unsupported tensor datatype conversion from %s to %s", t.DataType, DT_BF16)
	}
}

func (t *Tensor) ToFloat32() (*Tensor, error) {
	if t.DataType == DT_F32 {
		return t, nil
	}
	inputItemSize := t.DataType.ItemSize

	switch t.DataType {
	case DT_BF16:
		dst := NewEmptyTensorEx(t.Name, t.Size, DT_F32, true)
		dstItemSize := dst.DataType.ItemSize

		tPtr := unsafe.Pointer(&t.RawData[0])
		dstPtr := unsafe.Pointer(&dst.RawData[0])

		writeOffset := 0
		for readOffset := 0; readOffset < t.GetBytesCount(); readOffset += inputItemSize {
			resultItem := dtype.BFloat16bitsToFloat32(*(*uint16)(unsafe.Add(tPtr, readOffset)))
			*(*float32)(unsafe.Add(dstPtr, writeOffset)) = resultItem
			writeOffset += dstItemSize
		}
		return dst, nil
	default:
		return nil, fmt.Errorf("unsupported tensor datatype conversion from %s to %s", t.DataType, DT_F32)
	}
}

func (t *Tensor) ViewAsComplex64() (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.view_as_complex.html
	/*
		Comment from Pytorch's documentation (link aboe):
		Torch's view_as_complex() is only supported for tensors with torch.dtype torch.float64 and torch.float32.
		The input is expected to have the last dimension of size 2. In addition, the tensor must have a stride of 1 for
		its last dimension. The strides of all other dimensions must be even numbers.
	*/
	if t.DataType != DT_F32 {
		return nil, fmt.Errorf("tensor must be in float32 data type")
	}
	if t.Size[len(t.Size)-1] != 2 {
		return nil, fmt.Errorf("last dimension of size must be 2, got size %v", t.Size)
	}
	dstSize := t.Size[0 : len(t.Size)-1]
	dst := NewEmptyTensorEx(t.Name, dstSize, DT_COMPLEX, false)
	dst.RawData = t.RawData
	return dst, nil
}

func (t *Tensor) ViewAsComplex64WithReshape() (*Tensor, error) {
	// t example shape=[5,32,128] dtype=DT_BF16
	newShape := append(append([]int{}, t.Size[0:len(t.Size)-1]...), t.Size[len(t.Size)-1]/2, 2)
	t_, err := t.ToFloat32() // example shape=[5,32,128] dtype=DT_F32
	if err != nil {
		return nil, err
	}
	if t_, err = t_.Reshape(newShape); err != nil { // example shape=[5,32,64,2] dtype=DT_F32
		return nil, err
	}
	if t_, err = t_.ViewAsComplex64(); err != nil { // example shape=[5,32,64] dtype=DT_COMPLEX
		return nil, err
	}
	return t_, nil
}

func (t *Tensor) ViewAsFloat32() (*Tensor, error) {
	// See: https://pytorch.org/docs/stable/generated/torch.view_as_real.html
	if t.DataType != DT_COMPLEX {
		return nil, fmt.Errorf("tensor must be in complex64 data type")
	}
	dstSize := append(t.Size, 2)
	dst := NewEmptyTensorEx(t.Name, dstSize, DT_F32, false)
	dst.RawData = t.RawData
	return dst, nil
}

func (t *Tensor) ViewAsFloat32WithReshape() (*Tensor, error) {
	// t example shape=[5,32,64] dtype=DT_COMPLEX
	t_, err := t.ViewAsFloat32() // example shape=[5,32,64,2] dtype=DT_F32
	if err != nil {
		return nil, err
	}
	newShape := append(append([]int{}, t_.Size[0:len(t_.Size)-2]...), t_.Size[len(t_.Size)-2]*2)
	if t_, err = t_.Reshape(newShape); err != nil { // example shape=[5,32,128] dtype=DT_F32
		return nil, err
	}
	return t_, nil
}

func (t *Tensor) Transpose(dim1 int, dim2 int) (*Tensor, error) {
	if dim1 < 0 || dim1 >= len(t.Size) {
		return nil, fmt.Errorf("incompatible dimension argument %d for shape %v", dim1, t.Size)
	}
	if dim2 < 0 || dim2 >= len(t.Size) {
		return nil, fmt.Errorf("incompatible dimension argument %d for shape %v", dim2, t.Size)
	}
	if dim1 == dim2 {
		return nil, fmt.Errorf("dim1 and dim2 can't be equal: %d", dim1)
	}
	if dim1 > dim2 {
		temp := dim1
		dim1 = dim2
		dim2 = temp
	}
	dstSize := make([]int, len(t.Size))
	copy(dstSize, t.Size)
	dstSize[dim1] = t.Size[dim2]
	dstSize[dim2] = t.Size[dim1]
	dst := NewEmptyTensor(dstSize, t.DataType)

	tLoc := make([]int, len(t.Size))
	dstLoc := make([]int, len(t.Size))
	blockSize := 1
	for dimension := dim2 + 1; dimension < len(t.Size); dimension++ {
		blockSize = blockSize * t.Size[dimension]
	}
	blockSize = blockSize * t.DataType.ItemSize
	for iteratorFirstPart := IterateOverSize(dst.Size[0:dim1], 0); iteratorFirstPart.HasNext(); {
		locFirstPart := iteratorFirstPart.Next()
		copy(tLoc, locFirstPart)
		copy(dstLoc, locFirstPart)
		for i := 0; i < dst.Size[dim1]; i++ {
			tLoc[dim2] = i
			dstLoc[dim1] = i
			for iteratorMiddlePart := IterateOverSize(dst.Size[dim1+1:dim2], 0); iteratorMiddlePart.HasNext(); {
				locMiddlePart := iteratorMiddlePart.Next()
				copy(tLoc[dim1+1:], locMiddlePart)
				copy(dstLoc[dim1+1:], locMiddlePart)

				for j := 0; j < dst.Size[dim2]; j++ {
					tLoc[dim1] = j
					dstLoc[dim2] = j

					readOffsetStart := t.calculateByteOffset(tLoc)
					readOffsetEnd := readOffsetStart + blockSize
					writeOffsetStart := dst.calculateByteOffset(dstLoc)

					if bytesCount := copy(dst.RawData[writeOffsetStart:], t.RawData[readOffsetStart:readOffsetEnd]); bytesCount != readOffsetEnd-readOffsetStart {
						return nil, fmt.Errorf("error while copying bytes in Transpose, expected: %d, actual: %d", readOffsetEnd-readOffsetStart, bytesCount)
					}
				}
			}
		}
	}
	return dst, nil
}

package ml

import (
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"strings"
	"unsafe"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

type UnquantizedDataType = DataType

var (
	DT_BF16    = UnquantizedDataType{"BF16", reflect.TypeOf(dtype.BFloat16(0))}
	DT_F32     = DataType{"Float32", reflect.TypeOf(float32(0))}
	DT_UINT16  = DataType{"UInt16", reflect.TypeOf(uint16(0))}
	DT_COMPLEX = DataType{"Complex", reflect.TypeOf(complex64(complex(0.0, 0.0)))}
)

type DataType struct {
	name     string
	dataType reflect.Type
}

func (dt DataType) ItemSize() int {
	return int(dt.dataType.Size())
}

func (dt DataType) GetName() string {
	return dt.name
}

func (dt DataType) String() string {
	return dt.name
}

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

func NewEmptyTensorEx(name string, size []int, dataType DataType) *Tensor {
	result := &Tensor{
		Name:       name,
		Size:       size,
		DataType:   dataType,
		ByteStride: calculateByteStride(size, dataType),
	}
	result.RawData = make([]byte, result.GetBytesCount())
	return result
}

func NewEmptyTensor(size []int, dataType DataType) *Tensor {
	return NewEmptyTensorEx("", size, dataType)
}

func DuplicateTensor(input *Tensor) *Tensor {
	dst := NewEmptyTensorEx(input.Name, input.Size, input.DataType)
	copy(dst.RawData, input.RawData)
	return dst
}

func (t *Tensor) GetShape() []int {
	return t.Size
}

func (t *Tensor) GetElementCount() int {
	result := 1
	for _, shapeItem := range t.GetShape() {
		result = result * shapeItem
	}
	return result
}

func (t *Tensor) GetBytesCount() int {
	return t.GetElementCount() * t.DataType.ItemSize()
}

func (t *Tensor) String() string {
	shorten := true
	if len(t.Size) == 1 {
		shorten = false
	}
	return fmt.Sprintf("[Tensor \"%s\"](%s, shape=%v, dtype=%s)", t.Name, t.dimensionToString([]int{}, shorten), t.Size, t.DataType.name)
}

func (t *Tensor) StringLong() string {
	return fmt.Sprintf("[Tensor \"%s\"](%s, shape=%v, dtype=%s)", t.Name, t.dimensionToString([]int{}, false), t.Size, t.DataType.name)
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
	switch item := item.(type) {
	case dtype.BFloat16:
		return fmt.Sprintf("%.4e", item.Float32())
	case float32, float64:
		return fmt.Sprintf("%.4e", item)
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", item)
	case complex64:
		realPart := real(item)
		imagPart := imag(item)

		str := fmt.Sprintf("%.4e+%.4ej", realPart, imagPart)
		return str
	default:
		return fmt.Sprintf("%v", item)
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
	switch t.DataType {
	case DT_BF16:
		return dtype.ReadBFloat16LittleEndian(t.RawData[offset:])
	case DT_UINT16:
		return binary.LittleEndian.Uint16(t.RawData[offset:])
	case DT_F32:
		return math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[offset:]))
	case DT_COMPLEX:
		realPart := math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[offset:]))
		imagPart := math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[offset+int(unsafe.Sizeof(realPart)):]))
		return complex64(complex(realPart, imagPart))
	}
	return fmt.Errorf("unsupported tensor datatype %s", t.DataType)
}

func (t *Tensor) SetItemByOffset(offset int, val any) error {
	switch t.DataType {
	case DT_BF16:
		convVal, ok := val.(dtype.BFloat16)
		if !ok {
			return fmt.Errorf("uncompatible types BFloat16 and %v", reflect.TypeOf(val))
		}
		dtype.WriteBFloat16LittleEndian(t.RawData[offset:], convVal)
		return nil
	case DT_UINT16:
		convVal, ok := val.(uint16)
		if !ok {
			return fmt.Errorf("uncompatible types uint16 and %v", reflect.TypeOf(val))
		}
		binary.LittleEndian.PutUint16(t.RawData[offset:], convVal)
		return nil
	case DT_F32:
		convVal, ok := val.(float32)
		if !ok {
			return fmt.Errorf("uncompatible types float32 and %v", reflect.TypeOf(val))
		}
		binary.LittleEndian.PutUint32(t.RawData[offset:], math.Float32bits(convVal))
		return nil

	case DT_COMPLEX:
		convVal, ok := val.(complex64)
		if !ok {
			return fmt.Errorf("uncompatible types complex64 and %v", reflect.TypeOf(val))
		}
		realPartBits := math.Float32bits(real(convVal))
		imagPartBits := math.Float32bits(imag(convVal))
		binary.LittleEndian.PutUint32(t.RawData[offset:], realPartBits)
		binary.LittleEndian.PutUint32(t.RawData[offset+int(unsafe.Sizeof(realPartBits)):], imagPartBits)
		return nil
	}
	return fmt.Errorf("unsupported tensor datatype %s", t.DataType)
}

func (t *Tensor) Item() any {
	return t.GetItemByOffset(0)
}

func (t *Tensor) Apply(fn func(val any) any) error {
	for offset := 0; offset < len(t.RawData); offset += t.DataType.ItemSize() {
		val := t.GetItemByOffset(offset)
		val = fn(val)
		if err := t.SetItemByOffset(offset, val); err != nil {
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

	result[len(size)-1] = dataType.ItemSize()
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
		if locStart[dimension] < 0 || locStart[dimension] >= t.Size[dimension] ||
			locEnd[dimension] < 0 || locEnd[dimension] >= t.Size[dimension] ||
			locEnd[dimension]-locStart[dimension] < 0 {
			return nil, fmt.Errorf("uncompatible locStart, locEnd values and tensor")
		}
		dstSize[dimension] = locEnd[dimension] - locStart[dimension]
	}
	for dimension := len(locStart); dimension < len(t.Size); dimension++ {
		dstSize[dimension] = t.Size[dimension]
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

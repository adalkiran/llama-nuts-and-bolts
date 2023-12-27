package ml

import (
	"encoding/binary"
	"fmt"
	"reflect"

	"github.com/x448/float16"
)

// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

type UnquantizedDataType = DataType

var (
	DT_BF16 = UnquantizedDataType{"BF16", reflect.TypeOf(uint16(0))}
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

func (t *Tensor) IsVector() bool {
	return len(t.Size) == 1
}

func (t *Tensor) IsMatrix() bool {
	return len(t.Size) == 2
}

func (t *Tensor) GetItem(loc []int) (float16.Float16, error) {
	if len(loc) != len(t.Size) {
		return 0, fmt.Errorf("dimensions are not compatible: tensor is %dD, loc is %dD", len(t.Size), len(loc))
	}
	offset := t.calculateByteOffset(loc)
	return float16.Frombits(binary.BigEndian.Uint16(t.RawData[offset:])), nil
}

func (t *Tensor) SetItem(loc []int, val float16.Float16) error {
	if len(loc) != len(t.Size) {
		return fmt.Errorf("dimensions are not compatible: tensor is %dD, loc is %dD", len(t.Size), len(loc))
	}
	offset := t.calculateByteOffset(loc)
	binary.BigEndian.PutUint16(t.RawData[offset:], val.Bits())
	return nil
}

func (t *Tensor) GetItemByOffset(offset int) float16.Float16 {
	return float16.Frombits(binary.BigEndian.Uint16(t.RawData[offset : offset+2]))
}

func (t *Tensor) SetItemByOffset(offset int, val float16.Float16) {
	binary.BigEndian.PutUint16(t.RawData[offset:], val.Bits())
}

func (t *Tensor) Apply(fn func(val float16.Float16) float16.Float16) {
	for offset := 0; offset < len(t.RawData); offset += t.DataType.ItemSize() {
		val := t.GetItemByOffset(offset)
		val = fn(val)
		t.SetItemByOffset(offset, val)
	}
}

func (t *Tensor) calculateByteOffset(loc []int) int {
	offset := 0
	for i := 0; i < len(loc); i++ {
		offset += loc[i] * t.ByteStride[i]
	}
	return offset
}

func calculateByteStride(size []int, dataType DataType) []int {
	result := make([]int, len(size))

	result[len(size)-1] = dataType.ItemSize()
	for i := len(size) - 2; i >= 0; i-- {
		result[i] = result[i+1] * size[i+1]
	}
	return result
}

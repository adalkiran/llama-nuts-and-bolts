package torch

import (
	"fmt"
	"reflect"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
)

// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

type UnquantizedDataType = DataType

var (
	DT_BF16 = UnquantizedDataType{"BF16", reflect.TypeOf(uint16(0))}
)

type Tensor struct {
	size          []int
	stride        []int
	dataType      DataType
	description   string
	storageOffset int64

	storage TorchStorage
}

func (t Tensor) GetDataType() DataType {
	return t.dataType
}

func (t Tensor) GetShape() []int {
	return t.size
}

func (t Tensor) GetElementCount() int {
	result := 1
	for _, shapeItem := range t.GetShape() {
		result = result * shapeItem
	}
	return result
}

func (t Tensor) GetBytesCount() int {
	return t.GetElementCount() * t.dataType.itemSize()
}

func (t Tensor) GetWeights() []byte {
	return t.storage.rawData
}

var TORCH_CLASSES = map[string]interface{}{
	// getattr used here as a workaround for mypy not being smart enough to detrmine
	// the staticmethods have a __func__ attribute.
	//('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'),
	"torch._utils._rebuild_tensor_v2": rebuild_tensor_v2,

	"torch.BFloat16Storage": StorageKind{DT_BF16},

	//('torch', 'HalfStorage'): LazyStorageKind(DT_F16),
	//('torch', 'FloatStorage'): LazyStorageKind(DT_F32),
	//('torch', 'IntStorage'): LazyStorageKind(DT_I32),
	//('torch', 'Tensor'): LazyTensor,
}

func rebuild_tensor_v2(storage TorchStorage, storageOffset int, size pickle.PickleTuple, stride pickle.PickleTuple,
	requires_grad bool, backward_hooks interface{}, metadata interface{}) (*Tensor, error) {

	sizeInt, err := common.InterfaceArrToIntArr(size)
	if err != nil {
		return nil, err
	}

	strideInt, err := common.InterfaceArrToIntArr(stride)
	if err != nil {
		return nil, err
	}
	description := fmt.Sprintf("pickled storage_offset=%d in %s", storageOffset, storage.description)
	return &Tensor{size: sizeInt, stride: strideInt, dataType: storage.kind.dataType, description: description, storageOffset: storage.storageOffset, storage: storage}, nil
}

type DataType struct {
	name     string
	dataType reflect.Type
}

func (dt DataType) itemSize() int {
	return int(dt.dataType.Size())
}

func (dt DataType) GetName() string {
	return dt.name
}

type StorageKind struct {
	dataType DataType
}

type TorchStorage struct {
	filename      string
	kind          StorageKind
	storageOffset int64
	description   string

	rawData []byte
}

func (ts *TorchStorage) Load(memoryMapper *common.MemoryMapper, elmCount int) error {
	offset := int(ts.storageOffset)
	size := elmCount * ts.kind.dataType.itemSize()
	ts.rawData = memoryMapper.Data[offset : offset+size]
	return nil
}

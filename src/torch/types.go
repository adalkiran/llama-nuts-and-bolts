package torch

import (
	"bufio"
	"fmt"
	"io"
	"reflect"

	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
)

// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

type UnquantizedDataType = DataType

var (
	DT_BF16 = UnquantizedDataType{"BF16", reflect.TypeOf(uint16(0))}
)

type Tensor interface {
}

type TensorDescriptor struct {
	size          []int
	stride        []int
	dataType      DataType
	description   string
	storageOffset int

	storage StorageDescriptor
}

func (lt TensorDescriptor) GetDataType() DataType {
	return lt.dataType
}

func (lt TensorDescriptor) GetShape() []int {
	return lt.size
}

func (lt TensorDescriptor) Load(tmr *TorchModelReader) (Tensor, error) {
	elmCount := lt.stride[0] * lt.size[0]
	lt.storage.Load(tmr, lt.storageOffset, elmCount)
	return nil, nil
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

func rebuild_tensor_v2(storage StorageDescriptor, storageOffset int, size pickle.PickleTuple, stride pickle.PickleTuple,
	requires_grad bool, backward_hooks interface{}, metadata interface{}) (*TensorDescriptor, error) {

	sizeInt, err := pickle.InterfaceArrToIntArr(size)
	if err != nil {
		return nil, err
	}

	strideInt, err := pickle.InterfaceArrToIntArr(stride)
	if err != nil {
		return nil, err
	}
	description := fmt.Sprintf("pickled storage_offset=%d in %s", storageOffset, storage.description)
	return &TensorDescriptor{size: sizeInt, stride: strideInt, dataType: storage.kind.dataType, description: description, storageOffset: storageOffset, storage: storage}, nil
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

type StorageDescriptor struct {
	filename    string
	kind        StorageKind
	description string
}

func (sd StorageDescriptor) Load(tmr *TorchModelReader, offset int, elmCount int) (string, error) {
	dataFile, err := tmr.inputZipReader.Open(sd.filename)
	if err != nil {
		return "", err
	}
	defer dataFile.Close()
	if offset != 0 {
		return "", fmt.Errorf("offset value other than 0 is not supported for storage")
	}
	dataType := sd.kind.dataType

	dataFileReader := bufio.NewReader(dataFile)
	buf := make([]byte, elmCount*dataType.itemSize())
	readCount, err := io.ReadFull(dataFileReader, buf)
	if err != nil {
		return "", err
	}
	if readCount != len(buf) {
		return "", fmt.Errorf("cannot read all of tensor bytes")
	}
	return "asasdf", nil
}

package torch

import (
	"fmt"
	"os"
	"reflect"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
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
	storageOffset int64

	storage StorageDescriptor
}

func (td TensorDescriptor) GetDataType() DataType {
	return td.dataType
}

func (td TensorDescriptor) GetShape() []int {
	return td.size
}

func (td TensorDescriptor) GetElementCount() int {
	result := 1
	for _, shapeItem := range td.GetShape() {
		result = result * shapeItem
	}
	return result
}

func (td TensorDescriptor) GetBytesCount() int {
	return td.GetElementCount() * td.dataType.itemSize()
}

func (td TensorDescriptor) Load(tmr *TorchModelReader) (Tensor, error) {
	elmCount := td.stride[0] * td.size[0]
	td.storage.Load(tmr, elmCount)
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

	sizeInt, err := common.InterfaceArrToIntArr(size)
	if err != nil {
		return nil, err
	}

	strideInt, err := common.InterfaceArrToIntArr(stride)
	if err != nil {
		return nil, err
	}
	description := fmt.Sprintf("pickled storage_offset=%d in %s", storageOffset, storage.description)
	return &TensorDescriptor{size: sizeInt, stride: strideInt, dataType: storage.kind.dataType, description: description, storageOffset: storage.storageOffset, storage: storage}, nil
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
	filename      string
	kind          StorageKind
	storageOffset int64
	description   string
}

func (sd StorageDescriptor) Load(tmr *TorchModelReader, elmCount int) (string, error) {
	file, err := os.OpenFile(tmr.modelFilePath, os.O_RDONLY, 0)
	if err != nil {
		return "asdsd", err
	}
	defer file.Close()
	_, err = file.Seek(sd.storageOffset, 0)
	if err != nil {
		return "asdsd", err
	}
	buf := make([]byte, 1024)
	file.Read(buf)
	buf = buf

	/*
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

		cnt := 0
		for i, byt := range buf {
			if byt == 21 && buf[i+1] == 0 {
				cnt += i
			}
		}*/
	return fmt.Sprintf("asasdf"), nil
}

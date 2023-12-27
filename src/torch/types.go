package torch

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
)

var TORCH_CLASSES = map[string]interface{}{
	// getattr used here as a workaround for mypy not being smart enough to detrmine
	// the staticmethods have a __func__ attribute.
	//('torch._tensor', '_rebuild_from_type_v2'): getattr(rebuild_from_type_v2, '__func__'),
	"torch._utils._rebuild_tensor_v2": rebuild_tensor_v2,

	"torch.BFloat16Storage": StorageKind{ml.DT_BF16},

	//('torch', 'HalfStorage'): LazyStorageKind(DT_F16),
	//('torch', 'FloatStorage'): LazyStorageKind(DT_F32),
	//('torch', 'IntStorage'): LazyStorageKind(DT_I32),
	//('torch', 'Tensor'): LazyTensor,
}

func rebuild_tensor_v2(storage TorchStorage, storageOffset int, size pickle.PickleTuple, stride pickle.PickleTuple,
	requires_grad bool, backward_hooks interface{}, metadata interface{}) (*ml.Tensor, error) {

	sizeInt, err := common.InterfaceArrToIntArr(size)
	if err != nil {
		return nil, err
	}

	strideInt, err := common.InterfaceArrToIntArr(stride)
	if err != nil {
		return nil, err
	}
	return ml.NewTensor("", sizeInt, strideInt, storage.kind.dataType, storage.rawData), nil
}

type StorageKind struct {
	dataType ml.DataType
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
	size := elmCount * ts.kind.dataType.ItemSize()
	ts.rawData = memoryMapper.Data[offset : offset+size]
	return nil
}

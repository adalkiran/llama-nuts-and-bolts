package model

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
)

type ModelArchitecture uint8

func (ma ModelArchitecture) String() string {
	switch ma {
	case ModelArchitectureLlama:
		return "Llama"
	}
	return "UNKNOWN"
}

type ModelType uint8

func (mt ModelType) String() string {
	switch mt {
	case ModelType7B:
		return "7B"
	}
	return "UNKNOWN"
}

type TokenId int32
type Position int32
type SequenceId int32

const (
	ModelArchitectureUnknown ModelArchitecture = 0
	ModelArchitectureLlama   ModelArchitecture = 1

	ModelTypeUnknown ModelType = 0
	ModelType7B      ModelType = 1
)

type Model struct {
	Tensors    *pickle.PickleDict[*ml.Tensor]
	ModelArgs  *ModelArgs
	Vocabulary *Vocabulary

	Transformer *LlamaTransformer

	ModelArchitecture ModelArchitecture
	ModelType         ModelType

	MemoryMapper *common.MemoryMapper
}

func (m *Model) Free() error {
	return m.MemoryMapper.Unmap()
}

func (m *Model) GetElementCount() int {
	result := 0
	for _, key := range m.Tensors.GetKeys() {
		tensor, _ := m.Tensors.Get(key)
		result += tensor.GetElementCount()
	}
	return result
}

func (m *Model) GetBytesCount() int {
	result := 0
	for _, key := range m.Tensors.GetKeys() {
		tensor, _ := m.Tensors.Get(key)
		result += tensor.GetBytesCount()
	}
	return result
}

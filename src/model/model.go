package model

import (
	"fmt"

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

type TokenPiece struct {
	Piece string // piece must not be empty.
	Rank  int32

	IsByte       bool
	ByteFallback []byte
}

func (tp TokenPiece) String() string {
	var pieceType, pieceStr string
	if tp.IsByte {
		pieceType = "BYTE"
		pieceStr = tp.ByteFallbackString()
	} else {
		pieceType = "NORMAL"
		pieceStr = tp.Piece
	}
	return fmt.Sprintf("\"%s\" rank: %d, type: %s", pieceStr, tp.Rank, pieceType)
}

func (tp TokenPiece) ByteFallbackString() string {
	if tp.ByteFallback == nil {
		return ""
	}
	result := ""
	for _, b := range tp.ByteFallback {
		result += fmt.Sprintf("<0x%02X>", b)
	}
	return result
}

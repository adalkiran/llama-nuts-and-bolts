package model

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
	"github.com/adalkiran/llama-nuts-and-bolts/src/torch"
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

type Layer struct {
	// normalization
	attn_norm *torch.TensorDescriptor // Original: "layers.0.attention_norm.weight"  |  ggml: "blk.0.attn_norm.weight" | shape: [4096] -> [n_embd]

	// attention
	attn_wq *torch.TensorDescriptor // Original: "layers.0.attention.wq.weight"  |  ggml: "blk.0.attn_q.weight" | shape: [4096 4096] -> [n_embd, n_embd]
	attn_wk *torch.TensorDescriptor // Original: "layers.0.attention.wk.weight"  |  ggml: "blk.0.attn_k.weight" | shape: [4096 4096] -> [n_embd, n_embd_grouped_query_attn]
	attn_wv *torch.TensorDescriptor // Original: "layers.0.attention.wv.weight"  |  ggml: "blk.0.attn_v.weight" | shape: [4096 4096] -> [n_embd, n_embd_grouped_query_attn]
	attn_wo *torch.TensorDescriptor // Original: "layers.0.attention.wo.weight"  |  ggml: "blk.0.attn_output.weight" | shape: [4096 4096] -> [n_embd, n_embd]

	// feed forward normalization
	ffn_norm *torch.TensorDescriptor // Original: "layers.0.ffn_norm.weight"  |  ggml: "blk.0.ffn_norm.weight" | shape: [4096] -> [n_embd]

	// feed forward
	ffn_gate *torch.TensorDescriptor // Original: "layers.0.feed_forward.w1.weight"  |  ggml: "blk.0.ffn_gate.weight" | shape: [11008 4096] -> [n_embd, n_ff] | w1
	ffn_down *torch.TensorDescriptor // Original: "layers.0.feed_forward.w2.weight"  |  ggml: "blk.0.ffn_down.weight" | shape: [4096 11008] -> [n_ff, n_embd] | w2
	ffn_up   *torch.TensorDescriptor // Original: "layers.0.feed_forward.w3.weight"  |  ggml: "blk.0.ffn_up.weight" | shape: [4096 11008] -> [n_embd, n_ff] | w3
}

type Model struct {
	Tensors    *pickle.PickleDict[*torch.TensorDescriptor]
	Config     *Config
	Vocabulary *Vocabulary

	ModelArchitecture ModelArchitecture
	ModelType         ModelType

	tok_embd    *torch.TensorDescriptor // Original: "tok_embeddings.weight"  |  ggml: "token_embd.weight" | shape: [32000 4096] -> [n_embd, n_vocab]
	output_norm *torch.TensorDescriptor // Original: "norm.weight"  |  ggml: "output_norm.weight" | shape: [4096] -> [n_embd]
	output      *torch.TensorDescriptor // Original: "output.weight"  |  ggml: "output.weight" | shape: [32000 4096] -> [n_embd, n_vocab]

	Layers []*Layer
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

package model

import (
	"encoding/json"
	"io"
	"os"
)

// See: https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L20
// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

type ModelArgs struct {
	Dim              int     `json:"dim"`
	N_Layers         int     `json:"n_layers"`
	N_Heads          int     `json:"n_heads"`
	N_KVHeads        int     `json:"n_kv_heads"`
	VocabSize        int     `json:"vocab_size"`         // defined later by tokenizer
	MultipleOf       int     `json:"multiple_of"`        // make SwiGLU hidden layer size multiple of large power of 2
	FFNDimMultiplier float64 `json:"ffn_dim_multiplier"` // Optional
	NormEpsilon      float32 `json:"norm_eps"`

	MaxSequenceLength int

	N_Rep   int
	HeadDim int
}

func NewModelArgs() *ModelArgs {
	return &ModelArgs{
		Dim:        4096,
		N_Layers:   32,
		N_Heads:    32,
		N_KVHeads:  -1,
		VocabSize:  -1,
		MultipleOf: 256,

		FFNDimMultiplier:  -1,
		NormEpsilon:       1e-5,
		MaxSequenceLength: 4096,
	}
}

func (ma ModelArgs) String() string {
	result, _ := json.Marshal(ma)
	return string(result)
}

func loadModelArgsFromFile(configFilePath string) (*ModelArgs, error) {
	jsonFile, err := os.Open(configFilePath)
	if err != nil {
		return nil, err
	}
	defer jsonFile.Close()
	byteValue, _ := io.ReadAll(jsonFile)

	var modelArgs = NewModelArgs()
	if err := json.Unmarshal([]byte(byteValue), &modelArgs); err != nil {
		return nil, err
	}
	return modelArgs, nil
}

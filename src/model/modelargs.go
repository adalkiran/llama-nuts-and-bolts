package model

import (
	"encoding/json"
	"io"
	"os"
)

// See: https://github.com/meta-llama/llama-models/blob/6214a21dc837ce63983ef3fd7b172a6ed16e4905/models/llama3_1/api/args.py#L13
// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

type ModelArgs struct {
	Dim               int     `json:"dim"`
	N_Layers          int     `json:"n_layers"`
	N_Heads           int     `json:"n_heads"`
	N_KVHeads         int     `json:"n_kv_heads"`
	VocabSize         int     `json:"vocab_size"`         // defined later by tokenizer
	MultipleOf        int     `json:"multiple_of"`        // make SwiGLU hidden layer size multiple of large power of 2
	FFNDimMultiplier  float64 `json:"ffn_dim_multiplier"` // Optional
	NormEpsilon       float32 `json:"norm_eps"`
	UseScaledRope     bool    `json:"use_scaled_rope"`
	RopeTheta         float64 `json:"rope_theta"` // Optional
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
		RopeTheta:         500000,
		UseScaledRope:     false,
		MaxSequenceLength: 2048,
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

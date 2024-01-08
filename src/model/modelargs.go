package model

import (
	"encoding/json"
	"fmt"
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

	MaxBatchSize      int
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
		MaxBatchSize:      32,
		MaxSequenceLength: 32,
	}
}

func (ma ModelArgs) String() string {
	result, _ := json.Marshal(ma)
	return string(result)
}

func loadModelArgsFromFile(configFilePath string) (*ModelArgs, error) {
	jsonFile, err := os.Open(configFilePath)
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := io.ReadAll(jsonFile)

	var modelArgs = NewModelArgs()
	if err := json.Unmarshal([]byte(byteValue), &modelArgs); err != nil {
		return nil, err
	}
	return modelArgs, nil
	/*
		// hack to determine LLaMA v1 vs v2 vs CodeLlama
		norm_eps := config["norm_eps"].(float64)
		n_ctx := 0
		if config["rope_theta"] == 1000000 {
			// CodeLlama
			n_ctx = 16384
		} else if norm_eps == 1e-05 || norm_eps == 1e-06 {
			// LLaMA v2
			n_ctx = 4096
		} else {
			// LLaMA v1
			n_ctx = 2048
		}
		n_vocab, ok := config["vocab_size"]
		if !ok {
			tensor, _ := model.Tensors.Get("tok_embeddings.weight")
			n_vocab = tensor.GetShape()[0]
		}
		tensor, _ := model.Tensors.Get("layers.0.feed_forward.w1.weight")
		n_ff := tensor.GetShape()[0]
		n_head := int(config["n_heads"].(float64))
		var n_head_kv int
		n_head_kv_val, ok := config["n_kv_heads"].(float64)
		if ok {
			n_head_kv = int(n_head_kv_val)
		} else {
			n_head_kv = n_head
		}
		var rope_theta float32
		rope_theta_val, ok := config["rope_theta"].(float64)
		if ok {
			rope_theta = float32(rope_theta_val)
		} else {
			rope_theta = -1
		}
		result := &Config{
			Dim:              int(n_vocab.(float64)),
			N_embd:           int(config["dim"].(float64)),
			N_layer:          int(config["n_layers"].(float64)),
			N_ctx:            n_ctx,
			N_ff:             n_ff,
			N_head:           n_head,
			N_head_kv:        n_head_kv,
			F_norm_rms_eps:   float32(config["norm_eps"].(float64)),
			F_rope_freq_base: rope_theta,
		}
		return result, nil
	*/
}

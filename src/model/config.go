package model

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/adalkiran/llama-nuts-and-bolts/src/torch"
)

// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

type Config struct {
	N_vocab    int
	N_embd     int
	N_layer    int
	N_ctx      int
	N_ff       int
	N_head     int
	N_head_kv  int
	N_norm_eps float32

	F_rope_freq_base float32
}

func loadConfigFromFile(configFilePath string, model *Model) (*Config, error) {
	jsonFile, err := os.Open(configFilePath)
	// if we os.Open returns an error then handle it
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()
	byteValue, _ := io.ReadAll(jsonFile)

	var config map[string]interface{}
	json.Unmarshal([]byte(byteValue), &config)
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
		layer, _ := model.Layers.Get("tok_embeddings.weight")
		n_vocab = layer.(torch.TensorDescriptor).GetShape()[0]
	}
	layer, _ := model.Layers.Get("layers.0.feed_forward.w1.weight")
	n_ff := layer.(*torch.TensorDescriptor).GetShape()[0]
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
		N_vocab:          int(n_vocab.(float64)),
		N_embd:           int(config["dim"].(float64)),
		N_layer:          int(config["n_layers"].(float64)),
		N_ctx:            n_ctx,
		N_ff:             n_ff,
		N_head:           n_head,
		N_head_kv:        n_head_kv,
		N_norm_eps:       float32(config["norm_eps"].(float64)),
		F_rope_freq_base: rope_theta,
	}
	return result, nil
}

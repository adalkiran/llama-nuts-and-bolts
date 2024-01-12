package model

import (
	"math/rand"
	"time"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
)

type InferenceContext struct {
	SequenceLength int // context size used during inference

	randomNumberGenerator *rand.Rand

	CacheK []*ml.Tensor
	CacheV []*ml.Tensor
}

func NewInferenceContext(model *Model, inferenceArgs common.InferenceArgs) *InferenceContext {
	// See: https://github.com/ggerganov/llama.cpp/blob/a7aee47b98e45539d491071b25778b833b77e387/llama.cpp#L9304C14-L9304C14
	context := &InferenceContext{}
	if inferenceArgs.Seed == -1 {
		inferenceArgs.Seed = time.Now().UnixNano()
	}
	if inferenceArgs.SequenceLength > 0 {
		context.SequenceLength = inferenceArgs.SequenceLength
	} else {
		context.SequenceLength = model.ModelArgs.MaxSequenceLength
	}
	context.randomNumberGenerator = rand.New(rand.NewSource(inferenceArgs.Seed))

	modelArgs := model.ModelArgs
	context.CacheK = make([]*ml.Tensor, modelArgs.N_Layers)
	context.CacheV = make([]*ml.Tensor, modelArgs.N_Layers)
	for layerIdx := 0; layerIdx < modelArgs.N_Layers; layerIdx++ {
		context.CacheK[layerIdx], _ = ml.Zeros([]int{
			modelArgs.MaxSequenceLength, // 32
			modelArgs.N_KVHeads,         // 32
			modelArgs.HeadDim,           // 128
		}, ml.DT_BF16)

		context.CacheV[layerIdx], _ = ml.Zeros([]int{
			modelArgs.MaxSequenceLength, // 32
			modelArgs.N_KVHeads,         // 32
			modelArgs.HeadDim,           // 128
		}, ml.DT_BF16)
	}
	return context
}

package model

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
)

type InferenceContext struct {
	SequenceLength int // context size used during inference

	CacheK []*ml.Tensor
	CacheV []*ml.Tensor

	logFn func(format string, v ...any)
}

func NewInferenceContext(model *Model, inferenceArgs common.InferenceArgs, logFn func(format string, v ...any)) *InferenceContext {
	// See: https://github.com/ggerganov/llama.cpp/blob/a7aee47b98e45539d491071b25778b833b77e387/llama.cpp#L9304C14-L9304C14
	context := &InferenceContext{
		logFn: logFn,
	}
	if inferenceArgs.SequenceLength > 0 {
		context.SequenceLength = inferenceArgs.SequenceLength
	} else {
		context.SequenceLength = model.ModelArgs.MaxSequenceLength
	}

	modelArgs := model.ModelArgs
	context.CacheK = make([]*ml.Tensor, modelArgs.N_Layers)
	context.CacheV = make([]*ml.Tensor, modelArgs.N_Layers)
	for layerIdx := 0; layerIdx < modelArgs.N_Layers; layerIdx++ {
		context.CacheK[layerIdx], _ = ml.Zeros([]int{
			inferenceArgs.SequenceLength, // specified argument value (default 4096)
			modelArgs.N_KVHeads,          // 32
			modelArgs.HeadDim,            // 128
		}, ml.DT_BF16)

		context.CacheV[layerIdx], _ = ml.Zeros([]int{
			inferenceArgs.SequenceLength, // specified argument value (default 4096)
			modelArgs.N_KVHeads,          // 32
			modelArgs.HeadDim,            // 128
		}, ml.DT_BF16)
	}
	common.GLogger.DebugPrintf("Inference Context created with SequenceLength: %d", context.SequenceLength)
	return context
}

func (ic *InferenceContext) Logf(format string, v ...any) {
	if ic.logFn != nil {
		ic.logFn(format, v...)
	}
}

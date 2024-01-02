package model

import (
	"math/rand"
	"time"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
)

type InferenceContext struct {
	SequenceLength int // context size used during inference

	randomNumberGenerator *rand.Rand
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

	return context
}

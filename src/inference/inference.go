package inference

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

type InferenceEngine struct {
	model         *model.Model
	inferenceArgs common.InferenceArgs
}

func NewInferenceEngine(model *model.Model, inferenceArgs common.InferenceArgs) *InferenceEngine {
	return &InferenceEngine{
		model:         model,
		inferenceArgs: inferenceArgs,
	}
}

func (ie *InferenceEngine) Generate(promptTokens []model.TokenId) ([]model.TokenId, error) {
	context := ie.CreateInferenceContext()

	inputTokens, err := ml.Full([]int{context.SequenceLength}, ml.DT_UINT16, uint16(ie.model.Vocabulary.PadId))
	if err != nil {
		return nil, err
	}
	for i, token := range promptTokens {
		if err := inputTokens.SetItem([]int{i}, uint16(token)); err != nil {
			return nil, err
		}
	}
	prevPos := 0
	minPromptLength := len(promptTokens)
	for curPos := minPromptLength; curPos < context.SequenceLength; curPos++ {
		inputTokensSlice, err := inputTokens.Slice([]int{prevPos}, []int{curPos})
		if err != nil {
			return nil, err
		}
		logits, err := ie.model.Transformer.Forward(context, inputTokensSlice, prevPos)
		if err != nil {
			return nil, err
		}
		logits = logits
	}
	return nil, nil
}

func (ie *InferenceEngine) CreateInferenceContext() *model.InferenceContext {
	return model.NewInferenceContext(ie.model, ie.inferenceArgs)
}

package inference

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
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
	context := model.NewInferenceContext(ie.model, ie.inferenceArgs)

	sequence, minPromptLength, sequenceLength := ie.createTokenSequence(context, promptTokens)
	prevPos := 0
	for curPos := minPromptLength; curPos < sequenceLength; curPos++ {
		input := make([]model.TokenId, len(sequence))
		input = sequence[prevPos:curPos]

		/*logits := */
		_, err := ie.model.Transformer.Forward(context, input, prevPos)
		if err != nil {
			return nil, err
		}
	}
	return nil, nil
}

func (ie *InferenceEngine) createInferenceContext() *model.InferenceContext {
	return &model.InferenceContext{}
}

func (ie *InferenceEngine) createTokenSequence(context *model.InferenceContext, promptTokens []model.TokenId) (result []model.TokenId, minBatchLength int, sequenceLength int) {
	// This part was planned to process token batches (multiple prompts at a time), but currently reverted.
	sequenceLength = context.SequenceLength
	minBatchLength = len(promptTokens)
	batch := createEmptyTokenBatch(sequenceLength, ie.model.Vocabulary.PadId)
	copy(batch[:len(promptTokens)], promptTokens)
	result = batch
	return
}

func createEmptyTokenBatch(sequenceLength int, padId model.TokenId) []model.TokenId {
	result := make([]model.TokenId, sequenceLength)
	for i := 0; i < sequenceLength; i++ {
		result[i] = padId
	}
	return result
}

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

	tokens, err := ml.Full([]int{context.SequenceLength}, ml.DT_INT32, int32(ie.model.Vocabulary.PadId))
	if err != nil {
		return nil, err
	}
	for i, token := range promptTokens {
		if err := tokens.SetItem([]int{i}, int32(token)); err != nil {
			return nil, err
		}
	}
	prevPos := 0
	minPromptLength := len(promptTokens)
	for curPos := minPromptLength; curPos < context.SequenceLength; curPos++ {
		inputTokensSlice, err := tokens.Slice([]int{prevPos}, []int{curPos})
		if err != nil {
			return nil, err
		}
		logits, err := ie.model.Transformer.Forward(context, inputTokensSlice, prevPos)
		if err != nil {
			return nil, err
		}
		if logits, err = logits.Slice([]int{logits.Size[0] - 1}, []int{logits.Size[0]}); err != nil {
			return nil, err
		}
		nextToken, err := ml.Argmax(logits, len(logits.Size)-1) // shape=[1,1] dtype=DT_INT32
		if err != nil {
			return nil, err
		}
		nextTokenId := model.TokenId(nextToken.Item().(int32))
		// Comment in original Python code: only replace token if prompt has already been generated
		existingToken, err := tokens.GetItem([]int{curPos})
		if err != nil {
			return nil, err
		}
		existingTokenId := model.TokenId(existingToken.(int32))
		if existingTokenId != ie.model.Vocabulary.PadId {
			nextTokenId = existingTokenId
		}
		if err = tokens.SetItem([]int{curPos}, int32(nextTokenId)); err != nil {
			return nil, err
		}
		eosReached := nextTokenId == ie.model.Vocabulary.EndOfSentenceId
		prevPos = curPos
		if eosReached {
			break
		}
	}
	outputTokenIds := make([]model.TokenId, 0)
	for i := minPromptLength; i < context.SequenceLength; i++ {
		tokenItem, err := tokens.GetItem([]int{i})
		if err != nil {
			return nil, err
		}
		outputTokenIds = append(outputTokenIds, model.TokenId(tokenItem.(int32)))
	}

	return outputTokenIds, nil
}

func (ie *InferenceEngine) CreateInferenceContext() *model.InferenceContext {
	return model.NewInferenceContext(ie.model, ie.inferenceArgs)
}

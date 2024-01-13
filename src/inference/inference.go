package inference

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

type InferenceEngine struct {
	model         *model.Model
	inferenceArgs common.InferenceArgs
	logFn         func(format string, v ...any)
}

func NewInferenceEngine(model *model.Model, inferenceArgs common.InferenceArgs, logFn func(format string, v ...any)) *InferenceEngine {
	return &InferenceEngine{
		model:         model,
		inferenceArgs: inferenceArgs,
		logFn:         logFn,
	}
}

func (ie *InferenceEngine) Generate(promptTokens []model.TokenId) (<-chan model.TokenId, <-chan error) {
	// See: https://betterprogramming.pub/writing-a-stream-api-in-go-afbc3c4350e2
	generatedTokensCh := make(chan model.TokenId)
	errorCh := make(chan error)
	go func() {
		defer func() {
			close(errorCh)
			close(generatedTokensCh)
		}()
		ie.generateInternal(promptTokens, generatedTokensCh, errorCh)
	}()
	return generatedTokensCh, errorCh
}

func (ie *InferenceEngine) generateInternal(promptTokens []model.TokenId, generatedTokensCh chan<- model.TokenId, errorCh chan<- error) {
	context := ie.CreateInferenceContext()

	tokens, err := ml.Full([]int{context.SequenceLength}, ml.DT_INT32, int32(ie.model.Vocabulary.PadId))
	if err != nil {
		errorCh <- err
		return
	}
	for i, token := range promptTokens {
		if err := tokens.SetItem([]int{i}, int32(token)); err != nil {
			errorCh <- err
			return
		}
	}
	prevPos := 0
	minPromptLength := len(promptTokens)
	for curPos := minPromptLength; curPos < context.SequenceLength; curPos++ {
		inputTokensSlice, err := tokens.Slice([]int{prevPos}, []int{curPos})
		if err != nil {
			errorCh <- err
			return
		}
		logits, err := ie.model.Transformer.Forward(context, inputTokensSlice, prevPos)
		if err != nil {
			errorCh <- err
			return
		}
		if logits, err = logits.Slice([]int{logits.Size[0] - 1}, []int{logits.Size[0]}); err != nil {
			errorCh <- err
			return
		}
		nextToken, err := ml.Argmax(logits, len(logits.Size)-1) // shape=[1,1] dtype=DT_INT32
		if err != nil {
			errorCh <- err
			return
		}
		nextTokenId := model.TokenId(nextToken.Item().(int32))
		// Comment in original Python code: only replace token if prompt has already been generated
		existingToken, err := tokens.GetItem([]int{curPos})
		if err != nil {
			errorCh <- err
			return
		}
		existingTokenId := model.TokenId(existingToken.(int32))
		if existingTokenId != ie.model.Vocabulary.PadId {
			nextTokenId = existingTokenId
		}
		if err = tokens.SetItem([]int{curPos}, int32(nextTokenId)); err != nil {
			errorCh <- err
			return
		}
		generatedTokensCh <- nextTokenId
		eosReached := nextTokenId == ie.model.Vocabulary.EndOfSentenceId
		prevPos = curPos
		if eosReached {
			break
		}
	}
}

func (ie *InferenceEngine) CreateInferenceContext() *model.InferenceContext {
	return model.NewInferenceContext(ie.model, ie.inferenceArgs, ie.logFn)
}

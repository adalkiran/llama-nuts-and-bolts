package inference

import (
	"fmt"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
)

type GenerationState byte

const (
	GSInProgress               GenerationState = 1
	GSFinishedByReachingEOS    GenerationState = 2
	GSFinishedByReachingSeqLen GenerationState = 3
)

type GeneratedPart struct {
	DecodedString     string
	TokenId           model.TokenId
	Token             sentencepiece.SentencePiece
	AddedToWaiting    bool
	IsResendOfWaiting bool
	GenerationState   GenerationState
}

type generationStepResult[T any] struct {
	state GenerationState
	value T
}

type InferenceEngine struct {
	model         *model.Model
	inferenceArgs common.InferenceArgs
	logFn         func(format string, v ...any)
}

type TokenGeneratorFn = func(promptTokens []model.TokenId, generatedTokensCh chan<- generationStepResult[model.TokenId], errorCh chan<- error)

func NewInferenceEngine(model *model.Model, inferenceArgs common.InferenceArgs, logFn func(format string, v ...any)) *InferenceEngine {
	return &InferenceEngine{
		model:         model,
		inferenceArgs: inferenceArgs,
		logFn:         logFn,
	}
}

func (ie *InferenceEngine) GenerateString(promptTokens []model.TokenId) (<-chan GeneratedPart, <-chan error) {
	return ie.GenerateStringGeneric(promptTokens, ie.generateTokensInternal)
}

func (ie *InferenceEngine) GenerateStringFromOutputTokens(outputTokens []model.TokenId) (<-chan GeneratedPart, <-chan error) {
	return ie.GenerateStringGeneric([]model.TokenId{}, func(promptTokens []model.TokenId, generatedTokensCh chan<- generationStepResult[model.TokenId], errorCh chan<- error) {
		bytes := []byte{0xF0, 0x9F, 0x87, 0xB9, 0xF0, 0x9F, 0x87, 0xB7}
		for _, b := range bytes {
			nextTokenId := ie.model.Vocabulary.TokenToId[fmt.Sprintf("<0x%02X>", b)]
			generatedTokensCh <- generationStepResult[model.TokenId]{
				state: GSInProgress,
				value: nextTokenId,
			}
		}
	})
}

func (ie *InferenceEngine) GenerateStringGeneric(promptTokens []model.TokenId, tokenGeneratorFn TokenGeneratorFn) (<-chan GeneratedPart, <-chan error) {
	// See: https://betterprogramming.pub/writing-a-stream-api-in-go-afbc3c4350e2
	outputCh := make(chan GeneratedPart, 1)
	outputErrorCh := make(chan error)

	go func() {
		defer func() {
			close(outputCh)
			close(outputErrorCh)
		}()
		ie.generateStringInternal(promptTokens, outputCh, outputErrorCh, tokenGeneratorFn)
	}()
	return outputCh, outputErrorCh
}

func (ie *InferenceEngine) generateStringInternal(promptTokens []model.TokenId, outputCh chan<- GeneratedPart, outputErrorCh chan<- error, tokenGeneratorFn TokenGeneratorFn) {
	generatedWaitingBytes := make([]byte, 0)
	generatedWaitingParts := make([]GeneratedPart, 0)
	lastGenerationState := GSInProgress

	generatedTokensCh, errorCh := ie.GenerateTokensGeneric(promptTokens, tokenGeneratorFn)
	loop := true
	for loop {
		select {
		case generatedTokenIdResult, ok := <-generatedTokensCh:
			if !ok {
				loop = false
				break
			}
			generatedToken, generatedTokenStr, addedToWaiting := ie.TokenToString(generatedTokenIdResult.value, &generatedWaitingBytes)
			common.GLogger.DebugPrintf("Generated token string: \"%s\", addedToWaiting: %v, details: %s", generatedTokenStr, addedToWaiting, ie.TokenBatchToDebugString([]model.TokenId{generatedTokenIdResult.value}))
			result := GeneratedPart{
				TokenId:         generatedTokenIdResult.value,
				Token:           generatedToken,
				DecodedString:   generatedTokenStr,
				AddedToWaiting:  addedToWaiting,
				GenerationState: GSInProgress,
			}
			if generatedTokenIdResult.state != GSInProgress && len(generatedWaitingParts) == 0 {
				result.GenerationState = generatedTokenIdResult.state
			}
			lastGenerationState = generatedTokenIdResult.state
			if result.AddedToWaiting {
				generatedWaitingParts = append(generatedWaitingParts, result)
			} else {
				if len(generatedWaitingParts) > 0 {
					generatedWaitingParts = make([]GeneratedPart, 0)
				}
			}
			outputCh <- result
		case err, ok := <-errorCh:
			if !ok || err == nil {
				continue
			}
			outputErrorCh <- err
			return
		}
	}
	if len(generatedWaitingParts) > 0 {
		for i, waitingPart := range generatedWaitingParts {
			result := GeneratedPart{
				TokenId:           waitingPart.TokenId,
				Token:             waitingPart.Token,
				DecodedString:     fmt.Sprintf("<0x%02X>", waitingPart.Token.ByteFallback),
				AddedToWaiting:    false,
				IsResendOfWaiting: true,
				GenerationState:   GSInProgress,
			}
			if i+1 == len(generatedWaitingParts) {
				result.GenerationState = lastGenerationState
			}
			outputCh <- result
		}
	}
}

func (ie *InferenceEngine) GenerateTokensGeneric(promptTokens []model.TokenId, tokenGeneratorFn TokenGeneratorFn) (<-chan generationStepResult[model.TokenId], <-chan error) {
	// See: https://betterprogramming.pub/writing-a-stream-api-in-go-afbc3c4350e2
	generatedTokensCh := make(chan generationStepResult[model.TokenId])
	errorCh := make(chan error)
	go func() {
		defer func() {
			close(errorCh)
			close(generatedTokensCh)
		}()
		tokenGeneratorFn(promptTokens, generatedTokensCh, errorCh)
	}()
	return generatedTokensCh, errorCh
}

func (ie *InferenceEngine) generateTokensInternal(promptTokens []model.TokenId, generatedTokensCh chan<- generationStepResult[model.TokenId], errorCh chan<- error) {
	infContext := ie.CreateInferenceContext()
	promptLength := len(promptTokens)
	if promptLength >= infContext.SequenceLength {
		errorCh <- fmt.Errorf("context SequenceLength %d must be higher than prompt tokens length %d", infContext.SequenceLength, promptLength)
		return
	}

	tokens, err := ml.Full([]int{infContext.SequenceLength}, ml.DT_INT32, int32(ie.model.Vocabulary.PadId))
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
	common.GLogger.DebugPrintf("Created input tokens tensor: shape(%v)", tokens.Size)
	prevPos := 0
	for curPos := promptLength; curPos < infContext.SequenceLength; curPos++ {
		inputTokensSlice, err := tokens.Slice([]int{prevPos}, []int{curPos})
		if err != nil {
			errorCh <- err
			return
		}
		common.GLogger.DebugPrintf("=======================================\n\n")
		common.GLogger.DebugPrintf("Running Transformer.Forward for curPos: %d, prevPos: %d, inputTokensSlice: shape(%v)", curPos, prevPos, inputTokensSlice.Size)
		logits, err := ie.model.Transformer.Forward(infContext, inputTokensSlice, prevPos)
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
		common.GLogger.DebugPrintf("Generated token for curPos: %d, prevPos: %d, token id: %d", curPos, prevPos, nextTokenId)

		eosReached := nextTokenId == ie.model.Vocabulary.EndOfSentenceId
		prevPos = curPos
		if eosReached {
			generatedTokensCh <- generationStepResult[model.TokenId]{
				state: GSFinishedByReachingEOS,
				value: nextTokenId,
			}
			break
		}
		if curPos+1 == infContext.SequenceLength {
			generatedTokensCh <- generationStepResult[model.TokenId]{
				state: GSFinishedByReachingSeqLen,
				value: nextTokenId,
			}
			break
		}
		generatedTokensCh <- generationStepResult[model.TokenId]{
			state: GSInProgress,
			value: nextTokenId,
		}
	}
}

func (ie *InferenceEngine) CreateInferenceContext() *model.InferenceContext {
	return model.NewInferenceContext(ie.model, ie.inferenceArgs, ie.logFn)
}

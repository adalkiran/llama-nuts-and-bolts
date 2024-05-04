package inference

import (
	"fmt"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

type GenerationState byte

const (
	GSInProgress               GenerationState = 1
	GSFinishedByReachingEOS    GenerationState = 2
	GSFinishedByReachingSeqLen GenerationState = 3
)

type GeneratedPart struct {
	DecodedString        string
	TokenId              model.TokenId
	Token                model.TokenPiece
	AddedToWaiting       bool
	WaitingRunesExtraStr string
	IsResendOfWaiting    bool
	GenerationState      GenerationState
}

type generationStepResult[T any] struct {
	state GenerationState
	value T
}

type generationDecodingContext struct {
	waitingBytes         []byte
	waitingParts         []GeneratedPart
	waitingRunes         string
	waitingRunesExtraStr string
	decodingFinished     bool
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
		for _, outputToken := range outputTokens {
			generatedTokensCh <- generationStepResult[model.TokenId]{
				state: GSInProgress,
				value: outputToken,
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
	decodingContext := &generationDecodingContext{
		waitingBytes: make([]byte, 0),
		waitingParts: make([]GeneratedPart, 0),
	}
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
			generatedToken, generatedTokenStr, addedToWaiting := ie.TokenToString(generatedTokenIdResult.value, decodingContext)
			common.GLogger.DebugPrintf("Generated token string: \"%s\", addedToWaiting: %v, details: %s", generatedTokenStr, addedToWaiting, ie.TokenBatchToDebugString([]model.TokenId{generatedTokenIdResult.value}))
			result := GeneratedPart{
				TokenId:              generatedTokenIdResult.value,
				Token:                generatedToken,
				DecodedString:        generatedTokenStr,
				AddedToWaiting:       addedToWaiting,
				WaitingRunesExtraStr: decodingContext.waitingRunesExtraStr,
				GenerationState:      GSInProgress,
			}
			if generatedTokenIdResult.state != GSInProgress && len(decodingContext.waitingParts) == 0 {
				result.GenerationState = generatedTokenIdResult.state
			}
			lastGenerationState = generatedTokenIdResult.state
			if result.AddedToWaiting {
				decodingContext.waitingParts = append(decodingContext.waitingParts, result)
			} else {
				if len(decodingContext.waitingParts) > 0 {
					decodingContext.waitingParts = make([]GeneratedPart, 0)
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
	decodingContext.decodingFinished = true
	if len(decodingContext.waitingParts) > 0 {
		for i, waitingPart := range decodingContext.waitingParts {
			result := GeneratedPart{
				TokenId:              waitingPart.TokenId,
				Token:                waitingPart.Token,
				DecodedString:        waitingPart.Token.ByteFallbackString(),
				AddedToWaiting:       false,
				WaitingRunesExtraStr: "",
				IsResendOfWaiting:    true,
				GenerationState:      GSInProgress,
			}
			if len(decodingContext.waitingRunesExtraStr) > 0 {
				result.DecodedString = decodingContext.waitingRunesExtraStr + result.DecodedString
				decodingContext.waitingRunes = ""
				decodingContext.waitingRunesExtraStr = ""
			}
			if i+1 == len(decodingContext.waitingParts) {
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

		_, eosReached := ie.model.Vocabulary.StopTokenIds[nextTokenId]
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

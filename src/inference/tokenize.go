package inference

import (
	"fmt"
	"math"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

const (
	B_TXT    = "<|begin_of_text|>"
	B_HEADER = "<|start_header_id|>"
	E_HEADER = "<|end_header_id|>"
	E_TURN   = "<|eot_id|>"
)

type PromptPart struct {
	Header          string
	Content         string
	IsLastAssistant bool
}

func (ie *InferenceEngine) Tokenize(promptParts []PromptPart) ([]model.TokenId, error) {
	result := make([]model.TokenId, 0)
	text := ""
	vocabulary := ie.model.Vocabulary

	// <|begin_of_text|>
	result = append(result, vocabulary.TokenToId[B_TXT])
	text += B_TXT

	promptParts = append(promptParts, PromptPart{
		Header:          "assistant",
		Content:         "",
		IsLastAssistant: true,
	})

	for _, promptPart := range promptParts {
		// <|start_header_id|>
		result = append(result, vocabulary.TokenToId[B_HEADER])
		text += B_HEADER

		partTokens, err := ie.TokenizeString(promptPart.Header, false)
		if err != nil {
			return nil, err
		}
		result = append(result, partTokens...)
		text += promptPart.Header

		// <|end_header_id|>
		result = append(result, vocabulary.TokenToId[E_HEADER])
		text += E_HEADER

		// \n\n
		partTokens, err = ie.TokenizeString("\n\n", false)
		if err != nil {
			return nil, err
		}
		result = append(result, partTokens...)
		text += "\n\n"

		// content
		partTokens, err = ie.TokenizeString(promptPart.Content, false)
		if err != nil {
			return nil, err
		}
		result = append(result, partTokens...)
		text += promptPart.Content

		if !promptPart.IsLastAssistant {
			// <|eot_id|>
			result = append(result, vocabulary.TokenToId[E_TURN])
			text += E_TURN
		}
	}

	common.GLogger.DebugPrintf("Tokenizing prompt: \"%s\"", text)

	/*
			if addBeginOfSentence && vocabulary.BeginOfSentenceId != -1 {
				result = append(result, vocabulary.BeginOfSentenceId)
			}
		result = append(result, separatePieces(text, vocabulary)...)
	*/
	common.GLogger.DebugPrintf("Prompt token ids: \"%v\"", result)
	common.GLogger.DebugPrintf("Prompt tokens: \"%v\"", ie.TokenBatchToDebugString(result))
	return result, nil
}

func (ie *InferenceEngine) TokenizeBatch(prompts [][]PromptPart) (result [][]model.TokenId, err error) {
	result = make([][]model.TokenId, len(prompts))
	for i, promptParts := range prompts {
		tokenIds, err := ie.Tokenize(promptParts)
		if err != nil {
			return nil, err
		}
		result[i] = tokenIds
	}
	return result, nil
}

func (ie *InferenceEngine) bytePairMerge(piece string, ranks map[string]model.TokenId) ([]model.TokenId, []string, error) {
	// Ported from Tiktoken Rust code
	// See: https://github.com/openai/tiktoken/blob/1b9faf2779855124f05174adf1383e53689ed94b/src/lib.rs
	type rankTuple struct {
		rank model.TokenId
		idx  int
	}
	parts := make([]rankTuple, len(piece)+1)
	min_rank := rankTuple{rank: math.MaxInt32, idx: math.MaxInt32}
	for i := 0; i < len(piece)-1; i++ {
		var rank model.TokenId
		var ok bool
		if i+1 < len(piece) {
			if rank, ok = ranks[piece[i:i+2]]; !ok {
				rank = math.MaxInt32
			}
		} else {
			rank = math.MaxInt32
		}
		if rank < min_rank.rank {
			min_rank = rankTuple{rank: rank, idx: i}
		}
		parts[i] = rankTuple{rank: rank, idx: i}
	}
	parts[len(piece)-1] = rankTuple{rank: math.MaxInt32, idx: len(piece) - 1}
	parts[len(piece)] = rankTuple{rank: math.MaxInt32, idx: len(piece)}

	getRankFn := func(parts []rankTuple, i int) model.TokenId {
		var newRank model.TokenId
		var ok bool
		if i+3 < len(parts) {
			pieceToSearch := piece[parts[i].idx:parts[i+3].idx]
			if newRank, ok = ranks[pieceToSearch]; !ok {
				newRank = math.MaxInt32
			}
		} else {
			newRank = math.MaxInt32
		}
		return newRank
	}

	for min_rank.rank != math.MaxInt32 {
		i := min_rank.idx
		// Update parts[i] and parts[i - 1] before removing parts[i + 1], since
		// `parts.remove(i + 1)` will thrash the cache.
		if i > 0 {
			parts[i-1].rank = getRankFn(parts, i-1)
		}
		parts[i].rank = getRankFn(parts, i)
		parts = append(parts[:i+1], parts[i+1+1:]...) // remove parts[i + 1]

		min_rank = rankTuple{rank: math.MaxInt32, idx: math.MaxInt}
		for i = 0; i < len(parts)-1; i++ {
			if parts[i].rank < min_rank.rank {
				min_rank = rankTuple{rank: parts[i].rank, idx: i}
			}
		}
	}

	splitRanks := make([]model.TokenId, 0)
	splitPieces := make([]string, 0)
	for i := 0; i < len(parts)-1; i++ {
		subPiece := piece[parts[i].idx:parts[i+1].idx]
		splitPieces = append(splitPieces, subPiece)
		splitRanks = append(splitRanks, ranks[subPiece])
	}
	return splitRanks, splitPieces, nil
}

func (ie *InferenceEngine) TokenizeString(text string, addBeginOfSentence bool) ([]model.TokenId, error) {
	vocabulary := ie.model.Vocabulary

	result := make([]model.TokenId, 0)

	for _, match := range vocabulary.SplitRegexp.FindAllStringSubmatch(text, -1) {
		if token, ok := vocabulary.TokenToId[match[0]]; ok {
			result = append(result, token)
			continue
		}
		splitTokens, _, err := ie.bytePairMerge(match[0], vocabulary.TokenToId)
		if err != nil {
			return nil, err
		}
		result = append(result, splitTokens...)
	}
	return result, nil
}

func (ie *InferenceEngine) TokenToString(tokenId model.TokenId, decodingContext *generationDecodingContext) (token model.TokenPiece, resultString string, addedToWaiting bool) {
	vocabulary := ie.model.Vocabulary
	if tokenId < 0 || int(tokenId) >= len(vocabulary.IdToToken) {
		return model.TokenPiece{Piece: "<UNKNOWN>"}, "not used anymore", false
	}
	token = model.TokenPiece{
		Piece: vocabulary.IdToToken[tokenId],
		Rank:  int32(tokenId),
	}
	if !utf8.Valid([]byte(token.Piece)) {
		token.ByteFallback = []byte(token.Piece)
		token.IsByte = true
	} else {
		r, rsize := utf8.DecodeRune([]byte(token.Piece))
		if rsize == len(token.Piece) && rsize > 2 && (unicode.IsMark(r) || r == zwj) {
			token.ByteFallback = []byte(token.Piece)
			token.IsByte = true
		}
	}
	if len(decodingContext.waitingRunesExtraStr) > 0 && !token.IsByte {
		resultString = decodingContext.waitingRunesExtraStr
		decodingContext.waitingRunes = ""
		decodingContext.waitingRunesExtraStr = ""
	}

	if token.IsByte {
		if decodingContext.waitingBytes == nil {
			decodingContext.waitingBytes = make([]byte, 0)
		}
		decodingContext.waitingBytes = append(decodingContext.waitingBytes, token.ByteFallback...)
		if utf8.Valid(decodingContext.waitingBytes) {
			r, rsize := utf8.DecodeRune(decodingContext.waitingBytes)
			decodingContext.waitingBytes = decodingContext.waitingBytes[rsize:]
			resultString += processEmoji(decodingContext, r)
		} else {
			addedToWaiting = true
		}
		return
	} else {
		resultString += token.Piece
		return
	}
}

func (ie *InferenceEngine) TokenBatchToString(tokenIdBatch []model.TokenId) ([]model.TokenPiece, string) {
	decodingContext := &generationDecodingContext{
		waitingBytes: make([]byte, 0),
		waitingParts: make([]GeneratedPart, 0),
	}
	resultTokens := make([]model.TokenPiece, 0)
	resultStr := ""
	for _, tokenId := range tokenIdBatch {
		if tokenId == ie.model.Vocabulary.PadId {
			break
		}
		token, tokenStr, addedToWaiting := ie.TokenToString(tokenId, decodingContext)
		resultTokens = append(resultTokens, token)
		if !addedToWaiting {
			resultStr += tokenStr
		}
	}
	return resultTokens, resultStr
}

func (ie *InferenceEngine) TokenBatchToDebugString(tokenIdBatch []model.TokenId) string {
	vocabulary := ie.model.Vocabulary
	resultStrArray := make([]string, 0)
	for _, tokenId := range tokenIdBatch {
		if tokenId == ie.model.Vocabulary.PadId {
			break
		}
		if tokenId < 0 || int(tokenId) >= len(vocabulary.IdToToken) {
			resultStrArray = append(resultStrArray, fmt.Sprintf("[id: %d, UNKNOWN ID]", tokenId))
		}
		token := vocabulary.IdToToken[tokenId]
		resultStrArray = append(resultStrArray, fmt.Sprintf("[id: %d, %s]", tokenId, token))
	}
	return strings.Join(resultStrArray, ", ")
}

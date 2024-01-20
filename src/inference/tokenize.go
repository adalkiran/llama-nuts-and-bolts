package inference

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
)

const (
	whitespaceEscapeToken = "\xe2\x96\x81"
	unknownOutputToken    = "\xe2\x96\x85"
)

func (ie *InferenceEngine) Tokenize(text string, addBeginOfSentence bool) ([]model.TokenId, error) {
	common.GLogger.DebugPrintf("Tokenizing prompt: \"%s\", addBeginOfSentence: %v", text, addBeginOfSentence)
	result := make([]model.TokenId, 0)
	vocabulary := ie.model.Vocabulary

	text = " " + text
	text = escapeWhitespace(text)

	if addBeginOfSentence && vocabulary.BeginOfSentenceId != -1 {
		result = append(result, vocabulary.BeginOfSentenceId)
	}
	result = append(result, separatePieces(text, vocabulary)...)
	common.GLogger.DebugPrintf("Prompt token ids: \"%v\"", result)
	common.GLogger.DebugPrintf("Prompt tokens: \"%v\"", ie.TokenBatchToDebugString(result))
	return result, nil
}

func (ie *InferenceEngine) TokenizeBatch(texts []string, addBeginOfSentence bool) (result [][]model.TokenId, err error) {
	result = make([][]model.TokenId, len(texts))
	for i, text := range texts {
		tokenIds, err := ie.Tokenize(text, addBeginOfSentence)
		if err != nil {
			return nil, err
		}
		result[i] = tokenIds
	}
	return result, nil
}

func (ie *InferenceEngine) TokenToString(tokenId model.TokenId, waitingBytes *[]byte) (token sentencepiece.SentencePiece, resultString string, addedToWaiting bool) {
	vocabulary := ie.model.Vocabulary
	if tokenId < 0 || int(tokenId) >= len(vocabulary.IdToToken) {
		return sentencepiece.SentencePiece{PieceType: sentencepiece.UNKNOWN}, unknownOutputToken, false
	}
	token = vocabulary.IdToToken[tokenId]
	switch token.PieceType {
	case sentencepiece.CONTROL:
		// Do nothing
	case sentencepiece.BYTE:
		if waitingBytes == nil {
			*waitingBytes = make([]byte, 0)
		}
		*waitingBytes = append(*waitingBytes, token.ByteFallback)
		if utf8.Valid(*waitingBytes) {
			r, rsize := utf8.DecodeRune(*waitingBytes)
			*waitingBytes = (*waitingBytes)[rsize:]
			resultString = unescapeWhitespace(string(r))
		} else {
			addedToWaiting = true
		}
		return
	case sentencepiece.NORMAL:
		resultString = unescapeWhitespace(token.Piece)
		return
	}
	return token, "", false
}

func (ie *InferenceEngine) TokenBatchToString(tokenIdBatch []model.TokenId) ([]sentencepiece.SentencePiece, string) {
	resultTokens := make([]sentencepiece.SentencePiece, 0)
	resultStr := ""
	generatedWaitingBytes := make([]byte, 0)
	for _, tokenId := range tokenIdBatch {
		if tokenId == ie.model.Vocabulary.PadId {
			break
		}
		token, tokenStr, addedToWaiting := ie.TokenToString(tokenId, &generatedWaitingBytes)
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
		resultStrArray = append(resultStrArray, fmt.Sprintf("[id: %d, %s]", tokenId, token.String()))
	}
	return strings.Join(resultStrArray, ", ")
}

func separatePieces(text string, vocabulary *model.Vocabulary) []model.TokenId {
	tokens := make([]string, 0)

	for len(text) > 0 {
		// Find the longest matching subword unit
		var matchedSubword string
		incrementCursor := 0
		for i := 1; i <= utf8.RuneCountInString(text); i++ {
			subword := string([]rune(text)[:i])
			if _, ok := vocabulary.TokenToId[subword]; ok {
				matchedSubword = subword
				incrementCursor = len(matchedSubword)
			} else if len(subword) == 1 {
				r := []rune(text)[:i][0]
				if utf8.RuneLen(r) == 1 {
					matchedSubword = fmt.Sprintf("<0x%02X>", r)
					incrementCursor = 1 // 1-byte
					break
				}
			} else {
				continue
			}
		}

		// If no matching subword is found, treat it as an unknown token
		if matchedSubword == "" {
			matchedSubword = "<unk>"
			incrementCursor = 1 // skip 1-byte
		}

		// Add the matched subword to the list of tokens
		tokens = append(tokens, matchedSubword)

		// Move to the next unmatched part of the input
		text = text[incrementCursor:]
	}

	result := make([]model.TokenId, len(tokens))

	for i, token := range tokens {
		if id, ok := vocabulary.TokenToId[token]; ok {
			result[i] = id
		} else {
			result[i] = vocabulary.UnknownId
		}
	}

	return result
}

func escapeWhitespace(text string) string {
	return strings.ReplaceAll(text, " ", whitespaceEscapeToken)
}

func unescapeWhitespace(text string) string {
	return strings.ReplaceAll(text, whitespaceEscapeToken, " ")
}

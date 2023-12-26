package inference

import (
	"strings"
	"unicode/utf8"

	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
)

const (
	whitespaceEscapeToken = "\xe2\x96\x81"
	unknownOutputToken    = "\xe2\x96\x85"
)

func (ie *InferenceEngine) tokenize(text string, addBeginOfSentence bool) ([]model.TokenId, error) {
	result := make([]model.TokenId, 0)
	vocabulary := ie.context.model.Vocabulary

	text = " " + text
	text = escapeWhitespace(text)

	if addBeginOfSentence && vocabulary.BeginOfSentenceId != -1 {
		result = append(result, vocabulary.BeginOfSentenceId)
	}
	result = append(result, separatePieces(text, vocabulary)...)

	return result, nil
}

func (ie *InferenceEngine) TokenizeBatch(texts []string, addBeginOfSentence bool) (result [][]model.TokenId, err error) {
	result = make([][]model.TokenId, len(texts))
	for i, text := range texts {
		tokenIds, err := ie.tokenize(text, addBeginOfSentence)
		if err != nil {
			return nil, err
		}
		result[i] = tokenIds
	}
	return result, nil
}

func (ie *InferenceEngine) TokenToString(tokenId model.TokenId) string {
	vocabulary := ie.context.model.Vocabulary
	if tokenId < 0 || int(tokenId) >= len(vocabulary.IdToToken) {
		return unknownOutputToken
	}
	token := vocabulary.IdToToken[tokenId]
	switch token.PieceType {
	case sentencepiece.CONTROL:
		// Do nothing
	case sentencepiece.NORMAL:
		return unescapeWhitespace(token.Piece)
	}
	return ""
}

func (ie *InferenceEngine) TokenBatchToString(tokenIdBatch []model.TokenId) string {
	result := ""
	for _, tokenId := range tokenIdBatch {
		if tokenId == ie.context.model.Vocabulary.PadId {
			break
		}
		result += ie.TokenToString(tokenId)
	}
	return result
}

func separatePieces(text string, vocabulary *model.Vocabulary) []model.TokenId {
	tokens := make([]string, 0)

	for len(text) > 0 {
		// Find the longest matching subword unit
		var matchedSubword string
		for i := 1; i <= utf8.RuneCountInString(text); i++ {
			subword := string([]rune(text)[:i])
			if _, ok := vocabulary.TokenToId[subword]; ok {
				matchedSubword = subword
			} else {
				break
			}
		}

		// If no matching subword is found, treat it as an unknown token
		if matchedSubword == "" {
			matchedSubword = "<unk>"
		}

		// Add the matched subword to the list of tokens
		tokens = append(tokens, matchedSubword)

		// Move to the next unmatched part of the input
		text = text[len(matchedSubword):]
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

/*
	func separateRunes(text string) []string {
		result := make([]string, 0)
		for offset := 0; offset < len(text); {
			r, size := utf8.DecodeRuneInString(text)
			text = text[size:]
			result = append(result, string(r))
		}
		return result
	}

	func separateBigram(symbols []string, vocabulary *Vocabulary) []sentencepiece.SentencePiece {
		queue := make([]sentencepiece.SentencePiece, 0)

		for i := 1; i < len(symbols); i++ {

			bigramToken := strings.Join(symbols[i-1:i+1], "")
			if id, ok := vocabulary.tokenToId[bigramToken]; ok {
				piece := vocabulary.idToToken[id]
				queue = append(queue, piece)
			}
		}
		return queue
	}
*/
func escapeWhitespace(text string) string {
	return strings.ReplaceAll(text, " ", whitespaceEscapeToken)
}

func unescapeWhitespace(text string) string {
	return strings.ReplaceAll(text, whitespaceEscapeToken, " ")
}

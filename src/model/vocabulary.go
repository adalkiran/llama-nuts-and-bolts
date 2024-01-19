package model

import (
	"encoding/hex"
	"regexp"

	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
)

const (
	unknownToken          = "<unk>"
	beginOfSentenceToken  = "<s>"
	endOfSentenceToken    = "</s>"
	whitespaceEscapeToken = "\xe2\x96\x81"
	unknownOutputToken    = "\xe2\x96\x85"
)

var extractHexadecimalPiece = *regexp.MustCompile(`<0x(\w+)>`)

type Vocabulary struct {
	TokenToId map[string]TokenId
	IdToToken []sentencepiece.SentencePiece

	BeginOfSentenceId TokenId
	EndOfSentenceId   TokenId
	UnknownId         TokenId
	PadId             TokenId
}

func NewVocabulary(vocabModelProto *sentencepiece.ModelProto) *Vocabulary {
	result := &Vocabulary{
		TokenToId:         make(map[string]TokenId, len(*vocabModelProto.Pieces)),
		IdToToken:         *vocabModelProto.Pieces,
		UnknownId:         -1,
		BeginOfSentenceId: -1,
		EndOfSentenceId:   -1,
		PadId:             -1,
	}
	for i, token := range result.IdToToken {
		match := extractHexadecimalPiece.FindStringSubmatch(token.Piece)

		if len(match) >= 2 {
			byteValue, err := hex.DecodeString(match[1])
			if err == nil && len(byteValue) == 1 {
				token.Piece = string(byteValue[0])
				result.IdToToken[i] = token
			}
		}
		result.TokenToId[token.Piece] = TokenId(i)
	}
	if id, ok := result.TokenToId[unknownToken]; ok {
		result.UnknownId = id
	}
	if id, ok := result.TokenToId[beginOfSentenceToken]; ok {
		result.BeginOfSentenceId = id
	}
	if id, ok := result.TokenToId[endOfSentenceToken]; ok {
		result.EndOfSentenceId = id
	}
	return result
}

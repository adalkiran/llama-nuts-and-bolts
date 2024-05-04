package model

import (
	"regexp"

	"github.com/adalkiran/llama-nuts-and-bolts/src/tiktoken"
)

type Vocabulary struct {
	TokenToId map[string]TokenId
	IdToToken []string

	BeginOfSentenceId TokenId
	EndOfSentenceId   TokenId
	UnknownId         TokenId
	PadId             TokenId
	StopTokenIds      map[TokenId]struct{}

	SplitRegexp *regexp.Regexp
}

func NewVocabulary(vocabBpe *tiktoken.ModelData) *Vocabulary {
	result := &Vocabulary{
		TokenToId:         make(map[string]TokenId, len(vocabBpe.MergeableRanks)+len(vocabBpe.SpecialTokens)),
		IdToToken:         make([]string, len(vocabBpe.MergeableRanks)+len(vocabBpe.SpecialTokens)),
		UnknownId:         TokenId(vocabBpe.UnknownId),
		BeginOfSentenceId: TokenId(vocabBpe.BeginOfSentenceId),
		EndOfSentenceId:   TokenId(vocabBpe.EndOfSentenceId),
		PadId:             TokenId(vocabBpe.PadId),
		StopTokenIds:      make(map[TokenId]struct{}, len(vocabBpe.StopTokenIds)),
		//SplitRegexp:       regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`),
		SplitRegexp: regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`),
	}
	for token, rank := range vocabBpe.MergeableRanks {
		result.TokenToId[token] = TokenId(rank)
	}
	for token, rank := range vocabBpe.SpecialTokens {
		result.TokenToId[token] = TokenId(rank)
	}
	for token, rank := range vocabBpe.MergeableRanks {
		result.IdToToken[rank] = token
	}
	for token, rank := range vocabBpe.SpecialTokens {
		result.IdToToken[rank] = token
	}
	for _, stopTokenId := range vocabBpe.StopTokenIds {
		result.StopTokenIds[TokenId(stopTokenId)] = struct{}{}
	}
	return result
}

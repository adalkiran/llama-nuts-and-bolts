package model

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
)

type Model struct {
	Layers     *pickle.PickleDict
	Config     *Config
	VocabModel *sentencepiece.ModelProto
}

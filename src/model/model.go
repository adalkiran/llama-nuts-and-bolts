package model

import (
	"github.com/adalkiran/llama-nuts-and-bolts/src/pickle"
)

type Model struct {
	Layers *pickle.PickleDict
	Config *Config
}

package model

import (
	"path/filepath"

	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
	"github.com/adalkiran/llama-nuts-and-bolts/src/torch"
)

func LoadModel(modelFilePath string) (*Model, error) {
	torchModelReader := torch.NewTorchModelReader(modelFilePath)
	modelLayers, err := torchModelReader.Load()
	if err != nil {
		return nil, err
	}
	model := &Model{Layers: modelLayers}
	err = loadConfig(modelFilePath, model)
	if err != nil {
		return nil, err
	}
	err = loadVocab(modelFilePath, model)
	if err != nil {
		return nil, err
	}
	return model, nil
}

func loadConfig(modelFilePath string, model *Model) error {
	configFilePath := filepath.Dir(modelFilePath) + "/params.json"
	config, err := loadConfigFromFile(configFilePath, model)
	if err != nil {
		return err
	}
	model.Config = config
	return nil
}

func loadVocab(modelFilePath string, model *Model) error {
	vocabFilePath := filepath.Dir(modelFilePath) + "/tokenizer.model"
	resultArr, err := sentencepiece.Load(vocabFilePath)
	if err != nil {
		return err
	}
	resultArr = resultArr //Currently not used
	return nil
}

package model

import (
	"fmt"
	"path/filepath"

	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
	"github.com/adalkiran/llama-nuts-and-bolts/src/torch"
)

func LoadModel(modelFilePath string) (*Model, error) {
	torchModelReader := torch.NewTorchModelReader(modelFilePath)
	fmt.Printf("Loading model file: \"%s\"...\n", modelFilePath)
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
	fmt.Printf("Found %d layers in the model.\n", len(model.Layers.GetKeys()))
	return model, nil
}

func loadConfig(modelFilePath string, model *Model) error {
	configFilePath := filepath.Dir(modelFilePath) + "/params.json"
	fmt.Printf("Loading model configuration file: \"%s\"...\n", configFilePath)
	config, err := loadConfigFromFile(configFilePath, model)
	if err != nil {
		return err
	}
	model.Config = config
	fmt.Printf("Model configuration: %v\n", *model.Config)
	return nil
}

func loadVocab(modelFilePath string, model *Model) error {
	vocabFilePath := filepath.Dir(modelFilePath) + "/tokenizer.model"
	fmt.Printf("Loading vocabulary/tokens file: \"%s\"...\n", vocabFilePath)
	vocabModel, err := sentencepiece.Load(vocabFilePath)
	if err != nil {
		return err
	}
	model.VocabModel = vocabModel
	fmt.Printf("Found %d tokens in the model.\n", len(*model.VocabModel.Pieces))
	return nil
}

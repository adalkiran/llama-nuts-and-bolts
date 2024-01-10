package main

import (
	"fmt"
	"log"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/inference"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

func main() {
	fmt.Println("Welcome to Llama Nuts and Bolts!")
	fmt.Print("=================================\n\n\n")
	modelFilePath := "../models-original/7B/consolidated.00.pth"

	prompt := "Hello my name is"

	llamaModel, err := model.LoadModel(modelFilePath)
	if err != nil {
		log.Fatal(err)
	}

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 8

	engine := inference.NewInferenceEngine(llamaModel, inferenceArgs)

	tokens, err := engine.Tokenize(prompt, true)
	if err != nil {
		llamaModel.Free()
		log.Fatal(err)
	}

	fmt.Println()
	fmt.Print(engine.TokenBatchToString(tokens))

	generatedTokens, err := engine.Generate(tokens)
	if err != nil {
		log.Fatal(err)
	}
	generatedTokens = generatedTokens

	llamaModel.Free()
}

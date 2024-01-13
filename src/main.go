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
	defer llamaModel.Free()

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 20

	engine := inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn)

	tokens, err := engine.Tokenize(prompt, true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println()
	fmt.Print(engine.TokenBatchToString(tokens))

	generatedTokensCh, errorCh := engine.Generate(tokens)
	for {
		select {
		case generatedToken, ok := <-generatedTokensCh:
			if !ok {
				return
			}
			fmt.Print(engine.TokenToString(generatedToken))
		case err := <-errorCh:
			log.Fatal(err)
		}
	}
}

func logFn(format string, v ...any) {
	log.Printf(format, v...)
}

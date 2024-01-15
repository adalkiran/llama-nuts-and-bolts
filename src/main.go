package main

import (
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/inference"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
)

const B_INST, E_INST = "[INST]", "[/INST]"
const esc = 27

var latestGeneratedToken sentencepiece.SentencePiece
var generatedText string
var latestLogText string
var progressText string
var startTimeTotal time.Time
var startTimeToken time.Time

func main() {
	fmt.Println("Welcome to Llama Nuts and Bolts!")
	fmt.Print("=================================\n\n\n")
	modelFilePath := "../models-original/7B-chat/consolidated.00.pth"
	isChatModel := true

	generatedText = ""
	latestLogText = ""
	progressText = fmt.Sprintf("Loading model \"%s\"...", modelFilePath)
	updateOutput()

	prompt := "You are Einstein. Describe your theory. "
	if isChatModel {
		prompt = fmt.Sprintf("%s %s %s", B_INST, strings.TrimSpace(prompt), E_INST)
	}

	llamaModel, err := model.LoadModel(modelFilePath)
	if err != nil {
		log.Fatal(err)
	}
	defer llamaModel.Free()

	fmt.Println()
	fmt.Println()
	fmt.Println()

	generatedText = ""
	latestLogText = ""
	progressText = fmt.Sprintf("Model \"%s\" was loaded, starting inference...", modelFilePath)
	updateOutput()

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 100

	engine := inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn)

	tokens, err := engine.Tokenize(prompt, true)
	if err != nil {
		log.Fatal(err)
	}

	generatedText += engine.TokenBatchToString(tokens)

	promptTokenCount := len(tokens)
	generatedTokenCount := 0

	startTimeTotal = time.Now()
	startTimeToken = startTimeTotal
	progressText = generateProgressText(promptTokenCount, 0, inferenceArgs.SequenceLength)
	updateOutput()

	generatedTokensCh, errorCh := engine.Generate(tokens)
	for {
		select {
		case generatedTokenId, ok := <-generatedTokensCh:
			if !ok {
				return
			}
			generatedTokenCount++
			generatedToken, generatedTokenStr := engine.TokenToString(generatedTokenId)
			latestGeneratedToken = generatedToken
			generatedText += generatedTokenStr
			progressText = generateProgressText(promptTokenCount, generatedTokenCount, inferenceArgs.SequenceLength)
			updateOutput()
			startTimeToken = time.Now()
		case err := <-errorCh:
			log.Fatal(err)
		}
	}
}

func logFn(format string, v ...any) {
	latestLogText = fmt.Sprintf(format, v...)
	updateOutput()
}

func generateProgressText(promptTokenCount int, generatedTokenCount int, sequenceLength int) string {
	var latestGeneratedTokenStr string
	if latestGeneratedToken.PieceType == 0 {
		latestGeneratedTokenStr = "(generating)"
	} else {
		latestGeneratedTokenStr = latestGeneratedToken.String()
	}
	return fmt.Sprintf("%c[1mGenerating tokens %d / %d, including %d prompt tokens...%c[0m Latest generated token: %s",
		esc, promptTokenCount+generatedTokenCount+1, sequenceLength, promptTokenCount, esc, latestGeneratedTokenStr)
}

func updateOutput() {
	// See: https://github.com/apoorvam/goterminal/blob/master/writer_posix.go
	lineCount := 4

	if generatedText != "" {
		for i := 0; i < lineCount; i++ {
			fmt.Printf("%c[2K\r", esc)   // Clear current line
			fmt.Printf("%c[%dA", esc, 1) // Move cursor 1 upper line
		}
		fmt.Printf("%c[2K\r", esc) // Clear current line
	} else {
		generatedText = " "
	}
	if latestLogText == "" {
		latestLogText = "..."
	}
	fmt.Println(progressText)
	fmt.Printf("Total elapsed: %c[1m%.4f sec(s)%c[0m, elapsed for next token: %c[1m%.4f sec(s)%c[0m\n", esc, time.Since(startTimeTotal).Seconds(), esc, esc, time.Since(startTimeToken).Seconds(), esc)
	fmt.Println("Running for next token: " + latestLogText)
	fmt.Println()
	fmt.Print(generatedText)
}

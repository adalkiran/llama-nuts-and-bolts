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

var appState = AppState{
	generatedTokens: make([]sentencepiece.SentencePiece, 0),
}

func main() {
	fmt.Println("Welcome to Llama Nuts and Bolts!")
	fmt.Print("=================================\n\n\n")
	modelFilePath := "../models-original/7B-chat/consolidated.00.pth"
	isChatModel := true

	appState.literalProgressText = fmt.Sprintf("Loading model \"%s\"...", modelFilePath)
	appState.updateOutput()

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

	appState.literalProgressText = fmt.Sprintf("Model \"%s\" was loaded, starting inference...", modelFilePath)
	appState.updateOutput()

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 100

	engine := inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn)

	tokens, err := engine.Tokenize(prompt, true)
	if err != nil {
		log.Fatal(err)
	}

	appState.promptTokens, appState.promptText = engine.TokenBatchToString(tokens)

	appState.startTimeTotal = time.Now()
	appState.startTimeToken = appState.startTimeTotal

	appState.sequenceLength = inferenceArgs.SequenceLength
	appState.literalProgressText = ""
	appState.updateOutput()

	generatedTokensCh, errorCh := engine.Generate(tokens)
	for {
		select {
		case generatedTokenId, ok := <-generatedTokensCh:
			if !ok {
				fmt.Println()
				return
			}
			generatedToken, generatedTokenStr := engine.TokenToString(generatedTokenId)
			appState.generatedTokens = append(appState.generatedTokens, generatedToken)
			appState.generatedText += generatedTokenStr
			appState.updateOutput()
			appState.startTimeToken = time.Now()
		case err := <-errorCh:
			if err == nil {
				continue
			}
			fmt.Println()
			log.Fatal(err)
		}
	}
}

func logFn(format string, v ...any) {
	appState.latestLogText = fmt.Sprintf(format, v...)
	appState.updateOutput()
}

type AppState struct {
	latestLogText string

	sequenceLength      int
	promptText          string
	promptTokens        []sentencepiece.SentencePiece
	generatedText       string
	literalProgressText string

	generatedTokens []sentencepiece.SentencePiece
	startTimeTotal  time.Time
	startTimeToken  time.Time
}

func (as AppState) updateOutput() {
	// See: https://github.com/apoorvam/goterminal/blob/master/writer_posix.go
	lineCount := 5

	if as.startTimeTotal.Year() > 1 {
		for i := 0; i < lineCount; i++ {
			fmt.Printf("%c[2K\r", esc)   // Clear current line
			fmt.Printf("%c[%dA", esc, 1) // Move cursor 1 upper line
		}
		fmt.Printf("%c[2K\r", esc) // Clear current line
	}
	if as.latestLogText == "" {
		as.latestLogText = "..."
	}
	elapsedTotalStr, elapsedTokenStr := as.durationsToStr()
	fmt.Println(as.generateProgressText())
	fmt.Printf("Total elapsed: %c[1m%s%c[0m, elapsed for next token: %c[1m%s%c[0m\n", esc, elapsedTotalStr, esc, esc, elapsedTokenStr, esc)
	fmt.Println("Running for next token: " + as.latestLogText)
	fmt.Println()
	if as.promptText != "" {
		generatedText := as.generatedText
		if generatedText == "" {
			generatedText = "..."
		}
		fmt.Print(fmt.Sprintf("%c[1mPrompt:%c[0m ", esc, esc) + as.promptText +
			fmt.Sprintf("\n%c[1mAssistant:%c[0m ", esc, esc) + generatedText)
	} else {
		fmt.Print("...")
	}
}

func (as AppState) generateProgressText() string {
	if as.literalProgressText != "" {
		return as.literalProgressText
	}
	var latestGeneratedTokenStr string
	if len(as.generatedTokens) == 0 {
		latestGeneratedTokenStr = "(generating)"
	} else {
		latestGeneratedTokenStr = as.generatedTokens[0].String()
	}
	nextTokenNum := len(as.promptTokens) + len(as.generatedTokens)
	if nextTokenNum < as.sequenceLength {
		nextTokenNum++
	}
	return fmt.Sprintf("%c[1mGenerating tokens %d / %d, including %d prompt tokens...%c[0m Latest generated token: %s",
		esc, nextTokenNum, as.sequenceLength, len(as.promptTokens), esc, latestGeneratedTokenStr)
}

func (as AppState) durationsToStr() (elapsedTotalStr string, elapsedTokenStr string) {
	elapsedTotalStr = "..:.."
	elapsedTokenStr = "..:.."
	if as.startTimeTotal.Year() > 1 {
		// See: https://stackoverflow.com/questions/47341278/how-to-format-a-duration
		totalElapsed := time.Since(as.startTimeTotal).Round(time.Second)
		totalElapsedHourPart := totalElapsed / time.Hour
		totalElapsed -= totalElapsedHourPart * time.Minute
		totalElapsedMinPart := totalElapsed / time.Minute
		totalElapsed -= totalElapsedMinPart * time.Minute
		totalElapsedSecPart := totalElapsed / time.Second
		elapsedTotalStr = fmt.Sprintf("%02dh:%02dm:%02ds", totalElapsedHourPart, totalElapsedMinPart, totalElapsedSecPart)

		elapsedTokenStr = fmt.Sprintf("%.4f sec(s)", time.Since(as.startTimeToken).Seconds())
	}
	return
}

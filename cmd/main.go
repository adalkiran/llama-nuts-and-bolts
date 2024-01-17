package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/inference"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
	"github.com/apoorvam/goterminal"
)

const B_INST, E_INST = "[INST]", "[/INST]"
const esc = 27

var appState = &AppState{
	generatedTokens:               make([]sentencepiece.SentencePiece, 0),
	prevLineWidths:                make([]int, 0),
	consoleMeasure:                goterminal.New(os.Stdout),
	excludeEscapeDirectivesRegexp: *regexp.MustCompile(string(rune(esc)) + "\\[\\d+[a-zA-Z]"),
}

func main() {
	exePath, err := os.Executable()
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	// Get the directory containing the executable
	exeDir := filepath.Dir(exePath)

	fmt.Println(exeDir)

	fmt.Println("Welcome to Llama Nuts and Bolts!")
	fmt.Print("=================================\n\n\n")
	modelFilePath := "../models-original/7B-chat/consolidated.00.pth"
	isChatModel := true

	appState.literalProgressText = fmt.Sprintf("Loading model \"%s\"...", modelFilePath)
	appState.updateOutput()

	prompt := "Can you explain what is Theory of relativity, shortly?"
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

	appState.resetConsoleState()
	appState.literalProgressText = fmt.Sprintf("Model \"%s\" was loaded, starting inference...", modelFilePath)
	appState.updateOutput()

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 200

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
	mu                            sync.Mutex
	consoleMeasure                *goterminal.Writer
	excludeEscapeDirectivesRegexp regexp.Regexp

	prevLineWidths []int
	latestLogText  string

	sequenceLength      int
	promptText          string
	promptTokens        []sentencepiece.SentencePiece
	generatedText       string
	literalProgressText string

	generatedTokens []sentencepiece.SentencePiece
	startTimeTotal  time.Time
	startTimeToken  time.Time
}

func (as *AppState) updateOutput() {
	// See: https://github.com/apoorvam/goterminal/blob/master/writer_posix.go
	as.cleanupConsole()
	if as.latestLogText == "" {
		as.latestLogText = "..."
	}

	elapsedTotalStr, elapsedTokenStr := as.durationsToStr()
	as.printLinef(as.generateProgressText())
	as.printLinef("Total elapsed: %c[1m%s%c[0m, elapsed for next token: %c[1m%s%c[0m", esc, elapsedTotalStr, esc, esc, elapsedTokenStr, esc)
	as.printLinef("Running for next token: " + as.latestLogText)
	as.printLinef("")
	if as.promptText != "" {
		generatedText := as.generatedText
		if generatedText == "" {
			generatedText = "..."
		}
		as.printLinef(fmt.Sprintf("%c[1mPrompt:%c[0m ", esc, esc) + as.promptText +
			fmt.Sprintf("\n%c[1mAssistant:%c[0m ", esc, esc) + generatedText)
	} else {
		as.printLinef("...")
	}
}

func (as *AppState) printLinef(format string, v ...any) {
	s := fmt.Sprintf(format, v...)
	lines := strings.Split(s, "\n")
	for _, line := range lines {
		line = as.excludeEscapeDirectivesRegexp.ReplaceAllString(line, "")
		if len(as.prevLineWidths) == 0 || as.prevLineWidths[len(as.prevLineWidths)-1] > 0 {
			as.prevLineWidths = append(as.prevLineWidths, len(line))
		} else {
			as.prevLineWidths[len(as.prevLineWidths)-1] = len(line)
		}
	}
	if len(as.prevLineWidths) == 0 || as.prevLineWidths[len(as.prevLineWidths)-1] > 0 {
		s += "\n"
		as.prevLineWidths = append(as.prevLineWidths, 0)
	}
	fmt.Print(s)
}

func (as *AppState) measureConsoleWidth() int {
	var measureMu sync.Mutex
	defer measureMu.Unlock()
	measureMu.Lock()
	w, _ := as.consoleMeasure.GetTermDimensions()
	for {
		time.Sleep(300 * time.Millisecond)
		w2, _ := as.consoleMeasure.GetTermDimensions()
		if w == w2 {
			return w
		}
		w = w2
	}
}

func (as *AppState) cleanupConsole() {
	defer as.mu.Unlock()
	as.mu.Lock()
	if as.prevLineWidths != nil && len(as.prevLineWidths) > 0 {
		lineCountToClean := 0
		currentConsoleWidth := as.measureConsoleWidth()
		for i := len(as.prevLineWidths) - 1; i >= 0; i-- {
			prevLineWidth := as.prevLineWidths[i]
			lineCountByCurrentConsoleWidth := int(math.Ceil(float64(prevLineWidth) / float64(currentConsoleWidth)))
			if prevLineWidth == 0 {
				lineCountByCurrentConsoleWidth = 1
			}
			lineCountToClean += lineCountByCurrentConsoleWidth
		}
		for i := 0; i < lineCountToClean; i++ {
			fmt.Printf("%c[2K\r", esc) // Clear current line
			if i < lineCountToClean-1 {
				fmt.Printf("%c[%dA", esc, 1) // Move cursor upper line
			}
		}
	}
	as.resetConsoleState()
}

func (as *AppState) resetConsoleState() {
	as.prevLineWidths = nil
	as.prevLineWidths = make([]int, 0)
}

func (as *AppState) generateProgressText() string {
	if as.literalProgressText != "" {
		return as.literalProgressText
	}
	var latestGeneratedTokenStr string
	if len(as.generatedTokens) == 0 {
		latestGeneratedTokenStr = "(generating)"
	} else {
		latestGeneratedTokenStr = as.generatedTokens[len(as.generatedTokens)-1].String()
	}
	nextTokenNum := len(as.promptTokens) + len(as.generatedTokens)
	if nextTokenNum < as.sequenceLength {
		nextTokenNum++
	}
	return fmt.Sprintf("%c[1mGenerating tokens %d / %d, including %d prompt tokens...%c[0m Latest generated token: %s",
		esc, nextTokenNum, as.sequenceLength, len(as.promptTokens), esc, latestGeneratedTokenStr)
}

func (as *AppState) durationsToStr() (elapsedTotalStr string, elapsedTokenStr string) {
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

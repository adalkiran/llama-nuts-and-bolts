package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
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
const waitingByteTempChar = "\u2026" // Unicode character ellipsis …
const debugMode = false

var predefinedPrompts = []PromptInput{
	{IsChatMode: false, Prompt: "Hello, my name is "},
	{IsChatMode: false, Prompt: "You are Einstein. Describe your theory. "},
	{IsChatMode: true, Prompt: "Can you explain what is Theory of relativity, shortly?"},
	{IsChatMode: true, Prompt: "<<SYS>>\nAlways answer with emojis\n<</SYS>>\n\nHow to go from Beijing to NY?"},
}

var appState = &AppState{
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

	var debugLogWriter io.Writer = nil
	if debugMode {
		debugLogWriter, err = os.Create("debug.log")
		if err != nil {
			panic(err)
		}
	}

	common.GLogger, err = common.NewLogger(os.Stdout, debugLogWriter)
	if err != nil {
		panic(err)
	}
	defer common.GLogger.Close()

	modelFilePath := "../models-original/7B-chat/consolidated.00.pth"

	common.GLogger.ConsolePrintf("Loading model \"%s\"...", modelFilePath)

	llamaModel, err := model.LoadModel(modelFilePath)
	if err != nil {
		common.GLogger.ConsoleFatal(err)
	}
	defer llamaModel.Free()

	fmt.Printf("Model \"%s\" was loaded.\n", modelFilePath)

	fmt.Printf("\n\n\n")

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 200

	engine := inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn)

	userPrompt := askUserPromptChoice()
	if userPrompt.IsChatMode {
		userPrompt.Prompt = fmt.Sprintf("%s %s %s", B_INST, userPrompt.Prompt, E_INST)
	}

	fmt.Printf("\n\n\n")

	tokens, err := engine.Tokenize(userPrompt.Prompt, true)
	if err != nil {
		common.GLogger.ConsoleFatal(err)
	}

	appState.promptTokens, appState.promptText = engine.TokenBatchToString(tokens)

	appState.startTimeTotal = time.Now()
	appState.startTimeToken = appState.startTimeTotal

	appState.sequenceLength = inferenceArgs.SequenceLength
	appState.resetConsoleState()
	appState.literalProgressText = ""
	appState.updateOutput()

	appState.generatedTokenIds = make([]model.TokenId, 0)
	appState.generatedTokens = make([]sentencepiece.SentencePiece, 0)

	var wg sync.WaitGroup

	generatedPartCh, errorCh := engine.GenerateString(tokens)

	wg.Add(1)
	go listenGenerationChannels(&wg, generatedPartCh, errorCh)

	wg.Wait()

	finishReason := "unknown"
	switch appState.generationState {
	case inference.GSFinishedByReachingEOS:
		finishReason = "reaching EOS token"
	case inference.GSFinishedByReachingSeqLen:
		finishReason = "reaching sequence length"
	}

	fmt.Printf("\n\nFinished %c[1mby %s%c[0m.\n", esc, finishReason, esc)
}

func listenGenerationChannels(wg *sync.WaitGroup, generatedPartCh <-chan inference.GeneratedPart, errorCh <-chan error) {
	defer wg.Done()
	loop := true
	for loop {
		select {
		case generatedPart, ok := <-generatedPartCh:
			if !ok {
				loop = false
				fmt.Println()
				break
			}
			if !generatedPart.IsResendOfWaiting {
				appState.generatedTokenIds = append(appState.generatedTokenIds, generatedPart.TokenId)
				appState.generatedTokens = append(appState.generatedTokens, generatedPart.Token)
			}
			if generatedPart.AddedToWaiting {
				appState.generatedText += waitingByteTempChar
			} else {
				// Check if the text ends with waitingByteTempChar, if true, remove it.
				for strings.HasSuffix(appState.generatedText, waitingByteTempChar) {
					appState.generatedText = strings.TrimSuffix(appState.generatedText, waitingByteTempChar)
				}
				appState.generatedText += generatedPart.DecodedString
			}
			appState.generationState = generatedPart.GenerationState
			appState.updateOutput()
			appState.startTimeToken = time.Now()

		case err := <-errorCh:
			if err == nil {
				continue
			}
			fmt.Println()
			common.GLogger.ConsoleFatal(err)
		}
	}
}

func askUserPromptChoice() PromptInput {
	extraChoiceCount := 2
	for {
		fmt.Printf("%c[1mSelect from our predefined prompts:%c[0m\n", esc, esc)
		for i, predefinedPrompt := range predefinedPrompts {
			isChatModeText := "Chat mode"
			if !predefinedPrompt.IsChatMode {
				isChatModeText = "Text completion"
			}
			fmt.Printf("%2d. [%s] %s\n", i+1, isChatModeText, predefinedPrompt.Prompt)
		}
		fmt.Printf("%c[1m%2d.%c[0m [%s] %s\n", esc, len(predefinedPrompts)+1, esc, "Text completion", "Other, manual input")
		fmt.Printf("%c[1m%2d.%c[0m [%s] %s\n", esc, len(predefinedPrompts)+2, esc, "Chat mode", "Other, manual input")

		reader := bufio.NewReader(os.Stdin)
		fmt.Printf("\nYour choice (choose %d to %d and press Enter): ", 1, len(predefinedPrompts)+extraChoiceCount)
		userChoice, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("\nerror: %v\n\n", err)
			continue
		}
		userChoiceNum, err := strconv.Atoi(strings.TrimSpace(userChoice))
		if err != nil {
			fmt.Printf("\nNot a valid number.\n\n")
			continue
		}
		if userChoiceNum < 1 || userChoiceNum > len(predefinedPrompts)+extraChoiceCount {
			fmt.Printf("\nChoice must be between %d and %d.\n\n", 0, len(predefinedPrompts)+extraChoiceCount)
			continue
		}

		if userChoiceNum <= len(predefinedPrompts) {
			return predefinedPrompts[userChoiceNum-1]
		}
		userPromptInput := PromptInput{
			IsChatMode: userChoiceNum == len(predefinedPrompts)+2,
		}
		fmt.Print("Write down your prompt and press Enter: ")
		if userPromptInput.Prompt, err = reader.ReadString('\n'); err != nil {
			fmt.Printf("\nerror: %v\n", err)
			continue
		}
		userPromptInput.Prompt = strings.TrimRight(userPromptInput.Prompt, "\r\n")
		if len(userPromptInput.Prompt) == 0 {
			fmt.Printf("\nThe prompt you entered is empty.")
			continue
		}
		return userPromptInput
	}
}

type PromptInput struct {
	Prompt     string
	IsChatMode bool
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

	generationState   inference.GenerationState
	generatedTokenIds []model.TokenId
	generatedTokens   []sentencepiece.SentencePiece
	startTimeTotal    time.Time
	startTimeToken    time.Time
}

func (as *AppState) updateOutput() {
	// See: https://github.com/apoorvam/goterminal/blob/master/writer_posix.go
	as.cleanupConsole()
	if as.latestLogText == "" {
		as.latestLogText = waitingByteTempChar
	}

	elapsedTotalStr, elapsedTokenStr := as.durationsToStr()
	as.printLinef("Press Ctrl+C to exit.")
	as.printLinef(as.generateProgressText())
	as.printLinef("%-23s: %c[1m%s%c[0m, elapsed for next token: %c[1m%s%c[0m", "Total elapsed", esc, elapsedTotalStr, esc, esc, elapsedTokenStr, esc)
	as.printLinef("%-23s: %s", "Running for next token", as.latestLogText)
	as.printLinef("")
	if as.promptText != "" {
		generatedText := as.generatedText
		if generatedText == "" {
			generatedText = waitingByteTempChar
		}
		as.printLinef("%c[1m%-23s:%c[0m \"%s\"", esc, "Prompt", esc, as.promptText)
		as.printLinef("%c[1m%-23s:%c[0m \"%s\"", esc, "Assistant", esc, generatedText)
	} else {
		as.printLinef(waitingByteTempChar)
	}
}

func (as *AppState) printLinef(format string, v ...any) {
	s := fmt.Sprintf(format, v...)
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		line = as.excludeEscapeDirectivesRegexp.ReplaceAllString(line, "")
		if len(as.prevLineWidths) == 0 || i > 0 || as.prevLineWidths[len(as.prevLineWidths)-1] > 0 {
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
	return fmt.Sprintf("%c[1m%-23s: %d / %d, including %d prompt tokens...%c[0m",
		esc, "Generating tokens", nextTokenNum, as.sequenceLength, len(as.promptTokens), esc) + "\n" +
		fmt.Sprintf("%-23s: %s", "Latest generated token", latestGeneratedTokenStr)
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
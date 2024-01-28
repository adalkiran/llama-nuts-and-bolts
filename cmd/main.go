package main

import (
	"bufio"
	"context"
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
const B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
const waitingByteTempChar = "\u2026" // Unicode character ellipsis …
const modelsDirName = "models-original"
const debugMode = false

var predefinedPrompts = []PromptInput{
	{IsChatMode: false, Prompt: "Hello, my name is"},
	{IsChatMode: true, SystemPrompt: "You are Einstein", Prompt: "Describe your theory."},
	{IsChatMode: true, SystemPrompt: "Answer in 20 words, directly, and without an introduction", Prompt: "Can you explain what is Theory of relativity?"},
	{IsChatMode: true, SystemPrompt: "You are a pirate", Prompt: "Explain what is quantum computer in 20 words."},
	{IsChatMode: true, SystemPrompt: "Always answer with emojis", Prompt: "How to go from Beijing to NY?"},
	{IsChatMode: true, SystemPrompt: "Answer with only one emoji", Prompt: "What is the flag of Turkey?"},
}

var appState = &AppState{
	consoleOutWriter:              os.Stdout,
	prevLineWidths:                make([]int, 0),
	consoleMeasure:                goterminal.New(os.Stdout),
	excludeEscapeDirectivesRegexp: *regexp.MustCompile(string(rune('\033')) + "\\[\\d+[a-zA-Z]"),
}

func main() {
	fmt.Println("Welcome to Llama Nuts and Bolts!")
	fmt.Print("=================================\n\n\n")

	var err error
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

	machineEndian := common.DetermineMachineEndian()
	common.GLogger.ConsolePrintf("Determined machine endianness: %s", machineEndian)
	if machineEndian != "LITTLE_ENDIAN" {
		panic(fmt.Errorf("error: Endianness of your machine is not supported. Expected LITTLE_ENDIAN but got %s", machineEndian))
	}

	modelDir, err := searchForModelPath(modelsDirName, "7B-chat")
	if err != nil {
		panic(err)
	}

	common.GLogger.ConsolePrintf("Found model files in \"%s\"...", modelDir)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	llamaModel, err := model.LoadModel(modelDir)
	if err != nil {
		common.GLogger.ConsoleFatal(err)
	}
	defer llamaModel.Free()

	fmt.Printf("Model \"%s\" was loaded.\n", modelDir)

	fmt.Printf("\n\n\n")

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 200

	engine := inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn)

	userPrompt := askUserPromptChoice()
	userPromptStr := userPrompt.Prompt
	if userPrompt.IsChatMode {
		systemPrompt := ""
		if userPrompt.SystemPrompt != "" {
			systemPrompt = fmt.Sprintf("%s%s%s", B_SYS, userPrompt.SystemPrompt, E_SYS)
		}
		userPromptStr = fmt.Sprintf("%s %s%s %s", B_INST, systemPrompt, userPrompt.Prompt, E_INST)
	} else {
		if !strings.HasSuffix(userPromptStr, " ") {
			userPromptStr += " "
		}
	}

	fmt.Printf("\n\n\n")

	tokens, err := engine.Tokenize(userPromptStr, true)
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
	go listenGenerationChannels(&wg, ctx, generatedPartCh, errorCh)

	wg.Wait()

	finishReason := "unknown"
	switch appState.generationState {
	case inference.GSFinishedByReachingEOS:
		finishReason = "reaching EOS token"
	case inference.GSFinishedByReachingSeqLen:
		finishReason = "reaching sequence length"
	}

	fmt.Printf("\n\nFinished \033[1mby %s\033[0m.\n", finishReason)
}

func listenGenerationChannels(wg *sync.WaitGroup, ctx context.Context, generatedPartCh <-chan inference.GeneratedPart, errorCh <-chan error) {
	defer wg.Done()
	loop := true
	for loop {
		select {
		case generatedPart, ok := <-generatedPartCh:
			if !ok {
				loop = false
				fmt.Fprintln(appState.consoleOutWriter)
				break
			}
			if !generatedPart.IsResendOfWaiting {
				appState.generatedTokenIds = append(appState.generatedTokenIds, generatedPart.TokenId)
				appState.generatedTokens = append(appState.generatedTokens, generatedPart.Token)
			}
			if generatedPart.AddedToWaiting {
				appState.addedToWaitingCount++
			} else {
				appState.addedToWaitingCount = 0
				appState.generatedText += generatedPart.DecodedString
			}
			appState.waitingRunesExtraStr = generatedPart.WaitingRunesExtraStr
			appState.generationState = generatedPart.GenerationState
			appState.updateOutput()
			appState.startTimeToken = time.Now()

		case err := <-errorCh:
			if err == nil {
				continue
			}
			fmt.Fprintln(appState.consoleOutWriter)
			common.GLogger.ConsoleFatal(err)
		case <-ctx.Done():
			loop = false
		}
	}
	if len(appState.waitingRunesExtraStr) > 0 {
		appState.generatedText += appState.waitingRunesExtraStr
		appState.updateOutput()
	}
}

func searchForModelPath(modelsDirName string, modelName string) (string, error) {
	exePath, err := os.Executable()
	if err != nil {
		return "", err
	}
	// Get the directory containing the executable
	exeDir := filepath.Dir(exePath)

	fileNamesToLookFor := []string{"consolidated.00.pth", "params.json", "tokenizer.model"}

	rootPathAlternatives := [][2]string{
		{exeDir, "."}, {exeDir, ".."}, {"", "."}, {"", ".."},
	}
	searchedDirectories := make([]string, 0)
	for _, rootPathAlternative := range rootPathAlternatives {
		found := true
		modelDir := filepath.Join(rootPathAlternative[0], rootPathAlternative[1], modelsDirName, modelName)
		for _, fileNameToLookFor := range fileNamesToLookFor {
			if _, err := os.Stat(filepath.Join(modelDir, fileNameToLookFor)); err != nil {
				found = false
				break
			}
		}
		if found {
			return modelDir, nil
		}
		searchedDirectories = append(searchedDirectories, modelDir)
	}
	return "", fmt.Errorf("model directory \"%s\" and related files could not be found in:\n%s", modelsDirName, strings.Join(searchedDirectories, "\n"))
}

func askUserPromptChoice() PromptInput {
	extraChoiceCount := 2
	for {
		fmt.Printf("\033[1mSelect from our predefined prompts (latest two are for manual input):\033[0m\n")
		for i, predefinedPrompt := range predefinedPrompts {
			if predefinedPrompt.IsChatMode {
				systemPrompt := predefinedPrompt.SystemPrompt
				if systemPrompt == "" {
					systemPrompt = "(empty)"
				}
				fmt.Printf("%2d. %-17s \033[1mSystem Prompt:\033[0m %s\n%22s\033[1mPrompt:\033[0m %s\n",
					i+1, "[Chat mode]", systemPrompt, " ", predefinedPrompt.Prompt)

			} else {
				fmt.Printf("%2d. %-17s \033[1mPrompt:\033[0m %s\n", i+1, "[Text completion]", predefinedPrompt.Prompt)

			}
		}
		fmt.Printf("%2d. %-17s %s\n", len(predefinedPrompts)+1, "[Text completion]", "Other, manual input")
		fmt.Printf("%2d. %-17s %s\n", len(predefinedPrompts)+2, "[Chat mode]", "Other, manual input")

		reader := bufio.NewReader(os.Stdin)
		fmt.Printf("\n\033[1mYour choice (choose %d to %d and press Enter):\033[0m ", 1, len(predefinedPrompts)+extraChoiceCount)
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
		if userPromptInput.IsChatMode {
			fmt.Print("\033[1mWrite down your \"system prompt\" (optional, will be surrounded by <<SYS>> and <</SYS>>) and press Enter:\033[0m ")
			if userPromptInput.Prompt, err = reader.ReadString('\n'); err != nil {
				fmt.Printf("\nerror: %v\n", err)
				continue
			}
			userPromptInput.SystemPrompt = strings.TrimRight(userPromptInput.Prompt, "\r\n")
		}
		if userPromptInput.IsChatMode {
			fmt.Print("\033[1mWrite down your prompt (will be surrounded by [INST] and [/INST]) and press Enter:\033[0m ")
		} else {
			fmt.Print("\033[1mWrite down your prompt and press Enter:\033[0m ")
		}
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
	SystemPrompt string
	Prompt       string
	IsChatMode   bool
}

func logFn(format string, v ...any) {
	appState.latestLogText = fmt.Sprintf(format, v...)
	appState.updateOutput()
}

type AppState struct {
	mu                            sync.Mutex
	consoleMeasure                *goterminal.Writer
	excludeEscapeDirectivesRegexp regexp.Regexp

	prevLineWidths   []int
	latestLogText    string
	printStrBuilder  strings.Builder
	consoleOutWriter io.Writer

	sequenceLength       int
	promptText           string
	promptTokens         []sentencepiece.SentencePiece
	generatedText        string
	waitingRunesExtraStr string
	literalProgressText  string

	generationState     inference.GenerationState
	generatedTokenIds   []model.TokenId
	generatedTokens     []sentencepiece.SentencePiece
	addedToWaitingCount int
	startTimeTotal      time.Time
	startTimeToken      time.Time
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
	as.printLinef("%-23s: \033[1m%s\033[0m, elapsed for next token: \033[1m%s\033[0m", "Total elapsed", elapsedTotalStr, elapsedTokenStr)
	as.printLinef("%-23s: %s", "Running for next token", as.latestLogText)
	as.printLinef("")
	if as.promptText != "" {
		generatedText := as.generatedText
		generatedText += as.waitingRunesExtraStr
		for i := 0; i < as.addedToWaitingCount; i++ {
			generatedText += waitingByteTempChar
		}
		if generatedText == "" {
			generatedText = waitingByteTempChar
		}
		waitingTokensText := ""
		if appState.addedToWaitingCount > 0 {
			waitingTokensTextItems := make([]string, appState.addedToWaitingCount)
			addedToWaitingTokens := appState.generatedTokens[len(appState.generatedTokens)-appState.addedToWaitingCount : len(appState.generatedTokens)]
			for i, addedToWaitingToken := range addedToWaitingTokens {
				waitingTokensTextItems[i] = fmt.Sprintf("\"%s\"", addedToWaitingToken.Piece)
			}
			waitingTokensText = fmt.Sprintf("%s, possibly a part of an upcoming emoji)", strings.Join(waitingTokensTextItems, ", "))
		}
		as.printLinef("\033[1m%-23s:\033[0m \"%s\"", "Prompt", as.promptText)
		as.printLinef("\033[1m%-23s:\033[0m \"%s\"", "Assistant", generatedText)
		if waitingTokensText != "" {
			as.printLinef("\033[1m%-23s:\033[0m %s", "Tokens waiting to be processed further", waitingTokensText)
		}
	} else {
		as.printLinef(waitingByteTempChar)
	}
	as.flushConsolePrint()
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
	// While testing on some Windows machines, directly printing out the output caused a "flash" on the screen.
	// To prevent this, the output is collected in a string builder, and then printed at one time via flushConsolePrint() function.
	as.printStrBuilder.WriteString(s)
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
	var sb strings.Builder
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
			sb.WriteString("\033[2K\r") // Clear current line
			if i < lineCountToClean-1 {
				sb.WriteString(fmt.Sprintf("\033[%dA", 1)) // Move cursor upper line
			}
		}
	}
	as.printStrBuilder.WriteString(sb.String())
	as.resetConsoleState()
}

func (as *AppState) flushConsolePrint() {
	fmt.Fprint(as.consoleOutWriter, as.printStrBuilder.String())
	as.printStrBuilder.Reset()
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
	return fmt.Sprintf("\033[1m%-23s: %d / %d, including %d prompt tokens...\033[0m",
		"Generating tokens", nextTokenNum, as.sequenceLength, len(as.promptTokens)) + "\n" +
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

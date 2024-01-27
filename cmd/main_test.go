package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"regexp"
	"runtime"
	"sync"
	"testing"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/inference"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

// "\x1b[1mAssistant              :\x1b[0m \"…\""
var assistantLineRegexp = *regexp.MustCompile(`.*\[1mAssistant\s*\:\x1b\[0m \"(.+)\"`)

// "\x1b[1mTokens waiting to be processed further:\x1b[0m \"<0xF0>\", possibly a part of an upcoming emoji)"
var tokensWaitingLineRegexp = *regexp.MustCompile(`.*\[1mTokens waiting to be processed further\s*\:\x1b\[0m (.+), possibly`)

type InterceptorWriter struct {
	Target       io.Writer
	ListenerChan chan<- string
}

func (iw *InterceptorWriter) Write(p []byte) (n int, err error) {
	if iw.Target != nil {
		n, err = iw.Target.Write(p)
	}
	iw.ListenerChan <- string(p)
	return n, err
}

func prepareInferenceEngine(t *testing.T) (*inference.InferenceEngine, chan string) {
	var err error
	common.GLogger, err = common.NewLogger(os.Stdout, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer common.GLogger.Close()

	appState.promptText = "Dummy prompt text"
	consoleListenerChan := make(chan string, 1000)
	appState.consoleOutWriter = &InterceptorWriter{
		Target:       nil,
		ListenerChan: consoleListenerChan,
	}

	modelDir := "../models-original/7B"
	if _, err := os.Stat(modelDir); err != nil {
		t.Skipf("Model directory \"%s\" is not found, passing this test: %s", modelDir, "TestSimulated")
		return nil, nil
	}

	llamaModel, err := model.LoadModel(modelDir)
	if err != nil {
		common.GLogger.ConsoleFatal(err)
	}
	defer llamaModel.Free()

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 200

	return inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn), consoleListenerChan
}

func TestSimulatedEmojiOutput(t *testing.T) {
	engine, consoleListenerChan := prepareInferenceEngine(t)
	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	bytesInput := []byte{0xF0, 0x9F, 0x87, 0xB9, 0xF0, 0x9F, 0x87, 0xB7}
	bytesInputStr := ""
	for _, b := range bytesInput {
		bytesInputStr += fmt.Sprintf("<0x%02X>", b)
	}

	expectedAssistantLines := []string{
		"…",
		"……",
		"………",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]…",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]……",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]………",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]🇷[:REGIONAL INDICATOR SYMBOL LETTER R:\\U0001F1F7]",
	}
	expectedWaitingLines := []string{
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x87>\"",
		"",
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x87>\"",
		"",
	}

	tokens, err := engine.Tokenize(bytesInputStr, true)
	if err != nil {
		common.GLogger.ConsoleFatal(err)
	}

	generatedPartCh, errorCh := engine.GenerateStringFromOutputTokens(tokens)

	wg.Add(1)
	go listenGenerationChannels(&wg, ctx, generatedPartCh, errorCh)
	runtime.Gosched()

	loop := true
	iteration := 0
	for loop {
		select {
		case printedStr, ok := <-consoleListenerChan:
			if !ok {
				loop = false
				break
			}
			expectedAssistantLine := "!IGNORE"
			expectedWaitingLine := "!IGNORE"

			if iteration < len(expectedAssistantLines) {
				expectedAssistantLine = expectedAssistantLines[iteration]
			}
			if iteration < len(expectedWaitingLines) {
				expectedWaitingLine = expectedWaitingLines[iteration]
			}
			var match []string
			match = assistantLineRegexp.FindStringSubmatch(printedStr)
			if expectedAssistantLine != "!IGNORE" {
				actual := ""
				if len(match) >= 2 {
					actual = match[1]
				}
				if expectedAssistantLine != "" && actual == "" {
					t.Fatalf("Iteration %d. Expected \"Assistant\" output line: \"%s\", but not found in: %s", iteration, expectedAssistantLine, printedStr)
				}
				if actual != expectedAssistantLine {
					t.Fatalf("Iteration %d. Expected \"Assistant\" output line: \"%s\", but got \"%s\"", iteration, expectedAssistantLine, actual)
				}
			}
			match = tokensWaitingLineRegexp.FindStringSubmatch(printedStr)
			if expectedWaitingLine != "!IGNORE" {
				actual := ""
				if len(match) >= 2 {
					actual = match[1]
				}
				if expectedWaitingLine != "" && actual == "" {
					t.Fatalf("Iteration %d. Expected \"Tokens waiting...\" output line: \"%s\", but not found in: %s", iteration, expectedWaitingLine, printedStr)
				}
				if actual != expectedWaitingLine {
					t.Fatalf("Iteration %d. Expected \"Tokens waiting...\" output line: \"%s\", but got \"%s\"", iteration, expectedWaitingLine, actual)
				}
			}
			iteration++
		case <-common.WaitGroupDone(&wg):
			close(consoleListenerChan)
		}
	}
}

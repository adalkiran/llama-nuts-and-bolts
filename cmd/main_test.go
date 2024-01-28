package main

import (
	"context"
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
	t.Helper()
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

	llamaModel, err := model.LoadModelEx(modelDir, false, true)
	if err != nil {
		common.GLogger.ConsoleFatal(err)
	}
	defer llamaModel.Free()

	inferenceArgs := common.NewInferenceArgs()
	inferenceArgs.Seed = 1234
	inferenceArgs.SequenceLength = 200

	return inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn), consoleListenerChan
}

func testSimulatedEmojiOutput(t *testing.T, inputStr string, expectedAssistantLines []string, expectedWaitingLines []string) {
	t.Helper()
	engine, consoleListenerChan := prepareInferenceEngine(t)
	var wg sync.WaitGroup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tokens, err := engine.Tokenize(inputStr, false)
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
					t.Fatalf("Iteration %d. Expected \"Assistant\" output line:\n\"%s\",\nbut not found in:\n%s", iteration, expectedAssistantLine, printedStr)
				}
				if actual != expectedAssistantLine {
					t.Fatalf("Iteration %d. Expected \"Assistant\" output line:\n\"%s\",\nbut got\n\"%s\"", iteration, expectedAssistantLine, actual)
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

func TestSimulatedEmojiOutputTurkeyFlag(t *testing.T) {
	inputPartT := "<0xF0><0x9F><0x87><0xB9>" // Character: 🇹
	inputPartR := "<0xF0><0x9F><0x87><0xB7>" // Character: 🇷
	inputPartEOS := "</s>"
	inputStr := inputPartT + inputPartR + inputPartEOS // Character: 🇹🇷
	expectedAssistantLines := []string{
		"…",
		"……",
		"………",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]…",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]……",
		"🇹[:REGIONAL INDICATOR SYMBOL LETTER T:\\U0001F1F9]………",
		"🇹🇷[:flag_for_turkey:\\U0001F1F9\\U0001F1F7]",
		"🇹🇷[:flag_for_turkey:\\U0001F1F9\\U0001F1F7]",
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
		"",
	}
	testSimulatedEmojiOutput(t, inputStr, expectedAssistantLines, expectedWaitingLines)
}

func TestSimulatedEmojiOutputEmojiWithText(t *testing.T) {
	whitespaceEscapeToken := "\xe2\x96\x81"
	inputPartEyes := "<0xF0><0x9F><0x91><0x80>" //Character: 👀
	inputPart_I := whitespaceEscapeToken + "I"
	inputStr := inputPartEyes + inputPart_I
	expectedAssistantLines := []string{
		"…",
		"……",
		"………",
		"👀[:eyes:\\U0001F440]",
		"👀[:eyes:\\U0001F440] I",
	}
	expectedWaitingLines := []string{
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x91>\"",
		"",
		"",
	}
	testSimulatedEmojiOutput(t, inputStr, expectedAssistantLines, expectedWaitingLines)
}

func TestSimulatedEmojiOutputMultipleEmojis(t *testing.T) {
	inputPartArrivingAirplane := "<0xF0><0x9F><0x9B><0xAC>" //Character: 🛬
	inputPartMantelpieceClock := "<0xF0><0x9F><0x95><0xB0>" //Character: 🕰
	inputPartLocomotive := "<0xF0><0x9F><0x9A><0x82>"       //Character: 🚂
	inputPartSunriseMountains := "<0xF0><0x9F><0x8C><0x84>" //Character: 🌄
	inputStr := inputPartArrivingAirplane + inputPartMantelpieceClock + inputPartLocomotive + inputPartSunriseMountains
	expectedAssistantLines := []string{
		"…",
		"……",
		"………",
		"🛬[:airplane_arrival:\\U0001F6EC]",
		"🛬[:airplane_arrival:\\U0001F6EC]…",
		"🛬[:airplane_arrival:\\U0001F6EC]……",
		"🛬[:airplane_arrival:\\U0001F6EC]………",
		"🛬🕰[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570]",
		"🛬🕰[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570]…",
		"🛬🕰[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570]……",
		"🛬🕰[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570]………",
		"🛬🕰🚂[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570][:locomotive:\\U0001F682]",
		"🛬🕰🚂[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570][:locomotive:\\U0001F682]…",
		"🛬🕰🚂[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570][:locomotive:\\U0001F682]……",
		"🛬🕰🚂[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570][:locomotive:\\U0001F682]………",
		"🛬🕰🚂🌄[:airplane_arrival:\\U0001F6EC][:MANTELPIECE CLOCK:\\U0001F570][:locomotive:\\U0001F682][:sunrise_over_mountains:\\U0001F304]",
	}
	expectedWaitingLines := []string{
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x9B>\"",
		"",
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x95>\"",
		"",
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x9A>\"",
		"",
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x8C>\"",
		"",
	}
	testSimulatedEmojiOutput(t, inputStr, expectedAssistantLines, expectedWaitingLines)
}

func TestSimulatedEmojiOutputMultipleCompositeEmojis(t *testing.T) {
	inputZwj := "<0xE2><0x80><0x8D>" // ZWJ (Zero Width Joiner)

	inputSuperhero := "<0xF0><0x9F><0xA6><0xB8>"   //Character: 🦸 U+1F9B8
	inpuutMaleSign := "<0xE2><0x99><0x82>"         //Character: ♂ U+2642
	inputVariationSelector := "<0xEF><0xB8><0x8F>" //Character: ◌️ U+FE0F - Variation Selector-16 (VS16)

	inputCompositeManSuperhero := inputSuperhero + inputZwj + inpuutMaleSign + inputVariationSelector

	inputPartMan := "<0xF0><0x9F><0x91><0xA8>"
	inputPartWoman := "<0xF0><0x9F><0x91><0xA9>"
	inputPartGirl := "<0xF0><0x9F><0x91><0xA7>"
	inputPartBoy := "<0xF0><0x9F><0x91><0xA6>"

	inputCompositeFamily := inputPartMan + inputZwj + inputPartWoman + inputZwj + inputPartGirl + inputZwj + inputPartBoy

	inputStr := inputCompositeManSuperhero + inputCompositeFamily

	expectedAssistantLines := []string{
		/*itr  0*/ "…",
		/*itr  1*/ "……",
		/*itr  2*/ "………",

		/*itr  3*/ "🦸[:superhero:\\U0001F9B8]",
		/*itr  4*/ "🦸[:superhero:\\U0001F9B8]…",
		/*itr  5*/ "🦸[:superhero:\\U0001F9B8]……",
		/*itr  6*/ "🦸‍[:superhero:\\U0001F9B8][:ZERO WIDTH JOINER:\\U0000200D]",
		/*itr  7*/ "🦸‍[:superhero:\\U0001F9B8][:ZERO WIDTH JOINER:\\U0000200D]…",
		/*itr  8*/ "🦸‍[:superhero:\\U0001F9B8][:ZERO WIDTH JOINER:\\U0000200D]……",
		/*itr  9*/ "🦸‍♂[:superhero:\\U0001F9B8][:ZERO WIDTH JOINER:\\U0000200D][:MALE SIGN:\\U00002642]",
		/*itr 10*/ "🦸‍♂[:superhero:\\U0001F9B8][:ZERO WIDTH JOINER:\\U0000200D][:MALE SIGN:\\U00002642]…",
		/*itr 11*/ "🦸‍♂[:superhero:\\U0001F9B8][:ZERO WIDTH JOINER:\\U0000200D][:MALE SIGN:\\U00002642]……",
		/*itr 12*/ "🦸‍♂️[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F]",
		/*itr 13*/ "🦸‍♂️[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F]…",
		/*itr 14*/ "🦸‍♂️[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F]……",
		/*itr 15*/ "🦸‍♂️[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F]………",

		/*itr 16*/ "🦸‍♂️👨[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468]",
		/*itr 17*/ "🦸‍♂️👨[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468]…",
		/*itr 18*/ "🦸‍♂️👨[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468]……",
		/*itr 19*/ "🦸‍♂️👨‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D]",
		/*itr 20*/ "🦸‍♂️👨‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D]…",
		/*itr 21*/ "🦸‍♂️👨‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D]……",
		/*itr 22*/ "🦸‍♂️👨‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D]………",
		/*itr 23*/ "🦸‍♂️👨‍👩[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D][:woman:\\U0001F469]",
		/*itr 24*/ "🦸‍♂️👨‍👩[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D][:woman:\\U0001F469]…",
		/*itr 25*/ "🦸‍♂️👨‍👩[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D][:woman:\\U0001F469]……",
		/*itr 26*/ "🦸‍♂️👨‍👩‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D][:woman:\\U0001F469][:ZERO WIDTH JOINER:\\U0000200D]",
		/*itr 27*/ "🦸‍♂️👨‍👩‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D][:woman:\\U0001F469][:ZERO WIDTH JOINER:\\U0000200D]…",
		/*itr 28*/ "🦸‍♂️👨‍👩‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D][:woman:\\U0001F469][:ZERO WIDTH JOINER:\\U0000200D]……",
		/*itr 29*/ "🦸‍♂️👨‍👩‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:man:\\U0001F468][:ZERO WIDTH JOINER:\\U0000200D][:woman:\\U0001F469][:ZERO WIDTH JOINER:\\U0000200D]………",
		/*itr 30*/ "🦸‍♂️👨‍👩‍👧[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467]",
		/*itr 31*/ "🦸‍♂️👨‍👩‍👧[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467]…",
		/*itr 32*/ "🦸‍♂️👨‍👩‍👧[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467]……",
		/*itr 33*/ "🦸‍♂️👨‍👩‍👧‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467][:ZERO WIDTH JOINER:\\U0000200D]",
		/*itr 34*/ "🦸‍♂️👨‍👩‍👧‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467][:ZERO WIDTH JOINER:\\U0000200D]…",
		/*itr 35*/ "🦸‍♂️👨‍👩‍👧‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467][:ZERO WIDTH JOINER:\\U0000200D]……",
		/*itr 36*/ "🦸‍♂️👨‍👩‍👧‍[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467][:ZERO WIDTH JOINER:\\U0000200D]………",
		/*itr 37*/ "🦸‍♂️👨‍👩‍👧‍👦[:man_superhero:\\U0001F9B8\\U0000200D\\U00002642\\U0000FE0F][:family_man_woman_girl_boy:\\U0001F468\\U0000200D\\U0001F469\\U0000200D\\U0001F467\\U0000200D\\U0001F466]",
		"",
	}
	expectedWaitingLines := []string{
		//inputCompositeManSuperhero
		//  - inputSuperhero
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0xA6>\"",
		"",

		//  - inputZwj
		"\"<0xE2>\"",
		"\"<0xE2>\", \"<0x80>\"",
		"",

		//  - inpuutMaleSign
		"\"<0xE2>\"",
		"\"<0xE2>\", \"<0x99>\"",
		"",

		//  - inputVariationSelector
		"\"<0xEF>\"",
		"\"<0xEF>\", \"<0xB8>\"",
		"",

		//inputCompositeFamily
		//  - inputPartMan
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x91>\"",
		"",

		//  - inputZwj
		"\"<0xE2>\"",
		"\"<0xE2>\", \"<0x80>\"",
		"",

		//  - inputPartWoman
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x91>\"",
		"",

		//  - inputZwj
		"\"<0xE2>\"",
		"\"<0xE2>\", \"<0x80>\"",
		"",

		//  - inputPartGirl
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x91>\"",
		"",

		//  - inputZwj
		"\"<0xE2>\"",
		"\"<0xE2>\", \"<0x80>\"",
		"",

		//  - inputPartBoy
		"\"<0xF0>\"",
		"\"<0xF0>\", \"<0x9F>\"",
		"\"<0xF0>\", \"<0x9F>\", \"<0x91>\"",
		"",
	}
	testSimulatedEmojiOutput(t, inputStr, expectedAssistantLines, expectedWaitingLines)
}

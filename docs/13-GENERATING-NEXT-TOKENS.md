# **13. GENERATING NEXT TOKENS**

What we've done so far has been a preparatory stage for generating new tokens through text completion methods from the prompt tokens we have on hand, thus creating new outputs in natural language. Now, let the show begin!

## **13.1. Preliminary Concepts**

### **13.1.1. Context**

The Go language offers lots of robust tools to do parallel programming, with ensuring efficient management of concurrency.

>**From the [context](https://pkg.go.dev/context) package documentation:**<br>
>Package context defines the Context type, which carries deadlines, cancellation signals, and other request-scoped values across API boundaries and between processes.

In this project, we use ```context.WithCancel(...)``` to create a new context with a cancel function for generating cancellation signal. In Go, we use lots of [goroutines](https://gobyexample.com/goroutines) to run functions parallel. Mostly these ```goroutine```s initiates an infinite loop to wait some inputs from like [go channels](https://gobyexample.com/channels) or signals like context cancellation signal.

This design sometimes leads to side-effects such as unfinished goroutines remaining active upon the intentional or unintentional termination of the main process. For instance, while you have unfinished goroutines running, one CTRL+C keystroke to terminate the process may not be enough, it requires you to press CTRL+C multiple times.

To prevent these types of side-effects and undesirable occurrences, we create a ```ctx context``` along with a cancellation signal function named ```cancel```. When the ```cancel``` function is invoked, all goroutines that check the status of the given ```ctx context``` in every iteration of their infinite loops, will break their loops, and exit from their goroutine functions. This approach ensures a healthy and controlled termination process.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
func main() {
    ...
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    ...
    go listenGenerationChannels(&wg, ctx, generatedPartCh, errorCh)
    ...
}
```

>Note that, we use ```context.WithCancel(...)``` method in our project, but there are more alternatives to instantiate a new context, check out [the documentation](https://pkg.go.dev/context).

See also:

* [Documentation of context package](https://pkg.go.dev/context)
* [How To Use Contexts in Go](https://www.digitalocean.com/community/tutorials/how-to-use-contexts-in-go)
* [goroutines](https://gobyexample.com/goroutines)
* [Go channels](https://gobyexample.com/channels)

### **13.1.2. Waiting loop**

It starts with creating a wait group, it can be thought as a counter for parallel running goroutines/threads that needs to be added to the wait list. The *waitGroup.Wait()* method runs in a loop that waits until waitGroup item count becomes zero, so the process doesn't end.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
func main() {
    ...
    var wg sync.WaitGroup
    ...
    wg.Add(1)
    go listenGenerationChannels(&wg, ctx, generatedPartCh, errorCh)
    ...
    wg.Wait()
    ...
}
```

See also:

* [How to wait for all goroutines to finish in Golang](https://codewithyury.com/golang-wait-for-all-goroutines-to-finish/)
* [Using WaitGroup in Golang](https://www.geeksforgeeks.org/using-waitgroup-in-golang/)

## **13.2. Calling GenerateString(...)**

We call the [InferenceEngine.GenerateString(...)](../src/inference/inference.go) method to start the generation process and retrieve channels ```generatedPartCh``` and ```errorCh```. Then, we listen these two channels via goroutine [listenGenerationChannels(...)](../cmd/main.go). At the end, the generation process may be finished by user termination (CTRL+C), unexpected error panic, reaching an EOS (End of Sentence) token, or reaching maximum sequence length specified at ```inferenceArgs.SequenceLength```. Then the application will print out the generated text on the console, then exit.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
func main() {
    ...
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    ...
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
```

## **13.3. Internals of GenerateString(...)**

In this project, our objective was to implement *streaming* allowing us to print the generated text without waiting for the completion of the generation process, like what the ChatGPT does. This approach enables us to print out each generated token immediately after it was generated. However, this approach, while bringing some advantages, also comes with certain challenges that need to be tackled.

>If you aren't familiar with this approach, please refer to [Writing a Stream API in Go](https://betterprogramming.pub/writing-a-stream-api-in-go-afbc3c4350e2).

In our approach, we have separated the process into multiple methods and defined the ```InferenceEngine.GenerateStringGeneric(...)``` method, because we call ```InferenceEngine.GenerateString(...)``` method in the main application code, and we call ```InferenceEngine.GenerateStringFromOutputTokens(...)``` method in the unit test of main application, which both commonly call ```InferenceEngine.GenerateStringGeneric(...)``` method.

The ```InferenceEngine.GenerateStringGeneric(...)``` method creates a channel of [GeneratedPart](../src/inference/inference.go) and a channel of ```error```. Then calls ```InferenceEngine.generateStringInternal(...)``` by giving these channels. This call is done as a goroutine to make it run parallel. At the end, the function returns these channels.

<sup>from [src/inference/inference.go](../src/inference/inference.go)</sup>

```go
func (ie *InferenceEngine) GenerateString(promptTokens []model.TokenId) (<-chan GeneratedPart, <-chan error) {
    return ie.GenerateStringGeneric(promptTokens, ie.generateTokensInternal)
}

func (ie *InferenceEngine) GenerateStringGeneric(promptTokens []model.TokenId, tokenGeneratorFn TokenGeneratorFn) (<-chan GeneratedPart, <-chan error) {
    // See: https://betterprogramming.pub/writing-a-stream-api-in-go-afbc3c4350e2
    outputCh := make(chan GeneratedPart, 1)
    outputErrorCh := make(chan error)

    go func() {
        defer func() {
            close(outputCh)
            close(outputErrorCh)
        }()
        ie.generateStringInternal(promptTokens, outputCh, outputErrorCh, tokenGeneratorFn)
    }()
    return outputCh, outputErrorCh
}

func (ie *InferenceEngine) generateStringInternal(promptTokens []model.TokenId, outputCh chan<- GeneratedPart, outputErrorCh chan<- error, tokenGeneratorFn TokenGeneratorFn) {
    ...
}
```

## **13.4. Internals of generateStringInternal(...)**

This method creates a [generationDecodingContext](../src/inference/inference.go) object that carries waiting bytes and waiting parts. Then it calls ```InferenceEngine.GenerateTokensGeneric(...)``` method to make it create and return channels ```generatedTokensCh``` and ```errorCh```. ```InferenceEngine.GenerateTokensGeneric(...)``` runs parallelly, and calls the function given as ```tokenGeneratorFn``` argument. This ```tokenGeneratorFn``` function is ```InferenceEngine.generateTokensInternal(...)``` for normal application, not for unit test.

Then, it initiates an infinite loop to wait upcoming new token from ```generatedTokensCh``` or an error from ```errorCh```. If a new token was came from  ```generatedTokensCh```, it processes the new token in consideration with waiting bytes and waiting parts, then sends it via ```outputCh```. If an error was came from ```errorCh```, it returns the error and exits.

While exiting, it checks if there's an item in ```decodingContext.waitingParts```, it processes them and sends them via ```outputCh```.

>Waiting bytes and waiting parts are a temporary place for not processed tokens. For e.g., Llama models can generate emoji characters. More than one bytes form an emoji, as a [Unicode](https://en.wikipedia.org/wiki/Unicode) character. Llama uses [UTF-8](https://en.wikipedia.org/wiki/UTF-8) standard to encode a unicode character.<br>
>And, Llama models represent emojis with byte tokens like "<0xE2><0x9C>", "<0x88>", "<0xEF><0xB8><0x8F>". Let's say an emoji was encoded in 6 bytes, Llama encodes it as multiple different byte tokens in a few iterations. So, if we get a new generated byte token, we need to check if it requires another byte token to be rendered. Also, emojis can consist of multiple emojis, because of this, we need to handle these types of situations.

>For instance, the :airplane: ([Airplane](https://www.iemoji.com/view/emoji/145/travel-places/airplane)) emoji is formed by 6 bytes, as 3 different byte tokens: ```0xE2 0x9C 0x88 0xEF 0xB8 0x8F```. ```"<0xE2><0x9C>"```, ```"<0x88>"```, and ```"<0xEF><0xB8><0x8F>"``` byte tokens are generated respectively. You must be able to handle the first ```"<0xE2><0x9C>"``` byte token when it was generated. When you take this new token, should you send it directly to the output channel to render, or add it to waiting bytes list to wait for the next ```"<0x88>"``` and other required bytes to represent a valid UTF-8 character, that's the problem.

>The emoji rendering process will be discussed in a dedicated chapter further. But if you want to learn more, you can check out the unit tests prefixed as ```TestSimulatedEmojiOutput...``` in the [main unit test](../cmd/main_test.go).

<sup>from [src/inference/inference.go](../src/inference/inference.go)</sup>

```go
func (ie *InferenceEngine) generateStringInternal(promptTokens []model.TokenId, outputCh chan<- GeneratedPart, outputErrorCh chan<- error, tokenGeneratorFn TokenGeneratorFn) {
    decodingContext := &generationDecodingContext{
        waitingBytes: make([]byte, 0),
        waitingParts: make([]GeneratedPart, 0),
    }
    lastGenerationState := GSInProgress

    generatedTokensCh, errorCh := ie.GenerateTokensGeneric(promptTokens, tokenGeneratorFn)
    loop := true
    for loop {
        select {
        case generatedTokenIdResult, ok := <-generatedTokensCh:
            if !ok {
                loop = false
                break
            }
            generatedToken, generatedTokenStr, addedToWaiting := ie.TokenToString(generatedTokenIdResult.value, decodingContext)
            ...
            result := GeneratedPart{
                ...
            }
            ...
            outputCh <- result
        case err, ok := <-errorCh:
            if !ok || err == nil {
                continue
            }
            outputErrorCh <- err
            return
        }
    }
    decodingContext.decodingFinished = true
    if len(decodingContext.waitingParts) > 0 {
        for i, waitingPart := range decodingContext.waitingParts {
            result := GeneratedPart{
                ...
            }
            ...
            outputCh <- result
        }
    }
}
```

## **13.5. Internals of generateTokensInternal(...)**

### **13.5.1. Preparing the input tokens tensor**

We instantiate a new [model.InferenceContext](../src/model/inferencecontext.go) object as ```infContext``` to keep temporary data about the current generation process, especially the ```CacheK``` and ```CacheV``` tensors that keep key and value. These concepts will be discussed in further chapters.

<sup>from [src/model/inferencecontext.go](../src/model/inferencecontext.go)</sup>

```go
type InferenceContext struct {
    SequenceLength int // context size used during inference

    CacheK []*ml.Tensor
    CacheV []*ml.Tensor

    logFn func(format string, v ...any)
}
```

![STAGE 2: Creating tokens tensor Diagram](./images/DIAG01-STAGE02-creating-tokens-tensor.drawio.svg)
<sup>*Diagram: **Creating tokens tensor**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

A tensor named ```tokens``` with ```DT_INT32``` data type and with shape of ```{infContext.SequenceLength}``` (in our default case it's ```{200}```) is instantiated. Then, this tensor will be filled with integer value in ```ie.model.Vocabulary.PadId```, which is ```-1``` default.

After instantiation, prompt tokens is put into this ```tokens``` tensor.

The ```tokens``` tensor with shape ```{200}``` for the prompt is:

```go
promptString = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Einstein<|eot_id|><|start_header_id|>user<|end_header_id|>

Describe your theory.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"

[128000, 128006, 9125, 128007, 271, 2675, 527, 55152, 128009, 128006, 882, 
128007, 271, 75885, 701, 10334, 13, 128009, 128006, 78191, 128007, 271, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
shape=[200], dtype=Int32
```

<sup>from [src/inference/inference.go](../src/inference/inference.go)</sup>

```go
func (ie *InferenceEngine) generateTokensInternal(promptTokens []model.TokenId, generatedTokensCh chan<- generationStepResult[model.TokenId], errorCh chan<- error) {
    infContext := ie.CreateInferenceContext()
    ...
    tokens, err := ml.Full([]int{infContext.SequenceLength}, ml.DT_INT32, int32(ie.model.Vocabulary.PadId))
    ...
    for i, token := range promptTokens {
        if err := tokens.SetItem([]int{i}, int32(token)); err != nil {
            errorCh <- err
            return
        }
    }
    common.GLogger.DebugPrintf("Created input tokens tensor: shape(%v)", tokens.Size)
    ...
}
```

### **13.5.2. Looping through sequence length**

Now, we have a ```tokens``` tensor with shape ```{200}```. This tensor's first 22 items ```tokens[0:22], indices between 0 and 21``` are filled with our prompt tokens, remaining items are ```-1```.

We initiate a for loop with ```curPos``` variable from ```22 to 200-1```, which is ```promptLength to infContext.SequenceLength - 1```.

The first iteration of this loop takes all of prompt tokens as input, then other iterations takes the latest generated one token as input. Because other iterations use the data at the KV Cache (key-value cache) in ```infContext```.

>**Info:** The term [logits](https://en.wikipedia.org/wiki/Logit) stands for a tensor containing probabilities of each alternative. In machine learning, particularly in neural networks, classification models provide their results in the form of logits. Output layers have neurons in count of output alternatives, each containing a float representing the probability of "this item is the prediction".<br>
>In our scenario, the vocabulary in the "tokenizer.model" file contains 128,256 tokens. Consequently, our logits tensor will have 128,256 items in one of its dimensions. Then we perform the [Argmax](https://en.wikipedia.org/wiki/Arg_max) operation over the logits tensor, which returns the index (token id) of the item with the highest probability.

>**Additional info:** In the LLM domain, we can use the [temperature](https://www.promptingguide.ai/introduction/settings) value to randomly select from the most likely alternative tokens, allowing for the generation of different outputs with each run. However, in our project, we haven't implemented this functionality, instead, we just return exactly the token with the highest probability.

![STAGE 3: Looping through sequence length Diagram](./images/DIAG01-STAGE03-looping-through-sequence-length.drawio.svg)
<sup>*Diagram: **Looping through sequence length**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

The flow is:

* The first iteration:

    * ```curPos = 22```, ```prevPos = 0```,
    * ```inputTokensSlice``` has 22 items, which are prompt tokens,
    * Execute ```ie.model.Transformer.Forward(...)``` to do first inference with our transformer model to retrieve logits,
    * We have ```logits``` tensor with ```DT_F32``` data type and with shape of ```{22, 128256}```.
    * But, we need only probabilities of the last sequence, we take the last one via ```logits.Slice([]int{21}, []int{22}```:

        ```go
        logits, err = logits.Slice([]int{logits.Size[0] - 1}, []int{logits.Size[0]})
        ```

    * Now, our ```logits``` tensor's shape was become ```{1, 128256}```,
    * Execute [ml.Argmax](../src/ml/operations_impl.go) function over our ```logits```, it will return ```nextToken``` tensor with ```DT_INT32``` data type and with shape of ```{1, 1}```,
    * Take *the only one item* from the ```nextToken``` token via ```nextToken.Item()``` as ```int32``` into ```nextTokenId``` variable, the value in our case is ```7979```,
    * Take ```32th``` item from the ```tokens``` tensor into ```existingToken``` variable, then if it is not ```-1``` (```ie.model.Vocabulary.PadId```), take the existing token as next token. This step was implemented by the [original Llama 3.1 Python repository of Meta](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/reference_impl/model.py), and I have kept it as is,
    * Set ```22th``` item of the ```tensor``` to ```7979``` (value of ```nextTokenId```),
    * Check if the ```nextTokenId``` equals to ```ie.model.Vocabulary.EndOfSentenceId```, if yes, send a signal of EOS reached via ```generatedTokensCh``` channel,
    * Check if ```curPos``` reached to the sequence length, if yes, send a signal of maximum sequence length reached via ```generatedTokensCh``` channel,
    * Otherwise, send ```nextTokenId``` with a signal of in progress via ```generatedTokensCh``` channel,
    * Continue to loop with next iteration.
* Other iterations:
    * ```curPos = 23```, ```prevPos = 22```,
    * Take ```32th``` item from the ```tokens``` tensor into ```inputTokensSlice```,
    * ```inputTokensSlice``` has 1 item, which is the last generated token,
    * Execute ```ie.model.Transformer.Forward(...)``` to do first inference with our transformer model to retrieve logits,
    * Perform the same steps defined for "The first iteration" above.

<sup>from [src/inference/inference.go](../src/inference/inference.go)</sup>

```go
func (ie *InferenceEngine) generateTokensInternal(promptTokens []model.TokenId, generatedTokensCh chan<- generationStepResult[model.TokenId], errorCh chan<- error) {
    ...
    prevPos := 0
    for curPos := promptLength; curPos < infContext.SequenceLength; curPos++ {
        inputTokensSlice, err := tokens.Slice([]int{prevPos}, []int{curPos})
        if err != nil {
            errorCh <- err
            return
        }
        common.GLogger.DebugPrintf("=======================================\n\n")
        common.GLogger.DebugPrintf("Running Transformer.Forward for curPos: %d, prevPos: %d, inputTokensSlice: shape(%v)", curPos, prevPos, inputTokensSlice.Size)
        logits, err := ie.model.Transformer.Forward(infContext, inputTokensSlice, prevPos)
        ...
        if logits, err = logits.Slice([]int{logits.Size[0] - 1}, []int{logits.Size[0]}); err != nil {
            errorCh <- err
            return
        }
        nextToken, err := ml.Argmax(logits, len(logits.Size)-1) // shape=[1,1] dtype=DT_INT32
        if err != nil {
            errorCh <- err
            return
        }
        nextTokenId := model.TokenId(nextToken.Item().(int32))
        // Comment in original Python code: only replace token if prompt has already been generated
        existingToken, err := tokens.GetItem([]int{curPos})
        if err != nil {
            errorCh <- err
            return
        }
        existingTokenId := model.TokenId(existingToken.(int32))
        if existingTokenId != ie.model.Vocabulary.PadId {
            nextTokenId = existingTokenId
        }
        if err = tokens.SetItem([]int{curPos}, int32(nextTokenId)); err != nil {
            errorCh <- err
            return
        }
        common.GLogger.DebugPrintf("Generated token for curPos: %d, prevPos: %d, token id: %d", curPos, prevPos, nextTokenId)

        eosReached := nextTokenId == ie.model.Vocabulary.EndOfSentenceId
        prevPos = curPos
        if eosReached {
            generatedTokensCh <- generationStepResult[model.TokenId]{
                state: GSFinishedByReachingEOS,
                value: nextTokenId,
            }
            break
        }
        if curPos+1 == infContext.SequenceLength {
            generatedTokensCh <- generationStepResult[model.TokenId]{
                state: GSFinishedByReachingSeqLen,
                value: nextTokenId,
            }
            break
        }
        generatedTokensCh <- generationStepResult[model.TokenId]{
            state: GSInProgress,
            value: nextTokenId,
        }
    }
}
```

## **13.6. Calling listenGenerationChannels(...)**

We've dove into the internals of some functions that generate new tokens and send them via the ```generatedPartCh``` channel, starting from the [InferenceEngine.GenerateString(...)](../src/inference/inference.go) method, so far.

We will discuss the details of ```ie.model.Transformer.Forward(...)``` function in further chapters.

Now, we explain the [listenGenerationChannels](../cmd/main.go) function shortly.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
func main() {
    ...
    generatedPartCh, errorCh := engine.GenerateString(tokens)

    wg.Add(1)
    go listenGenerationChannels(&wg, ctx, generatedPartCh, errorCh)

    wg.Wait()
    ...
}
```

This function runs as a ```goroutine``` parallelly, listens for incoming new tokens from ```generatedPartCh``` channel and errors from ```errorCh```, or a cancellation signal from ```ctx.Done()``` via initiating an infinite loop.

If it receives an [inference.GeneratedPart](../src/inference/inference.go) object from ```generatedPartCh``` channel, it precesses the object, updates the console screen via ```appState.updateOutput()``` method.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
func listenGenerationChannels(wg *sync.WaitGroup, ctx context.Context, generatedPartCh <-chan inference.GeneratedPart, errorCh <-chan error) {
	defer wg.Done()
	loop := true
	spacesAfterEmoji := ""
	for loop {
		select {
		case generatedPart, ok := <-generatedPartCh:
			if !ok {
				loop = false
				appState.waitingRunesExtraStr = ""
				fmt.Fprintln(appState.consoleOutWriter)
				break
			}
			if !generatedPart.IsResendOfWaiting {
				appState.generatedTokenIds = append(appState.generatedTokenIds, generatedPart.TokenId)
				appState.generatedTokens = append(appState.generatedTokens, generatedPart.Token)
			}

			if len(spacesAfterEmoji) > 0 && len(generatedPart.WaitingRunesExtraStr) == 0 {
				// If space characters should be added between the emoji and generatedPart.DecodedString
				// which generated at previous iteration, add them
				generatedPart.DecodedString = spacesAfterEmoji + generatedPart.DecodedString
				spacesAfterEmoji = ""
			} else {
				// If there is some emoji in the generated string, add space characters between the emoji and waitingRunesExtraStr
				spacesAfterEmoji = generateRequiredSpacesAfterEmoji(generatedPart.WaitingRunesExtraStr)
				generatedPart.WaitingRunesExtraStr = spacesAfterEmoji + generatedPart.WaitingRunesExtraStr
			}
			appState.waitingRunesExtraStr = generatedPart.WaitingRunesExtraStr

			if generatedPart.AddedToWaiting {
				appState.addedToWaitingCount++
			} else {
				appState.addedToWaitingCount = 0
				appState.generatedText += generatedPart.DecodedString
			}
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
		// If there is some emoji in the generated string, add space characters between the emoji and waitingRunesExtraStr
		appState.generatedText += generateRequiredSpacesAfterEmoji(appState.waitingRunesExtraStr)
		appState.generatedText += appState.waitingRunesExtraStr
		appState.updateOutput()
	}
}
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: TOKENIZATION](./12-TOKENIZATION.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: MAKING PREDICTION with LLAMA MODEL - 1&nbsp;&nbsp;&gt;](./14-MAKING-PREDICTION-WITH-LLAMA-MODEL-1.md)

</div>

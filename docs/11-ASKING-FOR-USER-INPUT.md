# **11. ASKING FOR USER INPUT**

At this stage, we have a complete model object containing Llama transformer architecture, weights, tokenizer, and all other required things should be loaded or instantiated.

Depending on user preferences and usage requirements, user interfaces and flows can be designed in various ways. Therefore, it is more appropriate to keep this chapter very brief.

First, we initiate the inference args, which has currently one variable ```SequenceLength```, which may vary and can be maximum of 2048 for Llama 3.1 models. This value makes our model break when reaching 200 total tokens (including prompt tokens). Our text generation process may be finished before reaching this limit, by generating an EOS (end of sentence) token too.

The application asks user for a prompt, with some predefined options in ```predefinedPrompts``` variable via ```askUserPromptChoice(...)``` function.

If user chose an option that have ```IsChatMode=true``` defined, the application surrounds the prompt with special tokens which means this prompt contains an *instruction*. The Llama Instruct/Chat models are trained with these instruction patterns, so these models can understand that it is a chat *instruction* that must be followed.

Also, if ```IsChatMode=true```, this means that our prompt has a *system prompt* alongside the normal prompt, in this case, we call the normal prompt as *instruction prompt*. Briefly, a *system prompt* is the prompt part that enables us to give some directions to the model independently from the instruction, like defining who should the model act as, how an output format we want, etc... To understand more about the system prompt, you can check out the system prompt parts of our ```predefinedPrompts``` variable.

Additionally, the Llama 3.1 Instruct model supports few-shot learning type prompts that has agent/user conversation examples, but in our project we didn't implemented multiple dialog conversations. If you want to learn more, you can find more complex examples in the ```dialogs: List[Dialog]``` variable at the [original Llama 3 Python repository of Meta's example_chat_completion.py file](https://github.com/meta-llama/llama3/blob/main/example_chat_completion.py).

With the ```[Text completion]``` choices, the model is used only to perform text completion task. New tokens will be generated according to the input prompt text.

With the ```[Chat mode]``` choices, the application starts the prompt with ```<|begin_of_text|>``` string to specify "this is an instruction prompt". Also it surrounds the system prompt part with ```<|start_header_id|>system<|end_header_id|>\n``` and ```<|eot_id|>``` strings to specify this part is a *system prompt*, surrounds the user prompt part with ```<|start_header_id|>user<|end_header_id|>\n``` and ```<|eot_id|>``` strings to specify this part is a *user prompt*.

At the end, a chat mode prompt that we have in ```userPromptStr``` string variable will be look like following:

```sh
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Always answer with emojis<|eot_id|><|start_header_id|>user<|end_header_id|>

How to go from Beijing to NY?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
```


>LINK FOR OLD VERSION: For more information about prompting Llama Chat models: **"How to Format Chat Prompts"** chapter of [A guide to prompting Llama 2](https://replicate.com/blog/how-to-prompt-llama)

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
...
var predefinedPrompts = []PromptInput{
    {IsChatMode: false, Prompt: "Hello, my name is"},
    {IsChatMode: true, SystemPrompt: "You are Einstein", Prompt: "Describe your theory."},
    {IsChatMode: true, SystemPrompt: "Answer in 20 words, directly, and without an introduction", Prompt: "Can you explain what is Theory of relativity?"},
    {IsChatMode: true, SystemPrompt: "You are a pirate", Prompt: "Explain what is quantum computer in 20 words."},
    {IsChatMode: true, SystemPrompt: "Always answer with emojis", Prompt: "How to go from Beijing to NY?"},
    {IsChatMode: true, SystemPrompt: "Answer with only one emoji", Prompt: "What is the flag of Turkey?"},
}
...
func main() {
    ...
    fmt.Printf("Model \"%s\" was loaded.\n", modelDir)

	fmt.Printf("Developed by: Adil Alper DALKIRAN")

    fmt.Printf("\n\n\n")

    inferenceArgs := common.NewInferenceArgs()
    inferenceArgs.SequenceLength = 200

    engine := inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn)

	userPrompt := askUserPromptChoice(llamaModel)

	var tokens []model.TokenId

	if userPrompt.IsChatMode {
		userPromptParts := []inference.PromptPart{
			{Header: "system", Content: userPrompt.SystemPrompt},
			{Header: "user", Content: userPrompt.Prompt},
		}
		tokens, err = engine.Tokenize(userPromptParts)
		if err != nil {
			common.GLogger.ConsoleFatal(err)
		}
	} else {
		userPromptStr := userPrompt.Prompt
		if !strings.HasSuffix(userPromptStr, " ") {
			userPromptStr += " "
		}
		tokens, err = engine.TokenizeString(userPromptStr, true)
		if err != nil {
			common.GLogger.ConsoleFatal(err)
		}
	}

    fmt.Printf("\n\n\n")
    ...
}
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: RoPE \(ROTARY POSITIONAL EMBEDDINGS\)](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: TOKENIZATION&nbsp;&nbsp;&gt;](./12-TOKENIZATION.md)

</div>

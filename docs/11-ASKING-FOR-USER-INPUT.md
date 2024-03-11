# **11. ASKING FOR USER INPUT**

At this stage, we have a complete model object containing LLaMa transformer architecture, weights, tokenizer, and all other required things should be loaded or instantiated.

Depending on user preferences and usage requirements, user interfaces and flows can be designed in various ways. Therefore, it is more appropriate to keep this chapter very brief.

First, we initiate the inference args, which has currently one variable ```SequenceLength```, which may vary and can be maximum of 4096 for LLaMa 2 models. This value makes our model break when reaching 200 total tokens (including prompt tokens). Our text generation process may be finished before reaching this limit, by generating an EOS (end of sentence) token too.

The application asks user for a prompt, with some predefined options in ```predefinedPrompts``` variable via ```askUserPromptChoice(...)``` function.

If user chose an option that have ```IsChatMode=true``` defined, the application surrounds the prompt with ```B_INST``` and ```E_INST``` strings (```"[INST]"``` and ```"[/INST]"```) which means this prompt contains an *instruction*. The LLaMa Chat models are trained with these instruction patterns, so these models can understand that it is a chat *instruction* that must be followed.

Also, if ```IsChatMode=true```, this means that our prompt has a *system prompt* alongside the normal prompt, in this case, we call the normal prompt as *instruction prompt*. Briefly, a *system prompt* is the prompt part that enables us to give some directions to the model independently from the instruction, like defining who should the model act as, how an output format we want, etc... To understand more about the system prompt, you can check out the system prompt parts of our ```predefinedPrompts``` variable. To differentiate the system prompt and the instruction in the prompt string, the application surrounds the system prompt with ```B_SYS``` and ```E_SYS``` strings (```<<SYS>>\n``` and ```\n<</SYS>>\n\n```) which means this part is a *system prompt*.

Additionally, the LLaMa 2 Chat model supports few-shot learning type prompts that has agent/user conversation examples, but in our project we didn't implemented multiple dialog conversations. If you want to learn more, you can find more complex examples in the ```dialogs: List[Dialog]``` variable at the [original LLaMa 2 Python repository of Meta's example_chat_completion.py file](https://github.com/facebookresearch/llama/blob/main/example_chat_completion.py).

At the end, a chat mode prompt that we have in ```userPromptStr``` string variable will be look like following:

```sh
"[INST] <<SYS>>
Always answer with emojis
<</SYS>>

How to go from Beijing to NY? [/INST]"
```


>For more information about prompting LLaMa Chat models: **"How to Format Chat Prompts"** chapter of [A guide to prompting Llama 2](https://replicate.com/blog/how-to-prompt-llama)

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

    fmt.Printf("\n\n\n")

    inferenceArgs := common.NewInferenceArgs()
    inferenceArgs.SequenceLength = 200

    engine := inference.NewInferenceEngine(llamaModel, inferenceArgs, logFn)

    userPrompt := askUserPromptChoice(llamaModel)
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
    ...
}
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: RoPE \(ROTARY POSITIONAL EMBEDDINGS\)](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: TOKENIZATION&nbsp;&nbsp;&gt;](./12-TOKENIZATION.md)

</div>

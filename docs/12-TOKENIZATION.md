# **12. TOKENIZATION**

> **Recap from [10. RoPE (ROTARY POSITIONAL EMBEDDINGS)](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md)**:<br>
>For decades, *embedding* has been the most commonly used technique to represent words, concepts, or tokens in NLP (Natural Language Processing) world. Typically, an embedding model is trained to let them learn frequencies of use of tokens together. The tokens are placed in suitable positions within a multi-dimensional space, where distances reflect the difference or similarity between them.

In computing, computers commonly process the information *digitally* as *bit*s. Even when emerging new computer system approaches like quantum computing come up, State of the Art of them also needs to convert their outcome into the way digital computers understand. Then, *bit*s come together to form *byte*s, and *byte*s create data types such as *array*s, *integer*s, and *float*s. Most of these types are different ways to represent a number, think of, *character*s in a *string* are represented with numbers, behind the curtains.

In NLP context, a natural language text string must be defined as *token*s at first, then commonly tokens are converted to *embedding* vectors. Tokens have a corresponding integer ID value, then they are represented as numbers.

>From [Understanding “tokens” and tokenization in large language models](https://blog.devgenius.io/understanding-tokens-and-tokenization-in-large-language-models-1058cd24b944): A token is typically not a word; it could be a smaller unit, like a character or a part of a word, or a larger one like a whole phrase.

Llama 3 and 3.1 models use [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/en/chapter6/5) tokenizer model which was saved as [Tiktoken tokenizer format](https://github.com/openai/tiktoken) file. Llama 2 models use a [SentencePiece (SPM)](https://github.com/google/sentencepiece) tokenizer.


![STAGE 1: Tokenization Diagram](./images/DIAG01-STAGE01-tokenization.drawio.svg)
<sup>*Diagram: **Tokenization**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

In our project, we get the ```userPromptStr``` prompt string and get token id integers into ```tokens``` variable via [InferenceEngine.Tokenize(...)](../src/inference/tokenize.go) method. This method separates the input text and gets corresponding token ids using [Model.Vocabulary](../src/model/model.go) that has been already loaded from "tokenizer.model" file.

Then, we regenerate our prompt string from token id integers, to cross-check the result.

<sup>from [cmd/main.go](../cmd/main.go)</sup>

```go
func main() {
    ...
    tokens, err := engine.Tokenize(userPromptStr, true)
    if err != nil {
        common.GLogger.ConsoleFatal(err)
    }

    appState.promptTokens, appState.promptText = engine.TokenBatchToString(tokens)
    ...
}
```

As an example, the following prompt will be separated to tokens, and represented as token id's via [separatePieces(...)](../src/inference/tokenize.go) and [InferenceEngine.Tokenize(...)](../src/inference/tokenize.go) functions as following:

Prompt string:

```go
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Einstein<|eot_id|><|start_header_id|>user<|end_header_id|>

Describe your theory.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"
```

Separated tokens:

```go
// Array of 22 strings:
"<|begin_of_text|>", "<|start_header_id|>", "system", "<|end_header_id|>", "\n\n", "You", " are", " Einstein", "<|eot_id|>", "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n", "Describe", " your", " theory", ".", "<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"
```

Token IDs corresponding each token:

```go
// Array of 32 integers
[128000, 128006, 9125, 128007, 271, 2675, 527, 55152, 128009, 128006, 882, 128007, 271, 75885, 701, 10334, 13, 128009, 128006, 78191, 128007, 271]
```

Sources:

* [Understanding “tokens” and tokenization in large language models](https://blog.devgenius.io/understanding-tokens-and-tokenization-in-large-language-models-1058cd24b944)

Now, we have token id integers of our input prompt string. The next is to generate new tokens using this input.

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: ASKING FOR USER INPUT](./11-ASKING-FOR-USER-INPUT.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: GENERATING NEXT TOKENS&nbsp;&nbsp;&gt;](./13-GENERATING-NEXT-TOKENS.md)

</div>

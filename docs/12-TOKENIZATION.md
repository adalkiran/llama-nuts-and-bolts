# **12. TOKENIZATION**

> **Recap from [10. RoPE (ROTARY POSITIONAL EMBEDDINGS)](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md)**:<br>
>For decades, *embedding* has been the most commonly used technique to represent words, concepts, or tokens in NLP (Natural Language Processing) world. Typically, an embedding model is trained to let them learn frequencies of use of tokens together. The tokens are placed in suitable positions within a multi-dimensional space, where distances reflect the difference or similarity between them.

In computing, computers commonly process the information *digitally* as *bit*s. Even when emerging new computer system approaches like quantum computing come up, State of the Art of them also needs to convert their outcome into the way digital computers understand. Then, *bit*s come together to form *byte*s, and *byte*s create data types such as *array*s, *integer*s, and *float*s. Most of these types are different ways to represent a number, think of, *character*s in a *string* are represented with numbers, behind the curtains.

In NLP context, a natural language text string must be defined as *token*s at first, then commonly tokens are converted to *embedding* vectors. Tokens have a corresponding integer ID value, then they are represented as numbers.

>From [Understanding “tokens” and tokenization in large language models](https://blog.devgenius.io/understanding-tokens-and-tokenization-in-large-language-models-1058cd24b944): A token is typically not a word; it could be a smaller unit, like a character or a part of a word, or a larger one like a whole phrase.

LLaMa models use a [SentencePiece (SPM)](https://github.com/google/sentencepiece) tokenizer.

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
"[INST] <<SYS>>\nYou are Einstein\n<</SYS>>\n\nDescribe your theory. [/INST]"
```

Separated tokens:

```go
// Array of 31 strings:
"[", "INST", "]", "▁<<", "SY", "S", ">>", "<0x0A>", "You", "▁are", "▁Eins", "tei", 
"n", "<0x0A>", "<<", "/", "SY", "S", ">>", "<0x0A>", "<0x0A>", "Desc", "rib", "e", "▁your", 
"▁theory", ".", "▁[", "/", "INST", "]"
```

>Note that, we've implemented a light version of [SentencePiece (SPM)](https://github.com/google/sentencepiece) tokenizer. The original one has more complex and optimized algorithm that considers *score*s of each token in the vocabulary, we don't use the *score* properties of tokens. This means that, separated outputs of the original SPM tokenizer and our implementation may differ.

Separated tokens (as [SentencePiece](../src/sentencepiece/model.go) objects array):

```yaml
# Array of 32 SentencePiece objects:
[id: 1, "<s>" score: 0.000000, type: CONTROL],
[id: 29961, "[" score: -29702.000000, type: NORMAL], 
[id: 25580, "INST" score: -25321.000000, type: NORMAL], 
[id: 29962, "]" score: -29703.000000, type: NORMAL], 
[id: 3532, "▁<<" score: -3273.000000, type: NORMAL], 
[id: 14816, "SY" score: -14557.000000, type: NORMAL], 
[id: 29903, "S" score: -29644.000000, type: NORMAL], 
[id: 6778, ">>" score: -6519.000000, type: NORMAL], 
[id: 13, "<0x0A>" score: 0.000000, type: BYTE], 
[id: 3492, "You" score: -3233.000000, type: NORMAL], 
[id: 526, "▁are" score: -267.000000, type: NORMAL], 
[id: 16943, "▁Eins" score: -16684.000000, type: NORMAL], 
[id: 15314, "tei" score: -15055.000000, type: NORMAL], 
[id: 29876, "n" score: -29617.000000, type: NORMAL], 
[id: 13, "<0x0A>" score: 0.000000, type: BYTE], 
[id: 9314, "<<" score: -9055.000000, type: NORMAL], 
[id: 29914, "/" score: -29655.000000, type: NORMAL], 
[id: 14816, "SY" score: -14557.000000, type: NORMAL], 
[id: 29903, "S" score: -29644.000000, type: NORMAL], 
[id: 6778, ">>" score: -6519.000000, type: NORMAL], 
[id: 13, "<0x0A>" score: 0.000000, type: BYTE], 
[id: 13, "<0x0A>" score: 0.000000, type: BYTE], 
[id: 19617, "Desc" score: -19358.000000, type: NORMAL], 
[id: 1091, "rib" score: -832.000000, type: NORMAL], 
[id: 29872, "e" score: -29613.000000, type: NORMAL], 
[id: 596, "▁your" score: -337.000000, type: NORMAL], 
[id: 6368, "▁theory" score: -6109.000000, type: NORMAL], 
[id: 29889, "." score: -29630.000000, type: NORMAL], 
[id: 518, "▁[" score: -259.000000, type: NORMAL], 
[id: 29914, "/" score: -29655.000000, type: NORMAL], 
[id: 25580, "INST" score: -25321.000000, type: NORMAL], 
[id: 29962, "]" score: -29703.000000, type: NORMAL]
```

>Note that, at first we had 31 token strings, but now we have 32 SentencePiece objects. Because while tokenizing the string, we have added a ```<s>``` *BOS(Begin of Sentence)* token at the beginning of the sequence: ```[id: 1, "<s>" score: 0.000000, type: CONTROL]```<br>
>Also check out the ```type```s of the SentencePiece objects, most of them are ```NORMAL``` pieces, some of them ```CONTROL``` and ```BYTE``` pieces. In this example, all of the ```BYTE``` pieces are ```"<0x0A>"``` correspond to ```"\n" newline character```.<br>
>The ```BYTE``` pieces are used also to generate emojis. We will discuss further in a dedicated chapter.

Token IDs corresponding each token:

```go
// Array of 32 integers
[1, 29961, 25580, 29962, 3532, 14816, 29903, 6778, 13, 3492, 526, 16943, 15314, 
29876, 13, 9314, 29914, 14816, 29903, 6778, 13, 13, 19617, 1091, 29872, 596, 
6368, 29889, 518, 29914, 25580, 29962]
```

Sources:

* [Understanding “tokens” and tokenization in large language models](https://blog.devgenius.io/understanding-tokens-and-tokenization-in-large-language-models-1058cd24b944)

Now, we have token id integers of our input prompt string. The next is to generate new tokens using this input.

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: ASKING FOR USER INPUT](./11-ASKING-FOR-USER-INPUT.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: GENERATING NEXT TOKENS&nbsp;&nbsp;&gt;](./13-GENERATING-NEXT-TOKENS.md)

</div>

# **5. LOADING TOKENIZER MODEL**

In this chapter, we'll walk through the process of loading tokenizer (vocabulary) model stored in the "tokenizer.model" file.

In our case, Llama 3.1's tokenizer file *tokenizer.model* stores a [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/en/chapter6/5) tokenizer model in text and base64 format.<br>

Since Llama 3 version, Llama models have started to use OpenAI's [Tiktoken tokenizer](https://github.com/openai/tiktoken).

## **5.1. Calling loadVocab()**

[loadVocab()](../src/model/loader.go) is called if ```includeVocab``` is true.

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func LoadModelEx(modelDir string, includeTensors bool, includeVocab bool) (*Model, error) {
    model := &Model{}
    ...
    if includeVocab {
        err := loadVocab(modelDir, model)
        if err != nil {
            return nil, err
        }
    }
    ...
}

func loadVocab(modelDir string, model *Model) error {
    vocabFilePath := filepath.Join(modelDir, "tokenizer.model")
    common.GLogger.ConsolePrintf("Loading vocabulary/tokens file: \"%s\"...", vocabFilePath)
    vocabBpe, err := tiktoken.Load(vocabFilePath)
    if err != nil {
        return err
    }

    model.Vocabulary = NewVocabulary(vocabModelProto)
    common.GLogger.ConsolePrintf("Found %d tokens in the model.", len(model.Vocabulary.IdToToken))
    return nil
}
```

In ```tiktoken.Load(...)``` function, we call ```loadTiktokenBpe(...)```function, we get a file instance by opening specified file. Then, we read this text file line-by-line, decode the base64 part, add hardcoded special tokens, and then return as [ModelData](../src/tiktoken/model.go).

For original implementation details, check out [this function](https://github.com/openai/tiktoken/blob/c0ba74c238d18b4824c25f3c27fc8698055b9a76/tiktoken/load.py#L143) and [this class](https://github.com/meta-llama/llama-models/blob/5ee9cb5eaf92d542f1b1ee595af64a9ffdc07bac/models/llama3_1/api/tokenizer.py#L44).

<sup>Some sample lines from Llama 3.1 tokenizer.model file, with base64 decoded forms</sup>

```go
IQ== 0 //"!"
Ig== 1 //"\""
Iw== 2 //"#"
JA== 3 //"$"
JQ== 4 //"%"
Jg== 5 //"&"
Jw== 6 //"'"
KA== 7 //"("
KQ== 8 //")"
Kg== 9 //"*"
Kw== 10 //"+"
...
IHRo 270 //" th"
Cgo= 271 //"\n\n"
IGM= 272 //" c"
bGU= 273 //"le"
IHM= 274 //" s"
aXQ= 275 //"it"
```

<sup>from [src/tiktoken/tiktokenreader.go](../src/tiktoken/tiktokenreader.go)</sup>

```go
func Load(vocabFilePath string) (*ModelData, error) {
    mergeableRanks, err := loadTiktokenBpe(vocabFilePath)
    if err != nil {
        return nil, err
    }
    baseTokensCount := len(mergeableRanks)

    reservedSpecialTokensCount := 256

    specialTokensArr := []string{
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>", // end of message
        "<|eot_id|>", // end of turn
        "<|python_tag|>",
    }

    reservedTokensArr := make([]string, reservedSpecialTokensCount-len(specialTokensArr))
    for i := 0; i < len(reservedTokensArr); i++ {
        reservedTokensArr[i] = fmt.Sprintf("<|reserved_special_token_%d|>", 2+i)
    }
    specialTokensArr = append(specialTokensArr, reservedTokensArr...)

    specialTokens := make(map[string]int)
    for i, token := range specialTokensArr {
        specialTokens[token] = baseTokensCount + i
    }

    result := &ModelData{
        MergeableRanks: mergeableRanks,
        SpecialTokens:  specialTokens,

        BeginOfSentenceId: specialTokens["<|begin_of_text|>"],
        EndOfSentenceId:   specialTokens["<|end_of_text|>"],
        PadId:             -1,
        UnknownId:         -1,
        StopTokenIds:      []int{specialTokens["<|eom_id|>"], specialTokens["<|eot_id|>"]},
    }

    return result, nil
}
```

### **5.2. Returning Vocabulary Model**

We get the [ModelData](../src/tiktoken/model.go) object as ```vocabBpe```, then we call [NewVocabulary(...)](../src/model/vocabulary.go) function by specifying it. This function creates and returns a [Vocabulary](../src/model/vocabulary.go) object that has ```TokenToId```, ```IdToTokenId``` maps to provide two-way querying.<br>
Then, we assign [Vocabulary](../src/model/vocabulary.go) object to ```model.Vocabulary``` property.

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func loadVocab(modelDir string, model *Model) error {
    vocabFilePath := filepath.Join(modelDir, "tokenizer.model")
    common.GLogger.ConsolePrintf("Loading vocabulary/tokens file: \"%s\"...", vocabFilePath)
    vocabBpe, err := tiktoken.Load(vocabFilePath)
    if err != nil {
        return err
    }

    model.Vocabulary = NewVocabulary(vocabBpe)
    common.GLogger.ConsolePrintf("Found %d tokens in the model.", len(model.Vocabulary.IdToToken))
    return nil
}
```

And we can see output lines in the console as follows:

```sh
[INFO] ... Loading vocabulary/tokens file: "/workspace/models-original/Meta-Llama-3.1-8B-Instruct/tokenizer.model"...
[INFO] ... Found 128256 tokens in the model.
Model "/workspace/models-original/Meta-Llama-3.1-8B-Instruct" was loaded.
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: LOADING MODEL ARGS](./04-LOADING-MODEL-ARGS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: OBSOLETE - LOADING LLAMA 2 TOKENIZER MODEL&nbsp;&nbsp;&gt;](./06-OBSOLETE-LOADING-LLAMA-2-TOKENIZER-MODEL.md)

</div>

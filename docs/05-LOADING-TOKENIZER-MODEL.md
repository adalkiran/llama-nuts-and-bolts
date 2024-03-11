# **5. LOADING TOKENIZER MODEL**

In this chapter, we'll walk through the process of loading tokenizer (vocabulary) model stored in the "tokenizer.model" file.

In our case, LLamA 2's tokenizer file *tokenizer.model* stores a [SentencePiece (SPM)](https://github.com/google/sentencepiece) tokenizer model in Protobuf message format.<br>
Protobuf operates by adhering to a descriptor, which serves as a blueprint or schema defining the structure and data types within a serialized message. This descriptor defines message structures and guides the serializer and deserializer.

>See: [Protocol Buffers](https://protobuf.dev/) | [Protobuf definition best practices](https://medium.com/@akhaku/protobuf-definition-best-practices-87f281576f31) | [Protocol Buffers (ProtoBuf) with GoLang](https://medium.com/trendyol-tech/protocol-buffers-protobuf-with-golang-41d0d332745d)

The descriptor for deserializing our SentencePiece model file is [this Protobuf structure](https://github.com/google/sentencepiece/blob/022f8c3fed4d2feb4e4c670949cf01cef477dcc4/src/sentencepiece_model.proto), we have defined it in Go language as ```modelprotoDescriptor``` variable, in [src/sentencepiece/model.go](../src/sentencepiece/model.go). This ```modelprotoDescriptor``` definition style is specific for our code infrastructure.

Although there are lots of libraries to implement this messaging format, in this project, we implement it from scratch ourselves as we always do in the "nuts and bolts" mindset.

## **5.1. Calling loadVocab() and Creating ProtobufReader**

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
    vocabModelProto, err := sentencepiece.Load(vocabFilePath)
    if err != nil {
        return err
    }
    model.Vocabulary = NewVocabulary(vocabModelProto)
    common.GLogger.ConsolePrintf("Found %d tokens in the model.", len(model.Vocabulary.IdToToken))
    return nil
}
```

In ```SentencePiece.Load(...)``` function, we get a file instance by opening specified file. Then, we call ```protobuf.NewProtobufReader(...)``` function by providing the file instance along with ```modelprotoDescriptor``` variable defined in [src/sentencepiece/model.go](../src/sentencepiece/model.go)

<sup>from [src/sentencepiece/sentencepiecereader.go](../src/sentencepiece/sentencepiecereader.go)</sup>

```go
func Load(vocabFilePath string) (*ModelProto, error) {
    vocabFile, err := os.Open(vocabFilePath)
    if err != nil {
        return nil, err
    }
    defer vocabFile.Close()

    vocabReader := protobuf.NewProtobufReader(vocabFile, modelprotoDescriptor)
    ...
}
```

## **5.2. Calling ProtobufReader.Unmarshal()**

When we call ```vocabReader.Unmarshal()```, it reads the given "tokenizer.model" file in guidance and help of the given ```modelprotoDescriptor```. At the end of this process, as ```modelprotoDescriptor``` helps, it returns a [ModelProto](../src/sentencepiece/model.go) object that contains Pieces (token definitions) and other specifications of the tokenizer model.

>**Note: If you're curious about the details of how the Protobuf file structure can be read, please refer to:** [6. LOADING TOKENIZER MODEL \(DETAILS\)](../docs/06-LOADING-TOKENIZER-MODEL-DETAILS.md)

<sup>from [src/sentencepiece/sentencepiecereader.go](../src/sentencepiece/sentencepiecereader.go)</sup>

```go
func Load(vocabFilePath string) (*ModelProto, error) {
    ...
    modelVal, err := vocabReader.Unmarshal()
    if err != nil {
        return nil, err
    }
    model, ok := modelVal.(*ModelProto)
    if !ok {
        return nil, fmt.Errorf("cannot convert %v to *ModelProto", model)
    }
    return &model, nil
}
```

### **5.3. Returning Vocabulary Model**

We get the [ModelProto](../src/sentencepiece/model.go) object as ```vocabModelProto```, then we call [NewVocabulary(...)](../src/model/vocabulary.go) function by specifying it. This function creates and returns a [Vocabulary](../src/model/vocabulary.go) object that has ```TokenToId```, ```IdToTokenId``` maps to provide two-way querying.<br>
Then, we assign [Vocabulary](../src/model/vocabulary.go) object to ```model.Vocabulary``` property.

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func loadVocab(modelDir string, model *Model) error {
    vocabModelProto, err := sentencepiece.Load(vocabFilePath)
    if err != nil {
        return err
    }
    model.Vocabulary = NewVocabulary(vocabModelProto)
    common.GLogger.ConsolePrintf("Found %d tokens in the model.", len(model.Vocabulary.IdToToken))
    return nil
}
```

And we can see output lines in the console as follows:

```sh
[INFO] ... Loading vocabulary/tokens file: "/workspace/models-original/7B-chat/tokenizer.model"...
[INFO] ... Found 32000 tokens in the model.
Model "/workspace/models-original/7B-chat" was loaded.
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: LOADING MODEL ARGS](./04-LOADING-MODEL-ARGS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: LOADING TOKENIZER MODEL \(DETAILS\)&nbsp;&nbsp;&gt;](./06-LOADING-TOKENIZER-MODEL-DETAILS.md)

</div>

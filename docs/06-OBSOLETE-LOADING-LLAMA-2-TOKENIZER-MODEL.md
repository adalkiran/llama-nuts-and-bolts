# **6. OBSOLETE - LOADING LLAMA 2 TOKENIZER MODEL**

## **Being Obsolete Note:**

### The contents of this chapter are about the older tokenizer ([SentencePiece (SPM) model](https://github.com/google/sentencepiece)) used by Llama 1 and Llama 2 models. We haven't preferred to remove this chapter completely, because it contains comprehensive information about [SentencePiece (SPM) model](https://github.com/google/sentencepiece) and [Protocol Buffers](https://protobuf.dev/).

**Ways to go:**
<br>

* If you only want to learn about things used in the latest Llama version (3.1), please continue with the next chapter: [BFLOAT16 DATA TYPE](./07-BFLOAT16-DATA-TYPE.md)

<br>

* Otherwise, if you are curious about different things used by older Llama versions, even if they aren't no longer used by Llama 3.1 anymore, please continue reading this chapter.

<br>


---
---

Obsolete Llama 2 content starts

---
---

<br>


## **6.0. The tokenizer model used by Llama 2 version**

In this chapter, we'll walk through the process of loading tokenizer (vocabulary) model stored in the "tokenizer.model" file.

In our case, Llama 2's tokenizer file *tokenizer.model* stores a [SentencePiece (SPM)](https://github.com/google/sentencepiece) tokenizer model in Protobuf message format.<br>
Protobuf operates by adhering to a descriptor, which serves as a blueprint or schema defining the structure and data types within a serialized message. This descriptor defines message structures and guides the serializer and deserializer.

>See: [Protocol Buffers](https://protobuf.dev/) | [Protobuf definition best practices](https://medium.com/@akhaku/protobuf-definition-best-practices-87f281576f31) | [Protocol Buffers (ProtoBuf) with GoLang](https://medium.com/trendyol-tech/protocol-buffers-protobuf-with-golang-41d0d332745d) | [Protocol Buffer Basics: Go](https://protobuf.dev/getting-started/gotutorial) |  [Protocol Buffer Encoding](https://protobuf.dev/programming-guides/encoding)

The descriptor for deserializing our SentencePiece model file is [this Protobuf structure](https://github.com/google/sentencepiece/blob/022f8c3fed4d2feb4e4c670949cf01cef477dcc4/src/sentencepiece_model.proto), we have defined it in Go language as ```modelprotoDescriptor``` variable, in [llama2/src/sentencepiece/model.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/model.go). This ```modelprotoDescriptor``` definition style is specific for our code infrastructure.

Although there are lots of libraries to implement this messaging format, in this project, we implement it from scratch ourselves as we always do in the "nuts and bolts" mindset.

### **6.0.1. Calling loadVocab() and Creating ProtobufReader**

[loadVocab()](../src/model/loader.go) is called if ```includeVocab``` is true.

<sup>from [llama2/src/model/loader.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/model/loader.go)</sup>

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

In ```SentencePiece.Load(...)``` function, we get a file instance by opening specified file. Then, we call ```protobuf.NewProtobufReader(...)``` function by providing the file instance along with ```modelprotoDescriptor``` variable defined in [llama2/src/sentencepiece/model.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/model.go)

<sup>from [llama2/src/sentencepiece/sentencepiecereader.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/sentencepiecereader.go)</sup>

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

### **6.0.2. Calling ProtobufReader.Unmarshal()**

When we call ```vocabReader.Unmarshal()```, it reads the given "tokenizer.model" file in guidance and help of the given ```modelprotoDescriptor```. At the end of this process, as ```modelprotoDescriptor``` helps, it returns a [ModelProto](../src/sentencepiece/model.go) object that contains Pieces (token definitions) and other specifications of the tokenizer model.

>**Note: If you're curious about the details of how the Protobuf file structure can be read, please refer to:** [6. LOADING TOKENIZER MODEL \(DETAILS\)](../docs/06-LOADING-TOKENIZER-MODEL-DETAILS.md)

<sup>from [llama2/src/sentencepiece/sentencepiecereader.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/sentencepiecereader.go)</sup>

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

### **6.0.3. Returning Vocabulary Model**

We get the [ModelProto](../src/sentencepiece/model.go) object as ```vocabModelProto```, then we call [NewVocabulary(...)](../src/model/vocabulary.go) function by specifying it. This function creates and returns a [Vocabulary](../src/model/vocabulary.go) object that has ```TokenToId```, ```IdToTokenId``` maps to provide two-way querying.<br>
Then, we assign [Vocabulary](../src/model/vocabulary.go) object to ```model.Vocabulary``` property.

<sup>from [llama2/src/model/loader.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/model/loader.go)</sup>

```go
func loadVocab(modelDir string, model *Model) error {
    ...
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

## **6.1. Creating Main Object**

Here, we'll walk through the process of reading some pieces (vocabulary token) stored in the "tokenizer.model" file, step by step.

Protobuf messages may consist of multiple separated sub-messages/objects/object types, but for deserializing process, we need to have a root destination object. This is called as "main object" in our project.

```ProtobufReader.Unmarshal()``` function calls given Protobuf descriptor's [ProtoDescriptor.MainObjectConstructorFn](../src/protobuf/protobufreader.go) function. In our case, it is [modelprotoDescriptor.MainObjectConstructorFn](../src/sentencepiece/model.go). It instantiates an empty ```ModelProto``` with an empty ```Pieces``` array.

Once we have a main object, we can start to read and process the "message"s in the file.

The Protobuf protocol/format consists of "message"s with a "number" (type identifier), and the reader has "message processor" functions corresponding "number"s (type identifier).

To read the Protobuf file/stream, we have a simple flow:

* We initiate a loop,
* Read a message (via [ProtobufReader.readMessage()](../src/protobuf/protobufreader.go)),
* If we are at EOF (end of file), break the loop,
* Find corresponding message processor function for ```message.Number``` in [ProtoDescriptor.MessageProcessorFns](../src/protobuf/protobufreader.go) function map,
* Execute it (this function makes changes on the main object if needed),
* Check for errors, and continue if no error.
* We return this main object.

## **6.2. Reading a Message**

>We initiate a loop that always calls the ```ProtobufReader.readMessage()``` method. This loop will continue until it encounters EOF (end of file) or an error.

```ProtobufReader.readMessage()``` method:

* Checks if we are at the EOF (end of file). If yes, returns with ```ok=false``` to notify finished,
* Calls ```ProtobufReader.readField(...)``` method to read "message number" and "message data",
* Returns successfully read [Message](../src/protobuf/protobufreader.go) object or ```ok=false```.

<sup>from [llama2/src/protobuf/protobufreader.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/protobuf/protobufreader.go)</sup>

```go
func (pbr *ProtobufReader) readMessage() (message *Message, ok bool) {
    _, err := pbr.fileReader.Peek(1)
    if err != nil {
        return nil, false
    }
    number, item, ok := pbr.readField(pbr.fileReader)
    if !ok {
        return nil, false
    }
    return &Message{number, item}, true
}
```

```ProtobufReader.readField(...)``` method:

> This part was inspired by Google's original Protobuf project's "protowire" implementation, see: [wire.go of original protobuf-go Project](https://github.com/protocolbuffers/protobuf-go/blob/e8baad6b6c9e2bb1c48e4963866248d4f35d4fd7/encoding/protowire/wire.go)

* Backs up current file position, because this method works not completely deterministic, includes some type of heuristics. It continues to read the file/stream with some assumptions and expectations, if it encounters a situation opposite to these assumptions/expectations, it reverts the stream position to backed up location, then tries other fallback strategies.
* Calls ```pbr.readTag(...)``` which reads a "varint" (uint64) and extracts ```number``` (int32) and ```type_``` (int8) values by bitwise operations on the read uint64 value.<br>
This ```number``` (int32) represents identifier number, which corresponds to key numbers in our ```MessageProcessorFns``` map in [modelprotoDescriptor](../src/sentencepiece/model.go). ```type_``` identifies the data type of current field/message data.
    >**Note that**, in this project, we only need to read Protobuf file/stream, not write, and we have implemented only required data types that used in the model file.

<sup>from [llama2/src/protobuf/protobufreader.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/protobuf/protobufreader.go)</sup>

```go
func (pbr *ProtobufReader) readField(r *bufio.Reader) (number Number, result interface{}, ok bool) {
    ...
}
```

## **6.3. Reading Tokens and Other Structures**

### **6.3.1. Reading 0th token**

>... Reading 0th token

* Call ```ProtobufReader.readMessage(...)``` method,
    >Depth: 0 (Reading one message, on ```pbr.fileReader```)
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```,
        > Depth: 1 (Reading one field for one message, on ```pbr.fileReader```)
        * Read tag. number: 1, type: 2 (BytesType)
        * BytesType means it contains a byte array or string,
        * Instantiate a ```resultMap``` which has keys as ```Number``` (int32) and ```interface{}```,
        * Call ```pbr.readValueBytes```:
            * Read a "varint": 14, this value is the length of the byte sequence,
            * Read 14 bytes into the buffer: "\n\x05\<unk\>\x15\x00\x00\x00\x00\x18\x02" (printed in string form). You can see a meaningful piece: "\<unk\>",
            * Return this byte sequence.
        * Instantiate another reader which dedicated to this 14-byte sequence, ```localReader```,
        * Initiate a loop (to traverse this 14-byte sequence)
            * Iteration 1:
                * Call ```ProtobufReader.readField(...)``` method on ```localReader```,
                    >Depth: 2 (to traverse this 14-byte sequence)
                    * Read tag. number: 1, type: 2 (BytesType)
                    * BytesType means it contains a byte array or string,
                    * Instantiate a ```resultMap``` which has keys as ```Number``` (int32) and ```interface{}```,
                    * Call ```pbr.readValueBytes``` method on ```localReader```:
                        * Read a "varint": 5, this value is the length of the byte sequence,
                        * Read 5 bytes into the buffer: "\<unk\>" (printed in string form). Yes, this is our first extracted token string: "\<unk\>",
                        * Return this byte sequence.
                    * Instantiate another reader which dedicated to this 5-byte sequence, ```localReader```,
                        >Don't forget, we are in another inner call, think of recursive
                    * Initiate a loop (to traverse this 5-byte sequence)
                        * Iteration 1:
                            * Call ```ProtobufReader.readField(...)``` method on ```localReader```,
                                >Depth: 3 (to traverse this 5-byte sequence)
                                * Read tag. number: 7, type: 4 (EndGroupType),
                                * Return ```ok=false```, to let parent method undo and fall in the string fallback.
                            * Because of returned ```allOk=false```, we do ```pbr.undoRead(...)``` on ```localReader``` to revert the reader position to previous position,
                            * Break the loop
                    * Loop was finished
                    * Reading BytesType failed, so it continues with trying to read the sequence as string,
                    * Check if the byte sequence valid for UTF-8 encoding with ```utf8.Valid```,
                    * Yes it's valid string, return: number: 1, result: "\<unk\>".
                * Set ```resultMap```s entry: key: 1, value: "\<unk\>"

                    ```yaml
                    resultMap: {
                        1: (string) "<unk>"
                    }
                    ```

            * Iteration 2:
                * Call ```ProtobufReader.readField(...)``` method on ```localReader```,
                    >Depth: 2 (to traverse this 14-byte sequence)
                    * Read tag. number: 2, type: 5 (Fixed32Type)
                    * Fixed32Type means it contains a 4-byte float32 value,
                    * Read 4 byte and convert it a float32 in little-endian form: Read value is 0,
                    * Return this float32 value.
                * Set ```resultMap```s entry: key: 2, value: 0 (float32)

                    ```yaml
                    resultMap: {
                        1: (string) "<unk>",
                        2: (float32) 0
                    }
                    ```

            * Iteration 3:
                * Call ```ProtobufReader.readField(...)``` method on ```localReader```,
                    >Depth: 2 (to traverse this 14-byte sequence)
                    * Read tag. number: 3, type: 0 (VarintType)
                    * VarintType means it contains a variable length signed integer,
                    * Read it and convert it an int64: Read value is 2,
                    * Return this int64 value.
                * Set ```resultMap```s entry: key: 3, value: 1 (int64)

                    ```yaml
                    resultMap: {
                        1: (string) "<unk>",
                        2: (float32) 0,
                        3: (int64) 2
                    }
                    ```

            * Iteration 4:
                * Check is EOF (end of file) for current 14-byte sequence: yes
                * Break the loop
        * Loop was finished
        * Return:

            ```yaml
            number: 1,
            value: {
                1: (string) "<unk>",
                2: (float32) 0,
                3: (int64) 2
            }
            ```

    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "<unk>", 2: 0, 3: 2}}
        ```

* Find the message processor function corresponding to our ```message.Number=1``` from [ProtoDescriptor.MessageProcessorFns](../src/protobuf/protobufreader.go) function map,
* Execute it,
* The ```modelprotoDescriptor.MessageProcessorFns[1]``` function converts ```message.Value``` as ```props: map[protobuf.Number]interface{}```,
    * ```props[1]```: Piece (string), string value of the token,
    * ```props[2]```: Score (float32), score of the token,
    * ```props[3]```: PieceType (Type/byte), token type of the token (can be ```sentencepiece.NORMAL```, ```sentencepiece.CONTROL```, ```sentencepiece.BYTE```, etc... constants were defined in [llama2/src/sentencepiece/model.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/model.go)). If not represented in props map, default is ```sentencepiece.NORMAL```,
    * Then instantiates new ```sentencepiece.SentencePiece``` from ```props``` map,
    * Appends it into main object's Pieces array.

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOW)}
}
```

<sup>from [llama2/src/sentencepiece/model.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/model.go)</sup>

```go
var modelprotoDescriptor = protobuf.ProtoDescriptor{
    ...
    MessageProcessorFns: map[protobuf.Number]func(interface{}, protobuf.Message){
        1: func(mainObject interface{}, message protobuf.Message) {
            mo := mainObject.(*ModelProto)
            props := message.Value.(map[protobuf.Number]interface{})
            pieceTypeVal, err := common.InterfaceToInt(props[3])
            if err != nil {
                pieceTypeVal = int(NORMAL)
            }
            item := newSentencePiece(props[1].(string), props[2].(float32), Type(pieceTypeVal))
            *mo.Pieces = append(*mo.Pieces, item)
        },
        ...
    }
}
```

### **6.3.2. Reading 1st token**

>... Reading 1st token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```,
        * Do things which we dove into for first token above,
        * Return:

            ```yaml
            number: 1,
            value: {
                1: (string) "<s>",
                2: (float32) 0,
                3: (int64) 3
            }
            ```

    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "<s>", 2: 0, 3: 3}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0x00},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0x00}
}
```

### **6.3.3. Reading some other tokens**

..... Some steps were taken

>... Reading 13th token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "<0x0A>", 2: 0, 3: 6}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0x00},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0x00},
    ...
    {Piece: "<0x0A>", Score: 0, PieceType: 6(sentencepiece.BYTE), ByteFallback: 0x0A},
}
```

..... Some steps were taken

>... Reading 259th token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "▁▁", 2: -1000000000}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0},
    ...
    {Piece: "<0x0A>", Score: 0, PieceType: 6(sentencepiece.BYTE), ByteFallback: 0x0A},
    ...
    {Piece: "▁▁", Score: -1000000000, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
}
```

>... Reading 260th token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "▁t", 2: -1}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0},
    ...
    {Piece: "<0x0A>", Score: 0, PieceType: 6(sentencepiece.BYTE), ByteFallback: 0x0A},
    ...
    {Piece: "▁▁", Score: -1000000000, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "▁t", Score: -1, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
}
```

>... Reading 261th token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "er", 2: -2}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0},
    ...
    {Piece: "<0x0A>", Score: 0, PieceType: 6(sentencepiece.BYTE), ByteFallback: 0x0A},
    ...
    {Piece: "▁▁", Score: -1000000000, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "▁t", Score: -1, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "er", Score: -2, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
}
```

..... Some steps were taken

>... Reading 1,001st token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "ied", 2: -741}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0},
    ...
    {Piece: "<0x0A>", Score: 0, PieceType: 6(sentencepiece.BYTE), ByteFallback: 0x0A},
    ...
    {Piece: "▁▁", Score: -1000000000, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "▁t", Score: -1, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "er", Score: -2, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
    ...
    {Piece: "ied", Score: -741, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
}
```

..... Some steps were taken

>... Reading 10,001st token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "ång", 2: -9741}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0},
    ...
    {Piece: "<0x0A>", Score: 0, PieceType: 6(sentencepiece.BYTE), ByteFallback: 0x0A},
    ...
    {Piece: "▁▁", Score: -1000000000, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "▁t", Score: -1, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "er", Score: -2, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
    ...
    {Piece: "ied", Score: -741, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
    ...
    {Piece: "ång", Score: -9741, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
}
```

..... Some steps were taken

>... Reading 31,001st token

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{Number: 1, Value: {1: "동", 2: -30741}}
        ```

* Do other stuff...

```yaml
mainObject.Pieces: {
    {Piece: "<unk>", Score: 0, PieceType: 2(sentencepiece.UNKNOWN), ByteFallback: 0},
    {Piece: "<s>", Score: 0, PieceType: 3(sentencepiece.CONTROL), ByteFallback: 0},
    ...
    {Piece: "<0x0A>", Score: 0, PieceType: 6(sentencepiece.BYTE), ByteFallback: 0x0A},
    ...
    {Piece: "▁▁", Score: -1000000000, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "▁t", Score: -1, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0},
    {Piece: "er", Score: -2, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
    ...
    {Piece: "ied", Score: -741, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
    ...
    {Piece: "ång", Score: -9741, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
    ...
    {Piece: "동", Score: -30741-30741, PieceType: 1(sentencepiece.NORMAL), ByteFallback: 0}
}
```

### **6.3.4. Reading TrainerSpec**

..... Some steps were taken

..... After finishing all of 32,000 tokens, we get a different message which has ```Number=2```

>... Reading TrainerSpec

> If you are curious about what is a TrainerSpec and what it contains, you can check out [this Protobuf structure](https://github.com/google/sentencepiece/blob/022f8c3fed4d2feb4e4c670949cf01cef477dcc4/src/sentencepiece_model.proto) for details.

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{
            Number: 2, 
            Value: {
                1: (string) "/large_experiments/theorem/datasets/MERGED/all.test1.merged"
                2: (string) "spm_model_32k_200M_charcov099995_allowWSO__v2"
                3: (int64) 2
                4: (int64) 128256
                6: (int64) 0
                7: (string) "text"
                10: (float32) 0.99995
                11: (int64) 200000000
                14: (int64) 1000000
                15: (float32) 0.75
                16: (int64) 80
                17: (int64) 2
                18: (int64) 4192
                19: (int64) 1
                20: (int64) 16
                21: (int64) 1
                22: (int64) 1
                23: (int64) 1
                24: (int64) 0
                25: (int64) 1
                26: (int64) 1
                32: (int64) 1
                33: (int64) 1
                34: (int64) 0
                35: (int64) 1
                36: ([]uint8) []
                40: (int64) 0
                41: (int64) 1
                42: (int64) 2
                43: (int64) -1
                44: (string) " ⁇ "
                45: (string) "<unk>"
                46: (string) "<s>"
                47: (string) "</s>"
                48: (string) "<pad>"
                49: (int64) 0
                50: (int64) 0
                51: (float32) 0
                52: (int64) 0
            }
        }
        ```

* We don't use this information, do nothing as following:

<sup>from [llama2/src/sentencepiece/model.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/model.go)</sup>

```go
var modelprotoDescriptor = protobuf.ProtoDescriptor{
    ...
    MessageProcessorFns: map[protobuf.Number]func(interface{}, protobuf.Message){
        ...
        2: func(mainObject interface{}, message protobuf.Message) {
            // Do nothing, we don't need TrainerSpec at this time.
        },
        ...
    }
}
```

### **6.3.5. Reading NormalizerSpec**

>... Reading NormalizerSpec

* Call ```ProtobufReader.readMessage(...)``` method,
    * Call ```ProtobufReader.readField(...)``` method on ```pbr.fileReader```, do things which we dove into for first token above,
    * Instantiate a Message object and return it.

        ```go
        Message{
            Number: 3, 
            Value: {
                1: (string) "identity"
                2: ([]uint8) []
                3: (int64) 1
                4: (int64) 0
                6: ([]uint8) []
            }
        }
        ```

* We don't use this information, but we convert it as following:

<sup>from [llama2/src/sentencepiece/model.go](https://github.com/adalkiran/llama-nuts-and-bolts/blob/llama2/src/sentencepiece/model.go)</sup>

```go
var modelprotoDescriptor = protobuf.ProtoDescriptor{
    ...
    MessageProcessorFns: map[protobuf.Number]func(interface{}, protobuf.Message){
        ...
        3: func(mainObject interface{}, message protobuf.Message) {
            mo := mainObject.(*ModelProto)
            props := message.Value.(map[protobuf.Number]interface{})
            ns := NormalizerSpec{}
            ns.Name = props[1].(string)
            ns.PrecompiledCharsmap = props[2].([]byte)

            ns.AddDummyPrefix = common.InterfaceToBool(props[3], true)
            ns.RemoveExtraWhitespaces = common.InterfaceToBool(props[4], true)
            ns.EscapeWhitespaces = common.InterfaceToBool(props[5], true)
            stringVal, ok := props[6].(string)
            if !ok {
                byteArrVal, ok := props[6].([]byte)
                if !ok {
                    stringVal = ""
                } else {
                    stringVal = string(byteArrVal)
                }
            }
            ns.NormalizationRuleTsv = stringVal
            mo.NormalizerSpec = &ns
        },
    }
}
```

* Finished.

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: LOADING TOKENIZER MODEL](./05-LOADING-TOKENIZER-MODEL.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: BFLOAT16 DATA TYPE&nbsp;&nbsp;&gt;](./07-BFLOAT16-DATA-TYPE.md)

</div>

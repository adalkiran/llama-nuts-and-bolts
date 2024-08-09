# **14. MAKING PREDICTION with LLAMA MODEL - 1**

We've delved into all details and internals of all stages. We've covered all steps of the complete end-to-end flow of our project, except one, I saved it for last: The [LlamaTransformer.Forward(...)](../src/model/llamatransformer.go) method.

If you are familiar with [PyTorch](https://pytorch.org/) framework, you should be aware of what the ```forward``` function means. In this project, this naming style was kept, it means, this function is the real place for performing calculations.

In this chapter, following [15. MAKING PREDICTION with LLAMA MODEL - 2](./15-MAKING-PREDICTION-WITH-LLAMA-MODEL-2.md), and [16. MAKING PREDICTION with LLAMA MODEL - 3](./16-MAKING-PREDICTION-WITH-LLAMA-MODEL-3.md) chapters, we walk through the [LlamaTransformer.Forward(...)](../src/model/llamatransformer.go) method and its components.

**Important note:** The given example tensor shapes and values are for the first iteration that the prompt tokens is processed at, for this input prompt string:

```go
promptString = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are Einstein<|eot_id|><|start_header_id|>user<|end_header_id|>

Describe your theory.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"

// the shape of original inputTokens is {200},
// prompt tokens length is 22,
// shape of inputTokens argument is {22}, because it is the part of inputTokens containing prompt tokens.
```

## **14.1. Warming up**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) Forward(infContext *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
    if inputTokens.Size[0] == 0 {
        return nil, fmt.Errorf("empty token array")
    }
    common.GLogger.DebugPrintf("LlamaTransformer.Forward started for inputTokens: shape(%v), startPos: %d -> tensor inputTensor", inputTokens.Size, startPos)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Created input tokens tensor: shape([200]) ...
[DEBUG] ... =======================================

[DEBUG] ... Running Transformer.Forward for curPos: 22, prevPos: 0, inputTokensSlice: shape([22])...
```

## **14.2. Making preparations**

Our ```LlamaTransformer.Forward(...)``` method calls the ```LlamaTransformer.prepare(...)``` method, to make it to create:

* ```inputTensor```: The tensor contains embedding vectors of the token ids in ```inputTokens``` tensor, with shape of ```{22, 4096}``` (remember the ```inputTokensSlice``` has shape of ```{22}``` in our first iteration example case and ```4096``` is one dimension size of Llama model's embedding layer),
* ```freqsCis```: The tensor contains a slice of ```LlamaTransformer.PrecomputedFreqsCis``` taken by between indices 0 and 21. The ```LlamaTransformer.PrecomputedFreqsCis``` tensor was precomputed before and we have covered this subject in previous chapters. The shape of the ```freqsCis``` tensor is ```{22, 64}``` in our first iteration example case,
* ```mask```: A triangular upper tensor consists of "negative infinities" and "ones (1s)", with shape of ```{22, 22}``` in our first iteration example case,
* ```err```: If the function finishes with an error, this represents it, otherwise ```nil```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) Forward(infContext *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
    ...
    inputTensor, freqsCis, mask, err := lt.prepare(inputTokens, startPos)
    ...
}
```

### **14.2.1. Creating the inputTensor**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) prepare(inputTokens *ml.Tensor, startPos int) (inputTensor *ml.Tensor, freqsCis *ml.Tensor, mask *ml.Tensor, err error) {
    sequenceLength := inputTokens.Size[0]
    common.GLogger.DebugPrintf("LlamaTransformer.prepare started for inputTokens: shape(%v), startPos: %d. sequenceLength: %d", inputTokens.Size, startPos, sequenceLength)
    inputTensor, err = ml.Fwd_Get_Rows(lt.tok_embd, inputTokens)
    ...
}
```

![STAGE 4: Creating inputTensor Diagram](./images/DIAG01-STAGE04-creating-input-tensor.drawio.svg)
<sup>*Diagram: **Creating inputTensor**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

The ```inputTensor``` is created via [ml.Fwd_Get_Rows](../src/ml/operations_impl.go) function.
This step aims to retrieve embedding vectors of the token ids in ```inputTokens``` tensor (in our case it contains 32 token id integers), the result will be a tensor with shape of ```{22, 4096}```.

We have the embedding layer tensor at ```lt.tok_embd``` already loaded to ```LlamaTransformer``` struct. In our case, shape of our embedding layer is ```{128256, 4096}```, which has 128256 rows corresponding to supported tokens in our vocabulary, 4096 columns corresponding to dimensions of one token embedding vector.

This function simply takes token embedding vectors with ```{1, 4096}``` shape by using our token ids as item indices/row indices in a list. It finds start and end indices at the ```embedding.RawData``` byte array of corresponding row, for each token independently, then collects all of them into ```dst``` tensor.

<sup>from [src/ml/operations_impl.go](../src/ml/operations_impl.go)</sup>

```go
func Fwd_Get_Rows(embedding *Tensor, tokens *Tensor) (*Tensor, error) {
    if err := processErrors(
        checkIsMatrix(embedding),
        checkIsVector(tokens),
    ); err != nil {
        return nil, err
    }

    if tokens.DataType != DT_INT32 {
        return nil, fmt.Errorf("tensor is not in data type %s: \"%s\" is %s", DT_INT32, tokens.Name, tokens.DataType)
    }

    sequenceLength := tokens.Size[0]
    embeddingDim := embedding.Size[1]
    dst := NewEmptyTensor([]int{sequenceLength, embeddingDim}, embedding.DataType)

    rowCount := tokens.GetElementCount()

    for rowIdx := 0; rowIdx < rowCount; rowIdx++ {
        rowVal, err := tokens.GetItem([]int{rowIdx})
        if err != nil {
            return nil, err
        }
        row := int(rowVal.(int32))
        readOffsetStart := embedding.calculateByteOffset([]int{row, 0})
        readOffsetEnd := embedding.calculateByteOffset([]int{row + 1, 0})
        rowBytes := embedding.RawData[readOffsetStart:readOffsetEnd]
        writeOffsetStart := dst.calculateByteOffset([]int{rowIdx, 0})
        copy(dst.RawData[writeOffsetStart:], rowBytes)
    }
    return dst, nil
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... LlamaTransformer.Forward started for inputTokens: shape([22]), startPos: 0 -> tensor inputTensor...
[DEBUG] ... LlamaTransformer.prepare started for inputTokens: shape([22]), startPos: 0. sequenceLength: 22...
```

### **14.2.2. Creating the freqsCis tensor**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) prepare(inputTokens *ml.Tensor, startPos int) (inputTensor *ml.Tensor, freqsCis *ml.Tensor, mask *ml.Tensor, err error) {
    ...
    freqsCis, err = lt.PrecomputedFreqsCis.Slice([]int{startPos}, []int{startPos + sequenceLength})
    ...
}
```

We've precomputed the ```lt.PrecomputedFreqsCis``` in previous steps. It has the shape of ```{4096, 64}```, which has 4096 rows corresponding to 2 times of 2048 max sequence length, 64 columns corresponding to half of dimensions of ```dim / N_Heads (head count) = 4096 / 32 = 128```. It is 64, because we take 128 numbers as pairs to create one complex number, one as real part and other one as imaginary part. For more information, refer to [10. ROPE ROTARY POSITIONAL EMBEDDINGS](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md) and [10.BONUS. PRECOMPUTING FREQUENCY TENSOR](./10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb).

Here, we take a slice of the ```lt.PrecomputedFreqsCis``` into ```freqsCis``` with shape of ```{22, 64}``` in our case.

### **14.2.3. Creating the mask tensor**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) prepare(inputTokens *ml.Tensor, startPos int) (inputTensor *ml.Tensor, freqsCis *ml.Tensor, mask *ml.Tensor, err error) {
    ...
    if sequenceLength > 1 {
        negativeInfinity := dtype.BFloat16fromFloat32(float32(math.Inf(-1)))
        if mask, err = ml.Full([]int{sequenceLength, sequenceLength}, ml.DT_BF16, negativeInfinity); err != nil {
            return
        }
        if mask, err = ml.TriangularUpper(mask, 1); err != nil {
            return
        }
    }
    var maskSize []int
    if mask != nil {
        maskSize = mask.Size
    }
    common.GLogger.DebugPrintf("LlamaTransformer.prepare finished inputTensor: shape(%v), freqsCis: shape(%v), mask: shape(%v)", inputTensor.Size, freqsCis.Size, maskSize)
    return
}
```

In [Wikipedia](https://en.wikipedia.org/wiki/Llama), it writes:

```sh
Llama (Large Language Model Meta AI) is a family of autoregressive large language models (LLMs), released by Meta AI starting in February 2023.
```

Llama is an auto-regressive model, which means, remember this is a time-series model, it always handles the information behind the current position. So, if we add ```-Inf``` (negative infinity) value to the values we want to ignore and add ```0``` (zero) to the values we want to pay attention to, our goal could be achieved.

To achieve this goal, we generate a ```mask``` tensor consisting of ```-Inf``` values at the upper triangle and ```0``` values at the lower triangle via [ml.TriangularUpper](../src/ml/operations_impl.go) function. This tensor will be used in our attention layer further.

```go
[
    [   0, -Inf, -Inf,    ..., -Inf, -Inf, -Inf],
    [   0,    0, -Inf,    ..., -Inf, -Inf, -Inf],
    [   0,    0,    0,    ..., -Inf, -Inf, -Inf],
    ...,
    [   0,    0,    0,    ...,    0, -Inf, -Inf], 
    [   0,    0,    0,    ...,    0,    0, -Inf],
    [   0,    0,    0,    ...,    0,    0,    0]
]
shape=[22 22], dtype=BF16
```

Sources:

* [Autoregressive Model - ScienceDirect](https://www.sciencedirect.com/topics/mathematics/autoregressive-model)
* [Youtube - Llama explained... - "KV Cache" section - Mask part](https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=3240s)

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... LlamaTransformer.prepare finished inputTensor: shape([22 4096]), freqsCis: shape([22 64]), mask: shape([22 22])...
```

## **14.3. Performing Forward Pass Through Each Transformer Block - LlamaTransformerBlock.Forward(...)**

>**Recap:**<br>
>The most important part of the transformer models that provide accurate outputs is [the attention mechanism](https://arxiv.org/abs/1706.03762). Each "block" of Llama consists of a self-attention and a feed-forward neural network parts. The details will be explained further, but also we call these "block"s as "layer"s.
>
>A ```LlamaTransformerBlock``` object consists of ```attn_norm``` (RMS normalization), ```attention``` (Attention mechanism), ```ffn_norm``` (RMS normalization), and ```feedForward``` (Feed Forward Neural Network) modules. These modules operate respectively.
>
>The model configuration file "params.json" is parsed as [ModelArgs](../src/model/modelargs.go) object via JSON parser.

![STAGE 5: Forward Pass Through Each Transformer Block Diagram](./images/DIAG01-STAGE05-forward-pass-through-each-transformer-block.drawio.svg)
<sup>*Diagram: **Forward Pass Through Each Transformer Block**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

These "blocks" are also called as "layer"s. In our case, we use Llama 3.1 8B-Instruct model, in 8B model has 32 layers. Because we have ```model.ModelArgs.N_Layers = 32``` as read from "params.json" configuration file. We've defined these "layers"/"transformer block layers" in previous steps as described in [09. IMPLEMENTING LLAMA MODEL ARCHITECTURE](./09-IMPLEMENTING-LLAMA-MODEL-ARCHITECTURE.md).

We initiate a for loop to iterate over 32 layers defined at ```lt.Layers``` and call ```LlamaTransformerBlock.Forward(...)``` method.

>**Important note:** These 32 layers couldn't be run parallelly, because they take the output of the previous layer as input. This is a handicap for running parallel, so we need to try to parallelize the steps inside of them.

In this chapter, we won't delve into the details of a ```LlamaTransformerBlock.Forward(...)```, it is subject to another dedicated chapter further.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) Forward(infContext *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
    ...
    currentTensor := inputTensor
    for layerIdx, layer := range lt.Layers  {
        startTime := time.Now()
        common.GLogger.DebugPrintf("=======================================\n")
        common.GLogger.DebugPrintf("Calling LlamaTransformerBlock.Forward for layer: %d / %d, startPos: %d -> tensor currentTensor", layerIdx+1, len(lt.Layers), startPos)
        if currentTensor, err = layer.Forward(infContext, currentTensor, startPos, freqsCis, mask); err != nil {
            return nil, err
        }
        infContext.Logf("Transformer block layer %d / %d was run, took %.4f sec(s)", layerIdx+1, len(lt.Layers), time.Since(startTime).Seconds())
    }
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... =======================================
[DEBUG] ... Calling LlamaTransformerBlock.Forward for layer: 1 / 32, startPos: 0 -> tensor currentTensor ...
...
[DEBUG] ... Calling LlamaTransformerBlock.Forward for layer: 2 / 32, startPos: 0 -> tensor currentTensor ...
...
[DEBUG] ... Calling LlamaTransformerBlock.Forward for layer: 32 / 32, startPos: 0 -> tensor currentTensor ...
[DEBUG] ... Returning tensor output: shape([22 4096]) ...
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: GENERATING NEXT TOKENS](./13-GENERATING-NEXT-TOKENS.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: MAKING PREDICTION with LLAMA MODEL - 2&nbsp;&nbsp;&gt;](./15-MAKING-PREDICTION-WITH-LLAMA-MODEL-2.md)

</div>

# **9. IMPLEMENTING LLAMA MODEL ARCHITECTURE**

In this chapter, we'll walk through the process of defining and implementing the Llama 3.1 Model architecture.

Llama 2, Llama 3 and Llama 3.1 transformer model architectures are very similar, but new versions have come with some improvements.

>Inspired by [original Llama 3.1 Python repository of Meta](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/reference_impl/model.py) | [llama.cpp](https://github.com/ggerganov/llama.cpp)

>**<u>A Quick Reminder:</u>**<br>
>We've loaded 291 tensors from the model file into a map (PickleDict) of tensors per tensor name via ```torchModelReader.Load()```.

Now, [NewLlamaTransformer(...)](../src/model/llamatransformer.go) is called to build operation sequence graph of Llama architecture.

<sup>from [src/model/loader.go](../src/model/loader.go)</sup>

```go
func LoadModelEx(modelDir string, includeTensors bool, includeVocab bool) (*Model, error) {
    model := &Model{}
    if includeTensors {
        ...
        modelTensors, err := torchModelReader.Load()
        ...
        model.Tensors = modelTensors
        ...
    }
    ...
    if includeTensors {
        ...
        if model.Transformer, err = NewLlamaTransformer(model); err != nil {
            return nil, err
        }
    }
    return model, nil
}
```

## **9.1. Preparation of modelArgs**

The [model.ModelArgs](../src/model/modelargs.go) was loaded from the JSON File, "params.json". But, some of them (```N_Rep``` and ```HeadDim```) are "fields that should be calculated" and some of others may have value ```-1``` meaning "default value".

In our project, we used Meta-Llama-3.1-8B-Instruct model. This model has following parameters that are loaded into ```model.ModelArgs```:

```yaml
Dim:        4096 //dim
N_Layers:   32   //n_layers
N_Heads:    32   //n_heads
N_KVHeads:  8   //n_kv_heads
VocabSize:  128256 //vocab_size
MultipleOf: 1024  //multiple_of

FFNDimMultiplier:  1.3 //ffn_dim_multiplier
NormEpsilon:       1e-5 //norm_eps
RopeTheta:         500000 //rope_theta
UseScaledRope:     true //use_scaled_rope
MaxSequenceLength: //to be calculated

N_Rep:   //to be calculated
HeadDim: //to be calculated
```

These preparations are done in this function:

* If ```modelArgs.VocabSize``` is ```-1``` in the file, which indicates it wants us to set the default value. modelArgs wants to obey the "tokenizer.model" file. In our case, the tokenizer file contains 128,256 tokens.
* If ```modelArgs.N_KVHeads``` is not specified in the file, which indicates it wants us to set the default value. The default value is ```N_Heads```.<br>
This ```N_KVHeads``` is equal to ```8``` for 8B/8B-Instruct Llama models.
* ```modelArgs.N_Rep``` is set to integer value of ```N_Heads / N_KVHeads```, the repetition count for the following operation in [original Llama code](https://github.com/meta-llama/llama-models/blob/f45cdfd624b98b6655540f7101d8d9cb432e631c/models/llama3_1/reference_impl/model.py#L122) and also in [our implementation](../src/model/llamatransformer.go). In our case, it is ```32 / 8 = 4```. This means, our ```keys``` and ```values``` have ```8``` heads, other parts have ```32``` heads, so the ```8``` heads are repeated/copied ```4 times``` to adapt ```32``` heads.
* ```modelArgs.HeadDim``` is set to integer value of ```modelArgs.Dim / modelArgs.N_Heads```. In our case, it is ```4096 / 32 = 128```. This means we have 32 different ```attention heads``` and the dimension of each of these heads is ```128```.

>Also, you can check out sources for **Grouped Multi-Query Attention** which isn't described here:
>
>* [Grouped Multi-Query Attention](https://paperswithcode.com/method/grouped-query-attention)
>* [What's Grouped-Query attention (GQA)? a paper from Google Research](https://aliissa99.medium.com/-a596e4d86f79)
>* [Llama explained... - "Grouped Multi-Query Attention" section - Youtube](https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=3240s)

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
    result := &LlamaTransformer{}
    modelArgs := model.ModelArgs

    var err error
    // Compare (VocabSize, Dim) vs. "tok_embeddings.weight" tensor shape
    dim := modelArgs.Dim             // 4096
    vocabSize := modelArgs.VocabSize // 128256

    if modelArgs.N_KVHeads < 0 {
        modelArgs.N_KVHeads = modelArgs.N_Heads
    }
    modelArgs.N_Rep = int(modelArgs.N_Heads / modelArgs.N_KVHeads)
    // Calculate dimension of each head
    modelArgs.HeadDim = int(modelArgs.Dim / modelArgs.N_Heads) // 128
    ...
}
```

## **9.2. The First Layer of our Model: Embeddings Layer (tok_embd)**

Yes! We are at the stage where we REALLY are starting to build the model by laying the first brick!

The [getTensor](../src/model/loader.go) function gets the weights tensor with the name we specified from [Model.Tensors](../src/model/model.go) map, checks if it has really the expected shape we specified, then returns the [ml.Tensor](../src/ml/tensor.go) object, or returns "incorrect shape" error.

The weights tensor with name "tok_embeddings.weight" is taken and set to ```result.tok_embd``` as first brick. Our ```result.tok_embd``` weights tensor is with shape of ```{vocabSize, dim} = {128256, 4096}```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
    result := &LlamaTransformer{}
    ...
    if result.tok_embd, err = getTensor(model, "tok_embeddings.weight", []int{vocabSize, dim}); err != nil {
        return nil, err
    }
    ...
}
```

## **9.3. Building Transformer Blocks**

The most important part of the transformer models that provide accurate outputs is [the attention mechanism](https://arxiv.org/abs/1706.03762). Each "block" of Llama consists of a self-attention and a feed-forward neural network parts. The details will be explained further, but also we call these "block"s as "layer"s.

The value of the ```modelArgs.N_Layers``` variable corresponds to the number of blocks we have. It is ```32```, so we will initiate 32 different transformer block [LlamaTransformerBlock](../src/model/llamatransformer.go) objects via [NewLlamaTransformerBlock(...)](../src/model/llamatransformer.go) function. To achieve this, we instantiate ```result.Layers``` array with 32 items, then set each item with instantiating each block.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
    result := &LlamaTransformer{}
    ...
    result.Layers = make([]*LlamaTransformerBlock, modelArgs.N_Layers)

    for layerIdx := 0; layerIdx < modelArgs.N_Layers; layerIdx++ {
        var layer *LlamaTransformerBlock
        if layer, err = NewLlamaTransformerBlock(model, layerIdx); err != nil {
            return nil, err
        }
        result.Layers[layerIdx] = layer
    }
    ...
}
```

## **9.4. Building Each Transformer Block (LlamaTransformerBlock)**

The ```LlamaTransformerBlock``` object consists of ```attn_norm``` (RMS normalization), ```attention``` (Attention mechanism), ```ffn_norm``` (RMS normalization), and ```feedForward``` (Feed Forward Neural Network) modules. These modules operate respectively.

**<u>Type definition:</u>**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
type LlamaTransformerBlock struct {
    LayerIndex int

    attn_norm *RMSNorm // Weights Original: "layers.0.attention_norm.weight"  |  ggml: "blk.0.attn_norm.weight" | shape: [4096] -> [Dim]
    ffn_norm  *RMSNorm // Weights Original: "layers.0.ffn_norm.weight"  |  ggml: "blk.0.ffn_norm.weight" | shape: [4096] -> [Dim]

    attention   *LlamaAttention
    feedForward *LlamaFeedForward
}
```

In Llama models, these normalization modules are operated before their pair modules, for e.g., ```attn_norm``` is operated before ```attention``` module, ```ffn_norm``` is operated before ```feedForward```. This approach is called as ```prenormalization```. [Root Mean Square Layer Normalization](https://paperswithcode.com/method/rmsnorm) is used as normalization technique.

We will dive into deeper the details in further chapters, at this stage, we should stay zoomed-out view.

At this stage, our steps are:

* Taking the weights tensor of attention norm corresponding to current layer index, ```"layers.%d.attention_norm.weight"```. In Llama model, these weight tensors are named like "layers.0.attention_norm.weight", "layers.1.attention_norm.weight", "layers.2.attention_norm.weight", ..., "layers.31.attention_norm.weight". This weights tensor is with shape of ```{dim} = {4096}```,
* Instantiating an [RMSNorm](../src/model/llamatransformer.go) object with specifying ```modelArgs.NormEpsilon``` (```1e-5``` as epsilon value) and ```attn_norm_weights``` tensor via [NewRMSNorm(...)](../src/model/llamatransformer.go). Then it is set to ```result.attn_norm```,
* Instantiating a [LlamaAttention](../src/model/llamatransformer.go) object via [NewLlamaAttention(...)](../src/model/llamatransformer.go). Then it is set to ```result.attention```,
* Taking the weights tensor of feed-forward neural network norm corresponding to current layer index, ```"layers.%d.ffn_norm.weight"```. In Llama model, these weight tensors are named like "layers.0.ffn_norm.weight", "layers.1.ffn_norm.weight", "layers.2.ffn_norm.weight", ..., "layers.31.ffn_norm.weight". This weights tensor is with shape of ```{dim} = {4096}```,
* Instantiating an [RMSNorm](../src/model/llamatransformer.go) object with specifying ```modelArgs.NormEpsilon``` (```1e-5``` as epsilon value) and ```ffn_norm_weights``` tensor via [NewRMSNorm(...)](../src/model/llamatransformer.go). Then it is set to ```result.ffn_norm```,
* Instantiating a [LlamaFeedForward](../src/model/llamatransformer.go) object via [NewLlamaFeedForward(...)](../src/model/llamatransformer.go). Then it is set to ```result.feedForward```,

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformerBlock(model *Model, layerIndex int) (*LlamaTransformerBlock, error) {
    result := &LlamaTransformerBlock{
        LayerIndex: layerIndex,
    }
    modelArgs := model.ModelArgs
    dim := modelArgs.Dim // 4096
    var err error

    // attention normalization
    attn_norm_weights, err := getLayerTensor(model, "layers.%d.attention_norm.weight", layerIndex, []int{dim})
    if err != nil {
        return nil, err
    }
    result.attn_norm = NewRMSNorm(modelArgs.NormEpsilon, attn_norm_weights)

    if result.attention, err = NewLlamaAttention(model, layerIndex); err != nil {
        return nil, err
    }

    // feed forward normalization
    ffn_norm_weights, err := getLayerTensor(model, "layers.%d.ffn_norm.weight", layerIndex, []int{dim})
    if err != nil {
        return nil, err
    }
    result.ffn_norm = NewRMSNorm(modelArgs.NormEpsilon, ffn_norm_weights)

    if result.feedForward, err = NewLlamaFeedForward(model, layerIndex); err != nil {
        return nil, err
    }

    return result, nil
}
```

### **9.4.1. Building an Attention Module (LlamaAttention)**

The ```LlamaAttention``` object consists of:

* ```attn_wq```: Attention query weights tensor with shape of ```{N_Heads * HeadDim, Dim} = {32 * 128, 4096} = {4096, 4096}```,
* ```attn_wk```: Attention key weights tensor with shape of ```{N_KVHeads * HeadDim, Dim} = {8 * 128, 4096} = {1024, 4096}```,
* ```attn_wv```: Attention value weights tensor with shape of ```{N_KVHeads * HeadDim, Dim} = {8 * 128, 4096} = {1024, 4096}```,
* ```attn_wo```: Attention output weights tensor with shape of ```{N_Heads * HeadDim, Dim} = {32 * 128, 4096} = {4096, 4096}```.

**<u>Type definition:</u>**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
type LlamaAttention struct {
    LayerIndex int

    N_Heads   int
    N_KVHeads int
    N_Rep     int
    HeadDim   int

    attn_wq *ml.Tensor // Original: "layers.0.attention.wq.weight"  |  ggml: "blk.0.attn_q.weight" | [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
    attn_wk *ml.Tensor // Original: "layers.0.attention.wk.weight"  |  ggml: "blk.0.attn_k.weight" | [out_features, in_features] -> shape: [1024 4096] -> [N_KVHeads * HeadDim, Dim]
    attn_wv *ml.Tensor // Original: "layers.0.attention.wv.weight"  |  ggml: "blk.0.attn_v.weight" | [out_features, in_features] -> shape: [1024 4096] -> [N_KVHeads * HeadDim, Dim]
    attn_wo *ml.Tensor // Original: "layers.0.attention.wo.weight"  |  ggml: "blk.0.attn_output.weight" | [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
}
```

```NewLlamaAttention(...)``` is called to instantiate a new LlamaAttention object for current layer.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformerBlock(model *Model, layerIndex int) (*LlamaTransformerBlock, error) {
    result := &LlamaTransformerBlock{
        LayerIndex: layerIndex,
    }
    ...
    if result.attention, err = NewLlamaAttention(model, layerIndex); err != nil {
        return nil, err
    }
    ...
}
```

In ```NewLlamaAttention(...)```:

* Calculating dimension of normal heads and KV heads (key-value heads). In our case, results are both 4096,
* Taking the weights tensor of attention query corresponding to current layer index, ```"layers.%d.attention.wq.weight"```. Then it is set to ```result.attn_wq```,
* Taking the weights tensor of attention key corresponding to current layer index, ```"layers.%d.attention.wk.weight"```. Then it is set to ```result.attn_wk```,
* Taking the weights tensor of attention value corresponding to current layer index, ```"layers.%d.attention.wv.weight"```. Then it is set to ```result.attn_wv```,
* Taking the weights tensor of attention output corresponding to current layer index, ```"layers.%d.attention.wo.weight"```. Then it is set to ```result.attn_wo```,

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaAttention(model *Model, layerIndex int) (*LlamaAttention, error) {
    result := &LlamaAttention{
        LayerIndex: layerIndex,
    }
    modelArgs := model.ModelArgs
    dim := modelArgs.Dim // 4096
    var err error

    result.N_Heads = modelArgs.N_Heads
    result.N_KVHeads = modelArgs.N_KVHeads
    result.N_Rep = modelArgs.N_Rep
    // Calculate dimension of each head
    result.HeadDim = modelArgs.HeadDim                        // 128
    normalHeadsTotalDim := modelArgs.N_Heads * result.HeadDim // 4096
    kvHeadsTotalDim := result.N_KVHeads * result.HeadDim      // 4096

    // attn_wq, attn_wk, attn_wv, attn_wo are Linear units, so weight shapes are ordered reversely as [out_features, in_features]
    if result.attn_wq, err = getLayerTensor(model, "layers.%d.attention.wq.weight", layerIndex, []int{normalHeadsTotalDim, dim}); err != nil {
        return nil, err
    }
    if result.attn_wk, err = getLayerTensor(model, "layers.%d.attention.wk.weight", layerIndex, []int{kvHeadsTotalDim, dim}); err != nil {
        return nil, err
    }
    if result.attn_wv, err = getLayerTensor(model, "layers.%d.attention.wv.weight", layerIndex, []int{kvHeadsTotalDim, dim}); err != nil {
        return nil, err
    }
    if result.attn_wo, err = getLayerTensor(model, "layers.%d.attention.wo.weight", layerIndex, []int{normalHeadsTotalDim, dim}); err != nil {
        return nil, err
    }

    return result, nil
}
```

### **9.4.2. Building a FeedForward Module (LlamaFeedForward)**

The ```LlamaFeedForward``` object consists of:

* ```ffn_gate```: Feed-forward gate weights tensor with shape of ```{FFNHiddenDim, Dim} = {14336, 4096}```,
* ```ffn_down```: Feed-forward down weights tensor with shape of ```{Dim, FFNHiddenDim} = {4096, 14336}```,
* ```ffn_up```: Feed-forward up weights tensor with shape of ```{FFNHiddenDim, Dim} = {14336, 4096}```,

>Note: ```FFNHiddenDim``` value is calculated as ```14336```, we will see how is it calculated below.

**<u>Type definition:</u>**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
type LlamaFeedForward struct {
    FFNHiddenDim int

    ffn_gate *ml.Tensor // Original: "layers.0.feed_forward.w1.weight"  |  ggml: "blk.0.ffn_gate.weight" | [out_features, in_features] -> shape: [14336 4096] -> [FFNHiddenDim, Dim] | w1
    ffn_down *ml.Tensor // Original: "layers.0.feed_forward.w2.weight"  |  ggml: "blk.0.ffn_down.weight" | [out_features, in_features] -> shape: [4096 14336] -> [Dim, FFNHiddenDim] | w2
    ffn_up   *ml.Tensor // Original: "layers.0.feed_forward.w3.weight"  |  ggml: "blk.0.ffn_up.weight" | [out_features, in_features] -> shape: [14336 4096] -> [FFNHiddenDim, Dim] | w3
}
```

```NewLlamaFeedForward(...)``` is called to instantiate a new LlamaFeedForward object for current layer.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformerBlock(model *Model, layerIndex int) (*LlamaTransformerBlock, error) {
    result := &LlamaTransformerBlock{
        LayerIndex: layerIndex,
    }
    ...
    if result.feedForward, err = NewLlamaFeedForward(model, layerIndex); err != nil {
        return nil, err
    }
    ...
}
```

In ```NewLlamaFeedForward(...)```:

* Calculating dimension of feed forward neural network's hidden layer ```result.FFNHiddenDim```. Actually, I couldn't reasonate well this part, calculation method was taken directly from [here](https://github.com/meta-llama/llama-models/blob/f45cdfd624b98b6655540f7101d8d9cb432e631c/models/llama3_1/reference_impl/model.py#L256) and [here](https://github.com/meta-llama/llama-models/blob/f45cdfd624b98b6655540f7101d8d9cb432e631c/models/llama3_1/reference_impl/model.py#L227),
* Taking the weights tensor of Feed-forward gate corresponding to current layer index, ```"layers.%d.feed_forward.w1.weight"```. Then it is set to ```result.ffn_gate```,
* Taking the weights tensor of Feed-forward down corresponding to current layer index, ```"layers.%d.feed_forward.w2.weight"```. Then it is set to ```result.ffn_down```,
* Taking the weights tensor of Feed-forward up corresponding to current layer index, ```"layers.%d.feed_forward.w3.weight"```. Then it is set to ```result.ffn_up```,

>Note: ```ffn_gate```, ```ffn_down```, ```ffn_up``` are Linear units, so weight shapes are ordered reversely as [out_features, in_features]. At first sight, it may confuse.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaFeedForward(model *Model, layerIndex int) (*LlamaFeedForward, error) {
    result := &LlamaFeedForward{}
    modelArgs := model.ModelArgs
    dim := modelArgs.Dim // 4096
    var err error

    // See: https://github.com/meta-llama/llama-models/blob/f45cdfd624b98b6655540f7101d8d9cb432e631c/models/llama3_1/reference_impl/model.py#L256
    // Set it to 4 * dim at first
    result.FFNHiddenDim = 4 * modelArgs.Dim
    // See: https://github.com/meta-llama/llama-models/blob/f45cdfd624b98b6655540f7101d8d9cb432e631c/models/llama3_1/reference_impl/model.py#L227
    // Then, do this calculation below:
    result.FFNHiddenDim = int(2 * result.FFNHiddenDim / 3)
    if modelArgs.FFNDimMultiplier > -1 {
        result.FFNHiddenDim = int(modelArgs.FFNDimMultiplier * float64(result.FFNHiddenDim))
    }
    // Ensure ffnHiddenDim is multiple of modelArgs.MultipleOf value
    result.FFNHiddenDim = int(modelArgs.MultipleOf * ((result.FFNHiddenDim + modelArgs.MultipleOf - 1) / modelArgs.MultipleOf))

    // ffn_gate, ffn_down, ffn_up are Linear units, so weight shapes are ordered reversely as [out_features, in_features]
    if result.ffn_gate, err = getLayerTensor(model, "layers.%d.feed_forward.w1.weight", layerIndex, []int{result.FFNHiddenDim, dim}); err != nil {
        return nil, err
    }
    if result.ffn_down, err = getLayerTensor(model, "layers.%d.feed_forward.w2.weight", layerIndex, []int{dim, result.FFNHiddenDim}); err != nil {
        return nil, err
    }
    if result.ffn_up, err = getLayerTensor(model, "layers.%d.feed_forward.w3.weight", layerIndex, []int{result.FFNHiddenDim, dim}); err != nil {
        return nil, err
    }

    return result, nil
}
```

After completion of this stage, our ```LlamaTransformerBlock``` object of the first layer has been built.

This part will be recurred for 32 times for the Llama 3.1 8B models.

## **9.5. Building the Output Layers of the LlamaTransformer**

>**<u>A Quick Reminder:</u>**<br>
>We've done following things until now:
>
>* Built the embedding layer,
>* Built 32 LlamaTransformerBlock objects, each containing an attention module and a feed-forward module with RMS prenormalization.

After executing these layers, we have a ```currentTensor``` object as output of previous "transformer blocks". Then, we need to prenormalize our tensor, then process it with the "output weights".

We continue with:

* Taking the weights tensor of output norm, ```"norm.weight"```. This weights tensor is with shape of ```{dim} = {4096}```,
* Instantiating an [RMSNorm](../src/model/llamatransformer.go) object with specifying ```modelArgs.NormEpsilon``` (```1e-5``` as epsilon value) and ```output_norm_weights``` tensor via [NewRMSNorm(...)](../src/model/llamatransformer.go). Then it is set to ```result.output_norm```,
* Taking the weights tensor of output, ```"output.weight"```. This weights tensor is with shape of ```{vocabSize, dim} = {128256, 4096}```. Then it is set to ```result.output```,

>Note: The ```output``` is a Linear unit, so weight shapes are ordered reversely as [out_features, in_features]. At first sight, it may confuse.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
    result := &LlamaTransformer{}
    ...
    output_norm_weights, err := getTensor(model, "norm.weight", []int{dim})
    if err != nil {
        return nil, err
    }
    result.output_norm = NewRMSNorm(modelArgs.NormEpsilon, output_norm_weights)

    // output is a Linear unit, so weight shape is ordered reversely as [out_features, in_features]
    if result.output, err = getTensor(model, "output.weight", []int{vocabSize, dim}); err != nil {
        return nil, err
    }
    ...
}
```

## **9.6. Precomputing the Frequency Tensor for Complex Exponentials (cis)**

The code comment [from the original Llama 2 Python code](https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L80) that explains this:

```py
"""
Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
and the end index 'end'. The 'theta' parameter scales the frequencies.
The returned tensor contains complex values in complex64 data type.

Args:
    dim (int): Dimension of the frequency tensor.
    end (int): End index for precomputing frequencies.
    theta (float, optional): Scaling factor for frequency computation. Defaults to 500000.0.

Returns:
    torch.Tensor: Precomputed frequency tensor with complex exponentials.
"""
```

```precomputeFreqsCis(...)``` is called to calculate the frequency tensor for complex exponentials (cis). This tensor values will be used by [applyRotaryEmbeddings](../src/model/llamatransformer.go) while applying Rotary Embeddings further.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
    result := &LlamaTransformer{}
    ...
    if result.PrecomputedFreqsCis, err = precomputeFreqsCis(int(dim/modelArgs.N_Heads), modelArgs.MaxSequenceLength*2, modelArgs.RopeTheta, modelArgs.UseScaledRope); err != nil {
        return nil, err
    }
    return result, nil
}
```

The details of ```precomputeFreqsCis(...)``` function is discussed in a dedicated chapter: [10. RoPE (ROTARY POSITIONAL EMBEDDINGS)](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md).

Now, we have a complete [Model](../src/model/model.go) object that contains model arguments, the tokenizer, the ```LlamaTransformer``` object at its ```model.Transformer``` field, which has a complete Llama 3.1 8B model architecture.

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: TENSOR](./08-TENSOR.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: RoPE \(ROTARY POSITIONAL EMBEDDINGS\)&nbsp;&nbsp;&gt;](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md)

</div>

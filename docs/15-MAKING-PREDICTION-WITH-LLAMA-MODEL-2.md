# **15. MAKING PREDICTION with LLAMA MODEL - 2**

In previous [14. MAKING PREDICTION with LLAMA MODEL - 1](./14-MAKING-PREDICTION-WITH-LLAMA-MODEL-1.md) chapter, this chapter, and following [16. MAKING PREDICTION with LLAMA MODEL - 3](./16-MAKING-PREDICTION-WITH-LLAMA-MODEL-3.md) chapter, we walk through the [LlamaTransformer.Forward(...)](../src/model/llamatransformer.go) method and its components.

In the previous chapter, we have initiated a for loop to iterate over 32 layers defined at ```lt.Layers``` and called ```LlamaTransformerBlock.Forward(...)``` method.

In this chapter, we will delve into details of the ```LlamaTransformerBlock.Forward(...)``` method.

## **15.1. Performing Forward Pass Through Attention Prenormalization - RMSNorm.Forward(...)**

>**Recap:**<br>
>The LLaMa 2 models use [Pre-RMSNorm (Root Mean Square Layer Normalization)](https://paperswithcode.com/method/rmsnorm). Because of we perform Root Mean Square Layer Normalization before performing multiplication of current tensor with normalization weights tensor, we call this normalization stage as "pre-normalization".

![STAGE 6: Forward Pass Through Attention Pre-normalization Diagram](./images/DIAG01-STAGE06-forward-pass-through-attention-prenormalization.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Pre-normalization**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

In this chapter, we only cover how to call RMSNorm succinctly, to give more space for other details. The details of prenormalization and RMSNorm (Root Mean Square Layer Normalization) will be explained in the chapter [16.1. Performing Forward Pass Through Output Prenormalization - RMSNorm.Forward(...)](./16-MAKING-PREDICTION-WITH-LLAMA-MODEL-3.md).

Now, we have the ```x``` tensor argument. In our case, at the first iteration, ```x``` is input tensor, at the other iterations, ```x``` is output of previous ```LlamaTransformerBlock```. In our case, at the first iteration, the shape of this tensor is ```{32, 4096}```. 32 stands for sequence length, 4096 stands for the embedding layer dimension. ```normalizedX``` which is the resulting tensor will have same shape as the input, ```{32, 4096}```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (ltb *LlamaTransformerBlock) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    var maskSize []int
    if mask != nil {
        maskSize = mask.Size
    }
    common.GLogger.DebugPrintf("LlamaTransformerBlock.Forward started for x: shape(%v), startPos: %d, freqsCis: shape(%v), mask: shape(%v)", x.Size, startPos, freqsCis.Size, maskSize)
    common.GLogger.DebugPrintf("Calling RMSNorm for tensor x shape(%v) and LlamaTransformerBlock.attn_norm weights shape(%v) -> tensor normalizedX", x.Size, ltb.attn_norm.weights.Size)
    normalizedX, err := ltb.attn_norm.Forward(infContext, x)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling LlamaTransformerBlock.Forward for layer: 1 / 32, startPos: 32 -> tensor currentTensor ...
[DEBUG] ... LlamaTransformerBlock.Forward started for x: shape([32 4096]), startPos: 32, freqsCis: shape([32 64]), mask: shape([]) ...
[DEBUG] ... Calling RMSNorm for tensor x shape([32 4096]) and LlamaTransformerBlock.attn_norm weights shape([4096]) -> tensor normalizedX ...
```

## **15.2. Performing Forward Pass Through Attention Module - Calling**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (ltb *LlamaTransformerBlock) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling LlamaAttention.Forward for tensor normalizedX shape(%v) and startPos: %d, freqsCis: shape(%v), mask: shape(%v) -> tensor h", normalizedX.Size, startPos, freqsCis.Size, maskSize)
    h, err := ltb.attention.Forward(infContext, normalizedX, startPos, freqsCis, mask)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling LlamaAttention.Forward for tensor normalizedX shape([32 4096]) and startPos: 0, freqsCis: shape([32 64]), mask: shape([32 32]) -> tensor h ...
```

## **15.3. Performing Forward Pass Through Attention Module - LlamaAttention.Forward(...)**

>**Recap:**<br>
>The most important part of the transformer models that provide accurate outputs is [the attention mechanism](https://arxiv.org/abs/1706.03762). Each "block" of LLaMa consists of a self-attention and a feed-forward neural network parts. The details will be explained further, but also we call these "block"s as "layer"s.

The attention mechanism is one of the important inventions that made language models more improved. LLaMa models have implemented "multi-head attention", so in our model case (LLaMA 2 7B-chat) we have 32 attention heads with some other supportive components. In the following steps, we will walk through details of an "attention module".

>**Important note:**<br>
>In our case model LLaMA 2 7B-chat has 32 layers of transformer blocks and each block contains an attention module containing 32 attention heads.  Both numbers are 32, but they specify numbers of different concepts, so pay more attention to avoid any confusion.

![STAGE 7: Forward Pass Through Attention Module Diagram](./images/DIAG01-STAGE07-forward-pass-through-attention-module.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

### **15.3.1. Recall: Structure of LlamaAttention**

The ```LlamaAttention``` object consists of:

* ```attn_wq```: Attention query weights tensor with shape of ```{N_Heads * HeadDim, Dim} = {32 * 128, 4096} = {4096, 4096}```,
* ```attn_wk```: Attention key weights tensor with shape of ```{N_KVHeads * HeadDim, Dim} = {32 * 128, 4096} = {4096, 4096}```,
* ```attn_wv```: Attention value weights tensor with shape of ```{N_KVHeads * HeadDim, Dim} = {32 * 128, 4096} = {4096, 4096}```,
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
    attn_wk *ml.Tensor // Original: "layers.0.attention.wk.weight"  |  ggml: "blk.0.attn_k.weight" | [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
    attn_wv *ml.Tensor // Original: "layers.0.attention.wv.weight"  |  ggml: "blk.0.attn_v.weight" | [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
    attn_wo *ml.Tensor // Original: "layers.0.attention.wo.weight"  |  ggml: "blk.0.attn_output.weight" | [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
}
```

### **15.3.2. Structure to run multiple linear transformations**

Now, we have the ```x``` tensor argument. In our case, at the first iteration, ```x``` is <u>the normalized form of</u> input tensor, at the other iterations, ```x``` is <u>the normalized form of</u> output of previous ```LlamaTransformerBlock```. In our case, at the first iteration, the shape of this tensor is ```{32, 4096}```. 32 stands for sequence length, 4096 stands for the embedding layer dimension.

In our attention module, we have ```LlamaAttention.attn_wq```, ```LlamaAttention.attn_wk```, and ```LlamaAttention.attn_wv```. They are weight tensors of current (one of 32 layers) layer, which stand for "attention query weights", "attention key weights", and "attention value weights" respectively.

We need to perform a linear transformation with our ```x``` tensor with each of these three weight tensors independently, then take the results into ```xq```, ```xk```, and ```xv``` tensors respectively. These operations can be done independently, so we can run them parallelly. In this step, we provide a structure to call them parallelly as follows.

The concepts that is used here were described in the chapter [13.1. Preliminary Concepts](./13-GENERATING-NEXT-TOKENS.md). For now, know that, the [context](https://pkg.go.dev/context) and [WaitGroup](https://www.geeksforgeeks.org/using-waitgroup-in-golang/) are used to manage parallel operations. We call the [ml.LinearTransformation](../src/ml/operations_impl.go) as [goroutines](https://gobyexample.com/goroutines).

Then, these 3 goroutines are performed and finished, we take the results ```xq```, ```xk```, and ```xv``` tensors from the ```parallelResults``` map.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    sequenceLength := x.Size[0]

    ctx, cancel := context.WithCancelCause(context.Background())
    var wg sync.WaitGroup
    var mu sync.Mutex
    parallelResults := make(map[string]*ml.Tensor)

    common.GLogger.DebugPrintf("[Scheduling goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wq weights shape(%v) -> tensor xq", x.Size, lat.attn_wq.Size)
    wg.Add(1)
    go func() {
        defer wg.Done()
        if ctx.Err() != nil {
            return
        }
        ...
    }()

    common.GLogger.DebugPrintf("[Scheduling goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wk weights shape(%v) -> tensor xk", x.Size, lat.attn_wk.Size)
    wg.Add(1)
    go func() {
        defer wg.Done()
        if ctx.Err() != nil {
            return
        }
        ...
    }()

    common.GLogger.DebugPrintf("[Scheduling goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wv weights shape(%v) -> tensor xv", x.Size, lat.attn_wv.Size)
    wg.Add(1)
    go func() {
        defer wg.Done()
        if ctx.Err() != nil {
            return
        }
        ...
    }()

    runtime.Gosched()

    select {
    case <-ctx.Done():
        // Cancellation signal was received
        return nil, context.Cause(ctx)
    case <-common.WaitGroupDone(&wg):
        runtime.Gosched()
    }

    xq := parallelResults["xq"]
    xk := parallelResults["xk"]
    xv := parallelResults["xv"]
}
```

### **15.3.3. Calculating xq, xk, and xv**

We perform three [ml.LinearTransformation](../src/ml/operations_impl.go) which set their results into the ```parallelResults``` map with mutex locks. Each of these three result tensors are with same shape of ```{32, 4096}```.

![STAGE 8: Forward Pass Through Attention Module - Calculating xq, xk, and xv Diagram](./images/DIAG01-STAGE08-attention-fwd-calculating-xq-xk-xv.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module - Calculating xq, xk, and xv**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    go func() {
        ...
        xq, err := ml.LinearTransformation(x, lat.attn_wq)
        ...
        mu.Lock()
        parallelResults["xq"] = xq
        mu.Unlock()
    }
    ...
     go func() {
        ...
        xk, err := ml.LinearTransformation(x, lat.attn_wk)
        ...
        mu.Lock()
        parallelResults["xk"] = xk
        mu.Unlock()
    }
    ...
     go func() {
        ...
        xv, err := ml.LinearTransformation(x, lat.attn_wv)
        ...
        mu.Lock()
        parallelResults["xv"] = xv
        mu.Unlock()
    }
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... [Scheduling goroutine] ml.LinearTransformation for x shape([32 4096]) and LlamaAttention.attn_wq weights shape([4096 4096]) -> tensor xq ...
[DEBUG] ... [Scheduling goroutine] ml.LinearTransformation for x shape([32 4096]) and LlamaAttention.attn_wk weights shape([4096 4096]) -> tensor xk ...
[DEBUG] ... [Scheduling goroutine] ml.LinearTransformation for x shape([32 4096]) and LlamaAttention.attn_wv weights shape([4096 4096]) -> tensor xv ...
[DEBUG] ... [Calling in goroutine] ml.LinearTransformation for x shape([32 4096]) and LlamaAttention.attn_wv weights shape([4096 4096]) -> tensor xv ...
[DEBUG] ... [Calling in goroutine] ml.LinearTransformation for x shape([32 4096]) and LlamaAttention.attn_wq weights shape([4096 4096]) -> tensor xq ...
[DEBUG] ... [Calling in goroutine] ml.LinearTransformation for x shape([32 4096]) and LlamaAttention.attn_wk weights shape([4096 4096]) -> tensor xk ...
[DEBUG] ... Parallel results, xq: shape([32 4096]), xk: shape([32 4096]), xv: shape([32 4096]) ...
```

>Note: As you can see the logs above, the order of ```[Scheduling goroutine]``` and ```[Calling in goroutine]``` lines are different, it shows they were executed parallelly.

### **15.3.4. Do reshapings**

![STAGE 9: Forward Pass Through Attention Module - Do reshapings Diagram](./images/DIAG01-STAGE09-attention-do-reshapings.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module - Do reshapings**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

The resulting three tensors are with same shape of ```{32, 4096}```. In our case, we implement "multi-head attention" with ``32`` attention heads (according to ```modelArgs.N_Heads = 32```). Our resulting tensors have the values of each attention head combined into the dimension with size of ```4096```. Our ```modelArgs.HeadDim = 128```, so the dimension of each attention head is 128.

Now, we need to reshape our tensors to differentiate each attention head. The tensors with shape of ```{32, 4096}``` will be in shape of ```{32, 32, 128}```. The first ```32``` stands for sequence length, the second ```32``` stands for "attention head count" ```modelArgs.N_Heads```, the ```128``` stands for "attention head dimension" ```modelArgs.HeadDim```.

But, wait! We have mentioned the "attention head count" with ```modelArgs.N_Heads```, but in the code there are two concepts: ```lat.N_Heads``` and ```lat.N_KVHeads```. The ```modelArgs.N_Heads``` is used for specifying the shape of query tensor ```xq```. The ```modelArgs.N_KVHeads``` is used for specifying the shape of key ```xk``` and value ```xv``` tensors.

In our case model LLaMA 2 7B-chat, ```modelArgs.N_Heads``` is used as default value for ```modelArgs.N_KVHeads``` which equals to ```32```. The larger models apply [Grouped Multi-Query Attention](https://paperswithcode.com/method/grouped-query-attention) and in these larger models these two values are different. But we didn't implement Grouped Multi-Query Attention, it isn't necessary for 7B-chat model.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    common.GLogger.DebugPrintf("Parallel results, xq: shape(%v), xk: shape(%v), xv: shape(%v)", xq.Size, xk.Size, xv.Size)

    /*
        Do reshapings
    */
    var err error
    if xq, err = xq.Reshape([]int{sequenceLength, lat.N_Heads, lat.HeadDim}); err != nil {
        return nil, err
    }

    if xk, err = xk.Reshape([]int{sequenceLength, lat.N_KVHeads, lat.HeadDim}); err != nil {
        return nil, err
    }

    if xv, err = xv.Reshape([]int{sequenceLength, lat.N_KVHeads, lat.HeadDim}); err != nil {
        return nil, err
    }

    common.GLogger.DebugPrintf("Reshaping results, xq: shape(%v), xk: shape(%v), xv: shape(%v)", xq.Size, xk.Size, xv.Size)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Parallel results, xq: shape([32 4096]), xk: shape([32 4096]), xv: shape([32 4096]) ...
[DEBUG] ... Reshaping results, xq: shape([32 32 128]), xk: shape([32 32 128]), xv: shape([32 32 128]) ...
```

### **15.3.5. Apply Rotary Embeddings**

![STAGE 10: Forward Pass Through Attention Module - Apply Rotary Embeddings Diagram](./images/DIAG01-STAGE10-attention-apply-rotary-embeddings.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module - Apply Rotary Embeddings**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    /*
        Apply rotary embeddings
    */

    if xq, xk, err = applyRotaryEmbeddings(xq, xk, freqsCis); err != nil { // example shape=[5,32,128] dtype=DT_BF16
        return nil, err
    }

    common.GLogger.DebugPrintf("applyRotaryEmbeddings results, xq: shape(%v), xk: shape(%v)", xq.Size, xk.Size)
    ...
}
```

>For more information about how the ```freqsCis``` tensor is initiated, refer to [10. ROPE ROTARY POSITIONAL EMBEDDINGS](./10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md) and [10.BONUS. PRECOMPUTING FREQUENCY TENSOR](./10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb).

During this step, we apply the *RoPE (Rotary Positional Embeddings)* approach propoed by [RoFormer](https://arxiv.org/abs/2104.09864v5) paper over our query ```xq``` and key ```xk``` tensors.

> Note: The shape information here is for the first iteration of our sample case. The shape samples written in code descriptions are for a different case. The first dimension of the shapes stands for sequence length which varies by the prompt tokens count.

* Have the query tensor ```xq``` with shape of ```{32, 32, 128}```. Convert the tensor's data type from ```DT_BF16``` to ```DT_COMPLEX```, and change the shape to  ```{32, 32, 64}``` via [Tensor.ViewAsComplex64WithReshape(...)](../src/ml/tensor.go) method, the result is assigned into ```xq_``` variable,<br>
  This method:<br>
  * Converts the data type of the tensor to ```DT_F32``` (float32), the shape remains as  ```{32, 32, 128}```,
  * Reshapes the tensor with shape of ```{32, 32, 64, 2}```,
  * Converts each pair of float32 in the last dimension into a ```complex64``` data type via [Tensor.ViewAsComplex64(...)](../src/ml/tensor.go), the new shape is ```{32, 32, 64}``` with data type of ```DT_COMPLEX```.

  See [torch.view_as_complex](https://pytorch.org/docs/stable/generated/torch.view_as_complex.html) documentation for more information.

    > Comment from Pytorch's documentation (link above):<br>
    >Torch's view_as_complex() is only supported for tensors with torch.dtype torch.float64 and torch.float32.<br>
    >The input is expected to have the last dimension of size 2. In addition, the tensor must have a stride of 1 for its last dimension. The strides of all other dimensions must be even numbers.

* Have the key tensor ```xk``` with shape of ```{32, 32, 128}```. Convert the tensor's data type from ```DT_BF16``` to ```DT_COMPLEX```, and change the shape to  ```{32, 32, 64}``` via [Tensor.ViewAsComplex64WithReshape(...)](../src/ml/tensor.go) method, the result is assigned into ```xk_``` variable,
* Reshape the ```freqs_cis``` tensor with shape of ```{32, 64}``` to the shape ```{32, 1, 64}```,
* Process the ```xqOut```:
    * Perform an element-wise multiplication with ```xq_``` tensor with shape of ```{32, 32, 64}``` and ```freqs_cis``` tensor with shape of ```{32, 1, 64}``` via [ml.MultiplyElementwise](../src/ml/operations_impl.go). Output shape is ```{32, 32, 64}```, assign the result into ```xqOut``` variable,
    * Convert the ```xqOut``` tensor's data type from ```DT_COMPLEX``` to ```DT_F32``` (float32), and change the shape to  ```{32, 32, 128}``` via [Tensor.ViewAsComplex64WithReshape(...)](../src/ml/tensor.go) method (think as packing-unpacking the pairs in the last dimension),
    * Convert the ```xqOut``` tensor's data type from ```DT_F32``` (float32) to ```DT_BF16``` with same shape ```{32, 32, 128}```,
* Process the ```xkOut```:
    * Perform an element-wise multiplication with ```xk_``` tensor with shape of ```{32, 32, 64}``` and ```freqs_cis``` tensor with shape of ```{32, 1, 64}``` via [ml.MultiplyElementwise](../src/ml/operations_impl.go). Output shape is ```{32, 32, 64}```, assign the result into ```xkOut``` variable,
    * Convert the ```xkOut``` tensor's data type from ```DT_COMPLEX``` to ```DT_F32``` (float32), and change the shape to  ```{32, 32, 128}``` via [Tensor.ViewAsComplex64WithReshape(...)](../src/ml/tensor.go) method (think as packing-unpacking the pairs in the last dimension),
    * Convert the ```xkOut``` tensor's data type from ```DT_F32``` (float32) to ```DT_BF16``` with same shape ```{32, 32, 128}```,
* Return the tensors ```xqOut``` and ```xkOut``` tensors with shape ```{32, 32, 128}``` together as result.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func applyRotaryEmbeddings(xq *ml.Tensor, xk *ml.Tensor, freqs_cis *ml.Tensor) (xqOut *ml.Tensor, xkOut *ml.Tensor, err error) {
    // xq shape=[5,32,128] dtype=DT_BF16
    xq_, err := xq.ViewAsComplex64WithReshape() // shape=[5,32,64] dtype=DT_COMPLEX
    if err != nil {
        return nil, nil, err
    }
    // xk shape=[5,32,128] dtype=DT_BF16
    xk_, err := xk.ViewAsComplex64WithReshape() // shape=[5,32,64] dtype=DT_COMPLEX
    if err != nil {
        return nil, nil, err
    }

    // freqs_cis shape=[5, 64] dtype=DT_COMPLEX
    if freqs_cis, err = freqs_cis.Reshape([]int{xq_.Size[0], 1, xq_.Size[2]}); err != nil { // shape=[5,1,64] dtype=DT_COMPLEX
        return nil, nil, err
    }

    if xqOut, err = ml.MultiplyElementwise(xq_, freqs_cis); err != nil { // shape=[5,32,64] dtype=DT_COMPLEX
        return nil, nil, err
    }
    if xqOut, err = xqOut.ViewAsFloat32WithReshape(); err != nil { // shape=[5,32,128] dtype=DT_F32
        return nil, nil, err
    }
    if xqOut, err = xqOut.ToBFloat16(); err != nil { // shape=[5,32,128] dtype=DT_BF16
        return nil, nil, err
    }

    if xkOut, err = ml.MultiplyElementwise(xk_, freqs_cis); err != nil { // shape=[5,32,64] dtype=DT_COMPLEX
        return nil, nil, err
    }
    if xkOut, err = xkOut.ViewAsFloat32WithReshape(); err != nil { // shape=[5,32,128] dtype=DT_F32
        return nil, nil, err
    }
    if xkOut, err = xkOut.ToBFloat16(); err != nil { // shape=[5,32,128] dtype=DT_BF16
        return nil, nil, err
    }
    return xqOut, xkOut, nil
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... applyRotaryEmbeddings results, xq: shape([32 32 128]), xk: shape([32 32 128]) ...
```

### **15.3.6. Update KV cache**

![STAGE 11: Forward Pass Through Attention Module - Update KV cache Diagram](./images/DIAG01-STAGE11-attention-update-kv-cache.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module - Update KV cache**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

We have initiated an "inference context" with type of [model.InferenceContext](../src/model/inferencecontext.go) which we keep the state of one inference process. In this context object, we have two cache arrays: ```InferenceContext.CacheK``` and ```InferenceContext.CacheV``` which stand for "cache of keys" and "cache of values" respectively. These arrays have 32 items correspond to 32 layers. Each of these items consists of tensors with shape of ```{200, 32, 128}```. 200 stands for the maximum sequence length ```inferenceArgs.SequenceLength```, 32 stands for ```modelArgs.N_KVHeads```, 128 stands for ```modelArgs.HeadDim```.

Here, in our case of the first iteration, we set the cache of the ```0th``` layer. We set the slices of the ```CacheK``` and ```CacheV``` with index range ```0``` (startPos) to ```32``` (startPos + sequenceLength) to ```xk``` and ```xv``` tensors respectively. The ```sequenceLength``` is the first dimension of the shape of ```x``` tensor argument.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    /*
        Update KV cache
    */

    infContext.CacheK[lat.LayerIndex].SetSlice([]int{startPos}, []int{startPos + sequenceLength}, xk)
    infContext.CacheV[lat.LayerIndex].SetSlice([]int{startPos}, []int{startPos + sequenceLength}, xv)
    ...
}
```

### **15.3.7. Retrieve cached KV so far**

To make easy to understand how the KV cache is updated, think of a sample:

* Prompt tokens count is 32,
* While generation of the 1st token:
    * ```startPos``` is ```0```,
    * The shape of ```x``` argument of ```LlamaAttention.Forward(...)``` is ```{32, 4096}```,
    * The shapes of ```xk``` and ```xv``` are ```{32, 32, 128}```,
    * We update the indices of each cache ```0:32``` with ```xk``` and ```xv```.
* While generation of the 2nd token:
    * ```startPos``` is ```32```,
    * The shape of ```x``` argument of ```LlamaAttention.Forward(...)``` is ```{1, 4096}``` (in the iterations except first, the tokens are processed one by one, because of this, the first dimension is 1),
    * The shapes of ```xk``` and ```xv``` are ```{1, 32, 128}```,
    * We update the indices of each cache ```32:33``` with ```xk``` and ```xv```.
* While generation of the 3rd token:
    * ```startPos``` is ```33```,
    * The shape of ```x``` argument of ```LlamaAttention.Forward(...)``` is ```{1, 4096}``` (in the iterations except first, the tokens are processed one by one, because of this, the first dimension is 1),
    * The shapes of ```xk``` and ```xv``` are ```{1, 32, 128}```,
    * We update the indices of each cache ```33:34``` with ```xk``` and ```xv```.
* So on...

Now, we take the cached keys and values for the all positions so far. To make easy to understand:

* Prompt tokens count is 32,
* While generation of the 1st token:
    * ```startPos``` is ```0```,
    * The shape of ```x``` argument of ```LlamaAttention.Forward(...)``` is ```{32, 4096}```,
    * We take items at indices ```0:32``` of ```CacheK``` and ```CacheV``` into ```keys``` and ```values``` tensors respectively, because ```startPos + sequenceLength = 32```. The ```keys``` and ```values``` are with the shape of ```{32, 32, 128}```.
* While generation of the 2nd token:
    * ```startPos``` is ```32```,
    * The shape of ```x``` argument of ```LlamaAttention.Forward(...)``` is ```{1, 4096}``` (in the iterations except first, the tokens are processed one by one, because of this, the first dimension is 1),
    * We take items at indices ```0:33``` of ```CacheK``` and ```CacheV``` into ```keys``` and ```values``` tensors respectively, because ```startPos + sequenceLength = 33```. The ```keys``` and ```values``` are with the shape of ```{33, 32, 128}```.
* While generation of the 3rd token:
    * ```startPos``` is ```33```,
    * The shape of ```x``` argument of ```LlamaAttention.Forward(...)``` is ```{1, 4096}``` (in the iterations except first, the tokens are processed one by one, because of this, the first dimension is 1),
    * We take items at indices ```0:34``` of ```CacheK``` and ```CacheV``` into ```keys``` and ```values``` tensors respectively, because ```startPos + sequenceLength = 34```. The ```keys``` and ```values``` are with the shape of ```{34, 32, 128}```.

In this documentation, we cover only the first iteration of generating the first token. So, in our case, we retrieve items at indices ```0:32```, the shapes of our ```keys``` and ```values``` are ```{32, 32, 128}```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    /*
        Retrieve cached KV so far
    */

    keys, err := infContext.CacheK[lat.LayerIndex].Slice([]int{0}, []int{startPos + sequenceLength})
    if err != nil {
        return nil, err
    }
    values, err := infContext.CacheV[lat.LayerIndex].Slice([]int{0}, []int{startPos + sequenceLength})
    if err != nil {
        return nil, err
    }
    ...
}
```

### **15.3.8. Repeat K/V heads if N_KVHeads < N_Heads**

Repeating K/V heads step is only required if ```N_KVHeads < N_Heads```, so in our case (LLaMa 7B-chat model) we don't need to do this step. Nevertheless, we've defined a no-op function: [attentionRepeatKV](../src/model/llamatransformer.go). This step is required for larger models that apply [Grouped Multi-Query Attention](https://paperswithcode.com/method/grouped-query-attention).

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    /*
        Repeat k/v heads if N_KVHeads < N_Heads
    */

    if keys, err = attentionRepeatKV(keys, lat.N_Rep); err != nil { // example shape=[5, 32, 128] (cacheLen + sequenceLength, N_Heads, HeadDim)
        return nil, err
    }
    if values, err = attentionRepeatKV(values, lat.N_Rep); err != nil { // example shape=[5, 32, 128] (cacheLen + sequenceLength, N_Heads, HeadDim)
        return nil, err
    }
    ...
}
```

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func attentionRepeatKV(x *ml.Tensor, N_Rep int) (*ml.Tensor, error) {
    // See: https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L164
    // repeat_kv function was not implemented because currently we support only 7B model
    if N_Rep == 1 {
        return x, nil
    }
    return nil, fmt.Errorf("currently only 7B model is supported, N_Rep > 1 case was not implemented yet, because of this")
}
```

### **15.3.9. Do transposes**

![STAGE 12: Forward Pass Through Attention Module - Do transposes Diagram](./images/DIAG01-STAGE12-attention-do-transposes.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module - Do transposes**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

In this step, we need to perform some transpose operations:

* Transpose ```xq```'s ```0th``` and ```1st``` dimensions: from ```{sequenceLength, N_Heads, HeadDim} = {32, 32, 128}``` to ```{N_Heads, sequenceLength, HeadDim} = {32, 32, 128}```,<br>
    >In the sample at the code comments, sequenceLength is 5, and the operation is from ```{5, 32, 128}``` to ```{32, 5, 128}```
* Transpose ```keys```'s ```0th``` and ```1st``` dimensions: from ```{sequenceLength, N_Heads, HeadDim} = {32, 32, 128}``` to ```{N_Heads, sequenceLength, HeadDim} = {32, 32, 128}```,<br>
    >In the sample at the code comments, sequenceLength is 5, and the operation is from ```{5, 32, 128}``` to ```{32, 5, 128}```
* Transpose ```values```'s ```0th``` and ```1st``` dimensions: from ```{sequenceLength, N_Heads, HeadDim} = {32, 32, 128}``` to ```{N_Heads, sequenceLength, HeadDim} = {32, 32, 128}```,<br>
    >In the sample at the code comments, sequenceLength is 5, and the operation is from ```{5, 32, 128}``` to ```{32, 5, 128}```
* Transpose ```keys```'s ```1st``` and ```2nd``` dimensions: from ```{N_Heads, sequenceLength, HeadDim} = {32, 32, 128}``` to ```{N_Heads, HeadDim, sequenceLength} = {32, 128, 32}```.<br>
    >In the sample at the code comments, sequenceLength is 5, and the operation is from ```{32, 5, 128}``` to ```{32, 128, 5}```

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    /*
        Do transposes
    */

    if xq, err = xq.Transpose(0, 1); err != nil { // from [5, 32, 128] -> example shape=[32, 5, 128] (N_Heads, sequenceLength, HeadDim)
        return nil, err
    }

    if keys, err = keys.Transpose(0, 1); err != nil { // from [5, 32, 128] -> example shape=[32, 5, 128] (N_Heads, sequenceLength, HeadDim)
        return nil, err
    }

    if values, err = values.Transpose(0, 1); err != nil { // from [5, 32, 128] -> example shape=[32, 5, 128] (N_Heads, sequenceLength, HeadDim)
        return nil, err
    }

    if keys, err = keys.Transpose(1, 2); err != nil { // from [32, 5, 128] -> example shape=[32, 128, 5] (N_Heads, HeadDim, sequenceLength)
        return nil, err
    }

    common.GLogger.DebugPrintf("Multiple transposing results, xq: shape(%v), keys: shape(%v), values: shape(%v)", xq.Size, keys.Size, values.Size)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Multiple transposing results, xq: shape([32 32 128]), keys: shape([32 128 32]), values: shape([32 32 128]) ...
```

### **15.3.10. Calculate scores**

![STAGE 13: Forward Pass Through Attention Module - Calculate scores Diagram](./images/DIAG01-STAGE13-attention-calculate-scores.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module - Calculate scores**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

```py
# Goal in Python manner:
scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
```

We calculate the scores which we will perform [Softmax](https://en.wikipedia.org/wiki/Softmax_function) operation over.

* Perform a matrix multiplication over ```xq``` with shape of ```{32, 32, 128}``` and ```keys``` with shape of ```{32, 128, 32}``` (transpose operation has been performed in previous step already),
* Take square root of ```lat.HeadDim = 128``` is ```11.3125``` in BFloat16 form,
* Divide all items of the result of matrix multiplication ```xqMatMulKeys``` to ```11.3125```, assign the result into ```scores```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling ml.MatMul for xq shape(%v) and keys shape(%v) -> tensor xqMatMulKeys", xq.Size, keys.Size)
    xqMatMulKeys, err := ml.MatMul(xq, keys) // matmul([32,5,128], [32,128,5]) -> example shape=[32,5,5] (N_Heads, sequenceLength, sequenceLength)
    if err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Calling ml.DivToScalar for xqMatMulKeys shape(%v) and scalar -> tensor scores", xqMatMulKeys.Size)
    scores, err := ml.DivToScalar(xqMatMulKeys, dtype.BFloat16fromFloat32(float32(math.Sqrt(float64(lat.HeadDim))))) // example shape=[32,5,5]
    if err != nil {
        return nil, err
    }
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.MatMul for xq shape([32 32 128]) and keys shape([32 128 32]) -> tensor xqMatMulKeys ...
[DEBUG] ... Calling ml.DivToScalar for xqMatMulKeys shape([32 32 32]) and scalar -> tensor scores ...
```

### **15.3.11. Perform masking on scores**

If there is a given ```mask``` argument, perform masking operation. This is because LLaMa is an auto-regressive model and our mask contains triangular matrix consisting of ```0```s and ```-Inf (negative infinity)```s. For more information, refer to [14.2.3. Creating the mask tensor](./14-MAKING-PREDICTION-WITH-LLAMA-MODEL-1.md).

By performing [ml.Add(...)](../src/ml/operations_impl.go) operation over ```scores``` with shape of ```{32, 32, 32}``` and ```mask``` tensor with shape of ```{32, 32}```, we take the items corresponding on ```0``` mask values and ignore the items corresponding on ```-Inf``` mask values (adding ```-Inf``` to a number makes the number ```-Inf```).

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    if mask != nil {
        common.GLogger.DebugPrintf("Calling ml.Add to calculate scores shape(%v) + mask shape(%v) -> tensor scores", scores.Size, mask.Size)
        if scores, err = ml.Add(scores, mask); err != nil { // example shape=[32,5,5]
            return nil, err
        }
    } else {
        common.GLogger.DebugPrintf("Skipping addition scores + mask")
    }
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.Add to calculate scores shape([32 32 32]) + mask shape([32 32]) -> tensor scores ...
```

### **15.3.12. Apply Softmax over scores**

```py
# Goal in Python manner:
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

In this step, we perform [Softmax](https://en.wikipedia.org/wiki/Softmax_function) operation over the scores.

>For more information, refer to: [torch.nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html).

To achieve this:

* Convert the ```scores``` tensor data type to ```DT_F32``` (float32),
* Call [ml.Softmax](../src/ml/operations_impl.go) function,
* Convert the result data type to ```DT_BF16``` and assign into ```scores``` tensor.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Converting scores tensor shape(%v) to Float32 tensor -> tensor scores", scores.Size)
    scores, err = scores.ToFloat32() // example shape=[32,5,5] dtype=DT_F32
    if err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Calling ml.Softmax for scores shape(%v) and dim %d -> tensor scores", scores.Size, len(scores.Size)-1)
    if scores, err = ml.Softmax(scores, len(scores.Size)-1); err != nil { // example shape=[32,5,5] dtype=DT_F32
        return nil, err
    }
    common.GLogger.DebugPrintf("Converting scores tensor shape(%v) to BFloat16 tensor -> tensor scores", scores.Size)
    if scores, err = scores.ToBFloat16(); err != nil { // example shape=[32,5,5] (N_Heads, sequenceLength, sequenceLength) dtype=DT_BF16
        return nil, err
    }
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Converting scores tensor shape([32 32 32]) to Float32 tensor -> tensor scores ...
[DEBUG] ... Calling ml.Softmax for scores shape([32 32 32]) and dim 2 -> tensor scores ...
[DEBUG] ... Converting scores tensor shape([32 32 32]) to BFloat16 tensor -> tensor scores ...
```

### **15.3.13. Multiply values tensor and scores tensor**

![STAGE 14: Forward Pass Through Attention Module - Calculate output Diagram](./images/DIAG01-STAGE14-attention-calculate-output.drawio.svg)
<sup>*Diagram: **Forward Pass Through Attention Module - Calculate output**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

```py
# Goal in Python manner:
output = torch.matmul(scores, values)
output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
```

* Perform a matrix multiplication over ```values``` with shape of ```{32, 32, 32}``` and ```values``` with shape of ```{32, 32, 128}```, assign the result with shape of ```{32, 32, 128}``` into ```output``` tensor,
* Transpose the ```output```'s ```0th``` and ```1st``` dimensions to shape of ```{32, 32, 128}```,
* Reshape the ```output``` to the shape of ```{32, 4096} = {sequenceLength, output.GetElementCount() / sequenceLength}```,

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling ml.MatMul for scores shape(%v) and values shape(%v) -> tensor output", scores.Size, values.Size)
    output, err := ml.MatMul(scores, values)
    if err != nil {
        return nil, err
    }
    if output, err = output.Transpose(0, 1); err != nil {
        return nil, err
    }
    outputTrailingSize := output.GetElementCount() / sequenceLength
    if output, err = output.Reshape([]int{sequenceLength, outputTrailingSize}); err != nil {
        return nil, err
    }
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.MatMul for scores shape([32 32 32]) and values shape([32 32 128]) -> tensor output ...
```

### **15.3.14. Apply attention output weights**

We have the output weights of our attention module in the ```lat.attn_wo``` tensor. We perform a linear transformation with our ```output``` tensor (with the shape of ```{32, 4096}```) with the ```lat.attn_wo``` weights tensor (with shape of ```{4096, 4096}```). Then, we return this result with the shape of ```{32, 4096}``` as output of the attention model.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lat *LlamaAttention) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    /*
        Apply lat.attn_wo weights to output
    */

    common.GLogger.DebugPrintf("Calling ml.LinearTransformation for output shape(%v) and LlamaAttention.attn_wo weights shape(%v) -> tensor output", output.Size, lat.attn_wo.Size)
    // lat.attn_wo: [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
    if output, err = ml.LinearTransformation(output, lat.attn_wo); err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Returning tensor output: shape(%v)", output.Size)
    return output, nil
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.LinearTransformation for output shape([32 4096]) and LlamaAttention.attn_wo weights shape([4096 4096]) -> tensor output ...
[DEBUG] ... Returning tensor output: shape([32 4096]) ...
```

## **15.4. Adding the attention module output to current tensor**

![STAGE 15: Add attention module output and current tensor Diagram](./images/DIAG01-STAGE15-add-attention-output-and-current-tensor.drawio.svg)
<sup>*Diagram: **Add attention module output and current tensor**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

Now, we returned to our latest position in the ```LlamaTransformerBlock.Forward(...)``` method.

We had the ```x``` tensor argument. In our case, at the first iteration, ```x``` is input tensor, at the other iterations, ```x``` is output of previous ```LlamaTransformerBlock```. In our case, at the first iteration, the shape of this tensor is ```{32, 4096}```. 32 stands for sequence length, 4096 stands for the embedding layer dimension. ```normalizedX``` which is the resulting tensor will have same shape as the input, ```{32, 4096}```.

Also, we have the ```h``` tensor with the shape of ```{32, 4096}```, which is the output of our attention module ```LlamaAttention```.

We add ```x``` and ```h``` tensors via [ml.Add(...)](../src/ml/operations_impl.go) function and assign the result into ```h``` tensor.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (ltb *LlamaTransformerBlock) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling ml.Add to calculate x shape(%v) + h shape(%v) -> tensor h", x.Size, h.Size)
    if h, err = ml.Add(x, h); err != nil {
        return nil, err
    }
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.Add to calculate x shape([32 4096]) + h shape([32 4096]) -> tensor h ...
```

## **15.5. Performing Forward Pass Through Feed-Forward Prenormalization - RMSNorm.Forward(...)**

![STAGE 16: Forward Pass Through Feed-Forward Pre-normalization Diagram](./images/DIAG01-STAGE16-forward-pass-through-ffn-prenormalization.drawio.svg)
<sup>*Diagram: **Forward Pass Through Feed-Forward Pre-normalization**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

Now, we have the ```h``` tensor. We perform RMSNorm over ```h``` tensor with normalization weights of ```ltb.ffn_norm```, and assign the result into ```normalizedH``` which is the resulting tensor will have the same shape as the ```h```, ```{32, 4096}```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (ltb *LlamaTransformerBlock) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling RMSNorm for tensor h shape(%v) and LlamaTransformerBlock.ffn_norm weights shape(%v) -> tensor normalizedH", x.Size, ltb.ffn_norm.weights.Size)
    normalizedH, err := ltb.ffn_norm.Forward(infContext, h)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling RMSNorm for tensor h shape([32 4096]) and LlamaTransformerBlock.ffn_norm weights shape([4096]) -> tensor normalizedH ...
```

## **15.6. Performing Forward Pass Through Feed-Forward Module - Calling**

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (ltb *LlamaTransformerBlock) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling LlamaFeedForward.Forward for tensor normalizedH shape(%v) -> tensor ffnOutput", normalizedH.Size)
    ffnOutput, err := ltb.feedForward.Forward(normalizedH)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling LlamaFeedForward.Forward for tensor normalizedH shape([32 4096]) -> tensor ffnOutput ...
```

## **15.7. Performing Forward Pass Through Feed-Forward Module - LlamaFeedForward.Forward(...)**

![STAGE 17: Forward Pass Through Feed-Forward Module Diagram](./images/DIAG01-STAGE17-forward-pass-through-ffn-module.drawio.svg)
<sup>*Diagram: **Forward Pass Through Feed-Forward Module**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

```py
# Goal in Python manner:
self.w2(F.silu(self.w1(x)) * self.w3(x))

# Python code with our variable names:
self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
```

In this stage, we have a feed-forward neural network module consisting multiple weight tensors for each of 32 transformer block layers, with names: ```w1```, ```w2```, and ```w3``` in original Python repository, ```ffn_gate```, ```ffn_down```, and ```ffn_up``` in our project, respectively.

The steps are:

* Perform a linear transformation over ```x``` with shape of ```{32, 4096}``` and ```lff.ffn_gate``` weights with shape of ```{11008, 4096}```, assign the resulting tensor with shape of ```{32, 11008}``` into ```h```,
* Perform Sigmoid Linear Unit (SiLU) function over the ```h``` tensor via [ml.Silu(...)](../src/ml/activations.go) function,
    >For more information, refer to: [torch.nn.SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html).
* Perform a linear transformation over ```x``` with shape of ```{32, 4096}``` and ```lff.ffn_up``` weights with shape of ```{11008, 4096}```, assign the resulting tensor with shape of ```{32, 11008}``` into ```ffnUpX```,
* Perform an element-wise multiplication with ```h``` tensor with shape of ```{32, 11008}``` and ```ffnUpX``` tensor with shape of ```{32, 11008}``` via [ml.MultiplyElementwise](../src/ml/operations_impl.go). Output shape is ```{32, 11008}```, assign the result into ```h``` variable,
* Perform a linear transformation over ```h``` with shape of ```{32, 11008}``` and ```lff.ffn_down``` weights with shape of ```{4096, 11008}```, assign the resulting tensor with shape of ```{32, 4096}``` into ```output```,
* Return the ```output``` tensor.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lff *LlamaFeedForward) Forward(x *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling ml.LinearTransformation for x shape(%v) and LlamaFeedForward.ffn_gate weights shape(%v) -> tensor h", x.Size, lff.ffn_gate.Size)
    h, err := ml.LinearTransformation(x, lff.ffn_gate)
    if err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Calling ml.Silu for h shape(%v) -> tensor h", h.Size)
    if h, err = ml.Silu(h); err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Calling ml.LinearTransformation for x shape(%v) and LlamaFeedForward.ffn_up weights shape(%v) -> tensor ffnUpX", x.Size, lff.ffn_up.Size)
    ffnUpX, err := ml.LinearTransformation(x, lff.ffn_up)
    if err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Calling ml.MultiplyElementwise for h shape(%v) and ffnUpX weights shape(%v) -> tensor ffnUpX", h.Size, ffnUpX.Size)
    if h, err = ml.MultiplyElementwise(h, ffnUpX); err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Calling ml.LinearTransformation for h shape(%v) and LlamaFeedForward.ffn_down weights shape(%v) -> tensor output", h.Size, lff.ffn_down.Size)
    output, err := ml.LinearTransformation(h, lff.ffn_down)
    if err != nil {
        return nil, err
    }
    return output, nil
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.LinearTransformation for x shape([32 4096]) and LlamaFeedForward.ffn_gate weights shape([11008 4096]) -> tensor h ...
[DEBUG] ... Calling ml.LinearTransformation for x shape([32 4096]) and LlamaFeedForward.ffn_up weights shape([11008 4096]) -> tensor ffnUpX ...
[DEBUG] ... Calling ml.MultiplyElementwise for h shape([32 11008]) and ffnUpX weights shape([32 11008]) -> tensor ffnUpX ...
[DEBUG] ... Calling ml.LinearTransformation for h shape([32 11008]) and LlamaFeedForward.ffn_down weights shape([4096 11008]) -> tensor output ...
```

## **15.8. Adding the feed-forward network module output to current tensor**

![STAGE 18: Add Feed-Forward module output and current tensor Diagram](./images/DIAG01-STAGE18-add-ffn-output-and-current-tensor.drawio.svg)
<sup>*Diagram: **Add Feed-Forward module output and current tensor**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

Now, we returned to our latest position in the ```LlamaTransformerBlock.Forward(...)``` method.

* We have the ```ffnOutput``` tensor with the shape of ```{32, 4096}```, which is the output of our feed-forward neural network module ```LlamaFeedForward```.
* Also, we had the ```h``` tensor as current tensor with the shape of ```{32, 4096}```, which is the output of our attention module ```LlamaAttention```.
* We add ```h``` and ```ffnOutput``` tensors via [ml.Add(...)](../src/ml/operations_impl.go) function and assign the result with shape of ```{32, 4096}``` into ```output``` tensor,
* Return it as output of ```LlamaTransformerBlock```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (ltb *LlamaTransformerBlock) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling ml.Add to calculate h shape(%v) + ffnOutput shape(%v) -> tensor output", h.Size, ffnOutput.Size)
    output, err := ml.Add(h, ffnOutput)
    if err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Returning tensor output: shape(%v)", output.Size)
    return output, nil
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.Add to calculate h shape([32 4096]) + ffnOutput shape([32 4096]) -> tensor output ...
[DEBUG] ... Returning tensor output: shape([32 4096]) ...
````

The flow will continue with next ```LlamaTransformerBlock``` layer.

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: MAKING PREDICTION with LLAMA MODEL - 1](./14-MAKING-PREDICTION-WITH-LLAMA-MODEL-1.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: MAKING PREDICTION with LLAMA MODEL - 3&nbsp;&nbsp;&gt;](./16-MAKING-PREDICTION-WITH-LLAMA-MODEL-3.md)

</div>

# **16. MAKING PREDICTION with LLAMA MODEL - 3**

In previous [14. MAKING PREDICTION with LLAMA MODEL - 1](./14-MAKING-PREDICTION-WITH-LLAMA-MODEL-1.md), [15. MAKING PREDICTION with LLAMA MODEL - 2](./15-MAKING-PREDICTION-WITH-LLAMA-MODEL-2.md), and this chapters, we walk through the [LlamaTransformer.Forward(...)](../src/model/llamatransformer.go) method and its components.

Now, we have the ```currentTensor```, which is the resulting tensor after running all 32 transformer block layers. These transformer block layers were explained in the previous [15. MAKING PREDICTION with LLAMA MODEL - 2](./15-MAKING-PREDICTION-WITH-LLAMA-MODEL-2.md) chapter.

![STAGE 19: Forward Pass Through Output of The Transformer Diagram](./images/DIAG01-STAGE19-forward-pass-through-output.drawio.svg)
<sup>*Diagram: **Forward Pass Through Output of The Transformer**. For the complete diagram, [click here](./20-DIAGRAMS.md#complete-model-diagram).*</sup>

## **16.1. Performing Forward Pass Through Output Prenormalization - RMSNorm.Forward(...)**

The LLaMa 2 models use [Pre-RMSNorm (Root Mean Square Layer Normalization)](https://paperswithcode.com/method/rmsnorm). Because of we perform Root Mean Square Layer Normalization before performing multiplication of current tensor with normalization weights tensor, we call this normalization stage as "pre-normalization".

>In [this source](https://paperswithcode.com/method/rmsnorm), it writes:<br>
> RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) Forward(infContext *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling RMSNorm for currentTensor shape(%v) (result of all transformer blocks) and LlamaTransformer.output_norm weights shape(%v) -> tensor currentTensor", currentTensor.Size, lt.output_norm.weights.Size)
    if currentTensor, err = lt.output_norm.Forward(infContext, currentTensor); err != nil {
        return nil, err
    }
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling RMSNorm for currentTensor shape([32 4096]) (result of all transformer blocks) and LlamaTransformer.output_norm weights shape([4096]) -> tensor currentTensor ...
```

In [RMSNorm.Forward(...)](../src/model/llamatransformer.go) method, we call the ```RMSNorm.doNormalization(...)``` method, then perform an element-wise multiplication with ```LlamaTransformer.output_norm``` normalization weights tensor via [ml.MultiplyElementwise](../src/ml/operations_impl.go).

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (rms *RMSNorm) Forward(infContext *InferenceContext, x *ml.Tensor) (*ml.Tensor, error) {
    h, err := rms.doNormalization(x)
    if err != nil {
        return nil, err
    }
    return ml.MultiplyElementwise(h, rms.weights)
}
```

The ```RMSNorm.doNormalization(...)``` method consists of multiple steps:

* Take ```currentTensor``` which is the resulting tensor after running all 32 transformer block layers as input ```x``` tensor with shape of ```{32, 4096}```,
* Calculate square of each item in the ```x``` tensor via ```h, err = ml.Pow(x, 2)``` and assign it to ```h``` tensor,
* Calculate mean values of <u>last dimension</u>, <u>without removing the last dimension</u>: input shape was ```{32, 4096}```, output shape is ```{32, 1}```,
  >For further information, check out: [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html) documentation.
* Add scalar value of ```0.000001``` at [RMSNorm.epsilon](../src/model/llamatransformer.go) to each item in the ```h``` tensor via [ml.AddScalar](../src/ml/operations_impl.go). Because we have ```model.ModelArgs.NormEpsilon = 1e-06``` as read from "params.json" configuration file. Output shape is ```{32, 1}```,
  >The model configuration file "params.json" is parsed as [ModelArgs](../src/model/modelargs.go) object via JSON parser.
* Calculate reciprocal of the square-root of each item in the ```h``` tensor via [ml.RSqrt](../src/ml/operations_impl.go). Output shape is ```{32, 1}```,
  >For further information, check out: [torch.rsqrt](https://pytorch.org/docs/stable/generated/torch.rsqrt.html) documentation.<br>
  > The formula is:<br>

$$
out_i = \frac{1}{\sqrt{input_i}}
$$

* Perform an element-wise multiplication with ```x``` input tensor with shape of ```{32, 4096}``` and ```h``` normalization tensor with shape of ```{32, 1}``` via [ml.MultiplyElementwise](../src/ml/operations_impl.go). Output shape is ```{32, 4096}```,

Now, in ```RMSNorm.Forward(...)``` method, we have the ```h``` tensor with shape of ```{32, 4096}``` and ```RMSNorm.weights``` nromalization weights tensor with shape of ```{4096}```.

Then, we perform an element-wise multiplication with the ```h``` tensor with shape of ```{32, 4096}``` and ```RMSNorm.weights``` nromalization weights tensor with shape of ```{4096}```. Output shape is ```{32, 4096}```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (rms *RMSNorm) doNormalization(x *ml.Tensor) (*ml.Tensor, error) {
    var err error
    var h *ml.Tensor
    if h, err = ml.Pow(x, 2); err != nil {
        return nil, err
    }
    if h, err = ml.Mean(h, -1, true); err != nil {
        return nil, err
    }
    if h, err = ml.AddScalar(h, rms.epsilon); err != nil {
        return nil, err
    }
    if h, err = ml.RSqrt(h); err != nil {
        return nil, err
    }
    if h, err = ml.MultiplyElementwise(x, h); err != nil {
        return nil, err
    }
    return h, nil
}
```

Now, In [LlamaTransformer.Forward(...)](../src/model/llamatransformer.go) method, we have the pre-normalized via RMSNorm tensor ```currentTensor``` with shape of ```{32, 4096}```.

Sources:

* [RMSNorm (Root Mean Square Layer Normalization)](https://paperswithcode.com/method/rmsnorm)

## **16.2. Performing Linear Transformation with the Output Weights**

We've done pre-normalization over the ```currentTensor```.

We have the output weights tensor at ```lt.output``` already loaded to ```LlamaTransformer``` struct. In our case, shape of our output weights layer is ```{32000, 4096}```.

 Now we do matrix multiplication over ```currentTensor``` with shape of ```{32, 4096}``` and transpose of ```lt.output``` with shape of ```{32000, 4096}``` via [ml.LinearTransformation](../src/ml/operations_impl.go). Output shape is ```{32, 32000}```.

 >In our project, we have implemented two separate functions to perform matrix multiplication, because one is for direct matrix multiplication, other is for matrix multiplication with transpose of second argument (generally, a weights tensor). We've defined the first one as [ml.MatMul(...)](../src/ml/operations_impl.go) and the second one as [ml.LinearTransformation(...)](../src/ml/operations_impl.go).

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) Forward(infContext *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Calling ml.LinearTransformation for currentTensor (normalized result of all transformer blocks) shape(%v) and LlamaTransformer.output weights shape(%v) -> tensor output", currentTensor.Size, lt.output.Size)
    output, err := ml.LinearTransformation(currentTensor, lt.output)
    ...
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Calling ml.LinearTransformation for currentTensor (normalized result of all transformer blocks) shape([32 4096]) and LlamaTransformer.output weights shape([32000 4096]) -> tensor output ...
```

## **14.3. Converting the Output Tensor to Float32 Tensor and Returning It**

Now, we have the ```output``` tensor that contains probabilities of each alternative token in our vocabulary. In our case, at the first iteration, the shape of this tensor is ```{32, 32000}```. 32 stands for sequence length, 32,000 stands for the vocabulary size. This output tensor contains our ```logits```,  we perform the [Argmax](https://en.wikipedia.org/wiki/Arg_max) operation over this logits tensor. But here, we just convert our tensor items to float32, to make performing argmax easy via [Tensor.ToFloat32(...)](../src/ml/tensor.go) method.

How this ```output``` tensor is used was described in the chapter **"13.5.2. Looping through sequence length"** at [13. GENERATING NEXT TOKENS](./13-GENERATING-NEXT-TOKENS.md).

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func (lt *LlamaTransformer) Forward(infContext *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
    ...
    common.GLogger.DebugPrintf("Converting output tensor shape(%v) to Float32 tensor -> tensor output", output.Size)
    if output, err = output.ToFloat32(); err != nil {
        return nil, err
    }
    common.GLogger.DebugPrintf("Returning tensor output: shape(%v)", output.Size)
    return output, nil
}
```

We can see output lines in the "debug.log" file if debugging is enabled, as follows:

```sh
[DEBUG] ... Converting output tensor shape([32 32000]) to Float32 tensor -> tensor output ...
[DEBUG] ... Returning tensor output: shape([32 32000]) ...
```

With this step, we have finished all of the steps of ```LlamaTransformer.Forward(...)``` method and made prediction of the next token with probabilities. The journey continues in the loop at the chapter **"13.5.2. Looping through sequence length"** at [13. GENERATING NEXT TOKENS](./13-GENERATING-NEXT-TOKENS.md).

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: MAKING PREDICTION with LLAMA MODEL - 2](./15-MAKING-PREDICTION-WITH-LLAMA-MODEL-2.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: UNICODE, UTF-8 and EMOJIS&nbsp;&nbsp;&gt;](./17-UNICODE-UTF-8-EMOJIS.md)

</div>

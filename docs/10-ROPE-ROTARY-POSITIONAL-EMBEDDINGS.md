# **10. RoPE (ROTARY POSITIONAL EMBEDDINGS)**

The Llama model uses RoPE (Rotary Positional Embeddings) alongside the standard embedding layer to highlight the influence of token positions within a sequence.

For decades, *embedding* has been the most commonly used technique to represent words, concepts, or tokens in NLP (Natural Language Processing) world. Typically, an embedding model is trained to let them learn frequencies of use of tokens together. The tokens are placed in suitable positions within a multi-dimensional space, where distances reflect the difference or similarity between them.

There are a wide variety of different methods to implement embeddings. Some of them take token positions into account.

Taking the token positions into account is important. Think of, two sentences containing exactly the same words in different orders. If you don't take the positions into account, your system handles these two sentences (despite with different meanings) as the same thing.

## **10.1. Preliminary Concepts**

Let's try to explain with an example. Think of, we have a sequence of 5 tokens and an embedding layer with the shape of ```{128256, 4096}```. So, we have:

* A token embedding sequence which was calculated using the embedding layer. Our input tensor will be with the shape of ```{5, 4096}```,
* We have ```32``` "attention heads" (according to ```modelArgs.N_Heads = 32```). Our each attention head will have a dimension ```modelArgs.Dim / modelArgs.N_Heads```. In our case, it is ```4096 / 32 = 128```. So our positional embedding tensors will have ```128``` dimensions.
* We have an array of position indices of the tokens: ```{1, 2, 3, 4, 5}```.

We have several alternatives to calculate the positional embeddings. Some of them are:

* Taking the positions as they are: ```{0, 1, 2, 3, 4}```,
* Taking the positions in normalized values between ```0``` and ```1``` as ```{0., 0.25, 0.50, 0.75, 1.}```,
* As suggested in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), using sinusoidal functions. Each dimension of the positional encoding corresponds to a sinusoid:

    $$
    \begin{gathered}
        PE_(pos,2i) = \sin(pos/10000^\frac{2i}{d_{model}}) \\
        PE_(pos,2i+1) = \cos(pos/10000^\frac{2i}{d_{model}})
    \end{gathered}
    $$

    This means that, we have ```128``` dimensions for each position, ```i``` will loop from ```0``` to ```64``` (half of ```128```).
    
    The original paper suggests using ```10000``` as base theta value, and the Llama 2 model uses this value. But newer versions of Llama (3 and higher) started to use ```500000``` as base theta value, so, we will stick to using ```500000```.

    **<u>Update with Llama 3.1:</u>** The Llama 3.1 version comes with a small adjustment on frequencies. [apply_scaling(...)](https://github.com/meta-llama/llama-models/blob/5ee9cb5eaf92d542f1b1ee595af64a9ffdc07bac/models/llama3_1/api/model.py#L41) method was added into original Llama 3.1 implementation, that calculates wavelengths from these frequencies and applies some limitations on them. Implementation detail will be discussed in the following subchapters. Currently we represent this operation with $scl\left(...\right)$.

    Our ```PE``` positional embedding array for ```3th``` position will be like:

    $$
    \begin{gathered}
        PE = \left\lbrace
        \begin{array}{l}
        \sin\left(scl\left(\frac{3}{500000^\frac{0}{128}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^\frac{0}{128}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^\frac{2}{128}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^\frac{2}{128}}\right)\right), \\
        \dots, \\
        \sin\left(scl\left(\frac{3}{500000^\frac{124}{128}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^\frac{124}{128}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^\frac{126}{128}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^\frac{126}{128}}\right)\right)
        \end{array}
        \right\rbrace \\
    \end{gathered}
    $$

* As suggested in the paper [RoFormer](https://arxiv.org/abs/2104.09864v5), using sinusoidal functions in slightly different ways, as described following parts,
* Using the output of a custom function that takes the position indices as input.

We have several alternatives to integrate the positional embeddings with the token embedding vectors. If our positional embedding vectors and the token embedding vectors both have same dimension, we can sum or multiply them:

* Summation: As suggested in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) too, we can simply add two vectors as ```new_embeddings = token_embeddings + positional_embeddings```,
* Multiplication: As suggested in the paper [RoFormer](https://arxiv.org/abs/2104.09864v5) too, we can simply multiply element-wise two vectors as ```new_embeddings = token_embeddings * positional_embeddings```.

Think of, we can take the positions as they are, ```{0, 1, 2, 3, 4}```, and add them with each dimension of our embedding vectors. It may work, but sounds not so much meaningful, right? So, we need to find a more meaningful method.

You can find some introduction for *absolute position embedding* and *relative position embedding* in the paper [RoFormer](https://arxiv.org/abs/2104.09864v5) alongside the proposed approach *RoPE (Rotary Positional Embeddings)* by the same paper.

The main idea behind these approaches is to let the model effectively take into account token positions. *RoPE (Rotary Positional Embeddings)* approach represents the positions of tokens in the polar coordinate system, which employs angles and complex numbers.

With this approach, we have a chance to combine multiple approaches and we have the following advantages:

* To distribute our positional embeddings in the space in a limited range (```[-1, +1]``` because of limits of ```cos``` and ```sin``` functions),
* Other approaches prefer summation to integrate the positional embeddings with the token embedding vectors. However, the summation corrupts the exact data that the vector has. But in this approach, we prefer multiplication. And the value we multiply is a sinusoidal function, a sinusoid, we can think that we are only ***rotating*** the original embedding vector by an angle. So, theoretically, we don't corrupt the original data,
* In the approach of [RoFormer](https://arxiv.org/abs/2104.09864v5) that we used in this project, takes the items of ```128``` dimension of an attention head as ```64``` pairs. Then, it obtains a complex number from each pair by taking the first item of the pair as *real part* and the second item of the pair as *imaginary part* of a complex number.

$$
\begin{gathered}
    \text{where }i^2=-1, i\text{ is the imaginary unit,} \\
    out = abs \cdot \cos(angle) + abs \cdot \sin(angle) \cdot i
\end{gathered}
$$

* Taking the items of an attention head as float pairs and representing them as complex numbers in polar coordinate system makes our method more suitable for its mathematical nature. Also, this allows it to represent these complex numbers as a matrix, which allows us to perform matrix operations.
* With this approach, the influence of position on embeddings is high for lower dimensions (going from the first dimension throughout 128 dimensions) and converges to zero for higher dimensions. This makes higher dimensions of embeddings less sensitive to positional data than lower dimensions. Because the polar coordinates calculated for higher dimensions are nearly the same value.
    >**Important note:** I read it in a few sources then I saw it with the 3D charts that I drew in the notebook [10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb](./10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb), also can be found the bottom of this chapter.

>Note: In this approach, some concepts from computing, mathematics, geometry, and physics were combined. For example, we can think that:
>
> * Our positions are points in a time series,
> * Our positional encoding function as *a function with respect to time*, so we can say it is a signal,
> * Our angles as *angular frequency* (${\displaystyle \omega }$) and our positions as *real independent variable/time* of a sine wave.

In the following subchapters, we will see how these polar coordinates are precalculated. These values don't vary by input, so this calculation is made only once.

Sources:

* [A Guide on Word Embeddings in NLP](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp)
* [Word Embeddings in Natural Language Processing(NLP)](https://www.theaidream.com/post/word-embeddings-in-natural-language-processing-nlp)
* [RoFormer: Enhanced Transformer with Rotary Position Embedding Paper](https://arxiv.org/abs/2104.09864v5)
* [Youtube - RoPE (Rotary positional embeddings) explained: The positional workhorse of modern LLMs](https://www.youtube.com/watch?v=GQPOtyITy54)
* [Youtube - Llama explained... - "Rotary Positional Embeddings" section](https://www.youtube.com/watch?v=Mn_9W1nCFLo&t=1471s)

## **10.2. Introduction to Precomputing The Frequency Tensor**

>You can check out the following for more information:
>
> * The Python codes that create the sample data and graphs used here with this Python Notebook: [10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb](./10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb).
> * [A Gentle Introduction to Positional Encoding in Transformer Models, Part 1](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/) article.

The [precomputeFreqsCis(...)](../src/model/llamatransformer.go) function takes four arguments: ```dim```, ```end```, ```theta```, and ```useScaled```.

The argument ```dim``` is computed as ```int(modelArgs.Dim/modelArgs.N_Heads)```, ```end``` is computed as ```modelArgs.MaxSequenceLength*2```, ```theta``` is ```modelArgs.RopeTheta```, ```useScaled``` is ```modelArgs.UseScaledRope```. In our case, ```dim = 4096/32 = 128```,  ```end = 2048 * 2 = 4096```, ```theta = 500000```, ```useScaled = true```.

In our case, the [precomputeFreqsCis(...)](../src/model/llamatransformer.go) function is called with ```dim = 128```, ```end = 4096```, ```theta = 500000```, and ```useScaled = true```.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
    result := &LlamaTransformer{}
    ...
    if result.PrecomputedFreqsCis, err = precomputeFreqsCis(int(dim/modelArgs.N_Heads), modelArgs.MaxSequenceLength*2, modelArgs.RopeTheta, modelArgs.UseScaledRope); err != nil {
        return nil, err
    }
    ...
}
```

## **10.3. Initiating Angles of Frequency Tensor**

In the [original Llama 3.1 Python repository of Meta](https://github.com/meta-llama/llama-models/blob/5ee9cb5eaf92d542f1b1ee595af64a9ffdc07bac/models/llama3_1/api/model.py#L66), this Python code initiates the ```freqs``` array:

```py
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    ...
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    ...
```

>**Important note:** The ```theta``` variables in both Go and Python code are not an angle. They are explained as: *"Scaling factor for frequency computation. Defaults to 10000.0, but in our case, this value comes as 500000.0 for Llama 3.1."*.<br>
>Instead, the ```freqs``` is an array of angles, that corresponds to $\Theta$ and each item of ```freqs``` array corresponds to $\theta_i$ below.<br>
>Personally, at first sight, I was confused why they called *scaling factor* as ```theta``` which is a term that made me think *it is an angle*, but it isn't, items of the ```freqs``` are in an angle unit (radians), but at the end, they are only premise value (scaling factor) for the real angles!

The original equation in section "3.2.2 General form" of [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864):

$$
\begin{gathered}
\Theta = \left\lbrace \theta_i = 500000^{-\frac{2(i - 1)}{dim}}, i \in [1, 2, \dots, \frac{dim}{2}] \right\rbrace
\end{gathered}
$$

If we expand it for ```dim=128``` in our case:

$$
\begin{gathered}
\Theta = \left\lbrace
  \theta_1 = 500000^{-\frac{2(1 - 1)}{128}},
  \theta_2 = 500000^{-\frac{2(2 - 1)}{128}},
  \theta_3 = 500000^{-\frac{2(3 - 1)}{128}},
  \dots,
  \theta_{63} = 500000^{-\frac{2(63 - 1)}{128}},
  \theta_{64} = 500000^{-\frac{2(64 - 1)}{128}}
\right\rbrace \\
= \\
\Theta = \left\lbrace
  \theta_1 = 500000^{-\frac{0}{128}},
  \theta_2 = 500000^{-\frac{2}{128}},
  \theta_3 = 500000^{-\frac{4}{128}},
  \dots,
  \theta_{63} = 500000^{-\frac{124}{128}},
  \theta_{64} = 500000^{-\frac{126}{128}}
\right\rbrace \\
= \\
\Theta = \left\lbrace
  \theta_1 = \frac{1}{500000^{\frac{0}{128}}},
  \theta_2 = \frac{1}{500000^{\frac{2}{128}}},
  \theta_3 = \frac{1}{500000^{\frac{4}{128}}},
  \dots,
  \theta_{63} = \frac{1}{500000^{\frac{124}{128}}},
  \theta_{64} = \frac{1}{500000^{\frac{126}{128}}}
\right\rbrace
\end{gathered}
$$

**<u>Update with Llama 3.1:</u>** The Llama 3.1 version comes with a small adjustment on frequencies. Implementation detail will be discussed in the following subchapters. Currently we represent this operation with $scl\left(...\right)$.

If it will be expressed with variable names in the code and scaling is applied:

$$
\begin{gathered}
freqs =
\left\lbrace
    \text{item}_{i} = scl\left(\frac{1}{theta^{\frac{val}{dim}}}\right)
\right\rbrace
, val \in \lbrack0, 2, 4, ...,dim - 2\rbrack
, i \in \left[0, 1, 2, ..., \frac{dim}{2} -1\right] \\
\\
\\
\text{freqs} =
\left\lbrace
    \text{item}_0 = scl\left(\frac{1}{500000^{\frac{0}{128}}}\right),
    \text{item}_1 = scl\left(\frac{1}{500000^{\frac{2}{128}}}\right),
    \text{item}_2 = scl\left(\frac{1}{500000^{\frac{4}{128}}}\right),
    \dots,
    \text{item} _{62} = scl\left(\frac{1}{500000^{\frac{124}{128}}}\right),
    \text{item} _{63} = scl\left(\frac{1}{500000^{\frac{126}{128}}}\right)
\right\rbrace
\end{gathered}
$$

You can find original Python implementation of [apply_scaling(...)](https://github.com/meta-llama/llama-models/blob/6214a21dc837ce63983ef3fd7b172a6ed16e4905/models/llama3_1/api/model.py#L41) function which is represented as $scl\left(...\right)$ here.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func applyScaling(freqs *ml.Tensor) error {
	// See Llama 3.1 Code: https://github.com/meta-llama/llama-models/blob/6214a21dc837ce63983ef3fd7b172a6ed16e4905/models/llama3_1/api/model.py#L41
	// Values obtained from grid search
	scaleFactor := float32(8.0)
	lowFreqFactor := float32(1.0)
	highFreqFactor := float32(4.0)
	oldContextLen := float32(8192) // original llama3 length
	lowFreqWavelen := oldContextLen / lowFreqFactor
	highFreqWavelen := oldContextLen / highFreqFactor
	for i := 0; i < freqs.Size[0]; i++ {
		freq, err := freqs.GetItem_AsFloat32([]int{i})
		if err != nil {
			return err
		}
		var newFreq float32
		wavelen := 2 * math.Pi / freq
		if wavelen < highFreqWavelen {
			newFreq = freq
		} else if wavelen > lowFreqWavelen {
			newFreq = freq / scaleFactor
		} else {
			smooth := (oldContextLen/wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
			newFreq = (1-smooth)*freq/scaleFactor + smooth*freq

		}
		if err := freqs.SetItem_FromFloat32([]int{i}, newFreq); err != nil {
			return err
		}
	}
	return nil
}

func precomputeFreqsCis(dim int, end int, theta float64, useScaled bool) (*ml.Tensor, error) {
    ...
    dimFloat := float32(dim)
    freqs, err := ml.ARange(0, dim, 2, ml.DT_BF16)
    ...
    err = freqs.Apply_AsFloat32(func(val float32) float32 {
        return float32(1.0 / math.Pow(theta, float64(val/dimFloat)))
    })
    ...
	if useScaled {
		err = applyScaling(freqs)
		if err != nil {
			return nil, err
		}
	}
    ...
}
```

Go project (ours) values of the ```freqs``` array:

```yaml
freqs: {0, 2, 4, 6, 8, 10, 12, ..., 124, 126} # has 64 items, at this time, the freqs array contains "val" values that exists in the equations above.
# after running Apply_AsFloat32 and then applyScaling, freqs will be:
freqs: { # has 64 items, in radians.
    1.0000e+00, 8.1250e-01, 6.6016e-01, 5.3906e-01, 4.3945e-01, 3.5742e-01,
    2.9102e-01, 2.3730e-01, 1.9336e-01, 1.5723e-01, 1.2793e-01, 1.0449e-01,
    8.4961e-02, 6.9336e-02, 5.6641e-02, 4.6143e-02, 3.7598e-02, 3.0518e-02,
    2.4902e-02, 2.0264e-02, 1.6479e-02, 1.3489e-02, 1.0986e-02, 8.9111e-03,
    7.2632e-03, 5.9204e-03, 4.8218e-03, 3.9368e-03, 3.2043e-03, 2.1515e-03,
    1.3504e-03, 8.5068e-04, 5.1880e-04, 3.1090e-04, 1.7834e-04, 9.5367e-05,
    7.7724e-05, 6.2943e-05, 5.1498e-05, 4.1962e-05, 3.4094e-05, 2.7895e-05,
    2.2650e-05, 1.8477e-05, 1.5080e-05, 1.2279e-05, 1.0014e-05, 8.1062e-06,
    6.6459e-06, 5.3942e-06, 4.4107e-06, 3.5912e-06, 2.9206e-06, 2.3842e-06,
    1.9372e-06, 1.5795e-06, 1.2890e-06, 1.0431e-06, 8.5309e-07, 6.9663e-07,
    5.6624e-07, 4.6194e-07, 3.7625e-07, 3.0547e-07
}
```

This is the output of Python + Pytorch environment. There are slight differences between them because of floating point precision differences.

In the table below, you can find approximate equivalents of radian angle values in degrees, with corresponding "val" indices:

| **val** | **rad**       | **deg**   |     | **val** | **rad**       | **deg**   |     | **val** | **rad**       | **deg**   |     | **val** | **rad**       | **deg**   |
|---------|---------------|-----------|-----|---------|---------------|-----------|-----|---------|---------------|-----------|-----|---------|---------------|-----------|
| **0**   | 1.00000000    | 57.29578  |     | **32**  | 0.03760603    | 2.15467   |     | **64**  | 0.00052485    | 0.03007   |     | **96**  | 0.00000665    | 0.00038   |
| **2**   | 0.81461722    | 46.67413  |     | **34**  | 0.03063452    | 1.75523   |     | **66**  | 0.00031269    | 0.01792   |     | **98**  | 0.00000542    | 0.00031   |
| **4**   | 0.66360128    | 38.02155  |     | **36**  | 0.02495541    | 1.42984   |     | **68**  | 0.00017851    | 0.01023   |     | **100** | 0.00000441    | 0.00025   |
| **6**   | 0.54058099    | 30.97301  |     | **38**  | 0.02032910    | 1.16477   |     | **70**  | 0.00009556    | 0.00548   |     | **102** | 0.00000359    | 0.00021   |
| **8**   | 0.44036663    | 25.23115  |     | **40**  | 0.01656044    | 0.94884   |     | **72**  | 0.00007785    | 0.00446   |     | **104** | 0.00000293    | 0.00017   |
| **10**  | 0.35873023    | 20.55373  |     | **42**  | 0.01349042    | 0.77294   |     | **74**  | 0.00006342    | 0.00363   |     | **106** | 0.00000238    | 0.00014   |
| **12**  | 0.29222783    | 16.74342  |     | **44**  | 0.01098953    | 0.62965   |     | **76**  | 0.00005166    | 0.00296   |     | **108** | 0.00000194    | 0.00011   |
| **14**  | 0.23805381    | 13.63948  |     | **46**  | 0.00895226    | 0.51293   |     | **78**  | 0.00004208    | 0.00241   |     | **110** | 0.00000158    | 0.00009   |
| **16**  | 0.19392276    | 11.11096  |     | **48**  | 0.00729267    | 0.41784   |     | **80**  | 0.00003428    | 0.00196   |     | **112** | 0.00000129    | 0.00007   |
| **18**  | 0.15797281    | 9.05118   |     | **50**  | 0.00594073    | 0.34038   |     | **82**  | 0.00002793    | 0.00160   |     | **114** | 0.00000105    | 0.00006   |
| **20**  | 0.12868738    | 7.37324   |     | **52**  | 0.00483942    | 0.27728   |     | **84**  | 0.00002275    | 0.00130   |     | **116** | 0.00000086    | 0.00005   |
| **22**  | 0.10483095    | 6.00637   |     | **54**  | 0.00394228    | 0.22588   |     | **86**  | 0.00001853    | 0.00106   |     | **118** | 0.00000070    | 0.00004   |
| **24**  | 0.08539710    | 4.89289   |     | **56**  | 0.00321145    | 0.18400   |     | **88**  | 0.00001510    | 0.00086   |     | **120** | 0.00000057    | 0.00003   |
| **26**  | 0.06956595    | 3.98584   |     | **58**  | 0.00216657    | 0.12414   |     | **90**  | 0.00001230    | 0.00070   |     | **122** | 0.00000046    | 0.00003   |
| **28**  | 0.05666962    | 3.24693   |     | **60**  | 0.00137189    | 0.07860   |     | **92**  | 0.00001002    | 0.00057   |     | **124** | 0.00000038    | 0.00002   |
| **30**  | 0.04616405    | 2.64501   |     | **62**  | 0.00085675    | 0.04909   |     | **94**  | 0.00000816    | 0.00047   |     | **126** | 0.00000031    | 0.00002   |

Sources:

* RoFormer: Enhanced Transformer with Rotary Position Embedding: [Paper](https://arxiv.org/abs/2104.09864v5) | [Papers with Code](https://paperswithcode.com/paper/roformer-enhanced-transformer-with-rotary) | [LabML Annotated Implementation](https://nn.labml.ai/transformers/rope/index.html)
* Llama 2: Open Foundation and Fine-Tuned Chat Models
: [Paper](https://arxiv.org/abs/2307.09288)
* Llama: Open and Efficient Foundation Language Models
: [Paper](https://arxiv.org/abs/2302.13971v1)

## **10.4. Getting Outer Product of Frequency Tensor and Position Indices**

```py
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    ...
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    ...
```

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func precomputeFreqsCis(dim int, end int) (*ml.Tensor, error) {
    ...
    t, err := ml.ARange(0, end, 1, ml.DT_BF16)
    if err != nil {
        return nil, err
    }
    ...
}
```

The ```end``` is computed as ```modelArgs.MaxSequenceLength*2```. In our case, ```end = 2048 * 2 = 4096```.

> :warning: Note for weirdnes here: In original implementation, ```modelArgs.MaxSequenceLength``` value is equal to ```512``` which is limitation for input prompt token count. With multiplying to 2, they've aimed to avoid of unnecessary calculations.<br>
> However, in our implementation, we specified ```modelArgs.MaxSequenceLength``` as ```2048```, and when we multiply it with 2, we get ```4096```, an unnecessary and unmeaningful value. But I left it as it is, it doesn't hurt correctness, it causes only calculating unused unnecessary values.<br>
> We will continue with 4096, but know that, it is unnecessarily high.<br>
> On this issue, the original Python code has a comment (read this with considering this comment was taken from Llama 2 code, not Llama 3.1):<br>

```py
# Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
# Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.```
```

At first, we generate a tensor named ```t``` with 4096 items as: ```{0, 1, 2, 3, ..., 4093, 4094, 4095}```. This tensor contains our position indices.

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func precomputeFreqsCis(dim int, end int) (*ml.Tensor, error) {
    ...
    freqs, err = ml.Outer(t, freqs)
    if err != nil {
        return nil, err
    }
    ...
}
```

By calling [ml.Outer(t, freqs)](../src/ml/operations_impl.go) function, a tensor with shape ```{4096, 64}``` which is outer product of tensors ```t with shape {4096}```and ```freqs with shape {64}```.

This "outer product" function takes first argument as row vectors, second argument as column vectors.

In our case, we take items of ```t``` as rows, items of ```freqs``` as columns, then create 2D tensor called ```result``` as follows:

| row                  | column                        | set the result as                                       | python equivalent (rad) | python equivalent (deg)           |
|:---------------------|:------------------------------|:--------------------------------------------------------|-------------------------|-----------------------------------|
| ```t[0] = 0```       | ```freqs[0] = 1.0000e+00```   | ```result[0][0] = 0 * 1.0000e+00 = 0```                 | 0.00000                 | 0.00000                           |
|                      | ```freqs[1] = 8.1250e-01```   | ```result[0][1] = 0 * 8.1250e-01 = 0```                 | 0.00000                 | 0.00000                           |
| ...                  |                               |                                                         |                         |                                   |
| ```t[1] = 1```       | ```freqs[0] = 1.0000e+00```   | ```result[1][0] = 1 * 1.0000e+00 = 1.0000e+00```        | 1.00000000              | 57.29578                          |
|                      |  ```freqs[1] = 8.1250e-01```  | ```result[1][1] = 1 * 8.1250e-01 = 8.1250e-01```        | 0.81461722                 | 46.67413                            |
|                      | ...                           |                                                         |                         |                                   |
|                      | ```freqs[62] = 3.7625e-07```  | ```result[1][62] = 1 * 3.7625e-07 = 3.7625e-07```       | 0.00000038                 | 0.00002                             |
|                      | ```freqs[63] = 3.0547e-07```  | ```result[1][63] = 1 * 3.0547e-07 = 3.0547e-07```       | 0.00000031                 | 0.00002                             |
| ...                  |                               |                                                         |                         |                                   |
| ```t[2] = 2```       | ```freqs[0] = 1.0000e+00```   | ```result[2][0] = 2 * 1.0000e+00 = 2.0000e+00```        | 2.00000000                 | 114.59155                           |
|                      | ```freqs[1] = 8.1250e-01```   | ```result[2][1] = 2 * 8.1250e-01 = 1.6250e+00```        | 1.62923443                 | 93.34825                            |
|                      | ...                           |                                                         |                         |                                   |
|                      | ```freqs[62] = 3.7625e-07```  | ```result[2][62] = 2 * 3.7625e-07 = 7.5251e-07```       | 0.00000075                 | 0.00004                             |
|                      | ```freqs[63] = 3.0547e-07```  | ```result[2][63] = 2 * 3.0547e-07 = 6.1095e-07```       | 0.00000061                 | 0.00004                             |
| ...                  |                               |                                                         |                         |                                   |
| ```t[4094] = 8190``` | ```freqs[0] = 1.0000e+00```   | ```result[4094][0] = 4094 * 1.0000e+00 = 4.0800e+03```  | 4094.00000000              | 234568.90625 (normalized: -151.09375)  |
|                      |  ```freqs[1] = 8.1250e-01```  | ```result[4094][1] = 4094 * 8.1250e-01 = 3.3120e+03```  | 3335.04296875              | 191083.87500 (normalized: -76.12500)  |
|                      | ...                           |                                                         |                         |                                   |
|                      | ```freqs[62] = 3.7625e-07```  | ```result[4094][62] = 4094 * 3.7625e-07 = 1.5335e-03``` | 0.00154234                 | 0.08837                            |
|                      |  ```freqs[63] = 3.0547e-07``` | ```result[4094][63] = 4094 * 3.0547e-07 = 1.2436e-03``` | 0.00125642                 | 0.07199                            |
| ...                  |                               |                                                         |                         |                                   |
| ```t[4095] = 4095``` |  ```freqs[0] = 1.0000e+00```  | ```result[4095][0] = 4095 * 1.0000e+00 = 4.0800e+03```  | 4095.00000000              | 234626.20312 (normalized: -93.79688) |
|                      | ```freqs[1] = 8.1250e-01```   | ```result[4095][1] = 4095 * 8.1250e-01 = 3.3120e+03```  | 3335.85742188              | 191130.54688 (normalized: -29.45312)  |
|                      | ...                           |                                                         |                         |                                   |
|                      | ```freqs[62] = 3.7625e-07```  | ```result[4095][62] = 4095 * 3.7625e-07 = 1.5335e-03``` | 0.00154272                 | 0.00154272                            |
|                      | ```freqs[63] = 3.0547e-07```  | ```result[4095][63] = 4095 * 3.0547e-07 = 1.2436e-03``` | 0.00125673                 | 0.07201                            |
|                      | ...                           |                                                         |                         |                                   |

<sup>from [src/ml/operations_impl.go](../src/ml/operations_impl.go)</sup>

```go
func Outer(vec1 *Tensor, vec2 *Tensor) (*Tensor, error) {
    if err := processErrors(
        checkIsVector(vec1),
        checkIsVector(vec2),
        checkSameDataType(vec1, vec2),
    ); err != nil {
        return nil, err
    }
    itemSize := vec1.DataType.ItemSize
    result := NewEmptyTensor([]int{vec1.Size[0], vec2.Size[0]}, vec1.DataType)
    for i := 0; i < vec1.Size[0]; i++ {
        rowValF32, err := vec1.GetItemByOffset_AsFloat32(i * itemSize)
        if err != nil {
            return nil, err
        }
        for j := 0; j < vec2.Size[0]; j++ {
            colValF32, err := vec2.GetItemByOffset_AsFloat32(j * itemSize)
            if err != nil {
                return nil, err
            }
            valF32 := rowValF32 * colValF32
            if err := result.SetItem_FromFloat32([]int{i, j}, valF32); err != nil {
                return nil, err
            }
        }
    }
    return result, nil
}
```

## **10.5. Calculating Frequency Tensor as Cis (Polar Coordinates)**

[cis](https://en.wikipedia.org/wiki/Cis_%28mathematics%29) is described at Wikipedia:

>cis is a mathematical notation defined by cis x = cos x + i sin x,[nb 1] where cos is the cosine function, i is the imaginary unit and sin is the sine function. x is the argument of the complex number (angle between line to point and x-axis in polar form).

$$
\begin{gathered}
\text{where }i^2=-1, i\text{ is the imaginary unit,} \\
cis x = cos x + i \cdot sin x
\end{gathered}
$$

With this notation, we can express a point's location in cartesian coordinate system with cosine and sine of one angle, which is called as [polar coordinates](https://en.wikipedia.org/wiki/Polar_coordinate_system).

We've calculated angles of our polar coordinate points as ```freqs``` in previous chapter.

```py
import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0):
    ...
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
```

<sup>from [src/model/llamatransformer.go](../src/model/llamatransformer.go)</sup>

```go
func precomputeFreqsCis(dim int, end int) (*ml.Tensor, error) {
    ...
    ones, err := ml.OnesLike(freqs)
    if err != nil {
        return nil, err
    }
    freqs_cis, err := ml.Polar(ones, freqs)
    if err != nil {
        return nil, err
    }
    return freqs_cis, nil
}
```

We create a tensor which contains all ```1``` values with same shape ```{4096, 64}``` and data type as ```freqs``` tensor, via [ml.OnesLike(...))](../src/ml/operations_impl.go). These  ```1``` values will be the magnitude of our vector in polar coordinate system. We use 1 for magnitude to get identity vector for the angle.

By calling [ml.Polar(ones, freqs)](../src/ml/operations_impl.go) function, a tensor with shape ```{4096, 64}``` which is outer product of tensors ```t with shape {4096}```and ```freqs with shape {64}``` is got.

[Polar function](https://pytorch.org/docs/stable/generated/torch.polar.html) is described at Pytorch TORCH.POLAR documentation:

>Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value abs and angle.

$$
\begin{gathered}
\text{where }i^2=-1, i\text{ is the imaginary unit,} \\
out = abs \cdot \cos(angle) + abs \cdot \sin(angle) \cdot i
\end{gathered}
$$

In our case, the ```abs``` argument is a tensor full of ```1``` values. ```angle``` argument is our ```freqs``` variable.

For each item in the ```angle``` tensor:

* The cosine value of the angle is got, multiplied by ```absItemF64 = 1``` (it is always 1 in our case), set as the *real* part of the resulting complex number,
* The sine value of the angle is got, multiplied by ```absItemF64 = 1``` (it is always 1 in our case), set as the *imaginary* part of the resulting complex number.

Then, the ```dst``` tensor with ```DT_COMPLEX``` data type has cosine and sine values of our angles as complex numbers.

<sup>from [src/ml/operations_impl.go](../src/ml/operations_impl.go)</sup>

```go
func Polar(abs *Tensor, angle *Tensor) (*Tensor, error) {
    // See: (For formula) https://pytorch.org/docs/stable/generated/torch.polar.html
    ...
    for readOffset := 0; readOffset < abs.GetBytesCount(); readOffset += absItemSize {
        absItemF32, err := abs.GetItemByOffset_AsFloat32(readOffset)
        if err != nil {
            return nil, err
        }
        angleItemF32, err := angle.GetItemByOffset_AsFloat32(readOffset)
        if err != nil {
            return nil, err
        }

        absItemF64 := float64(absItemF32)
        angleItemF64 := float64(angleItemF32)

        realPart := absItemF64 * math.Cos(angleItemF64)
        imagPart := absItemF64 * math.Sin(angleItemF64)
        resultItem := complex64(complex(realPart, imagPart))
        if err := dst.SetItemByOffset(writeOffset, resultItem); err != nil {
            return nil, err
        }
        writeOffset += dstItemSize
    }
    return dst, nil
}
```

In our case, with ```1``` is always as ```absItemF64```:

| angleItemF64                       | realPart                           | imagPart                          | resultItem                |
|:-----------------------------------|:-----------------------------------|:----------------------------------|:--------------------------|
| ```freqs[0][0] = 0```              | ```cos(0) = 1```                   | ```sin(0) = 0```                  | (1 + 0i)                  |
| ```freqs[0][1] = 0```              | ```cos(0) = 1```                   | ```sin(0) = 0```                  | (1 + 0i)                  |
| ...                                |                                    |                                   |                           |
| ```freqs[1][0] = 1```              | ```cos(1) = 0.5403023```           | ```sin(1) = 0.84147096```         | (0.5403023 + 0.84147096i) |
| ...                                |                                    |                                   |                           |
| ```freqs[4095][63] = 0.0012436``` | ```cos(0.0012436) = 1``` | ```sin(0.0012436) = 0.0012436``` | (1 + 0.0012436i) |

## **10.6. The result**

Recap of what we have so far:

* An embedding layer with the shape of ```{128256, 4096}```, that contains vectors that have ```4096``` dimensions each, ```128256``` different token vectors.
* A token embedding sequence which was calculated using the embedding layer. Our input tensor will be with the shape of ```{SequenceLength, 4096}```,
* We have ```32``` "attention heads" (according to ```modelArgs.N_Heads = 32```). Our each attention head will have a dimension ```modelArgs.Dim / modelArgs.N_Heads```. In our case, it is ```4096 / 32 = 128```. So our positional embedding tensors will have ```128``` dimensions.
* Because of we have ```32``` attention heads and the dimesion of our each attention head is ```128```, we will separate our ```xq (queries)``` matrix into 32 equal pieces, then because of we have ```8``` key/value heads, we will separate our ```xk (keys)```matrix into 8 equal pieces, that end up with ```128``` at one dimension. Then, the integration with positional embeddings and the token embeddings is done on ```128``` dimension.
* Think of, we have 5 tokens to encode, so we have position indices of the tokens: ```{1, 2, 3, 4, 5}```.

After all of these processes, we will end up with a "positional encoding tensor" for a sequence having ```5``` positions as follows:

>Note: The $\LaTeX$ support of Github web app lacks and gives non-explanatory errors when you have more than a limit of superscipts/subscripts/fraction notations. So, it was a must to separate the biggest set notation into chunks.

The operation of ```applyScaling(...)``` function is represented with $scl\left(...\right)$.

$$
PE = \left\lbrace
\begin{array}{l}
p_{pos,2i} = \sin\left(pos \cdot scl\left(\frac{1}{500000^\frac{2i}{d_{model}}}\right)\right) \\
\\
p_{pos,2i+1} = \cos\left(pos \cdot scl\left(\frac{1}{500000^\frac{2i}{d_{model}}}\right)\right) \\
\end{array}
\right\rbrace
$$

* Positional Encoding tensor for 5 positions, without converting to complex number:

$$
\begin{gathered}
\text{where }PE_{pos,i} \in \mathbb{R}^{dim = 128}, \\
\\
PE = \left\lbrack
\begin{array}{l}
    PE_{pos0}, PE_{pos1}, PE_{pos2}, PE_{pos3}, PE_{pos4}
\end{array}
\right\rbrack
\end{gathered}
$$

<br>

$$
\begin{gathered}
    PE_{pos0}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{0}{500000^{\frac{0}{128}}}\right)\right),
        \cos\left(scl\left(\frac{0}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{0}{500000^{\frac{2}{128}}}\right)\right),
        \cos\left(scl\left(\frac{0}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{0}{500000^{\frac{4}{128}}}\right)\right),
        \cos\left(scl\left(\frac{0}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{0}{500000^{\frac{124}{128}}}\right)\right),
        \cos\left(scl\left(\frac{0}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{0}{500000^{\frac{126}{128}}}\right)\right),
        \cos\left(scl\left(\frac{0}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered}
$$

$$
\begin{gathered}
    PE_{pos1}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{1}{500000^{\frac{0}{128}}}\right)\right),
        \cos\left(scl\left(\frac{1}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{1}{500000^{\frac{2}{128}}}\right)\right),
        \cos\left(scl\left(\frac{1}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{1}{500000^{\frac{4}{128}}}\right)\right),
        \cos\left(scl\left(\frac{1}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{1}{500000^{\frac{124}{128}}}\right)\right),
        \cos\left(scl\left(\frac{1}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{1}{500000^{\frac{126}{128}}}\right)\right),
        \cos\left(scl\left(\frac{1}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered}
$$

$$
\begin{gathered}
    PE_{pos2}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{2}{500000^{\frac{0}{128}}}\right)\right),
        \cos\left(scl\left(\frac{2}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{2}{500000^{\frac{2}{128}}}\right)\right),
        \cos\left(scl\left(\frac{2}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{2}{500000^{\frac{4}{128}}}\right)\right),
        \cos\left(scl\left(\frac{2}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{2}{500000^{\frac{124}{128}}}\right)\right),
        \cos\left(scl\left(\frac{2}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{2}{500000^{\frac{126}{128}}}\right)\right),
        \cos\left(scl\left(\frac{2}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered}
$$

$$
\begin{gathered}
    PE_{pos3}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{3}{500000^{\frac{0}{128}}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^{\frac{2}{128}}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^{\frac{4}{128}}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{3}{500000^{\frac{124}{128}}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^{\frac{126}{128}}}\right)\right),
        \cos\left(scl\left(\frac{3}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered}
$$

$$
\begin{gathered}
    PE_{pos4}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{4}{500000^{\frac{0}{128}}}\right)\right),
        \cos\left(scl\left(\frac{4}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{4}{500000^{\frac{2}{128}}}\right)\right),
        \cos\left(scl\left(\frac{4}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{4}{500000^{\frac{4}{128}}}\right)\right),
        \cos\left(scl\left(\frac{4}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{4}{500000^{\frac{124}{128}}}\right)\right),
        \cos\left(scl\left(\frac{4}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{4}{500000^{\frac{126}{128}}}\right)\right),
        \cos\left(scl\left(\frac{4}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered}
$$

<br><br>

* Positional Encoding tensor for 5 positions, after converting to complex number:

$$
\begin{gathered}
\text{where }PE_{pos,i} \in \mathbb{C}^{\frac{dim}{2} = 64}, i^2=-1, i\text{ is the imaginary unit,}
\\
\\
PE = \left\lbrack
\begin{array}{l}
    PE_{pos0}, PE_{pos1}, PE_{pos2}, PE_{pos3}, PE_{pos4}
\end{array}
\right\rbrack
\end{gathered}
$$

<br>

$$
\begin{gathered}
    PE_{pos0}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{0}{500000^{\frac{0}{128}}}\right)\right) + i \cos\left(scl\left(\frac{0}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{0}{500000^{\frac{2}{128}}}\right)\right) + i \cos\left(scl\left(\frac{0}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{0}{500000^{\frac{4}{128}}}\right)\right) + i \cos\left(scl\left(\frac{0}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{0}{500000^{\frac{124}{128}}}\right)\right) + i \cos\left(scl\left(\frac{0}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{0}{500000^{\frac{126}{128}}}\right)\right) + i \cos\left(scl\left(\frac{0}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered} \\\\
$$

$$
\begin{gathered}
    PE_{pos1}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{1}{500000^{\frac{0}{128}}}\right)\right) + i \cos\left(scl\left(\frac{1}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{1}{500000^{\frac{2}{128}}}\right)\right) + i \cos\left(scl\left(\frac{1}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{1}{500000^{\frac{4}{128}}}\right)\right) + i \cos\left(scl\left(\frac{1}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{1}{500000^{\frac{124}{128}}}\right)\right) + i \cos\left(scl\left(\frac{1}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{1}{500000^{\frac{126}{128}}}\right)\right) + i \cos\left(scl\left(\frac{1}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\\\
\end{gathered} \\
$$

$$
\begin{gathered}
    PE_{pos2}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{2}{500000^{\frac{0}{128}}}\right)\right) + i \cos\left(scl\left(\frac{2}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{2}{500000^{\frac{2}{128}}}\right)\right) + i \cos\left(scl\left(\frac{2}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{2}{500000^{\frac{4}{128}}}\right)\right) + i \cos\left(scl\left(\frac{2}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{2}{500000^{\frac{124}{128}}}\right)\right) + i \cos\left(scl\left(\frac{2}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{2}{500000^{\frac{126}{128}}}\right)\right) + i \cos\left(scl\left(\frac{2}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\\\
\end{gathered} \\
$$

$$
\begin{gathered}
    PE_{pos3}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{3}{500000^{\frac{0}{128}}}\right)\right) + i \cos\left(scl\left(\frac{3}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^{\frac{2}{128}}}\right)\right) + i \cos\left(scl\left(\frac{3}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^{\frac{4}{128}}}\right)\right) + i \cos\left(scl\left(\frac{3}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{3}{500000^{\frac{124}{128}}}\right)\right) + i \cos\left(scl\left(\frac{3}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{3}{500000^{\frac{126}{128}}}\right)\right) + i \cos\left(scl\left(\frac{3}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered} \\\\
$$

$$
\begin{gathered}
    PE_{pos4}=
    \left\lbrace
    \begin{array}{l}
        \sin\left(scl\left(\frac{4}{500000^{\frac{0}{128}}}\right)\right) + i \cos\left(scl\left(\frac{4}{500000^{\frac{0}{128}}}\right)\right),
        \sin\left(scl\left(\frac{4}{500000^{\frac{2}{128}}}\right)\right) + i \cos\left(scl\left(\frac{4}{500000^{\frac{2}{128}}}\right)\right),
        \sin\left(scl\left(\frac{4}{500000^{\frac{4}{128}}}\right)\right) + i \cos\left(scl\left(\frac{4}{500000^{\frac{4}{128}}}\right)\right),
        \dots, \\\\
        \sin\left(scl\left(\frac{4}{500000^{\frac{124}{128}}}\right)\right) + i \cos\left(scl\left(\frac{4}{500000^{\frac{124}{128}}}\right)\right),
        \sin\left(scl\left(\frac{4}{500000^{\frac{126}{128}}}\right)\right) + i \cos\left(scl\left(\frac{4}{500000^{\frac{126}{128}}}\right)\right)
    \end{array}
    \right\rbrace \\
\end{gathered} \\
$$

## **10.7. Visualized Form of Some Samples from The Frequency Tensor**

The charts below aim to give you some insight into the values of angles and corresponding polar coordinates in the frequency tensor. The chart titles contain which index ranges are taken as samples in a particular chart.

These charts are drawn to make it easy for you to compare changes between positions and dimensions.

>You can check out the Python codes that create the sample data and charts used here with this Python Notebook: [10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb](./10.BONUS-PRECOMPUTING-FREQUENCY-TENSOR.ipynb)

3D charts of polar coordinates:

![Chart 3D - Polar coordinates of 0th position](./images/03-chart_polar_coordinates_3d_pos_0.svg)
![Chart 3D - Polar coordinates of 1st position](./images/03-chart_polar_coordinates_3d_pos_1.svg)
![Chart 3D - Polar coordinates of 2nd position](./images/03-chart_polar_coordinates_3d_pos_2.svg)
![Chart 3D - Polar coordinates of 5th position](./images/03-chart_polar_coordinates_3d_pos_5.svg)
![Chart 3D - Polar coordinates of 10th position](./images/03-chart_polar_coordinates_3d_pos_10.svg)
![Chart 3D - Polar coordinates of 300th position](./images/03-chart_polar_coordinates_3d_pos_300.svg)
![Chart 3D - Polar coordinates of 301st position](./images/03-chart_polar_coordinates_3d_pos_301.svg)
![Chart 3D - Polar coordinates of 1000th position](./images/03-chart_polar_coordinates_3d_pos_1000.svg)

2D charts of angles:

![Chart - Angles per sample positions and dimensions](./images/03-chart_angles_2d.svg)

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: IMPLEMENTING LLAMA MODEL ARCHITECTURE](./09-IMPLEMENTING-LLAMA-MODEL-ARCHITECTURE.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: ASKING FOR USER INPUT&nbsp;&nbsp;&gt;](./11-ASKING-FOR-USER-INPUT.md)

</div>

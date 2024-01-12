package model

import (
	"fmt"
	"math"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
)

type LlamaTransformer struct {
	tok_embd *ml.Tensor // Original: "tok_embeddings.weight"  |  ggml: "token_embd.weight" | shape: [32000 4096] -> [VocabSize, Dim]

	Layers []*LlamaTransformerBlock

	output_norm *RMSNorm   // Weights Original: "norm.weight"  |  ggml: "output_norm.weight" | shape: [4096] -> [Dim]
	output      *ml.Tensor // Original: "output.weight"  |  ggml: "output.weight" | [out_features, in_features] -> shape: [32000 4096] -> [VocabSize, Dim]

	PrecomputedFreqsCis *ml.Tensor // Precomputed frequency tensor for complex exponentials (cis)
}

type LlamaTransformerBlock struct {
	LayerIndex int

	attn_norm *RMSNorm // Weights Original: "layers.0.attention_norm.weight"  |  ggml: "blk.0.attn_norm.weight" | shape: [4096] -> [Dim]
	ffn_norm  *RMSNorm // Weights Original: "layers.0.ffn_norm.weight"  |  ggml: "blk.0.ffn_norm.weight" | shape: [4096] -> [Dim]

	attention   *LlamaAttention
	feedForward *LlamaFeedForward
}

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

type LlamaFeedForward struct {
	FFNHiddenDim int

	ffn_gate *ml.Tensor // Original: "layers.0.feed_forward.w1.weight"  |  ggml: "blk.0.ffn_gate.weight" | [out_features, in_features] -> shape: [11008 4096] -> [FFNHiddenDim, Dim] | w1
	ffn_down *ml.Tensor // Original: "layers.0.feed_forward.w2.weight"  |  ggml: "blk.0.ffn_down.weight" | [out_features, in_features] -> shape: [4096 11008] -> [Dim, FFNHiddenDim] | w2
	ffn_up   *ml.Tensor // Original: "layers.0.feed_forward.w3.weight"  |  ggml: "blk.0.ffn_up.weight" | [out_features, in_features] -> shape: [11008 4096] -> [FFNHiddenDim, Dim] | w3
}

type RMSNorm struct {
	epsilon float32
	weights *ml.Tensor
}

func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
	result := &LlamaTransformer{}
	modelArgs := model.ModelArgs

	var err error
	// Compare (VocabSize, Dim) vs. "tok_embeddings.weight" tensor shape
	dim := modelArgs.Dim             // 4096
	vocabSize := modelArgs.VocabSize // 32000

	if modelArgs.N_KVHeads < 0 {
		modelArgs.N_KVHeads = modelArgs.N_Heads
	}
	modelArgs.N_Rep = int(modelArgs.N_Heads / modelArgs.N_KVHeads)
	// Calculate dimension of each head
	modelArgs.HeadDim = int(modelArgs.Dim / modelArgs.N_Heads) // 128

	if result.tok_embd, err = getTensor(model, "tok_embeddings.weight", []int{vocabSize, dim}); err != nil {
		return nil, err
	}

	result.Layers = make([]*LlamaTransformerBlock, modelArgs.N_Layers)

	for layerIdx := 0; layerIdx < modelArgs.N_Layers; layerIdx++ {
		var layer *LlamaTransformerBlock
		if layer, err = NewLlamaTransformerBlock(model, layerIdx); err != nil {
			return nil, err
		}
		result.Layers[layerIdx] = layer
	}

	output_norm_weights, err := getTensor(model, "norm.weight", []int{dim})
	if err != nil {
		return nil, err
	}
	result.output_norm = NewRMSNorm(modelArgs.NormEpsilon, output_norm_weights)

	// output is a Linear unit, so weight shape is ordered reversely as [out_features, in_features]
	if result.output, err = getTensor(model, "output.weight", []int{vocabSize, dim}); err != nil {
		return nil, err
	}

	if result.PrecomputedFreqsCis, err = precomputeFreqsCis(int(dim/modelArgs.N_Heads), modelArgs.MaxSequenceLength*2); err != nil {
		return nil, err
	}
	return result, nil
}

func (lt *LlamaTransformer) prepare(inputTokens *ml.Tensor, startPos int) (inputTensor *ml.Tensor, freqsCis *ml.Tensor, mask *ml.Tensor, err error) {
	sequenceLength := inputTokens.Size[0]
	inputTensor, err = ml.Fwd_Get_Rows(lt.tok_embd, inputTokens)
	if err != nil {
		return
	}

	freqsCis, err = lt.PrecomputedFreqsCis.Slice([]int{startPos}, []int{startPos + sequenceLength})
	if err != nil {
		return
	}

	if sequenceLength > 1 {
		negativeInfinity := dtype.BFloat16fromFloat32(float32(math.Inf(-1)))
		if mask, err = ml.Full([]int{sequenceLength, sequenceLength}, ml.DT_BF16, negativeInfinity); err != nil {
			return
		}
		if mask, err = ml.TriangularUpper(mask, 1); err != nil {
			return
		}
	}
	return
}

func (lt *LlamaTransformer) Forward(context *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
	if inputTokens.Size[0] == 0 {
		return nil, fmt.Errorf("empty token array")
	}

	inputTensor, freqsCis, mask, err := lt.prepare(inputTokens, startPos)
	if err != nil {
		return nil, err
	}

	currentTensor := inputTensor
	for layerIdx, layer := range lt.Layers {
		fmt.Printf("Running transformer block layer: %d / %d\n", layerIdx, len(lt.Layers))
		if currentTensor, err = layer.Forward(context, currentTensor, startPos, freqsCis, mask); err != nil {
			return nil, err
		}
	}
	if currentTensor, err = lt.output_norm.Forward(context, currentTensor); err != nil {
		return nil, err
	}
	output, err := ml.LinearTransformation(currentTensor, lt.output)
	if err != nil {
		return nil, err
	}
	if output, err = output.ToFloat32(); err != nil {
		return nil, err
	}
	return output, nil
}

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

func (ltb *LlamaTransformerBlock) Forward(context *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
	normalizedX, err := ltb.attn_norm.Forward(context, x)
	if err != nil {
		return nil, err
	}
	h, err := ltb.attention.Forward(context, normalizedX, startPos, freqsCis, mask)
	if err != nil {
		return nil, err
	}
	if h, err = ml.Add(x, h); err != nil {
		return nil, err
	}

	normalizedH, err := ltb.ffn_norm.Forward(context, h)
	if err != nil {
		return nil, err
	}
	ffnOutput, err := ltb.feedForward.Forward(normalizedH)
	if err != nil {
		return nil, err
	}

	output, err := ml.Add(h, ffnOutput)
	if err != nil {
		return nil, err
	}

	return output, nil
}

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

func (lat *LlamaAttention) Forward(context *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
	sequenceLength := x.Size[0]

	// lat.attn_wq: [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
	xq, err := ml.LinearTransformation(x, lat.attn_wq)
	if err != nil {
		return nil, err
	}

	// lat.attn_wk: [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
	xk, err := ml.LinearTransformation(x, lat.attn_wk)
	if err != nil {
		return nil, err
	}

	// lat.attn_wv: [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
	xv, err := ml.LinearTransformation(x, lat.attn_wv)
	if err != nil {
		return nil, err
	}

	/*
		Do reshapings
	*/

	if xq, err = xq.Reshape([]int{sequenceLength, lat.N_Heads, lat.HeadDim}); err != nil {
		return nil, err
	}

	if xk, err = xk.Reshape([]int{sequenceLength, lat.N_KVHeads, lat.HeadDim}); err != nil {
		return nil, err
	}

	if xv, err = xv.Reshape([]int{sequenceLength, lat.N_KVHeads, lat.HeadDim}); err != nil {
		return nil, err
	}

	/*
		Apply rotary embeddings
	*/

	if xq, xk, err = applyRotaryEmbeddings(xq, xk, freqsCis); err != nil { // example shape=[5,32,128] dtype=DT_BF16
		return nil, err
	}

	/*
		Update KV cache
	*/

	context.CacheK[lat.LayerIndex].SetSlice([]int{startPos}, []int{startPos + sequenceLength}, xk)
	context.CacheV[lat.LayerIndex].SetSlice([]int{startPos}, []int{startPos + sequenceLength}, xv)

	/*
		Retrieve cached KV so far
	*/

	keys, err := context.CacheK[lat.LayerIndex].Slice([]int{0}, []int{startPos + sequenceLength})
	if err != nil {
		return nil, err
	}
	values, err := context.CacheV[lat.LayerIndex].Slice([]int{0}, []int{startPos + sequenceLength})
	if err != nil {
		return nil, err
	}

	/*
		Repeat k/v heads if N_KVHeads < N_Heads
	*/

	if keys, err = attentionRepeatKV(keys, lat.N_Rep); err != nil { // example shape=[5, 32, 128] (cacheLen + sequenceLength, N_Heads, HeadDim)
		return nil, err
	}
	if values, err = attentionRepeatKV(values, lat.N_Rep); err != nil { // example shape=[5, 32, 128] (cacheLen + sequenceLength, N_Heads, HeadDim)
		return nil, err
	}

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

	/*
		Goal in Python manner:
		scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
	*/

	xqMatMulKeys, err := ml.MatMul(xq, keys) // matmul([32,5,128], [32,128,5]) -> example shape=[32,5,5] (N_Heads, sequenceLength, sequenceLength)
	if err != nil {
		return nil, err
	}
	scores, err := ml.DivToScalar(xqMatMulKeys, dtype.BFloat16fromFloat32(float32(math.Sqrt(float64(lat.HeadDim))))) // example shape=[32,5,5]
	if err != nil {
		return nil, err
	}

	if mask != nil {
		if scores, err = ml.Add(scores, mask); err != nil { // example shape=[32,5,5]
			return nil, err
		}
	}

	/*
		Goal in Python manner:
		scores = F.softmax(scores.float(), dim=-1).type_as(xq)
	*/

	scores, err = scores.ToFloat32() // example shape=[32,5,5] dtype=DT_F32
	if err != nil {
		return nil, err
	}
	if scores, err = ml.Softmax(scores, len(scores.Size)-1); err != nil { // example shape=[32,5,5] dtype=DT_F32
		return nil, err
	}
	if scores, err = scores.ToBFloat16(); err != nil { // example shape=[32,5,5] (N_Heads, sequenceLength, sequenceLength) dtype=DT_BF16
		return nil, err
	}

	/*
		Goal in Python manner:
		output = torch.matmul(scores, values)
		output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
	*/

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

	/*
		Apply lat.attn_wo weights to output
	*/

	// lat.attn_wo: [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
	if output, err = ml.LinearTransformation(output, lat.attn_wo); err != nil {
		return nil, err
	}
	return output, nil
}

func attentionRepeatKV(x *ml.Tensor, N_Rep int) (*ml.Tensor, error) {
	// See: https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L164
	// repeat_kv function was not implemented because currently we support only 7B model
	if N_Rep == 1 {
		return x, nil
	}
	return nil, fmt.Errorf("currently only 7B model is supported, N_Rep > 1 case was not implemented yet, because of this")
}

func NewLlamaFeedForward(model *Model, layerIndex int) (*LlamaFeedForward, error) {
	result := &LlamaFeedForward{}
	modelArgs := model.ModelArgs
	dim := modelArgs.Dim // 4096
	var err error

	// See: https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L378
	// Set it to 4 * dim at first
	result.FFNHiddenDim = 4 * modelArgs.Dim
	// See: https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L331C4-L331C4
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

func (lff *LlamaFeedForward) Forward(x *ml.Tensor) (*ml.Tensor, error) {
	/*
		Goal in Python manner:
		self.w2(F.silu(self.w1(x)) * self.w3(x))
		-->
		self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
	*/
	h, err := ml.LinearTransformation(x, lff.ffn_gate)
	if err != nil {
		return nil, err
	}
	if h, err = ml.Silu(h); err != nil {
		return nil, err
	}
	ffnUpX, err := ml.LinearTransformation(x, lff.ffn_up)
	if err != nil {
		return nil, err
	}
	if h, err = ml.MultiplyElementwise(h, ffnUpX); err != nil {
		return nil, err
	}
	output, err := ml.LinearTransformation(h, lff.ffn_down)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func NewRMSNorm(epsilon float32, weights *ml.Tensor) *RMSNorm {
	return &RMSNorm{
		epsilon: epsilon,
		weights: weights,
	}
}

func (rms *RMSNorm) Forward(context *InferenceContext, x *ml.Tensor) (*ml.Tensor, error) {
	h, err := rms.doNormalization(x)
	if err != nil {
		return nil, err
	}
	return ml.MultiplyElementwise(h, rms.weights)
}

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

func precomputeFreqsCis(dim int, end int) (*ml.Tensor, error) {
	// Comment from Llama code
	// See: https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L80
	/*
		Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

		This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
		and the end index 'end'. The 'theta' parameter scales the frequencies.
		The returned tensor contains complex values in complex64 data type.

		Args:
			dim (int): Dimension of the frequency tensor.
			end (int): End index for precomputing frequencies.
			theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

		Returns:
			torch.Tensor: Precomputed frequency tensor with complex exponentials.
	*/
	theta := 10000.0

	dimFloat := float32(dim)
	freqs, err := ml.ARange(0, dim, 2, ml.DT_BF16)
	if err != nil {
		return nil, err
	}
	err = freqs.Apply(func(val any) any {
		f16val := val.(dtype.BFloat16)
		return dtype.BFloat16fromFloat32(float32(1.0 / math.Pow(theta, float64(f16val.Float32()/dimFloat))))
	})
	if err != nil {
		return nil, err
	}

	t, err := ml.ARange(0, end, 1, ml.DT_BF16)
	if err != nil {
		return nil, err
	}

	freqs, err = ml.Outer(t, freqs)
	if err != nil {
		return nil, err
	}

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

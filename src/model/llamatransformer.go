package model

import (
	"fmt"
	"math"

	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/x448/float16"
)

type LlamaTransformer struct {
	tok_embd *ml.Tensor // Original: "tok_embeddings.weight"  |  ggml: "token_embd.weight" | shape: [32000 4096] -> [VocabSize, Dim]

	Layers []*LlamaTransformerBlock

	output_norm *ml.Tensor // Original: "norm.weight"  |  ggml: "output_norm.weight" | shape: [4096] -> [Dim]
	output      *ml.Tensor // Original: "output.weight"  |  ggml: "output.weight" | [out_features, in_features] -> shape: [32000 4096] -> [VocabSize, Dim]

	Context *LlamaContext
}

type LlamaTransformerBlock struct {
	attn_norm *ml.Tensor // Original: "layers.0.attention_norm.weight"  |  ggml: "blk.0.attn_norm.weight" | shape: [4096] -> [Dim]
	ffn_norm  *ml.Tensor // Original: "layers.0.ffn_norm.weight"  |  ggml: "blk.0.ffn_norm.weight" | shape: [4096] -> [Dim]

	attention   *LlamaAttention
	feedForward *LlamaFeedForward
}

type LlamaAttention struct {
	N_KVHeads int
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

type LlamaContext struct {
	FreqsCis *ml.Tensor // Precomputed frequency tensor for complex exponentials (cis)
}

func NewLlamaTransformer(model *Model) (*LlamaTransformer, error) {
	result := &LlamaTransformer{
		Context: &LlamaContext{},
	}
	modelArgs := model.ModelArgs

	var err error
	// Compare (VocabSize, Dim) vs. "tok_embeddings.weight" tensor shape
	dim := modelArgs.Dim             // 4096
	vocabSize := modelArgs.VocabSize // 32000
	if result.tok_embd, err = getTensor(model, "tok_embeddings.weight", []int{vocabSize, dim}); err != nil {
		return nil, err
	}

	result.Layers = make([]*LlamaTransformerBlock, modelArgs.N_Layers)

	for i := 0; i < modelArgs.N_Layers; i++ {
		var layer *LlamaTransformerBlock
		if layer, err = NewLlamaTransformerBlock(model, i); err != nil {
			return nil, err
		}
		result.Layers[i] = layer
	}

	if result.output_norm, err = getTensor(model, "norm.weight", []int{dim}); err != nil {
		return nil, err
	}

	// output is a Linear unit, so weight shape is ordered reversely as [out_features, in_features]
	if result.output, err = getTensor(model, "output.weight", []int{vocabSize, dim}); err != nil {
		return nil, err
	}

	if result.Context.FreqsCis, err = precomputeFreqsCis(int(dim/modelArgs.N_Heads), modelArgs.MaxSequenceLength*2); err != nil {
		return nil, err
	}
	return result, nil
}

func (t *LlamaTransformer) Forward(tokens []TokenId, startPos int) ([]TokenId, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token array")
	}

	sequenceLength := len(tokens)
	inp_tokens := ml.NewEmptyTensorEx("inp_tokens", []int{sequenceLength}, ml.DT_UINT16)

	for i, token := range tokens {
		err := inp_tokens.SetItem([]int{i}, uint16(token))
		if err != nil {
			return nil, err
		}
	}

	inpL, err := ml.Fwd_Get_Rows(t.tok_embd, inp_tokens)
	if err != nil {
		return nil, err
	}
	inpL = inpL

	freqsCis, err := t.Context.FreqsCis.Slice([]int{startPos}, []int{startPos + sequenceLength})
	if err != nil {
		return nil, err
	}

	freqsCis = freqsCis
	return nil, nil
}

func NewLlamaTransformerBlock(model *Model, layerIndex int) (*LlamaTransformerBlock, error) {
	result := &LlamaTransformerBlock{}
	modelArgs := model.ModelArgs
	dim := modelArgs.Dim // 4096
	var err error

	// attention normalization
	if result.attn_norm, err = getLayerTensor(model, "layers.%d.attention_norm.weight", layerIndex, []int{dim}); err != nil {
		return nil, err
	}

	if result.attention, err = NewLlamaAttention(model, layerIndex); err != nil {
		return nil, err
	}

	// feed forward normalization
	if result.ffn_norm, err = getLayerTensor(model, "layers.%d.ffn_norm.weight", layerIndex, []int{dim}); err != nil {
		return nil, err
	}

	if result.feedForward, err = NewLlamaFeedForward(model, layerIndex); err != nil {
		return nil, err
	}

	return result, nil
}

func NewLlamaAttention(model *Model, layerIndex int) (*LlamaAttention, error) {
	result := &LlamaAttention{}
	modelArgs := model.ModelArgs
	dim := modelArgs.Dim // 4096
	var err error

	result.N_KVHeads = modelArgs.N_KVHeads
	if result.N_KVHeads < 0 {
		result.N_KVHeads = modelArgs.N_Heads
	}
	// Calculate dimension of each head
	result.HeadDim = int(modelArgs.Dim / modelArgs.N_Heads) // 128

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
	freqs.Apply(func(val any) any {
		f16val := val.(float16.Float16)
		return float16.Fromfloat32(float32(1.0 / math.Pow(theta, float64(f16val.Float32()/dimFloat))))
	})
	fmt.Printf("\n%s\n", freqs)

	t, err := ml.ARange(0, end, 1, ml.DT_BF16)
	if err != nil {
		return nil, err
	}
	fmt.Printf("\n%s\n", t)

	freqs, err = ml.Outer(t, freqs)
	if err != nil {
		return nil, err
	}
	fmt.Printf("\n%s\n", freqs)

	freqs_cis, err := ml.Polar(ml.OnesLike(freqs), freqs)
	if err != nil {
		return nil, err
	}
	return freqs_cis, nil
}

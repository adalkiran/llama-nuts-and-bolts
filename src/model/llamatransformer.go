package model

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"sync"
	"time"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
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

	if modelArgs.RopeTheta <= 0 {
		modelArgs.RopeTheta = 500000.0
	}

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

	if result.PrecomputedFreqsCis, err = precomputeFreqsCis(int(dim/modelArgs.N_Heads), modelArgs.MaxSequenceLength*2, modelArgs.RopeTheta, modelArgs.UseScaledRope); err != nil {
		return nil, err
	}
	return result, nil
}

func (lt *LlamaTransformer) prepare(inputTokens *ml.Tensor, startPos int) (inputTensor *ml.Tensor, freqsCis *ml.Tensor, mask *ml.Tensor, err error) {
	sequenceLength := inputTokens.Size[0]
	common.GLogger.DebugPrintf("LlamaTransformer.prepare started for inputTokens: shape(%v), startPos: %d. sequenceLength: %d", inputTokens.Size, startPos, sequenceLength)
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
	var maskSize []int
	if mask != nil {
		maskSize = mask.Size
	}
	common.GLogger.DebugPrintf("LlamaTransformer.prepare finished inputTensor: shape(%v), freqsCis: shape(%v), mask: shape(%v)", inputTensor.Size, freqsCis.Size, maskSize)
	return
}

func (lt *LlamaTransformer) Forward(infContext *InferenceContext, inputTokens *ml.Tensor, startPos int) (*ml.Tensor, error) {
	if inputTokens.Size[0] == 0 {
		return nil, fmt.Errorf("empty token array")
	}
	common.GLogger.DebugPrintf("LlamaTransformer.Forward started for inputTokens: shape(%v), startPos: %d -> tensor inputTensor", inputTokens.Size, startPos)
	inputTensor, freqsCis, mask, err := lt.prepare(inputTokens, startPos)
	if err != nil {
		return nil, err
	}

	currentTensor := inputTensor
	for layerIdx, layer := range lt.Layers {
		startTime := time.Now()
		common.GLogger.DebugPrintf("=======================================\n")
		common.GLogger.DebugPrintf("Calling LlamaTransformerBlock.Forward for layer: %d / %d, startPos: %d -> tensor currentTensor", layerIdx+1, len(lt.Layers), startPos)
		if currentTensor, err = layer.Forward(infContext, currentTensor, startPos, freqsCis, mask); err != nil {
			return nil, err
		}
		infContext.Logf("Transformer block layer %d / %d was run, took %.4f sec(s)", layerIdx+1, len(lt.Layers), time.Since(startTime).Seconds())
	}
	common.GLogger.DebugPrintf("Calling RMSNorm for currentTensor shape(%v) (result of all transformer blocks) and LlamaTransformer.output_norm weights shape(%v) -> tensor currentTensor", currentTensor.Size, lt.output_norm.weights.Size)
	if currentTensor, err = lt.output_norm.Forward(infContext, currentTensor); err != nil {
		return nil, err
	}
	common.GLogger.DebugPrintf("Calling ml.LinearTransformation for currentTensor (normalized result of all transformer blocks) shape(%v) and LlamaTransformer.output weights shape(%v) -> tensor output", currentTensor.Size, lt.output.Size)
	output, err := ml.LinearTransformation(currentTensor, lt.output)
	if err != nil {
		return nil, err
	}
	common.GLogger.DebugPrintf("Converting output tensor shape(%v) to Float32 tensor -> tensor output", output.Size)
	if output, err = output.ToFloat32(); err != nil {
		return nil, err
	}
	common.GLogger.DebugPrintf("Returning tensor output: shape(%v)", output.Size)
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

func (ltb *LlamaTransformerBlock) Forward(infContext *InferenceContext, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
	var maskSize []int
	if mask != nil {
		maskSize = mask.Size
	}
	common.GLogger.DebugPrintf("LlamaTransformerBlock.Forward started for x: shape(%v), startPos: %d, freqsCis: shape(%v), mask: shape(%v)", x.Size, startPos, freqsCis.Size, maskSize)
	common.GLogger.DebugPrintf("Calling RMSNorm for tensor x shape(%v) and LlamaTransformerBlock.attn_norm weights shape(%v) -> tensor normalizedX", x.Size, ltb.attn_norm.weights.Size)
	normalizedX, err := ltb.attn_norm.Forward(infContext, x)
	if err != nil {
		return nil, err
	}
	common.GLogger.DebugPrintf("Calling LlamaAttention.Forward for tensor normalizedX shape(%v) and startPos: %d, freqsCis: shape(%v), mask: shape(%v) -> tensor h", normalizedX.Size, startPos, freqsCis.Size, maskSize)
	h, err := ltb.attention.Forward(infContext, normalizedX, startPos, freqsCis, mask)
	if err != nil {
		return nil, err
	}
	common.GLogger.DebugPrintf("Calling ml.Add to calculate x shape(%v) + h shape(%v) -> tensor h", x.Size, h.Size)
	if h, err = ml.Add(x, h); err != nil {
		return nil, err
	}

	common.GLogger.DebugPrintf("Calling RMSNorm for tensor h shape(%v) and LlamaTransformerBlock.ffn_norm weights shape(%v) -> tensor normalizedH", x.Size, ltb.ffn_norm.weights.Size)
	normalizedH, err := ltb.ffn_norm.Forward(infContext, h)
	if err != nil {
		return nil, err
	}
	common.GLogger.DebugPrintf("Calling LlamaFeedForward.Forward for tensor normalizedH shape(%v) -> tensor ffnOutput", normalizedH.Size)
	ffnOutput, err := ltb.feedForward.Forward(normalizedH)
	if err != nil {
		return nil, err
	}

	common.GLogger.DebugPrintf("Calling ml.Add to calculate h shape(%v) + ffnOutput shape(%v) -> tensor output", h.Size, ffnOutput.Size)
	output, err := ml.Add(h, ffnOutput)
	if err != nil {
		return nil, err
	}
	common.GLogger.DebugPrintf("Returning tensor output: shape(%v)", output.Size)
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
		common.GLogger.DebugPrintf("[Calling in goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wq weights shape(%v) -> tensor xq", x.Size, lat.attn_wq.Size)
		// lat.attn_wq: [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
		xq, err := ml.LinearTransformation(x, lat.attn_wq)
		if err != nil {
			cancel(err)
			return
		}
		mu.Lock()
		parallelResults["xq"] = xq
		mu.Unlock()
	}()

	common.GLogger.DebugPrintf("[Scheduling goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wk weights shape(%v) -> tensor xk", x.Size, lat.attn_wk.Size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		if ctx.Err() != nil {
			return
		}
		common.GLogger.DebugPrintf("[Calling in goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wk weights shape(%v) -> tensor xk", x.Size, lat.attn_wk.Size)
		// lat.attn_wk: [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
		xk, err := ml.LinearTransformation(x, lat.attn_wk)
		if err != nil {
			cancel(err)
			return
		}
		mu.Lock()
		parallelResults["xk"] = xk
		mu.Unlock()
	}()

	common.GLogger.DebugPrintf("[Scheduling goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wv weights shape(%v) -> tensor xv", x.Size, lat.attn_wv.Size)
	wg.Add(1)
	go func() {
		defer wg.Done()
		if ctx.Err() != nil {
			return
		}
		common.GLogger.DebugPrintf("[Calling in goroutine] ml.LinearTransformation for x shape(%v) and LlamaAttention.attn_wv weights shape(%v) -> tensor xv", x.Size, lat.attn_wv.Size)
		// lat.attn_wv: [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
		xv, err := ml.LinearTransformation(x, lat.attn_wv)
		if err != nil {
			cancel(err)
			return
		}
		mu.Lock()
		parallelResults["xv"] = xv
		mu.Unlock()
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

	/*
		Apply rotary embeddings
	*/

	if xq, xk, err = applyRotaryEmbeddings(xq, xk, freqsCis); err != nil { // example shape=[5,32,128] dtype=DT_BF16
		return nil, err
	}

	common.GLogger.DebugPrintf("applyRotaryEmbeddings results, xq: shape(%v), xk: shape(%v)", xq.Size, xk.Size)

	/*
		Update KV cache
	*/

	infContext.CacheK[lat.LayerIndex].SetSlice([]int{startPos}, []int{startPos + sequenceLength}, xk)
	infContext.CacheV[lat.LayerIndex].SetSlice([]int{startPos}, []int{startPos + sequenceLength}, xv)

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

	common.GLogger.DebugPrintf("Multiple transposing results, xq: shape(%v), keys: shape(%v), values: shape(%v)", xq.Size, keys.Size, values.Size)

	/*
		Goal in Python manner:
		scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
	*/

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

	if mask != nil {
		common.GLogger.DebugPrintf("Calling ml.Add to calculate scores shape(%v) + mask shape(%v) -> tensor scores", scores.Size, mask.Size)
		if scores, err = ml.Add(scores, mask); err != nil { // example shape=[32,5,5]
			return nil, err
		}
	} else {
		common.GLogger.DebugPrintf("Skipping addition scores + mask")
	}

	/*
		Goal in Python manner:
		scores = F.softmax(scores.float(), dim=-1).type_as(xq)
	*/

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

	/*
		Goal in Python manner:
		output = torch.matmul(scores, values)
		output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
	*/

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

func attentionRepeatKV(x *ml.Tensor, N_Rep int) (*ml.Tensor, error) {
	// See: https://github.com/meta-llama/llama-models/blob/6214a21dc837ce63983ef3fd7b172a6ed16e4905/models/llama3_1/api/model.py#L99
	if N_Rep == 1 {
		return x, nil
	}
	sequenceLength, n_KVHeads, headDim := x.Size[0], x.Size[1], x.Size[2]

	expanded := ml.NewEmptyTensor([]int{sequenceLength, n_KVHeads, N_Rep, headDim}, x.DataType)
	var err error
	x, err = x.Reshape([]int{sequenceLength, n_KVHeads, 1, headDim})
	if err != nil {
		return nil, err
	}
	for i := 0; i < sequenceLength; i++ {
		for j := 0; j < n_KVHeads; j++ {
			slice, err := x.Slice([]int{i, j, 0}, []int{i, j, 1})
			if err != nil {
				return nil, err
			}
			for rep := 0; rep < N_Rep; rep++ {
				if err = expanded.SetSlice([]int{i, j, rep}, []int{i, j, rep + 1}, slice); err != nil {
					return nil, err
				}
			}
		}
	}
	if expanded, err = expanded.Reshape([]int{sequenceLength, n_KVHeads * N_Rep, headDim}); err != nil {
		return nil, err
	}
	return expanded, nil
}

func NewLlamaFeedForward(model *Model, layerIndex int) (*LlamaFeedForward, error) {
	result := &LlamaFeedForward{}
	modelArgs := model.ModelArgs
	dim := modelArgs.Dim // 4096
	var err error

	// See: https://github.com/meta-llama/llama-models/blob/6214a21dc837ce63983ef3fd7b172a6ed16e4905/models/llama3_1/api/model.py#L252
	// Set it to 4 * dim at first
	result.FFNHiddenDim = 4 * modelArgs.Dim
	// See: https://github.com/meta-llama/llama-models/blob/6214a21dc837ce63983ef3fd7b172a6ed16e4905/models/llama3_1/api/model.py#L223
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

func NewRMSNorm(epsilon float32, weights *ml.Tensor) *RMSNorm {
	return &RMSNorm{
		epsilon: epsilon,
		weights: weights,
	}
}

func (rms *RMSNorm) Forward(infContext *InferenceContext, x *ml.Tensor) (*ml.Tensor, error) {
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
	// Comment from Llama code
	// See Llama 2 Comment: https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L80
	// See Llama 3.1 Code: https://github.com/meta-llama/llama-models/blob/6214a21dc837ce63983ef3fd7b172a6ed16e4905/models/llama3_1/api/model.py#L66
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
	dimFloat := float32(dim)
	freqs, err := ml.ARange(0, dim, 2, ml.DT_BF16)
	if err != nil {
		return nil, err
	}
	err = freqs.Apply_AsFloat32(func(val float32) float32 {
		return float32(1.0 / math.Pow(theta, float64(val/dimFloat)))
	})
	if err != nil {
		return nil, err
	}

	t, err := ml.ARange(0, end, 1, ml.DT_BF16)
	if err != nil {
		return nil, err
	}

	if useScaled {
		err = applyScaling(freqs)
		if err != nil {
			return nil, err
		}
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

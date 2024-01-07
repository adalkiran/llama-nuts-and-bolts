package model

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
)

/*
	This simulation was added to ensure all steps are done correctly
	(until only first attention layer).
*/

func testTransformer_Prepare(t *testing.T, actualInputTensor *ml.Tensor, actualFreqsCis *ml.Tensor, actualMask *ml.Tensor) {
	expectedInputTensorSize := []int{5, 4096}
	// Shortened form as corresponding indices [0, 1, 2, 4093, 4094, 4095]
	expectedInputTensorShortened := [][]float32{
		{0.0019, -0.0034, 0.0004 /*...,*/, -0.0083, 0.0026, -0.0039},
		{0.0381, -0.0007, 0.0069 /*...,*/, -0.0168, 0.0024, 0.0157},
		{-0.0150, -0.0162, 0.0110 /*...,*/, 0.0082, -0.0283, 0.0049},
		{-0.0032, -0.0100, -0.0110 /*...,*/, -0.0033, -0.0038, -0.0117},
		{-0.0054, 0.0012, 0.0083 /*...,*/, 0.0112, -0.0043, -0.0077},
	}

	expectedFreqsCisSize := []int{5, 64}

	expectedMaskSize := []int{5, 5}
	negInf := float32(math.Inf(-1))
	expectedMask := [][]float32{
		{0, negInf, negInf, negInf, negInf},
		{0, 0, negInf, negInf, negInf},
		{0, 0, 0, negInf, negInf},
		{0, 0, 0, 0, negInf},
		{0, 0, 0, 0, 0},
	}

	if err := ml.CompareTestTensor(expectedInputTensorShortened, expectedInputTensorSize, actualInputTensor, common.THRESHOLD_F32, true); err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(expectedFreqsCisSize, actualFreqsCis.Size) {
		t.Errorf("expected size %v, but got %v", expectedFreqsCisSize, actualFreqsCis.Size)
	}

	if err := ml.CompareTestTensor(expectedMask, expectedMaskSize, actualMask, common.THRESHOLD_F32, false); err != nil {
		t.Error(err)
	}
}

func testTransformerBlock_AttnNorm_Forward(firstLayer *LlamaTransformerBlock, x *ml.Tensor) (*ml.Tensor, error) {
	/*
		normalizedX, err := ltb.attn_norm.Forward(context, x)
	*/
	expectedAttnNormPartSize := []int{5, 4096}
	expectedAttnNormPart := [][]float32{
		{0.2271, -0.4113, 0.0486 /*...,*/, -1.0125, 0.3145, -0.4802},
		{2.2847, -0.0412, 0.4119 /*...,*/, -1.0106, 0.1437, 0.9447},
		{-1.2023, -1.3054, 0.8833 /*...,*/, 0.6625, -2.2770, 0.3926},
		{-0.2195, -0.6924, -0.7599 /*...,*/, -0.2280, -0.2596, -0.8063},
		{-0.5375, 0.1223, 0.8214 /*...,*/, 1.1053, -0.4288, -0.7610},
	}

	expectedAttnNormalizedXSize := []int{5, 4096}
	expectedAttnNormalizedX := [][]float32{
		{6.7444e-03, -5.6152e-03, 9.5844e-05 /*...,*/, -1.0437e-02, 3.4485e-03, -2.9144e-03},
		{6.7871e-02, -5.6076e-04, 8.1253e-04 /*...,*/, -1.0315e-02, 1.5793e-03, 5.7373e-03},
		{-3.5645e-02, -1.7700e-02, 1.7395e-03 /*...,*/, 6.8054e-03, -2.5024e-02, 2.3804e-03},
		{-6.5308e-03, -9.3994e-03, -1.5030e-03 /*...,*/, -2.3346e-03, -2.8534e-03, -4.8828e-03},
		{-1.5991e-02, 1.6632e-03, 1.6174e-03 /*...,*/, 1.1292e-02, -4.7302e-03, -4.6387e-03},
	}

	actualAttnNormPart, err := firstLayer.attn_norm.doNormalization(x)
	if err != nil {
		return nil, err
	}
	if err := ml.CompareTestTensor(expectedAttnNormPart, expectedAttnNormPartSize, actualAttnNormPart, common.THRESHOLD_BF16, true); err != nil {
		return nil, err
	}

	actualAttnNormalizedX, err := ml.MultiplyElementwise(actualAttnNormPart, firstLayer.attn_norm.weights)
	if err != nil {
		return nil, err
	}
	if err := ml.CompareTestTensor(expectedAttnNormalizedX, expectedAttnNormalizedXSize, actualAttnNormalizedX, common.THRESHOLD_BF16, true); err != nil {
		return nil, err
	}
	return actualAttnNormalizedX, nil
}

func testTransformerBlock_Attention_Forward(attention *LlamaAttention, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
	expectedXqSize := []int{5, 4096}
	expectedXq := [][]float32{
		{0.1157, -0.4805, -0.4180 /*...,*/, 0.6250, -0.1670, 0.1602},
		{0.2598, -1.7734, -1.2266 /*..., */, 0.3789, -0.1729, -0.5195},
		{0.0747, -1.6172, -0.9336 /*...,*/, 0.3652, 0.0255, -0.1855},
		{0.1953, -1.8047, -1.1953 /*...,*/, 0.1611, -0.0103, -0.2363},
		{0.0112, -1.2500, -0.5391 /*...,*/, 0.1689, 0.1807, 0.0957},
	}

	expectedXkSize := []int{5, 4096}
	expectedXk := [][]float32{
		{-0.4512, -0.1523, -0.0170 /*...,*/, 0.2158, -0.3340, 0.2773},
		{-0.4258, 0.6836, 0.5156 /*...,*/, -0.6406, 0.7188, 0.3633},
		{-0.4824, 0.2930, 0.2285 /*..., */, 0.8438, -1.0469, -0.1099},
		{0.2051, -0.1377, -0.3711 /*...,*/, -0.5156, 0.6289, 0.1582},
		{-0.1514, -0.3613, -0.3457 /*...,*/, 0.9844, -0.9492, -0.2480},
	}

	expectedXvSize := []int{5, 4096}
	expectedXv := [][]float32{
		{-0.0060, -0.0064, 0.0056 /*...,*/, 0.0015, -0.0007, -0.0014},
		{0.0004, -0.0088, -0.0033 /* ...,*/, 0.0014, -0.0098, 0.0042},
		{0.0077, -0.0022, 0.0038 /* ...,*/, -0.0013, 0.0211, 0.0021},
		{-0.0008, -0.0053, 0.0113 /*...,*/, 0.0008, -0.0024, -0.0041},
		{0.0032, 0.0017, 0.0006 /*...,*/, -0.0003, 0.0056, 0.0035},
	}

	// lat.attn_wq: [out_features, in_features] -> shape: [4096 4096] -> [N_Heads * HeadDim, Dim]
	actualXq, err := ml.LinearTransformation(x, attention.attn_wq)
	if err != nil {
		return nil, err
	}

	if err := ml.CompareTestTensor(expectedXq, expectedXqSize, actualXq, common.THRESHOLD_BF16, true); err != nil {
		return nil, err
	}

	// lat.attn_wk: [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
	actualXk, err := ml.LinearTransformation(x, attention.attn_wk)
	if err != nil {
		return nil, err
	}

	if err := ml.CompareTestTensor(expectedXk, expectedXkSize, actualXk, common.THRESHOLD_BF16, true); err != nil {
		return nil, err
	}

	// lat.attn_wv: [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
	actualXv, err := ml.LinearTransformation(x, attention.attn_wv)
	if err != nil {
		return nil, err
	}

	if err := ml.CompareTestTensor(expectedXv, expectedXvSize, actualXv, common.THRESHOLD_BF16, true); err != nil {
		return nil, err
	}

	return nil, nil
}

func testTransformerBlock_Forward(firstLayer *LlamaTransformerBlock, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) (*ml.Tensor, error) {
	/*
		h, err := ltb.attention.Forward(context, normalizedX, startPos, freqsCis, mask)
	*/
	normalizedX, err := testTransformerBlock_AttnNorm_Forward(firstLayer, x)
	if err != nil {
		return nil, err
	}
	h, err := testTransformerBlock_Attention_Forward(firstLayer.attention, normalizedX, startPos, freqsCis, mask)
	if err != nil {
		return nil, err
	}
	return h, nil
}

func testTransformer_Forward(t *testing.T, transformer *LlamaTransformer) {
	// tokens: "<BOS>My name is"
	tokens := []TokenId{1, 15043, 590, 1024, 338}
	startPos := 0

	actualInputTensor, actualFreqsCis, actualMask, err := transformer.prepare(tokens, startPos)
	if err != nil {
		t.Error(err)
	}
	testTransformer_Prepare(t, actualInputTensor, actualFreqsCis, actualMask)

	currentTensor := actualInputTensor
	firstLayer := transformer.Layers[0]
	testTransformerBlock_Forward(firstLayer, currentTensor, startPos, actualFreqsCis, actualMask)
}

func TestSimulated(t *testing.T) {
	modelFilePath := "../../models-original/7B/consolidated.00.pth"
	if _, err := os.Stat(modelFilePath); err != nil {
		fmt.Printf("\nModel file \"%s\" is not found, passing this test: %s", modelFilePath, "TestSimulated")
		return
	}
	llamaModel, err := LoadModel(modelFilePath)
	if err != nil {
		t.Error(err)
	}

	testTransformer_Forward(t, llamaModel.Transformer)

	llamaModel.Free()
}

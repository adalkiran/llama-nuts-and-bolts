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

func testTransformerBlock_AttnNorm_Forward(t *testing.T, firstLayer *LlamaTransformerBlock, x *ml.Tensor) *ml.Tensor {
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
		t.Fatal(err)
	}
	if err := ml.CompareTestTensor(expectedAttnNormPart, expectedAttnNormPartSize, actualAttnNormPart, 2*common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}

	actualAttnNormalizedX, err := ml.MultiplyElementwise(actualAttnNormPart, firstLayer.attn_norm.weights)
	if err != nil {
		t.Fatal(err)
	}
	if err := ml.CompareTestTensor(expectedAttnNormalizedX, expectedAttnNormalizedXSize, actualAttnNormalizedX, common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}
	return actualAttnNormalizedX
}

func testTransformerBlock_Attention_Forward(t *testing.T, attention *LlamaAttention, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) *ml.Tensor {
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

	sequenceLength := x.Size[0]

	actualXq, err := ml.LinearTransformation(x, attention.attn_wq)
	if err != nil {
		t.Fatal(err)
	}

	if err := ml.CompareTestTensor(expectedXq, expectedXqSize, actualXq, 2*common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}

	actualXk, err := ml.LinearTransformation(x, attention.attn_wk)
	if err != nil {
		t.Fatal(err)
	}

	if err := ml.CompareTestTensor(expectedXk, expectedXkSize, actualXk, 2*common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}

	// lat.attn_wv: [out_features, in_features] -> shape: [4096 4096] -> [N_KVHeads * HeadDim, Dim]
	actualXv, err := ml.LinearTransformation(x, attention.attn_wv)
	if err != nil {
		t.Fatal(err)
	}

	if err := ml.CompareTestTensor(expectedXv, expectedXvSize, actualXv, 2*common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}

	/*
		Do reshapings
	*/

	expectedXqRsSize := []int{5, 32, 128}
	expectedXqRs := [][][]float32{
		{
			{0.1157, -0.4805, -0.4180 /*...,*/, 1.6875, 0.6250, -0.3145},
			{-0.1426, -1.0078, 0.2930 /*..., */, -0.6953, 0.0913, 0.5508},
			{-0.0281, 0.1206, 0.3184 /*..., */, 0.3574, -0.3418, 0.3496},
			/*...,*/
			{0.6992, 1.4062, -1.6094 /*...,*/, 0.5352, -0.2256, 1.2891},
			{0.7344, 1.9141, 1.0312 /*...,*/, 0.5234, -0.3809, 0.4863},
			{-1.3672, 0.7383, 1.3828 /*...,*/, 0.6250, -0.1670, 0.1602},
		},

		{
			{0.2598, -1.7734, -1.2266 /*...,*/, 0.5273, -1.1094, -0.7930},
			{0.0859, 0.0374, 0.1377 /*..., */, -0.9297, 0.1631, 0.5664},
			{0.5664, 0.5977, 0.0122 /*..., */, 0.3223, -0.2773, 0.3516},
			/*...,*/
			{0.5078, 2.9375, -2.4844 /* ..., */, 0.9219, -1.4141, 2.1094},
			{0.8242, 2.4531, 2.0469 /*...,*/, 0.5547, -0.4023, 0.5703},
			{-0.9297, 0.4766, -0.2754 /*...,*/, 0.3789, -0.1729, -0.5195},
		},

		{
			{0.0747, -1.6172, -0.9336 /* ..., */, 0.2871, -0.8203, -0.7422},
			{0.3301, -0.1230, 0.1807 /* ...,*/, -0.1309, -0.6719, -0.1494},
			{-0.5195, 0.8438, -0.7734 /* ...,*/, -0.1084, 0.0903, -0.0830},
			/*...,*/
			{0.4414, 1.3047, -1.0859 /* ..., */, 0.7578, -0.3613, 1.1875},
			{0.5820, 2.6094, 2.1250 /* ..., */, 0.6172, -0.4570, 0.6367},
			{-1.7734, 0.4219, -1.1250 /* ..., */, 0.3652, 0.0255, -0.1855},
		},

		{
			{0.1953, -1.8047, -1.1953 /* ...,*/, 0.7383, -1.0000, -1.0156},
			{0.0996, -0.0767, 0.0513 /* ..., */, -0.5312, -0.0591, 0.2129},
			{0.4375, 0.1270, 0.2285 /* ...,*/, 0.0737, -0.0613, 0.0986},
			/*...,*/
			{0.5195, 1.8828, -2.0469 /* ...,*/, 0.7461, -0.7031, 1.5938},
			{0.8320, 2.6094, 1.9922 /* ..., */, 0.6016, -0.4414, 0.6211},
			{-1.2812, 0.4707, -0.5156 /* ..., */, 0.1611, -0.0103, -0.2363},
		},

		{
			{0.0112, -1.2500, -0.5391 /* ..., */, 0.3301, -0.9062, -0.7461},
			{0.4512, -0.3379, 0.1299 /*...,  */, 0.1914, -0.9492, -0.3809},
			{-0.6211, 0.6094, -0.7656 /* ..., */, -0.2754, 0.2207, -0.2461},
			/* ...,*/
			{0.3516, 0.6523, -0.5547 /*  ..., */, 0.5547, -0.1299, 0.7070},
			{0.5508, 2.6562, 2.0156 /* ...,*/, 0.6289, -0.4688, 0.6445},
			{-1.9844, 0.4141, -1.2734 /* ...,*/, 0.1689, 0.1807, 0.0957},
		},
	}

	expectedXkRsSize := []int{5, 32, 128}
	expectedXkRs := [][][]float32{
		{
			{-0.4512, -0.1523, -0.0170 /*...,*/, -0.0238, -0.6211, -0.1260},
			{1.3516, -0.6094, 1.0781 /*...,*/, -0.3047, 0.0752, 0.4785},
			{0.0020, 0.3535, -0.2578 /*...,*/, 0.4980, -0.8320, 0.7617},
			/*...,*/
			{0.0247, -0.0693, -0.0049 /*...,*/, 0.9648, -0.4316, -0.4023},
			{0.1729, -0.7109, -0.5391 /*...,*/, 0.1367, -0.1855, 0.1406},
			{-0.4512, 0.1348, 1.2266 /*...,*/, 0.2158, -0.3340, 0.2773},
		},

		{
			{-0.4258, 0.6836, 0.5156 /*...,*/, -0.1240, 0.2754, 0.6055},
			{0.8867, -0.3125, 0.5547 /*...,*/, 0.3418, 0.2695, -0.2266},
			{-0.1836, 0.2852, -0.4941 /*..., */, -0.5156, 0.5898, -0.4668},
			/* ...,*/
			{-0.0615, -0.1543, 0.0630 /*...,*/, 0.5898, -0.1992, -0.1113},
			{0.2520, 0.2832, -0.2139 /*...,*/, -0.1226, 0.2383, -0.1182},
			{-1.0703, -0.2158, -0.8320 /*...,*/, -0.6406, 0.7188, 0.3633},
		},

		{
			{-0.4824, 0.2930, 0.2285 /*...,*/, -0.2520, 0.1758, 0.3535},
			{0.5977, 0.5234, 0.7812 /*...,*/, 0.2559, -0.1982, -0.6914},
			{-0.4004, -0.4609, 0.0496 /*...,*/, 1.8359, -1.7812, 1.6953},
			/* ...,*/
			{-1.2031, 0.3594, -0.2715 /*...,*/, -0.7148, 1.3750, -1.0703},
			{-0.4590, -0.1748, -0.3145 /*...,*/, 0.1357, -0.3457, 0.1387},
			{0.1157, -0.0874, -0.0542 /*...,*/, 0.8438, -1.0469, -0.1099},
		},

		{
			{0.2051, -0.1377, -0.3711 /*...,*/, 0.0244, -0.1035, 0.0479},
			{0.9414, 0.0113, 0.7305 /*...,*/, -0.0496, 0.3457, 0.4316},
			{0.0566, 0.3809, -0.2471 /*...,*/, -0.7852, 0.7891, -0.6797},
			/* ...,*/
			{-0.3301, 0.0918, -0.0408 /*...,*/, 0.0791, 0.3047, -0.5469},
			{0.2266, -0.0239, -0.4453 /*...,*/, -0.2559, 0.3516, -0.2520},
			{-0.6328, -0.3828, -0.5000 /*...,*/, -0.5156, 0.6289, 0.1582},
		},

		{
			{-0.1514, -0.3613, -0.3457 /*...,*/, 0.2314, -0.1050, -0.1689},
			{0.5117, 0.8516, 0.9414 /*...,*/, 0.0466, -0.3086, -0.3418},
			{-0.2910, -0.3867, 0.1621 /*...,*/, 1.8047, -1.7031, 1.6484},
			/* ...,*/
			{-1.7266, 0.5273, -0.3926 /*...,*/, -1.0391, 1.9141, -1.3359},
			{-0.3926, -0.4316, -0.5898 /*...,*/, 0.0928, -0.3438, 0.0918},
			{0.2100, -0.0898, 0.0125 /*...,*/, 0.9844, -0.9492, -0.2480},
		},
	}

	expectedXvRsSize := []int{5, 32, 128}
	expectedXvRs := [][][]float32{
		{
			{-6.0425e-03, -6.3782e-03, 5.5847e-03 /*...,*/, 1.3809e-03,
				2.0264e-02, -5.5237e-03},
			{2.0599e-03, 4.0283e-03, -9.7046e-03 /*..., */, -4.7607e-03,
				6.9275e-03, -1.9043e-02},
			{-3.8605e-03, -6.4392e-03, 1.4221e-02 /*..., */, 5.0354e-03,
				-1.4648e-02, 6.9885e-03},
			/*...,*/
			{-5.9509e-04, -2.0752e-03, -1.5747e-02 /*...,*/, 2.0752e-02,
				9.4604e-03, -5.1880e-03},
			{4.3106e-04, 1.5503e-02, -1.0254e-02 /*...,*/, -8.6060e-03,
				-4.5166e-02, -2.5024e-02},
			{-3.9978e-03, -3.9368e-03, 4.6730e-04 /*...,*/, 1.4572e-03,
				-6.7139e-04, -1.4420e-03},
		},

		{
			{3.8719e-04, -8.7891e-03, -3.2806e-03 /*...,*/, 5.3406e-03,
				1.2131e-03, -6.1646e-03},
			{8.1539e-05, -9.0790e-04, 2.3956e-03 /*...,*/, 9.7046e-03,
				2.1057e-03, -3.3417e-03},
			{7.9956e-03, -6.7139e-03, -6.6757e-04 /*...,*/, 5.0354e-03,
				2.9449e-03, -3.7231e-03},
			/*  ...,*/
			{-1.0376e-02, 4.0283e-02, -6.9336e-02 /*...,*/, -9.7656e-02,
				-3.1494e-02, -2.0752e-02},
			{-6.5918e-03, 2.0386e-02, -6.9885e-03 /*...,*/, -6.9580e-03,
				7.9632e-05, -4.5776e-03},
			{2.9144e-03, 1.0498e-02, -9.3384e-03 /*...,*/, 1.4191e-03,
				-9.7656e-03, 4.1809e-03},
		},

		{{7.6904e-03, -2.2430e-03, 3.8452e-03 /*...,*/, 6.8054e-03,
			1.2512e-03, 2.6245e-03},
			{3.4332e-03, 1.0986e-03, 1.6708e-03 /*...,*/, -5.3101e-03,
				-2.9144e-03, -4.0283e-03},
			{-7.9956e-03, 5.7373e-03, 8.3618e-03 /*...,*/, 1.8921e-03,
				7.5073e-03, 6.9885e-03},
			/*...,*/
			{-1.4954e-03, -2.1362e-02, 3.1738e-02 /*...,*/, 1.7624e-03,
				-2.0752e-03, 1.6724e-02},
			{1.0498e-02, -4.1199e-03, -1.6632e-03 /*...,*/, 6.0730e-03,
				-5.2795e-03, 1.2512e-03},
			{6.6223e-03, -1.5503e-02, -4.6997e-03 /*...,*/, -1.2894e-03,
				2.1118e-02, 2.1210e-03},
		},

		{
			{-7.7057e-04, -5.3406e-03, 1.1292e-02 /*...,*/, -5.9814e-03,
				5.8746e-04, -2.7084e-04},
			{-7.4768e-03, 1.3733e-03, -3.0212e-03 /*...,*/, 2.1820e-03,
				6.9275e-03, 7.2327e-03},
			{-1.0147e-03, -8.3618e-03, 4.1199e-03 /*...,*/, -8.4229e-03,
				-9.3994e-03, -6.9580e-03},
			/*...,*/
			{-9.4604e-03, -5.7373e-02, 2.2278e-03 /*...,*/, -5.6396e-02,
				-1.0071e-02, 1.3367e-02},
			{3.6926e-03, -3.4332e-03, 2.9449e-03 /*...,*/, -1.0254e-02,
				-1.2878e-02, 3.1738e-03},
			{-4.2343e-04, 9.5749e-04, -8.3618e-03 /*...,*/, 7.9346e-04,
				-2.4261e-03, -4.1199e-03},
		},

		{
			{3.1586e-03, 1.7166e-03, 6.1035e-04 /*...,*/, 6.3782e-03,
				2.3193e-03, -9.2506e-05},
			{-5.4321e-03, 1.8082e-03, -6.9275e-03 /*...,*/, 1.4420e-03,
				-4.1504e-03, 2.6131e-04},
			{-2.6131e-04, 1.1444e-03, -3.2663e-05 /*...,*/, 1.2112e-04,
				5.2490e-03, -1.1597e-03},
			/* ...,*/
			{-8.8501e-03, -2.4414e-02, -3.0884e-02 /*...,*/, 2.1606e-02,
				-4.6387e-03, -2.1729e-02},
			{1.7853e-03, 3.9062e-03, -7.5150e-04 /*...,*/, -2.8610e-04,
				5.4626e-03, 1.2207e-03},
			{2.9449e-03, -1.0529e-03, -6.4697e-03 /*...,*/, -3.0136e-04,
				5.5847e-03, 3.4790e-03},
		},
	}

	if actualXq, err = actualXq.Reshape([]int{sequenceLength, attention.N_Heads, attention.HeadDim}); err != nil {
		t.Fatal(err)
	}

	if actualXk, err = actualXk.Reshape([]int{sequenceLength, attention.N_KVHeads, attention.HeadDim}); err != nil {
		t.Fatal(err)
	}

	if actualXv, err = actualXv.Reshape([]int{sequenceLength, attention.N_KVHeads, attention.HeadDim}); err != nil {
		t.Fatal(err)
	}

	if err := ml.CompareTestTensor(expectedXqRs, expectedXqRsSize, actualXq, 4*common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}

	if err := ml.CompareTestTensor(expectedXkRs, expectedXkRsSize, actualXk, 4*common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}

	if err := ml.CompareTestTensor(expectedXvRs, expectedXvRsSize, actualXv, 4*common.THRESHOLD_BF16, true); err != nil {
		t.Fatal(err)
	}
	return nil
}

func testTransformerBlock_Forward(t *testing.T, firstLayer *LlamaTransformerBlock, x *ml.Tensor, startPos int, freqsCis *ml.Tensor, mask *ml.Tensor) *ml.Tensor {
	/*
		h, err := ltb.attention.Forward(context, normalizedX, startPos, freqsCis, mask)
	*/
	normalizedX := testTransformerBlock_AttnNorm_Forward(t, firstLayer, x)
	h := testTransformerBlock_Attention_Forward(t, firstLayer.attention, normalizedX, startPos, freqsCis, mask)
	return h
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
	currentTensor = testTransformerBlock_Forward(t, firstLayer, currentTensor, startPos, actualFreqsCis, actualMask)
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

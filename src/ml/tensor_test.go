package ml

import (
	"testing"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
)

func TestCheckBroadcastableOnce(t *testing.T) {
	expected := true
	actual := CheckBroadcastableOnce([]int{6, 3}, []int{2, 1})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = false
	actual = CheckBroadcastableOnce([]int{2, 1}, []int{6, 3})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = true
	actual = CheckBroadcastableOnce([]int{5, 2}, []int{5, 2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = false
	actual = CheckBroadcastableOnce([]int{5, 3}, []int{5, 2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = false
	actual = CheckBroadcastableOnce([]int{2}, []int{5, 2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = true
	actual = CheckBroadcastableOnce([]int{5, 2}, []int{2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}
}

func TestReshape(t *testing.T) {
	input, err := createTestInputTensor([]int{2, 6})
	if err != nil {
		t.Error(err)
	}

	expectedSize_input := []int{2, 6}
	expected_input := [][]float32{
		{1., 2., 3., 4., 5., 6.},
		{7., 8., 9., 10., 11., 12.},
	}

	if err := CompareTestTensor(expected_input, expectedSize_input, input, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}

	expectedSize_3_4 := []int{3, 4}
	expected_3_4 := [][]float32{
		{1., 2., 3., 4.},
		{5., 6., 7., 8.},
		{9., 10., 11., 12.},
	}
	actual_3_4, err := input.Reshape([]int{3, 4})
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected_3_4, expectedSize_3_4, actual_3_4, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}

	expectedSize_1_2_3_2 := []int{1, 2, 3, 2}
	expected_1_2_3_2 := [][][][]float32{
		{
			{
				{1., 2.},
				{3., 4.},
				{5., 6.},
			},
			{
				{7., 8.},
				{9., 10.},
				{11., 12.},
			},
		},
	}
	actual_1_2_3_2, err := input.Reshape([]int{1, 2, 3, 2})
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected_1_2_3_2, expectedSize_1_2_3_2, actual_1_2_3_2, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}

	expectedSize_4_3 := []int{4, 3}
	expected_4_3 := [][]float32{
		{1., 2., 3.},
		{4., 5., 6.},
		{7., 8., 9.},
		{10., 11., 12.},
	}
	actual_4_3, err := actual_1_2_3_2.Reshape([]int{4, 3})
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected_4_3, expectedSize_4_3, actual_4_3, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}

	expected_5_3 := "shape [5 3] is invalid for input of element count 12"
	actual_5_3, err := input.Reshape([]int{5, 3})
	if err == nil {
		t.Errorf("error expected, but got %v", actual_5_3)
	} else if err.Error() != expected_5_3 {
		t.Errorf("error expected \"%s\", but got error \"%s\"", expected_5_3, err)
	}
}

func TestSetSlice(t *testing.T) {
	input, err := createTestInputTensor([]int{4, 5})
	if err != nil {
		t.Error(err)
	}

	expectedSize := []int{4, 5}
	expected := [][]float32{
		{1., 2., 3., 4., 5.},
		{6., 7., 8., 9., 10.},
		{11., 12., 13., 14., 15.},
		{16., 17., 18., 19., 20.},
	}

	actual := NewEmptyTensor([]int{10, 5}, DT_BF16)
	if err := actual.SetSlice([]int{1}, []int{5}, input); err != nil {
		t.Error(err)
	}

	actualSlice, err := actual.Slice([]int{1}, []int{5})
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected, expectedSize, actualSlice, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}

	actual = NewEmptyTensor([]int{20, 10, 5}, DT_BF16)
	if err := actual.SetSlice([]int{19, 1}, []int{19, 5}, input); err != nil {
		t.Error(err)
	}

	actualSlice, err = actual.Slice([]int{19, 1}, []int{19, 5})
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected, expectedSize, actualSlice, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}
}

func TestTranspose_Simple(t *testing.T) {
	// [2, 3, 3]
	inputVals := [][][]float32{
		{
			{1., 2., 3.},
			{4., 5., 6.},
			{7., 8., 9.},
		},

		{
			{10., 11., 12.},
			{13., 14., 15.},
			{16., 17., 18.},
		},
	}

	expectedSize := []int{3, 2, 3}
	expected := [][][]float32{
		{
			{1., 2., 3.},
			{10., 11., 12.},
		},

		{
			{4., 5., 6.},
			{13., 14., 15.},
		},

		{
			{7., 8., 9.},
			{16., 17., 18.},
		},
	}

	actual := NewEmptyTensor([]int{2, 3, 3}, DT_F32)
	for iterator := IterateOver(actual, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := actual.SetItem(loc, inputVals[loc[0]][loc[1]][loc[2]]); err != nil {
			t.Error(err)
		}
	}

	var err error
	if actual, err = actual.Transpose(0, 1); err != nil { // transpose shape [2, 3, 3] to -> [3, 2, 3]
		t.Error(err)
	}

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}
}

func TestTranspose_Simple_Dim1_Dim3(t *testing.T) {
	// [2, 3, 1, 5]
	inputVals := [][][][]float32{
		{
			{
				{1., 2., 3., 4., 5.},
			},
			{
				{6., 7., 8., 9., 10.},
			},
			{
				{11., 12., 13., 14., 15.},
			},
		},
		{
			{
				{16., 17., 18., 19., 20.},
			},
			{
				{21., 22., 23., 24., 25.},
			},
			{
				{26., 27., 28., 39., 30.},
			},
		},
	}

	expectedSize := []int{2, 5, 1, 3}
	expected := [][][][]float32{
		{
			{
				{1., 6., 11.},
			},

			{
				{2., 7., 12.},
			},

			{
				{3., 8., 13.},
			},

			{
				{4., 9., 14.},
			},

			{
				{5., 10., 15.},
			},
		},

		{
			{
				{16., 21., 26.},
			},

			{
				{17., 22., 27.},
			},

			{
				{18., 23., 28.},
			},

			{
				{19., 24., 39.},
			},

			{
				{20., 25., 30.},
			},
		},
	}

	actual := NewEmptyTensor([]int{2, 3, 1, 5}, DT_F32)
	for iterator := IterateOver(actual, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := actual.SetItem(loc, inputVals[loc[0]][loc[1]][loc[2]][loc[3]]); err != nil {
			t.Fatal(err)
		}
	}

	expTensor := NewEmptyTensor([]int{2, 5, 1, 3}, DT_F32)
	for iterator := IterateOver(expTensor, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := expTensor.SetItem(loc, expected[loc[0]][loc[1]][loc[2]][loc[3]]); err != nil {
			t.Error(err)
		}
	}

	var err error
	if actual, err = actual.Transpose(1, 3); err != nil { // transpose shape [2, 3, 1, 5] to -> [2, 5, 1, 3]
		t.Error(err)
	}

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}
}

func TestTranspose_Large(t *testing.T) {
	// [5, 6, 6]
	inputVals := [][][]float32{
		{
			{0.1157, -0.4805, -0.4180, 1.6875, 0.6250, -0.3145},
			{-0.1426, -1.0078, 0.2930, -0.6953, 0.0913, 0.5508},
			{-0.0281, 0.1206, 0.3184, 0.3574, -0.3418, 0.3496},
			{0.6992, 1.4062, -1.6094, 0.5352, -0.2256, 1.2891},
			{0.7344, 1.9141, 1.0312, 0.5234, -0.3809, 0.4863},
			{-1.3672, 0.7383, 1.3828, 0.6250, -0.1670, 0.1602},
		},

		{
			{1.6328, -0.7383, -2.1094, 0.5273, -1.1094, -0.7930},
			{0.0150, 0.0923, -0.0737, -0.9297, 0.1631, 0.5664},
			{-0.1973, 0.8008, -0.5078, 0.3223, -0.2773, 0.3516},
			{-2.2031, 2.0156, -2.4062, 0.9219, -1.4141, 2.1094},
			{-1.6172, 2.0156, 2.4688, 0.5547, -0.4023, 0.5703},
			{-0.9023, -0.5234, -0.5391, 0.3789, -0.1729, -0.5195},
		},

		{
			{1.4375, 0.7422, -1.3125, 0.2871, -0.8203, -0.7422},
			{-0.0255, 0.3516, -0.0381, -0.1309, -0.6719, -0.1494},
			{-0.5508, -0.8242, -0.4004, -0.1084, 0.0903, -0.0830},
			{-1.3672, -0.1416, -0.1631, 0.7578, -0.3613, 1.1875},
			{-2.6094, -0.5586, 1.0547, 0.6172, -0.4570, 0.6367},
			{0.3535, -1.7891, -0.3477, 0.3652, 0.0255, -0.1855},
		},

		{
			{0.0613, 1.8125, 0.0732, 0.7383, -1.0000, -1.0156},
			{-0.0879, 0.0898, -0.0850, -0.5312, -0.0591, 0.2129},
			{-0.4512, -0.0640, -0.4258, 0.0737, -0.0613, 0.0986},
			{-0.7812, -1.7891, 1.3594, 0.7461, -0.7031, 1.5938},
			{-1.1953, -2.4688, -0.9141, 0.6016, -0.4414, 0.6211},
			{1.2031, -0.6484, 0.2012, 0.1611, -0.0103, -0.2363},
		},

		{
			{-0.9531, 0.8086, 0.9141, 0.3301, -0.9062, -0.7461},
			{-0.5508, -0.1206, -0.1953, 0.1914, -0.9492, -0.3809},
			{0.8672, 0.0718, 0.9180, -0.2754, 0.2207, -0.2461},
			{0.2637, -0.6914, 0.5547, 0.5547, -0.1299, 0.7070},
			{1.6484, -2.1562, -2.3281, 0.6289, -0.4688, 0.6445},
			{1.6094, 1.2344, 1.3750, 0.1689, 0.1807, 0.0957},
		},
	}

	expectedSize := []int{6, 5, 6}
	expected := [][][]float32{
		{
			{0.1157, -0.4805, -0.4180, 1.6875, 0.6250, -0.3145},
			{1.6328, -0.7383, -2.1094, 0.5273, -1.1094, -0.7930},
			{1.4375, 0.7422, -1.3125, 0.2871, -0.8203, -0.7422},
			{0.0613, 1.8125, 0.0732, 0.7383, -1.0000, -1.0156},
			{-0.9531, 0.8086, 0.9141, 0.3301, -0.9062, -0.7461},
		},

		{
			{-0.1426, -1.0078, 0.2930, -0.6953, 0.0913, 0.5508},
			{0.0150, 0.0923, -0.0737, -0.9297, 0.1631, 0.5664},
			{-0.0255, 0.3516, -0.0381, -0.1309, -0.6719, -0.1494},
			{-0.0879, 0.0898, -0.0850, -0.5312, -0.0591, 0.2129},
			{-0.5508, -0.1206, -0.1953, 0.1914, -0.9492, -0.3809},
		},

		{
			{-0.0281, 0.1206, 0.3184, 0.3574, -0.3418, 0.3496},
			{-0.1973, 0.8008, -0.5078, 0.3223, -0.2773, 0.3516},
			{-0.5508, -0.8242, -0.4004, -0.1084, 0.0903, -0.0830},
			{-0.4512, -0.0640, -0.4258, 0.0737, -0.0613, 0.0986},
			{0.8672, 0.0718, 0.9180, -0.2754, 0.2207, -0.2461},
		},

		{
			{0.6992, 1.4062, -1.6094, 0.5352, -0.2256, 1.2891},
			{-2.2031, 2.0156, -2.4062, 0.9219, -1.4141, 2.1094},
			{-1.3672, -0.1416, -0.1631, 0.7578, -0.3613, 1.1875},
			{-0.7812, -1.7891, 1.3594, 0.7461, -0.7031, 1.5938},
			{0.2637, -0.6914, 0.5547, 0.5547, -0.1299, 0.7070},
		},

		{
			{0.7344, 1.9141, 1.0312, 0.5234, -0.3809, 0.4863},
			{-1.6172, 2.0156, 2.4688, 0.5547, -0.4023, 0.5703},
			{-2.6094, -0.5586, 1.0547, 0.6172, -0.4570, 0.6367},
			{-1.1953, -2.4688, -0.9141, 0.6016, -0.4414, 0.6211},
			{1.6484, -2.1562, -2.3281, 0.6289, -0.4688, 0.6445},
		},

		{
			{-1.3672, 0.7383, 1.3828, 0.6250, -0.1670, 0.1602},
			{-0.9023, -0.5234, -0.5391, 0.3789, -0.1729, -0.5195},
			{0.3535, -1.7891, -0.3477, 0.3652, 0.0255, -0.1855},
			{1.2031, -0.6484, 0.2012, 0.1611, -0.0103, -0.2363},
			{1.6094, 1.2344, 1.3750, 0.1689, 0.1807, 0.0957},
		},
	}

	actual := NewEmptyTensor([]int{5, 6, 6}, DT_F32)
	for iterator := IterateOver(actual, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := actual.SetItem(loc, inputVals[loc[0]][loc[1]][loc[2]]); err != nil {
			t.Error(err)
		}
	}
	var err error
	if actual, err = actual.Transpose(0, 1); err != nil { // transpose shape [5, 6, 6] to -> [6, 5, 6]
		t.Error(err)
	}

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_EXACT, false); err != nil {
		t.Error(err)
	}
}

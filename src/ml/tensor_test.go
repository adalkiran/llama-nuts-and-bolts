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

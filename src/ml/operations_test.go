package ml

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

func createTestInputTensorDimension(tensor *Tensor, currentDimension int, loc []int, cnt *float32) error {
	loc = append(loc, 0)
	if currentDimension < len(tensor.Size)-1 {
		for i := 0; i < tensor.Size[currentDimension]; i++ {
			loc[currentDimension] = i
			if err := createTestInputTensorDimension(tensor, currentDimension+1, loc, cnt); err != nil {
				return err
			}
		}
	} else {
		for i := 0; i < tensor.Size[currentDimension]; i++ {
			loc[currentDimension] = i
			if err := tensor.SetItem(loc, dtype.BFloat16fromFloat32(*cnt)); err != nil {
				return err
			}
			*cnt++
		}
	}
	return nil
}

func createTestInputTensor(size []int) (*Tensor, error) {
	tensor := NewEmptyTensor(size, DT_BF16)
	cnt := float32(1)
	if err := createTestInputTensorDimension(tensor, 0, []int{}, &cnt); err != nil {
		return nil, err
	}
	return tensor, nil
}

func compareTestTensorDimension(expected interface{}, actual *Tensor, currentDimension int, loc []int) error {
	loc = append(loc, 0)
	if currentDimension < len(actual.Size)-1 {
		for i := 0; i < actual.Size[currentDimension]; i++ {
			loc[currentDimension] = i
			if err := compareTestTensorDimension(expected, actual, currentDimension+1, loc); err != nil {
				return err
			}
		}
	} else {
		expectedArr := reflect.ValueOf(expected)
		for dimension := 0; dimension < currentDimension; dimension++ {
			expectedArr = expectedArr.Index(loc[dimension])
		}
		if len(actual.Size) > 0 {
			expectedArrTyped := expectedArr.Interface().([]float32)
			for i := 0; i < actual.Size[currentDimension]; i++ {
				loc[currentDimension] = i
				actualItem, err := actual.GetItem(loc)
				if err != nil {
					return err
				}
				actualItemTyped := actualItem.(dtype.BFloat16)
				expectedItemTyped := expectedArrTyped[i]

				if actualItemTyped.Float32() != expectedItemTyped {
					return fmt.Errorf("Expected %g, but got %g at index: %v", expectedItemTyped, actualItemTyped.Float32(), loc)
				}
			}
		} else {
			expectedItemTyped, ok := expectedArr.Interface().(float32)
			if !ok {
				return fmt.Errorf("Expected type float32, but got given expected argument %v", expectedArr.Type())
			}
			actualItemTyped := actual.Item().(dtype.BFloat16)
			if actualItemTyped.Float32() != expectedItemTyped {
				return fmt.Errorf("Expected %g, but got %g at index: %v", expectedItemTyped, actualItemTyped.Float32(), loc)
			}
		}
	}
	return nil
}

func compareTestTensor(expected interface{}, expectedSize []int, actual *Tensor) error {
	if !reflect.DeepEqual(expectedSize, actual.Size) {
		return fmt.Errorf("Expected size %v, but got %v", expectedSize, actual.Size)
	}
	expectedArr := reflect.ValueOf(expected)
	for dimension := 0; dimension < len(expectedSize); dimension++ {
		if expectedArr.Kind() != reflect.Slice {
			return fmt.Errorf("given expected argument's dimension is not compatible: expected dimension: %d, current dimension: %d", len(expectedSize), dimension)
		}
		expectedArr = expectedArr.Index(0)
	}
	if len(expectedSize) > 0 && expectedArr.Kind() == reflect.Slice {
		return fmt.Errorf("given expected argument's dimension is not compatible: expected dimension: %d", len(expectedSize))
	}

	if err := compareTestTensorDimension(expected, actual, 0, []int{}); err != nil {
		return err
	}
	return nil
}

func TestMean3dKeepDimTrue(t *testing.T) {
	tensor, err := createTestInputTensor([]int{5, 4, 3})
	if err != nil {
		t.Error(err)
	}
	expectedSize := []int{5, 4, 1}
	expected := [][][]float32{
		{
			{2.}, {5.}, {8.}, {11.},
		},
		{
			{14.}, {17.}, {20.}, {23.},
		},
		{
			{26.}, {29.}, {32.}, {35.},
		},
		{
			{38.}, {41.}, {44.}, {47.},
		},
		{
			{50.}, {53.}, {56.}, {59.},
		},
	}
	actual, err := Mean(tensor, -1, true)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestMean3dKeepDimFalse(t *testing.T) {
	tensor, err := createTestInputTensor([]int{5, 4, 3})
	if err != nil {
		t.Error(err)
	}
	expectedSize := []int{5, 4}
	expected := [][]float32{
		{2., 5., 8., 11.},
		{14., 17., 20., 23.},
		{26., 29., 32., 35.},
		{38., 41., 44., 47.},
		{50., 53., 56., 59.},
	}
	actual, err := Mean(tensor, -1, false)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestMean2dKeepDimTrue(t *testing.T) {
	tensor, err := createTestInputTensor([]int{7, 9})
	if err != nil {
		t.Error(err)
	}
	expectedSize := []int{7, 1}
	expected := [][]float32{
		{5.}, {14.}, {23.}, {32.}, {41.}, {50.}, {59.},
	}
	actual, err := Mean(tensor, -1, true)
	if err != nil {
		t.Error(err)
	}
	fmt.Printf("%s", actual.StringLong())

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestMean2dKeepDimFalse(t *testing.T) {
	tensor, err := createTestInputTensor([]int{7, 9})
	if err != nil {
		t.Error(err)
	}
	expectedSize := []int{7}
	expected := []float32{
		5., 14., 23., 32., 41., 50., 59.,
	}
	actual, err := Mean(tensor, -1, false)
	if err != nil {
		t.Error(err)
	}
	fmt.Printf("%s", actual.StringLong())

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestMean1dKeepDimTrue(t *testing.T) {
	tensor, err := createTestInputTensor([]int{15})
	if err != nil {
		t.Error(err)
	}
	expectedSize := []int{1}
	expected := []float32{
		8.,
	}
	actual, err := Mean(tensor, -1, true)
	if err != nil {
		t.Error(err)
	}
	fmt.Printf("%s", actual.StringLong())

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestMean1dKeepDimFalse(t *testing.T) {
	tensor, err := createTestInputTensor([]int{15})
	if err != nil {
		t.Error(err)
	}
	expectedSize := []int{}
	expected := float32(8.)
	actual, err := Mean(tensor, -1, false)
	if err != nil {
		t.Error(err)
	}
	fmt.Printf("%s", actual.StringLong())

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

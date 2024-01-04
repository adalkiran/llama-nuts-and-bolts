package ml

import (
	"fmt"
	"math"
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
			if actual.DataType != DT_COMPLEX {
				expectedArrTyped, ok := expectedArr.Interface().([]float32)
				if !ok {
					return fmt.Errorf("given expected argument is in unsupported datatype %s, should be %s", reflect.TypeOf(expectedArr.Interface()), "float32")
				}
				for i := 0; i < actual.Size[currentDimension]; i++ {
					loc[currentDimension] = i
					actualItem, err := actual.GetItem(loc)
					if err != nil {
						return err
					}
					var actualItemTyped float32
					switch actualItem := actualItem.(type) {
					case dtype.BFloat16:
						actualItemTyped = actualItem.Float32()
					case float32:
						actualItemTyped = actualItem
					default:
						return fmt.Errorf("unsupported tensor datatype %s in compareTestTensorDimension function", reflect.TypeOf(actualItem))
					}
					expectedItemTyped := expectedArrTyped[i]

					if actualItemTyped != expectedItemTyped {
						return fmt.Errorf("Expected %g, but got %g at index: %v", expectedItemTyped, actualItemTyped, loc)
					}
				}
			} else {
				expectedArrTyped, ok := expectedArr.Interface().([]complex64)
				if !ok {
					return fmt.Errorf("given expected argument is in unsupported datatype %s, should be %s", reflect.TypeOf(expectedArr.Interface()), "complex64")
				}
				for i := 0; i < actual.Size[currentDimension]; i++ {
					loc[currentDimension] = i
					actualItem, err := actual.GetItem(loc)
					if err != nil {
						return err
					}
					actualItemTyped := actualItem.(complex64)
					expectedItemTyped := expectedArrTyped[i]

					// Comparison is done by converting values to string, because complex64 type may have
					// some slight differences due to high precision.
					if fmt.Sprintf("%.4e", actualItemTyped) != fmt.Sprintf("%.4e", expectedItemTyped) {
						return fmt.Errorf("Expected %g, but got %g at index: %v", expectedItemTyped, actualItemTyped, loc)
					}
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
	if len(expectedSize) > 0 && expectedArr.Kind() != reflect.Slice {
		return fmt.Errorf("given expected argument is not a slice/array, got %v", expectedArr.Type())
	}
	for dimension := 0; dimension < len(expectedSize); dimension++ {
		if expectedArr.Kind() != reflect.Slice {
			return fmt.Errorf("given expected argument's dimension is not compatible: expected dimension: %d, current dimension: %d", len(expectedSize), dimension)
		}
		if expectedArr.Len() != expectedSize[dimension] {
			return fmt.Errorf("given expected argument's shape is not compatible: expected length at dimension %d: %d, got length: %d", dimension, expectedSize[dimension], expectedArr.Len())
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

func TestARangeStep1BF16(t *testing.T) {
	expectedSize := []int{10}
	expected := []float32{
		0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
	}
	actual, err := ARange(0, 10, 1, DT_BF16)
	if err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestARangeMultipleCasesBF16(t *testing.T) {
	// Case 1
	expectedSize := []int{4}
	expected := []float32{
		1., 2., 3., 4.,
	}
	actual, err := ARange(1, 5, 1, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 2
	expectedSize = []int{7}
	expected = []float32{
		22., 23., 24., 25., 26., 27., 28.,
	}
	actual, err = ARange(22, 29, 1, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 3
	expectedSize = []int{7}
	expected = []float32{
		0., 2., 4., 6., 8., 10., 12.,
	}
	actual, err = ARange(0, 13, 2, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 4
	expectedSize = []int{6}
	expected = []float32{
		1., 3., 5., 7., 9., 11.,
	}
	actual, err = ARange(1, 13, 2, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 5
	expectedSize = []int{4}
	expected = []float32{
		1., 4., 7., 10.,
	}
	actual, err = ARange(1, 13, 3, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 6
	expectedSize = []int{6}
	expected = []float32{
		-4., -1., 2., 5., 8., 11.,
	}
	actual, err = ARange(-4, 13, 3, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 7
	expectedSize = []int{10}
	expected = []float32{
		-10., -5., 0., 5., 10., 15., 20., 25., 30., 35.,
	}
	actual, err = ARange(-10, 40, 5, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 8
	expectedSize = []int{4}
	expected = []float32{
		43., 48., 53., 58.,
	}
	actual, err = ARange(43, 59, 5, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestOuter(t *testing.T) {
	// See: (Sample) https://pytorch.org/docs/stable/generated/torch.outer.html
	expectedSize := []int{4, 3}
	expected := [][]float32{
		{1., 2., 3.},
		{2., 4., 6.},
		{3., 6., 9.},
		{4., 8., 12.},
	}

	inputVec1, err := ARange(1, 5, 1, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	inputVec2, err := ARange(1, 4, 1, DT_BF16)
	if err != nil {
		t.Error(err)
	}

	actual, err := Outer(inputVec1, inputVec2)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestFull(t *testing.T) {
	// See: (Sample) https://pytorch.org/docs/stable/generated/torch.outer.html
	// See: https://cloud.google.com/tpu/docs/bfloat16

	// Case 1: Shows how BFloat16 truncates mantissa part.
	expectedSize := []int{4, 3}
	expected := [][]float32{
		{11.3125, 11.3125, 11.3125},
		{11.3125, 11.3125, 11.3125},
		{11.3125, 11.3125, 11.3125},
		{11.3125, 11.3125, 11.3125},
	}

	actual, err := Full([]int{4, 3}, DT_BF16, dtype.BFloat16fromFloat32(11.34))
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 2: Shows how BFloat16 truncates mantissa part.
	expectedSize = []int{4, 3}
	expected = [][]float32{
		{1.3359375, 1.3359375, 1.3359375},
		{1.3359375, 1.3359375, 1.3359375},
		{1.3359375, 1.3359375, 1.3359375},
		{1.3359375, 1.3359375, 1.3359375},
	}

	actual, err = Full([]int{4, 3}, DT_BF16, dtype.BFloat16fromFloat32(1.34))
	if err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 3
	expectedSize = []int{4, 3}
	expected = [][]float32{
		{11.34, 11.34, 11.34},
		{11.34, 11.34, 11.34},
		{11.34, 11.34, 11.34},
		{11.34, 11.34, 11.34},
	}

	actual, err = Full([]int{4, 3}, DT_F32, float32(11.34))
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestZeros(t *testing.T) {
	expectedSize := []int{4, 3}
	expected := [][]float32{
		{0., 0., 0.},
		{0., 0., 0.},
		{0., 0., 0.},
		{0., 0., 0.},
	}

	actual, err := Zeros([]int{4, 3}, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	expectedItemTyped := dtype.BFloat16fromFloat32(0)
	actualItem, err := actual.GetItem([]int{0, 0})
	if err != nil {
		t.Error(err)
	}
	actualItemTyped, ok := actualItem.(dtype.BFloat16)
	if !ok {
		t.Errorf("Expected item type BFloat16, but got %v (%v)", actualItem, reflect.TypeOf(actualItem))
	}
	if actualItemTyped != expectedItemTyped {
		t.Errorf("Expected %v, but got %v", expectedItemTyped, actualItemTyped)
	}
}

func TestOnes(t *testing.T) {
	expectedSize := []int{4, 3}
	expected := [][]float32{
		{1., 1., 1.},
		{1., 1., 1.},
		{1., 1., 1.},
		{1., 1., 1.},
	}

	actual, err := Ones([]int{4, 3}, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	expectedItemTyped := dtype.BFloat16fromFloat32(1)
	actualItem, err := actual.GetItem([]int{0, 0})
	if err != nil {
		t.Error(err)
	}
	actualItemTyped, ok := actualItem.(dtype.BFloat16)
	if !ok {
		t.Errorf("Expected item type BFloat16, but got %v (%v)", actualItem, reflect.TypeOf(actualItem))
	}
	if actualItemTyped != expectedItemTyped {
		t.Errorf("Expected %v, but got %v", expectedItemTyped, actualItemTyped)
	}
}

func TestPolar(t *testing.T) {
	//See for sample: https://www.geeksforgeeks.org/python-pytorch-torch-polar-function/

	// Currently, Polar function supports only 2D matrices.
	// Because of this, {1, 5} is used as tensor shapes.

	// Output in the sample webpage:
	// tensor([-1.0054e-06+23.0000j,  3.1820e+01+31.8198j,  3.3500e+01+58.0237j,
	// 			4.3687e+01+31.7404j,  3.2000e+01+0.0000j], dtype=torch.complex64)
	expectedSize := []int{1, 5}
	expected := [][]complex64{
		{
			complex64(complex(-1.0054e-06, 23.)),
			complex64(complex(3.1820e+01, 31.8198)),
			complex64(complex(3.3500e+01, 58.0237)),
			complex64(complex(4.3687e+01, 31.7404)),
			complex64(complex(3.2000e+01, 0.)),
		},
	}

	// create absolute lengths of 5 with float type
	// abs = torch.tensor([23, 45, 67, 54, 32], dtype=torch.float32)
	abs := NewEmptyTensor([]int{1, 5}, DT_F32)
	if err := processErrors(
		abs.SetItem([]int{0, 0}, float32(23.)),
		abs.SetItem([]int{0, 1}, float32(45.)),
		abs.SetItem([]int{0, 2}, float32(67.)),
		abs.SetItem([]int{0, 3}, float32(54.)),
		abs.SetItem([]int{0, 4}, float32(32.))); err != nil {
		t.Error(err)
	}
	// create 5 angles with float type
	// angle = torch.tensor([numpy.pi / 2, numpy.pi / 4, numpy.pi /
	//                      3, numpy.pi / 5, 0], dtype=torch.float32)
	angle := NewEmptyTensor([]int{1, 5}, DT_F32)
	if err := processErrors(
		angle.SetItem([]int{0, 0}, float32(math.Pi/2)),
		angle.SetItem([]int{0, 1}, float32(math.Pi/4)),
		angle.SetItem([]int{0, 2}, float32(math.Pi/3)),
		angle.SetItem([]int{0, 3}, float32(math.Pi/5)),
		angle.SetItem([]int{0, 4}, float32(0))); err != nil {
		t.Error(err)
	}
	actual, err := Polar(abs, angle)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestTriangularUpperOnSquare(t *testing.T) {
	// See: https://pytorch.org/docs/stable/generated/torch.triu.html

	originalInput, err := Full([]int{5, 5}, DT_F32, float32(5.))
	if err != nil {
		t.Error(err)
	}

	// Case 1: Diagonal=1 as argument
	expectedSize := []int{5, 5}
	expected := [][]float32{
		{0., 5., 5., 5., 5.},
		{0., 0., 5., 5., 5.},
		{0., 0., 0., 5., 5.},
		{0., 0., 0., 0., 5.},
		{0., 0., 0., 0., 0.},
	}

	actual, err := TriangularUpper(originalInput, 1)
	if err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 2: Diagonal=0 as argument
	expectedSize = []int{5, 5}
	expected = [][]float32{
		{5., 5., 5., 5., 5.},
		{0., 5., 5., 5., 5.},
		{0., 0., 5., 5., 5.},
		{0., 0., 0., 5., 5.},
		{0., 0., 0., 0., 5.},
	}

	if actual, err = TriangularUpper(originalInput, 0); err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 3: Diagonal=-1 as argument
	expectedSize = []int{5, 5}
	expected = [][]float32{
		{5., 5., 5., 5., 5.},
		{5., 5., 5., 5., 5.},
		{0., 5., 5., 5., 5.},
		{0., 0., 5., 5., 5.},
		{0., 0., 0., 5., 5.},
	}

	if actual, err = TriangularUpper(originalInput, -1); err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 4: Diagonal=-2 as argument
	expectedSize = []int{5, 5}
	expected = [][]float32{
		{5., 5., 5., 5., 5.},
		{5., 5., 5., 5., 5.},
		{5., 5., 5., 5., 5.},
		{0., 5., 5., 5., 5.},
		{0., 0., 5., 5., 5.},
	}

	if actual, err = TriangularUpper(originalInput, -2); err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestTriangularUpperOnLandscapeRectangle(t *testing.T) {
	// See: https://pytorch.org/docs/stable/generated/torch.triu.html

	originalInput, err := Full([]int{4, 7}, DT_F32, float32(5.))
	if err != nil {
		t.Error(err)
	}

	// Case 1: Diagonal=1 as argument
	expectedSize := []int{4, 7}
	expected := [][]float32{
		{0., 5., 5., 5., 5., 5., 5.},
		{0., 0., 5., 5., 5., 5., 5.},
		{0., 0., 0., 5., 5., 5., 5.},
		{0., 0., 0., 0., 5., 5., 5.},
	}

	actual, err := TriangularUpper(originalInput, 1)
	if err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 2: Diagonal=0 as argument
	expectedSize = []int{4, 7}
	expected = [][]float32{
		{5., 5., 5., 5., 5., 5., 5.},
		{0., 5., 5., 5., 5., 5., 5.},
		{0., 0., 5., 5., 5., 5., 5.},
		{0., 0., 0., 5., 5., 5., 5.},
	}

	if actual, err = TriangularUpper(originalInput, 0); err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 3: Diagonal=-1 as argument
	expectedSize = []int{4, 7}
	expected = [][]float32{
		{5., 5., 5., 5., 5., 5., 5.},
		{5., 5., 5., 5., 5., 5., 5.},
		{0., 5., 5., 5., 5., 5., 5.},
		{0., 0., 5., 5., 5., 5., 5.},
	}

	if actual, err = TriangularUpper(originalInput, -1); err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestTriangularUpperOnPortraitRectangle(t *testing.T) {
	// See: https://pytorch.org/docs/stable/generated/torch.triu.html

	originalInput, err := Full([]int{7, 4}, DT_F32, float32(5.))
	if err != nil {
		t.Error(err)
	}

	// Case 1: Diagonal=1 as argument
	expectedSize := []int{7, 4}
	expected := [][]float32{
		{0., 5., 5., 5.},
		{0., 0., 5., 5.},
		{0., 0., 0., 5.},
		{0., 0., 0., 0.},
		{0., 0., 0., 0.},
		{0., 0., 0., 0.},
		{0., 0., 0., 0.},
	}

	actual, err := TriangularUpper(originalInput, 1)
	if err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 2: Diagonal=0 as argument
	expectedSize = []int{7, 4}
	expected = [][]float32{
		{5., 5., 5., 5.},
		{0., 5., 5., 5.},
		{0., 0., 5., 5.},
		{0., 0., 0., 5.},
		{0., 0., 0., 0.},
		{0., 0., 0., 0.},
		{0., 0., 0., 0.},
	}

	if actual, err = TriangularUpper(originalInput, 0); err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 3: Diagonal=-1 as argument
	expectedSize = []int{7, 4}
	expected = [][]float32{
		{5., 5., 5., 5.},
		{5., 5., 5., 5.},
		{0., 5., 5., 5.},
		{0., 0., 5., 5.},
		{0., 0., 0., 5.},
		{0., 0., 0., 0.},
		{0., 0., 0., 0.},
	}

	if actual, err = TriangularUpper(originalInput, -1); err != nil {
		t.Error(err)
	}

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

func TestPow(t *testing.T) {
	// Case 1: BFloat16 type, 1D
	expectedSize := []int{5}
	expected := []float32{
		9., 16., 25., 36., 49.,
	}

	input, err := ARange(3, 8, 1, DT_BF16)
	if err != nil {
		t.Error(err)
	}
	actual, err := Pow(input, 2)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 2: float32 type, 1D
	expectedSize = []int{5}
	expected = []float32{
		9., 16., 25., 36., 49.,
	}

	input, err = ARange(3, 8, 1, DT_F32)
	if err != nil {
		t.Error(err)
	}
	actual, err = Pow(input, 2)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}

	// Case 3: float32 type, 3D
	expectedSize3d := []int{2, 3, 1}
	expected3d := [][][]float32{
		{{0.}, {1.}, {4.}},
		{{9.}, {16.}, {25.}},
	}

	input3d := NewEmptyTensor([]int{2, 3, 1}, DT_F32)

	if err := processErrors(
		input3d.SetItem([]int{0, 0, 0}, float32(0.)),
		input3d.SetItem([]int{0, 1, 0}, float32(1.)),
		input3d.SetItem([]int{0, 2, 0}, float32(2.)),
		input3d.SetItem([]int{1, 0, 0}, float32(3.)),
		input3d.SetItem([]int{1, 1, 0}, float32(4.)),
		input3d.SetItem([]int{1, 2, 0}, float32(5.))); err != nil {
		t.Error(err)
	}

	actual3d, err := Pow(input3d, 2)
	if err != nil {
		t.Error(err)
	}
	if err := compareTestTensor(expected3d, expectedSize3d, actual3d); err != nil {
		t.Error(err)
	}
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

	if err := compareTestTensor(expected, expectedSize, actual); err != nil {
		t.Error(err)
	}
}

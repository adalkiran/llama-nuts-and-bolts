package ml

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
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
			var val any
			switch tensor.DataType {
			case DT_BF16:
				val = dtype.BFloat16fromFloat32(*cnt)
			case DT_F32:
				val = *cnt
			default:
				return fmt.Errorf("unsupported tensor datatype %s", tensor.DataType)
			}
			if err := tensor.SetItem(loc, val); err != nil {
				return err
			}
			*cnt++
		}
	}
	return nil
}

func createTestInputTensorEx(size []int, dataType DataType) (*Tensor, error) {
	tensor := NewEmptyTensor(size, dataType)
	cnt := float32(1)
	if err := createTestInputTensorDimension(tensor, 0, []int{}, &cnt); err != nil {
		return nil, err
	}
	return tensor, nil
}

func createTestInputTensor(size []int) (*Tensor, error) {
	return createTestInputTensorEx(size, DT_BF16)
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected3d, expectedSize3d, actual3d, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
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

	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
		t.Error(err)
	}
}

func TestLinearTransformationF32(t *testing.T) {
	inputRowSize := 2
	inputColSize := 3

	weightsOutputSize := 4
	weightsInputSize := 3

	expectedSize := []int{2, 4}
	expected := [][]float32{
		{0.014, 0.032, 0.05, 0.068},
		{0.032, 0.077, 0.122, 0.167},
	}

	weightVals := [][]float32{
		{0.01, 0.02, 0.03},
		{0.04, 0.05, 0.06},
		{0.07, 0.08, 0.09},
		{0.10, 0.11, 0.12},
	}
	weights := NewEmptyTensor([]int{weightsOutputSize, weightsInputSize}, DT_F32)
	for iterator := IterateOver(weights, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := weights.SetItem(loc, weightVals[loc[0]][loc[1]]); err != nil {
			t.Error(err)
		}
	}

	inputVals := [][]float32{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}

	input := NewEmptyTensor([]int{inputRowSize, inputColSize}, DT_F32)
	for iterator := IterateOver(input, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := input.SetItem(loc, inputVals[loc[0]][loc[1]]); err != nil {
			t.Error(err)
		}
	}

	actual, err := LinearTransformation(input, weights)
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
		t.Error(err)
	}
}

func TestLinearTransformationBF16(t *testing.T) {
	inputRowSize := 2
	inputColSize := 3

	weightsOutputSize := 4
	weightsInputSize := 3

	expectedSize := []int{2, 4}
	expected := [][]float32{
		{0.0138, 0.0317, 0.0495, 0.0673},
		{0.0317, 0.0761, 0.1210, 0.1660},
	}

	weightVals := [][]dtype.BFloat16{
		{dtype.BFloat16fromFloat32(0.01), dtype.BFloat16fromFloat32(0.02), dtype.BFloat16fromFloat32(0.03)},
		{dtype.BFloat16fromFloat32(0.04), dtype.BFloat16fromFloat32(0.05), dtype.BFloat16fromFloat32(0.06)},
		{dtype.BFloat16fromFloat32(0.07), dtype.BFloat16fromFloat32(0.08), dtype.BFloat16fromFloat32(0.09)},
		{dtype.BFloat16fromFloat32(0.10), dtype.BFloat16fromFloat32(0.11), dtype.BFloat16fromFloat32(0.12)},
	}
	weights := NewEmptyTensor([]int{weightsOutputSize, weightsInputSize}, DT_BF16)
	for iterator := IterateOver(weights, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := weights.SetItem(loc, weightVals[loc[0]][loc[1]]); err != nil {
			t.Error(err)
		}
	}

	inputVals := [][]dtype.BFloat16{
		{dtype.BFloat16fromFloat32(0.1), dtype.BFloat16fromFloat32(0.2), dtype.BFloat16fromFloat32(0.3)},
		{dtype.BFloat16fromFloat32(0.4), dtype.BFloat16fromFloat32(0.5), dtype.BFloat16fromFloat32(0.6)},
	}

	input := NewEmptyTensor([]int{inputRowSize, inputColSize}, DT_BF16)
	for iterator := IterateOver(input, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := input.SetItem(loc, inputVals[loc[0]][loc[1]]); err != nil {
			t.Error(err)
		}
	}

	actual, err := LinearTransformation(input, weights)
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
		t.Error(err)
	}
}

func TestMatMulBF16(t *testing.T) {
	groupSize := 2
	inputRowSize := 2
	inputColSize := 3

	otherRowSize := 3
	otherColSize := 4

	expectedSize := []int{2, 2, 4}
	expected := [][][]float32{
		{
			{3.7598e-02, 4.3457e-02, 4.9561e-02, 5.5420e-02},
			{8.2520e-02, 9.7168e-02, 1.1230e-01, 1.2695e-01},
		},
		{
			{3.7598e-02, 4.3457e-02, 4.9561e-02, 5.5420e-02},
			{8.2520e-02, 9.7168e-02, 1.1230e-01, 1.2695e-01},
		},
	}

	inputVals := [][][]dtype.BFloat16{
		{
			{dtype.BFloat16fromFloat32(0.1), dtype.BFloat16fromFloat32(0.2), dtype.BFloat16fromFloat32(0.3)},
			{dtype.BFloat16fromFloat32(0.4), dtype.BFloat16fromFloat32(0.5), dtype.BFloat16fromFloat32(0.6)},
		},
		{
			{dtype.BFloat16fromFloat32(0.1), dtype.BFloat16fromFloat32(0.2), dtype.BFloat16fromFloat32(0.3)},
			{dtype.BFloat16fromFloat32(0.4), dtype.BFloat16fromFloat32(0.5), dtype.BFloat16fromFloat32(0.6)},
		},
	}

	input := NewEmptyTensor([]int{groupSize, inputRowSize, inputColSize}, DT_BF16)
	for iterator := IterateOver(input, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := input.SetItem(loc, inputVals[loc[0]][loc[1]][loc[2]]); err != nil {
			t.Error(err)
		}
	}

	otherVals := [][][]dtype.BFloat16{
		{
			{dtype.BFloat16fromFloat32(0.01), dtype.BFloat16fromFloat32(0.02), dtype.BFloat16fromFloat32(0.03), dtype.BFloat16fromFloat32(0.04)},
			{dtype.BFloat16fromFloat32(0.05), dtype.BFloat16fromFloat32(0.06), dtype.BFloat16fromFloat32(0.07), dtype.BFloat16fromFloat32(0.08)},
			{dtype.BFloat16fromFloat32(0.09), dtype.BFloat16fromFloat32(0.10), dtype.BFloat16fromFloat32(0.11), dtype.BFloat16fromFloat32(0.12)},
		},
		{
			{dtype.BFloat16fromFloat32(0.01), dtype.BFloat16fromFloat32(0.02), dtype.BFloat16fromFloat32(0.03), dtype.BFloat16fromFloat32(0.04)},
			{dtype.BFloat16fromFloat32(0.05), dtype.BFloat16fromFloat32(0.06), dtype.BFloat16fromFloat32(0.07), dtype.BFloat16fromFloat32(0.08)},
			{dtype.BFloat16fromFloat32(0.09), dtype.BFloat16fromFloat32(0.10), dtype.BFloat16fromFloat32(0.11), dtype.BFloat16fromFloat32(0.12)},
		},
	}
	other := NewEmptyTensor([]int{groupSize, otherRowSize, otherColSize}, DT_BF16)
	for iterator := IterateOver(other, 0); iterator.HasNext(); {
		loc := iterator.Next()
		if err := other.SetItem(loc, otherVals[loc[0]][loc[1]][loc[2]]); err != nil {
			t.Error(err)
		}
	}

	actual, err := MatMul(input, other)
	if err != nil {
		t.Error(err)
	}
	if err := CompareTestTensor(expected, expectedSize, actual, common.THRESHOLD_F32, false); err != nil {
		t.Error(err)
	}
}

package ml

import (
	"fmt"
	"reflect"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

func calculateShortenedExpectedIndex(i int, dimensionSize int, shorten bool) int {
	if shorten {
		const shortenCount = 3
		if i >= shortenCount && dimensionSize > 2*shortenCount {
			if i >= shortenCount && i < dimensionSize-shortenCount {
				return -1
			} else if dimensionSize-i-1 < shortenCount {
				return 2*shortenCount - dimensionSize + i
			}
		}
	}
	return i
}

func CompareTestTensorDimension(expected interface{}, actual *Tensor, currentDimension int, loc []int, floatThreshold float64, shorten bool) error {
	loc = append(loc, 0)
	if currentDimension < len(actual.Size)-1 {
		var expectedIdx int
		for i := 0; i < actual.Size[currentDimension]; i++ {
			if expectedIdx = calculateShortenedExpectedIndex(i, actual.Size[currentDimension], shorten); expectedIdx == -1 {
				continue
			}
			loc[currentDimension] = i
			if err := CompareTestTensorDimension(expected, actual, currentDimension+1, loc, floatThreshold, shorten); err != nil {
				return err
			}
		}
	} else {
		expectedArr := reflect.ValueOf(expected)
		for dimension := 0; dimension < currentDimension; dimension++ {
			expectedIdx := calculateShortenedExpectedIndex(loc[dimension], expectedArr.Len(), shorten)
			expectedArr = expectedArr.Index(expectedIdx)
		}
		if len(actual.Size) > 0 {
			if actual.DataType != DT_COMPLEX {
				expectedArrTyped, ok := expectedArr.Interface().([]float32)
				if !ok {
					return fmt.Errorf("given expected argument is in unsupported datatype %s, should be %s", reflect.TypeOf(expectedArr.Interface()), "float32")
				}
				var expectedIdx int
				for i := 0; i < actual.Size[currentDimension]; i++ {
					if expectedIdx = calculateShortenedExpectedIndex(i, actual.Size[currentDimension], shorten); expectedIdx == -1 {
						continue
					}
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
					expectedItemTyped := expectedArrTyped[expectedIdx]

					if !common.AlmostEqualFloat32(actualItemTyped, expectedItemTyped, floatThreshold) {
						return fmt.Errorf("expected %g, but got %g at index: %v", expectedItemTyped, actualItemTyped, loc)
					}
				}
			} else {
				expectedArrTyped, ok := expectedArr.Interface().([]complex64)
				if !ok {
					return fmt.Errorf("given expected argument is in unsupported datatype %s, should be %s", reflect.TypeOf(expectedArr.Interface()), "complex64")
				}
				var expectedIdx int
				for i := 0; i < actual.Size[currentDimension]; i++ {
					if expectedIdx = calculateShortenedExpectedIndex(i, actual.Size[currentDimension], shorten); expectedIdx == -1 {
						continue
					}
					loc[currentDimension] = i
					actualItem, err := actual.GetItem(loc)
					if err != nil {
						return err
					}
					actualItemTyped := actualItem.(complex64)
					expectedItemTyped := expectedArrTyped[expectedIdx]

					// Comparison is done by converting values to string, because complex64 type may have
					// some slight differences due to high precision.
					if fmt.Sprintf("%.4e", actualItemTyped) != fmt.Sprintf("%.4e", expectedItemTyped) {
						return fmt.Errorf("expected %g, but got %g at index: %v", expectedItemTyped, actualItemTyped, loc)
					}
				}
			}
		} else {
			expectedItemTyped, ok := expectedArr.Interface().(float32)
			if !ok {
				return fmt.Errorf("expected type float32, but got given expected argument %v", expectedArr.Type())
			}
			actualItemTyped := actual.Item().(dtype.BFloat16)
			if !common.AlmostEqualFloat32(actualItemTyped.Float32(), expectedItemTyped, floatThreshold) {
				return fmt.Errorf("expected %g, but got %g at index: %v", expectedItemTyped, actualItemTyped.Float32(), loc)
			}
		}
	}
	return nil
}

func CompareTestTensor(expected interface{}, expectedSize []int, actual *Tensor, floatThreshold float64, shorten bool) error {
	if !reflect.DeepEqual(expectedSize, actual.Size) {
		return fmt.Errorf("expected size %v, but got %v", expectedSize, actual.Size)
	}
	expectedArr := reflect.ValueOf(expected)
	if len(expectedSize) > 0 && expectedArr.Kind() != reflect.Slice {
		return fmt.Errorf("given expected argument is not a slice/array, got %v", expectedArr.Type())
	}
	for dimension := 0; dimension < len(expectedSize); dimension++ {
		if expectedArr.Kind() != reflect.Slice {
			return fmt.Errorf("given expected argument's dimension is not compatible: expected dimension: %d, current dimension: %d", len(expectedSize), dimension)
		}
		expectedSizeForDim := expectedSize[dimension]
		if shorten && expectedSizeForDim > 6 {
			expectedSizeForDim = 6
		}
		if expectedArr.Len() != expectedSizeForDim {
			return fmt.Errorf("given expected argument's shape is not compatible (shorten=%v): expected length at dimension %d: %d, got length: %d", shorten, dimension, expectedSizeForDim, expectedArr.Len())
		}
		expectedArr = expectedArr.Index(0)
	}
	if len(expectedSize) > 0 && expectedArr.Kind() == reflect.Slice {
		return fmt.Errorf("given expected argument's dimension is not compatible: expected dimension: %d", len(expectedSize))
	}
	if err := CompareTestTensorDimension(expected, actual, 0, []int{}, floatThreshold, shorten); err != nil {
		return err
	}
	return nil
}
package common

import (
	"fmt"
	"math"
)

const (
	THRESHOLD_F32  = 1e-4
	THRESHOLD_BF16 = 2e-2
)

func InterfaceToInt(val interface{}) (int, error) {
	if x, ok := val.(int); ok {
		return int(x), nil
	}
	if x, ok := val.(uint8); ok {
		return int(x), nil
	}
	if x, ok := val.(uint16); ok {
		return int(x), nil
	}
	if x, ok := val.(uint32); ok {
		return int(x), nil
	}
	if x, ok := val.(uint64); ok {
		return int(x), nil
	}
	if x, ok := val.(int8); ok {
		return int(x), nil
	}
	if x, ok := val.(int16); ok {
		return int(x), nil
	}
	if x, ok := val.(int32); ok {
		return int(x), nil
	}
	if x, ok := val.(int64); ok {
		return int(x), nil
	}
	return 0, fmt.Errorf("cannot interfaceToInt done with value %v", val)
}

func InterfaceArrToIntArr(arr []interface{}) ([]int, error) {
	result := make([]int, len(arr))
	for i, val := range arr {
		intVal, err := InterfaceToInt(val)
		if err != nil {
			return nil, err
		}
		result[i] = intVal
	}
	return result, nil
}

func InterfaceToBool(val interface{}, defaultValue bool) bool {
	intVal, err := InterfaceToInt(val)
	if err != nil {
		return defaultValue
	}
	return intVal == 1
}

func AlmostEqualFloat32(a float32, b float32, threshold float64) bool {
	if a == b {
		// This check is for -Inf and +Inf values
		return true
	}
	return math.Abs(float64(a)-float64(b)) <= threshold
}

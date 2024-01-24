package common

import (
	"fmt"
	"math"
	"sync"
	"unsafe"
)

const (
	THRESHOLD_EXACT = 0
	THRESHOLD_F32   = 1e-3
	THRESHOLD_BF16  = 1e-2
)

func WaitGroupDone(wg *sync.WaitGroup) chan struct{} {
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	return done
}

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

func DetermineMachineEndian() string {
	// See: https://stackoverflow.com/questions/51332658/any-better-way-to-check-endianness-in-go
	buf := [2]byte{}
	*(*uint16)(unsafe.Pointer(&buf[0])) = uint16(0xABCD)

	switch buf {
	case [2]byte{0xCD, 0xAB}:
		return "LITTLE_ENDIAN"
	case [2]byte{0xAB, 0xCD}:
		return "BIG_ENDIAN"
	default:
		panic("Could not determine native endianness.")
	}
}

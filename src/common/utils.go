package common

import (
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
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

func AlmostEqualFloat32(a float32, b float32, threshold float64) bool {
	if a == b {
		// This check is for -Inf and +Inf values
		return true
	}
	return math.Abs(float64(a)-float64(b)) <= threshold
}

func ReplaceHexWithChar(input string) string {
	regex := regexp.MustCompile(`<0x([0-9A-Fa-f]{1,2})>`)

	output := ""
	pos := 0
	waitingBytes := make([]byte, 0)
	for _, match := range regex.FindAllStringIndex(input, -1) {
		matchPos := match[0]
		matchEndPos := match[1]
		if matchPos-pos > 0 {
			if len(waitingBytes) > 0 {
				output += string(waitingBytes)
				waitingBytes = waitingBytes[:0]
			}
			output += input[pos : matchPos-pos+1]
		}
		hexStr := input[matchPos:matchEndPos][3 : matchEndPos-matchPos-1]
		hexInt, err := strconv.ParseInt(hexStr, 16, 32)
		if err != nil {
			continue
		}
		waitingBytes = append(waitingBytes, byte(hexInt))
		pos = matchEndPos
	}
	if len(waitingBytes) > 0 {
		output += string(waitingBytes)
	}
	return output
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
		FriendlyPanic(fmt.Errorf("could not determine native endianness"))
		return ""
	}
}

func FriendlyPanic(err error) {
	fmt.Printf("\n\nOops! Something went wrong:\n\nError: %s\n\n", err.Error())
	os.Exit(1)
}

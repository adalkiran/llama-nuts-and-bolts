package dtype

import (
	"fmt"
	"testing"
)

func TestNotTruncatedDecimalPart(t *testing.T) {
	// See: https://en.wikipedia.org/wiki/Floating-point_arithmetic
	// See: https://cloud.google.com/tpu/docs/bfloat16

	//Case 1
	expected := float32(1.)
	actualBF16 := BFloat16fromFloat32(1.)
	actual := actualBF16.Float32()
	if actual != expected {
		t.Errorf("Expected %g, but got %g", expected, actual)
	}

	//Case 2
	expected = float32(6.25)
	actualBF16 = BFloat16fromFloat32(6.25)
	actual = actualBF16.Float32()
	if actual != expected {
		t.Errorf("Expected %g, but got %g", expected, actual)
	}
}

func TestTruncatedDecimalPart(t *testing.T) {
	// See: https://en.wikipedia.org/wiki/Floating-point_arithmetic
	// See: https://cloud.google.com/tpu/docs/bfloat16

	// Note that, this  an expected behaviour.

	//Case 1
	expected := float32(1.5234375)
	actualBF16 := BFloat16fromFloat32(1.53)
	actual := actualBF16.Float32()
	if actual != expected {
		t.Errorf("Expected %g, but got %g", expected, actual)
	}

	//Case 2
	expected = float32(6.5)
	actualBF16 = BFloat16fromFloat32(6.53)
	actual = actualBF16.Float32()
	if actual != expected {
		t.Errorf("Expected %g, but got %g", expected, actual)
	}

	//Case 3
	expected = float32(11.3125)
	actualBF16 = BFloat16fromFloat32(11.34)
	actual = actualBF16.Float32()
	if actual != expected {
		t.Errorf("Expected %g, but got %g", expected, actual)
	}

	//Case 3
	expected = float32(584)
	actualBF16 = BFloat16fromFloat32(586.25)
	actual = actualBF16.Float32()
	if actual != expected {
		t.Errorf("Expected %g, but got %g", expected, actual)
	}
}

func TestReadBFloat16LittleEndian(t *testing.T) {
	// Values are stored in little endian order

	testCases := []map[string]interface{}{
		{
			"input":              []byte{0xA5, 0x35}, //0.0000012293458
			"expectedUInt16Bits": uint16(0x35A5),
			"expectedF32":        float32(0.0000012293458),
		},
		{
			"input":              []byte{0xF4, 0xB5}, //-0.00000181794167
			"expectedUInt16Bits": uint16(0xB5F4),
			"expectedF32":        float32(-0.00000181794167),
		},
		{
			"input":              []byte{0x92, 0xB6}, //-0.000004351139
			"expectedUInt16Bits": uint16(0xB692),
			"expectedF32":        float32(-0.000004351139),
		},
	}

	for _, testCase := range testCases {
		input := testCase["input"].([]byte)
		expectedUInt16Bits := testCase["expectedUInt16Bits"].(uint16)
		expectedF32 := testCase["expectedF32"].(float32)

		actualBF16 := ReadBFloat16LittleEndian(input)

		actualUInt16Bits := uint16(actualBF16)
		if actualUInt16Bits != expectedUInt16Bits {
			t.Errorf("Expected uint16 0x%X, but got 0x%X", expectedUInt16Bits, actualUInt16Bits)
		}

		actualF32 := actualBF16.Float32()
		if actualF32 != expectedF32 {
			t.Errorf("Expected float32 %g, but got %g", expectedF32, actualF32)
		}
	}
}

func TestBFloat16ToString(t *testing.T) {
	// Values are stored in little endian order
	testCases := []map[string]interface{}{
		{
			"input":     []byte{0xA5, 0x35}, //0.0000012293458
			"expectedF": "0.0000012293458",
		},
		{
			"input":     []byte{0xF4, 0xB5}, //-0.00000181794167
			"expectedF": "-0.0000018179417",
		},
		{
			"input":     []byte{0x92, 0xB6}, //-0.000004351139
			"expectedF": "-0.000004351139",
		},
	}

	for _, testCase := range testCases {
		input := testCase["input"].([]byte)
		expected := testCase["expectedF"].(string)

		actualBF16 := ReadBFloat16LittleEndian(input)
		actual := actualBF16.String()
		if actual != expected {
			t.Errorf("Expected %s, but got %s", expected, actual)
		}
	}
}

func TestBFloat16ToFmtString(t *testing.T) {
	// Values are stored in little endian order
	testCases := []map[string]interface{}{
		{
			"input":     []byte{0xA5, 0x35}, //0.0000012293458
			"expectedF": "0.000001",
			"expectedG": "1.2293458e-06",
		},
		{
			"input":     []byte{0xF4, 0xB5}, //-0.00000181794167
			"expectedF": "-0.000002",
			"expectedG": "-1.8179417e-06",
		},
		{
			"input":     []byte{0x92, 0xB6}, //-0.000004351139
			"expectedF": "-0.000004",
			"expectedG": "-4.351139e-06",
		},
	}

	for _, testCase := range testCases {
		input := testCase["input"].([]byte)
		expectedF := testCase["expectedF"].(string)
		expectedG := testCase["expectedG"].(string)

		actualBF16 := ReadBFloat16LittleEndian(input)

		actualF := fmt.Sprintf("%f", actualBF16.Float32())
		if actualF != expectedF {
			t.Errorf("Expected for %%s: %s, but got %s for input 0x%X", expectedF, actualF, input)
		}

		actualG := fmt.Sprintf("%g", actualBF16.Float32())
		if actualG != expectedG {
			t.Errorf("Expected for %%g: %s, but got %s for input 0x%X", expectedG, actualG, input)
		}

	}
}

package ml

import (
	"fmt"
	"math"
	"reflect"
	"unsafe"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

type DataTypeFuncSet interface {
	IsCompatible(val any) bool
	FromFloat32(val float32) any
	ToFloat32(val any) float32

	ToString(val any) string

	ReadItem(rawDataPtr unsafe.Pointer) any
	WriteItem(rawDataPtr unsafe.Pointer, val any) error

	ReadItem_AsFloat32(rawDataPtr unsafe.Pointer) float32
	WriteItem_FromFloat32(rawDataPtr unsafe.Pointer, val float32)
}

/*
	DataTypeFuncSet_BF16
*/

type DataTypeFuncSet_BF16 struct{}

func (dtfs DataTypeFuncSet_BF16) IsCompatible(val any) bool {
	_, ok := val.(dtype.BFloat16)
	return ok
}

func (dtfs DataTypeFuncSet_BF16) FromFloat32(val float32) any {
	return dtype.BFloat16fromFloat32(val)
}

func (dtfs DataTypeFuncSet_BF16) ToFloat32(val any) float32 {
	return val.(dtype.BFloat16).Float32()
}

func (dtfs DataTypeFuncSet_BF16) ToString(val any) string {
	if !dtfs.IsCompatible(val) {
		return fmt.Sprintf("%v", val)
	}
	return fmt.Sprintf("%.4e", val.(dtype.BFloat16).Float32())
}

func (dtfs DataTypeFuncSet_BF16) ReadItem(rawDataPtr unsafe.Pointer) any {
	return dtype.BFloat16(*(*uint16)(rawDataPtr))
}

func (dtfs DataTypeFuncSet_BF16) WriteItem(rawDataPtr unsafe.Pointer, val any) error {
	convVal, ok := val.(dtype.BFloat16)
	if !ok {
		return fmt.Errorf("incompatible types BFloat16 and %v", reflect.TypeOf(val))
	}
	*(*dtype.BFloat16)(rawDataPtr) = convVal
	return nil
}

func (dtfs DataTypeFuncSet_BF16) ReadItem_AsFloat32(rawDataPtr unsafe.Pointer) float32 {
	return dtype.BFloat16bitsToFloat32((*(*uint16)(rawDataPtr)))
}

func (dtfs DataTypeFuncSet_BF16) WriteItem_FromFloat32(rawDataPtr unsafe.Pointer, val float32) {
	*(*dtype.BFloat16)(rawDataPtr) = dtype.BFloat16fromFloat32(val)
}

/*
	DataTypeFuncSet_F32
*/

type DataTypeFuncSet_F32 struct{}

func (dtfs DataTypeFuncSet_F32) IsCompatible(val any) bool {
	_, ok := val.(float32)
	return ok
}

func (dtfs DataTypeFuncSet_F32) FromFloat32(val float32) any {
	return val
}

func (dtfs DataTypeFuncSet_F32) ToFloat32(val any) float32 {
	return val.(float32)
}

func (dtfs DataTypeFuncSet_F32) ToString(val any) string {
	if !dtfs.IsCompatible(val) {
		return fmt.Sprintf("%v", val)
	}
	return fmt.Sprintf("%.4e", val)
}

func (dtfs DataTypeFuncSet_F32) ReadItem(rawDataPtr unsafe.Pointer) any {
	return math.Float32frombits(*(*uint32)(rawDataPtr))
}

func (dtfs DataTypeFuncSet_F32) WriteItem(rawDataPtr unsafe.Pointer, val any) error {
	convVal, ok := val.(float32)
	if !ok {
		return fmt.Errorf("incompatible types float32 and %v", reflect.TypeOf(val))
	}
	*(*uint32)(rawDataPtr) = math.Float32bits(convVal)
	return nil
}

func (dtfs DataTypeFuncSet_F32) ReadItem_AsFloat32(rawDataPtr unsafe.Pointer) float32 {
	return math.Float32frombits(*(*uint32)(rawDataPtr))
}

func (dtfs DataTypeFuncSet_F32) WriteItem_FromFloat32(rawDataPtr unsafe.Pointer, val float32) {
	*(*uint32)(rawDataPtr) = math.Float32bits(val)
}

/*
	DataTypeFuncSet_UINT16
*/

type DataTypeFuncSet_UINT16 struct{}

func (dtfs DataTypeFuncSet_UINT16) IsCompatible(val any) bool {
	_, ok := val.(uint16)
	return ok
}

func (dtfs DataTypeFuncSet_UINT16) FromFloat32(val float32) any {
	return uint16(val)
}

func (dtfs DataTypeFuncSet_UINT16) ToFloat32(val any) float32 {
	return float32(val.(uint16))
}

func (dtfs DataTypeFuncSet_UINT16) ToString(val any) string {
	if !dtfs.IsCompatible(val) {
		return fmt.Sprintf("%v", val)
	}
	return fmt.Sprintf("%d", val)
}

func (dtfs DataTypeFuncSet_UINT16) ReadItem(rawDataPtr unsafe.Pointer) any {
	return *(*uint16)(rawDataPtr)
}

func (dtfs DataTypeFuncSet_UINT16) WriteItem(rawDataPtr unsafe.Pointer, val any) error {
	convVal, ok := val.(uint16)
	if !ok {
		return fmt.Errorf("incompatible types uint16 and %v", reflect.TypeOf(val))
	}
	*(*uint16)(rawDataPtr) = convVal
	return nil
}

func (dtfs DataTypeFuncSet_UINT16) ReadItem_AsFloat32(rawDataPtr unsafe.Pointer) float32 {
	return float32(*(*uint16)(rawDataPtr))
}

func (dtfs DataTypeFuncSet_UINT16) WriteItem_FromFloat32(rawDataPtr unsafe.Pointer, val float32) {
	*(*uint16)(rawDataPtr) = uint16(val)
}

/*
	DataTypeFuncSet_INT32
*/

type DataTypeFuncSet_INT32 struct{}

func (dtfs DataTypeFuncSet_INT32) IsCompatible(val any) bool {
	_, ok := val.(int32)
	return ok
}

func (dtfs DataTypeFuncSet_INT32) FromFloat32(val float32) any {
	return int32(val)
}

func (dtfs DataTypeFuncSet_INT32) ToFloat32(val any) float32 {
	return float32(val.(int32))
}

func (dtfs DataTypeFuncSet_INT32) ToString(val any) string {
	if !dtfs.IsCompatible(val) {
		return fmt.Sprintf("%v", val)
	}
	return fmt.Sprintf("%d", val)
}

func (dtfs DataTypeFuncSet_INT32) ReadItem(rawDataPtr unsafe.Pointer) any {
	return int32(*(*uint32)(rawDataPtr))
}

func (dtfs DataTypeFuncSet_INT32) WriteItem(rawDataPtr unsafe.Pointer, val any) error {
	convVal, ok := val.(int32)
	if !ok {
		return fmt.Errorf("incompatible types int32 and %v", reflect.TypeOf(val))
	}
	*(*uint32)(rawDataPtr) = uint32(convVal)
	return nil
}

func (dtfs DataTypeFuncSet_INT32) ReadItem_AsFloat32(rawDataPtr unsafe.Pointer) float32 {
	return float32(int32(*(*uint32)(rawDataPtr)))
}

func (dtfs DataTypeFuncSet_INT32) WriteItem_FromFloat32(rawDataPtr unsafe.Pointer, val float32) {
	*(*uint32)(rawDataPtr) = uint32(int32(val))
}

/*
	DataTypeFuncSet_COMPLEX
*/

type DataTypeFuncSet_COMPLEX struct{}

func (dtfs DataTypeFuncSet_COMPLEX) IsCompatible(val any) bool {
	_, ok := val.(complex64)
	return ok
}

func (dtfs DataTypeFuncSet_COMPLEX) FromFloat32(val float32) any {
	return 0
}

func (dtfs DataTypeFuncSet_COMPLEX) ToFloat32(val any) float32 {
	return 0
}

func (dtfs DataTypeFuncSet_COMPLEX) ToString(val any) string {
	if !dtfs.IsCompatible(val) {
		return fmt.Sprintf("%v", val)
	}
	valConv := val.(complex64)
	realPart := real(valConv)
	imagPart := imag(valConv)

	return fmt.Sprintf("%.4e+%.4ej", realPart, imagPart)
}

func (dtfs DataTypeFuncSet_COMPLEX) ReadItem(rawDataPtr unsafe.Pointer) any {
	realPart := math.Float32frombits(*(*uint32)(rawDataPtr))
	imagPart := math.Float32frombits(*(*uint32)(unsafe.Add(rawDataPtr, unsafe.Sizeof(realPart))))

	return complex64(complex(realPart, imagPart))
}

func (dtfs DataTypeFuncSet_COMPLEX) WriteItem(rawDataPtr unsafe.Pointer, val any) error {
	convVal, ok := val.(complex64)
	if !ok {
		return fmt.Errorf("incompatible types complex64 and %v", reflect.TypeOf(val))
	}
	realPartBits := math.Float32bits(real(convVal))
	imagPartBits := math.Float32bits(imag(convVal))
	*(*uint32)(rawDataPtr) = realPartBits
	*(*uint32)(unsafe.Add(rawDataPtr, unsafe.Sizeof(realPartBits))) = imagPartBits
	return nil
}

func (dtfs DataTypeFuncSet_COMPLEX) ReadItem_AsFloat32(rawDataPtr unsafe.Pointer) float32 {
	return 0
}

func (dtfs DataTypeFuncSet_COMPLEX) WriteItem_FromFloat32(rawDataPtr unsafe.Pointer, val float32) {
	// Do nothing
}

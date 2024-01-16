package ml

import (
	"reflect"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

// See: https://github.com/ggerganov/llama.cpp/blob/master/convert.py

var (
	DT_BF16    = newDataType("BF16", dtype.BFloat16(0), DataTypeFuncSet_BF16{})
	DT_F32     = newDataType("Float32", float32(0), DataTypeFuncSet_F32{})
	DT_UINT16  = newDataType("UInt16", uint16(0), DataTypeFuncSet_UINT16{})
	DT_INT32   = newDataType("Int32", int32(0), DataTypeFuncSet_INT32{})
	DT_COMPLEX = newDataType("Complex", complex64(complex(0.0, 0.0)), DataTypeFuncSet_COMPLEX{})
)

type DataType struct {
	Name     string
	GoType   reflect.Type
	ItemSize int
	FuncSet  DataTypeFuncSet
}

func newDataType(name string, itemSample any, funcSet DataTypeFuncSet) DataType {
	result := DataType{
		Name:    name,
		GoType:  reflect.TypeOf(itemSample),
		FuncSet: funcSet,
	}
	result.ItemSize = int(result.GoType.Size())
	return result
}

func (dt DataType) String() string {
	return dt.Name
}

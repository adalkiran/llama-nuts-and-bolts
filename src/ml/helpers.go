package ml

import (
	"fmt"
	"os"
	"reflect"
)

func processErrors(errors ...error) error {
	for _, err := range errors {
		if err != nil {
			return err
		}
	}
	return nil
}

func checkIsVector(t *Tensor) error {
	if t.IsVector() {
		return nil
	}
	return fmt.Errorf("tensor \"%s\" with shape %v is not a vector", t.Name, t.GetShape())
}

func checkIsMatrix(t *Tensor) error {
	if t.IsMatrix() {
		return nil
	}
	return fmt.Errorf("tensor \"%s\" with shape %v is not a matrix", t.Name, t.GetShape())
}

func checkSameDataType(a *Tensor, b *Tensor) error {
	if a.DataType == b.DataType {
		return nil
	}
	return fmt.Errorf("tensors are not in same data type: \"%s\" is %s, \"%s\" is %s", a.Name, a.DataType, b.Name, b.DataType)
}

func checkSameShape(a *Tensor, b *Tensor) error {
	if reflect.DeepEqual(a.Size, b.Size) {
		return nil
	}
	return fmt.Errorf("tensors are not in same shape: \"%s\" is %v, \"%s\" is %v", a.Name, a.Size, b.Name, b.Size)
}

func DebugHelper(tensor *Tensor, filename string) {
	f, _ := os.Create(filename)
	rawData := tensor.RawData
	defer f.Close()
	for i := 0; i < int(len(rawData)/2); i++ {
		f.WriteString(fmt.Sprintf("%d: %v\n", i, rawData[i*2:i*2+2]))
	}
	fmt.Printf("debug_helper: File %s was written.\n", filename)
}

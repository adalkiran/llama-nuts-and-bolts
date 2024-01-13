package ml

import (
	"fmt"
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

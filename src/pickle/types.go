package pickle

import (
	"fmt"
)

var BASE_CLASSES = map[string]interface{}{
	"collections.OrderedDict": NewPickleDict[interface{}],
}

type PickleDict[T any] struct {
	keys  []string
	items map[string]T
}

func NewPickleDict[T any]() *PickleDict[T] {
	result := PickleDict[T]{}
	result.items = make(map[string]T)
	return &result
}

func (pd *PickleDict[T]) Get(key string) (T, bool) {
	val, ok := pd.items[key]
	return val, ok
}

func (pd *PickleDict[T]) Set(key string, val T) {
	_, ok := pd.items[key]
	if ok {
		for i, existingKey := range pd.keys {
			if existingKey == key {
				pd.keys = append(pd.keys[:i], pd.keys[i+1:]...)
				break
			}
		}
	}
	pd.keys = append(pd.keys, key)
	pd.items[key] = val
}

func (pd *PickleDict[T]) GetKeys() []string {
	return pd.keys
}

type PickleTuple = []interface{}

type StopSignal struct {
	Value *PickleDict[interface{}]
}

func (s *StopSignal) Error() string {
	return fmt.Sprintf("Stop signal: %s", s.Value)
}

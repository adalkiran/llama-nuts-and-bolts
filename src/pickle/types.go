package pickle

import (
	"fmt"
)

var BASE_CLASSES = map[string]interface{}{
	"collections.OrderedDict": NewPickleDict,
}

type PickleDict struct {
	keys  []string
	items map[string]interface{}
}

func NewPickleDict() *PickleDict {
	result := PickleDict{}
	result.items = make(map[string]interface{})
	return &result
}

func (pd *PickleDict) Get(key string) (interface{}, bool) {
	val, ok := pd.items[key]
	return val, ok
}

func (pd *PickleDict) Set(key string, val interface{}) {
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

func (pd *PickleDict) GetKeys() []string {
	return pd.keys
}

type PickleTuple = []interface{}

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

type StopSignal struct {
	Value *PickleDict
}

func (s *StopSignal) Error() string {
	return fmt.Sprintf("Stop signal: %s", s.Value)
}

package pickle

import (
	"bufio"
	"fmt"
	"io"
)

type PickleReader struct {
	fileReader *bufio.Reader

	proto     byte
	stack     []interface{}
	metastack []interface{}
	memo      map[int]interface{}

	FindClassFn      func(module string, name string) (interface{}, error)
	PersistentLoadFn func(pid []interface{}) (interface{}, error)
}

func NewPickleReader(fileReader io.ReadCloser) *PickleReader {
	result := new(PickleReader)
	result.fileReader = bufio.NewReader(fileReader)
	result.stack = make([]interface{}, 0)
	result.metastack = make([]interface{}, 0)
	result.memo = map[int]interface{}{}
	return result
}

func (pr *PickleReader) Load() (*PickleDict, error) {
	for {
		key, err := pr.ReadByte()
		if err != nil {
			return nil, err
		}
		err = dispatch(pr, key)
		if err != nil {
			if stopSignal, ok := err.(*StopSignal); ok {
				return stopSignal.Value, nil
			}
			return nil, err
		}
	}
}

func (pr *PickleReader) Read(byteCount int) ([]byte, error) {
	buf := make([]byte, byteCount)
	readCount, err := io.ReadFull(pr.fileReader, buf)
	if err != nil {
		return nil, err
	}
	if readCount < byteCount {
		return nil, fmt.Errorf("not found required bytes to read")
	}
	return buf, nil
}

func (pr *PickleReader) ReadByte() (byte, error) {
	return pr.fileReader.ReadByte()
}

func (pr *PickleReader) ReadLine() (string, error) {
	line, err := pr.fileReader.ReadString('\n')
	if err != nil {
		return "", err
	}
	return line[:len(line)-1], nil
}

func (pr *PickleReader) Append(item interface{}) {
	pr.stack = append(pr.stack, item)
}

func (pr *PickleReader) persistentLoad(pid []interface{}) (interface{}, error) {
	if pr.PersistentLoadFn != nil {
		return pr.PersistentLoadFn(pid)
	}
	return nil, fmt.Errorf("unimplemented method: PickleReader.persistentLoad")
}

func (pr *PickleReader) findClass(module string, name string) (interface{}, error) {
	var result interface{}
	var err error
	if pr.FindClassFn != nil {
		result, err = pr.FindClassFn(module, name)
		if err == nil {
			return result, nil
		}
	}
	result, ok := BASE_CLASSES[module+"."+name]
	if !ok {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("unknown class \"%s.%s\" not found", module, name)
	}
	return result, nil
}

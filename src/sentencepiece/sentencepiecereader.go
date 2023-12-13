package sentencepiece

import (
	"fmt"
	"os"

	"github.com/adalkiran/llama-nuts-and-bolts/src/protobuf"
)

func Load(vocabFilePath string) (*ModelProto, error) {
	vocabFile, err := os.Open(vocabFilePath)
	if err != nil {
		return nil, err
	}
	defer vocabFile.Close()

	vocabReader := protobuf.NewProtobufReader(vocabFile, modelprotoDescriptor)
	modelVal, err := vocabReader.Unmarshal()
	if err != nil {
		return nil, err
	}
	model, ok := modelVal.(ModelProto)
	if !ok {
		return nil, fmt.Errorf("cannot convert %v to ModelProto", model)
	}
	return &model, nil
}

package sentencepiece

import (
	"os"

	"github.com/adalkiran/llama-nuts-and-bolts/src/protobuf"
)

func Load(vocabFilePath string) ([]protobuf.Message, error) {
	vocabFile, err := os.Open(vocabFilePath)
	if err != nil {
		return nil, err
	}
	defer vocabFile.Close()

	vocabReader := protobuf.NewProtobufReader(vocabFile)
	resultArr := make([]protobuf.Message, 0)

	for {
		message, ok := vocabReader.ReadMessage()
		if !ok {
			break
		}
		resultArr = append(resultArr, *message)
		//fmt.Printf("\n%6d: %v", len(resultArr), itemPair)
	}
	return resultArr, nil
}

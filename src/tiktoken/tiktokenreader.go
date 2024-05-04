package tiktoken

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func loadTiktokenBpe(vocabFilePath string) (map[string]int, error) {
	vocabFile, err := os.Open(vocabFilePath)
	if err != nil {
		return nil, err
	}
	defer vocabFile.Close()

	fileScanner := bufio.NewScanner(vocabFile)
	fileScanner.Split(bufio.ScanLines)

	result := make(map[string]int)

	for fileScanner.Scan() {
		lineParts := strings.Split(fileScanner.Text(), " ")
		token, err := base64.StdEncoding.DecodeString(lineParts[0])
		if err != nil {
			return nil, err
		}
		rank, err := strconv.Atoi(lineParts[1])
		if err != nil {
			return nil, err
		}
		result[string(token)] = rank
	}
	return result, nil
}

func Load(vocabFilePath string) (*ModelData, error) {
	mergeableRanks, err := loadTiktokenBpe(vocabFilePath)
	if err != nil {
		return nil, err
	}
	baseTokensCount := len(mergeableRanks)

	reservedSpecialTokensCount := 256

	specialTokensArr := []string{
		"<|begin_of_text|>",
		"<|end_of_text|>",
		"<|reserved_special_token_0|>",
		"<|reserved_special_token_1|>",
		"<|reserved_special_token_2|>",
		"<|reserved_special_token_3|>",
		"<|start_header_id|>",
		"<|end_header_id|>",
		"<|reserved_special_token_4|>",
		"<|eot_id|>", // end of turn
	}

	for i := 5; i < reservedSpecialTokensCount-5; i++ {
		specialTokensArr = append(specialTokensArr, fmt.Sprintf("<|reserved_special_token_%d|>", i))
	}

	specialTokens := make(map[string]int)
	for i, token := range specialTokensArr {
		specialTokens[token] = baseTokensCount + i
	}

	result := &ModelData{
		MergeableRanks: mergeableRanks,
		SpecialTokens:  specialTokens,

		BeginOfSentenceId: specialTokens["<|begin_of_text|>"],
		EndOfSentenceId:   specialTokens["<|end_of_text|>"],
		PadId:             -1,
		UnknownId:         -1,
		StopTokenIds:      []int{specialTokens["<|end_of_text|>"], specialTokens["<|eot_id|>"]},
	}

	return result, nil
}

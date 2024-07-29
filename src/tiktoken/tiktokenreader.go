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
		"<|finetune_right_pad_id|>",
		"<|step_id|>",
		"<|start_header_id|>",
		"<|end_header_id|>",
		"<|eom_id|>", // end of message
		"<|eot_id|>", // end of turn
		"<|python_tag|>",
	}

	reservedTokensArr := make([]string, reservedSpecialTokensCount-len(specialTokensArr))
	for i := 0; i < len(reservedTokensArr); i++ {
		reservedTokensArr[i] = fmt.Sprintf("<|reserved_special_token_%d|>", 2+i)
	}
	specialTokensArr = append(specialTokensArr, reservedTokensArr...)

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
		StopTokenIds:      []int{specialTokens["<|eom_id|>"], specialTokens["<|eot_id|>"]},
	}

	return result, nil
}

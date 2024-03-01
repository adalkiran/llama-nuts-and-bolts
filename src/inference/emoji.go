package inference

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/enescakir/emoji"
	"golang.org/x/text/unicode/runenames"
)

const zwj = '\u200d' // ZWJ (Zero Width Joiner)
var emojiPartsRegexp = *regexp.MustCompile(`\<(\d+)\>`)

var emojiToAliasMap map[string]string

type RuneData struct {
	Rune          rune
	RuneName      string
	UnicodeEscape string
}

// See: https://tutorialedge.net/golang/the-go-init-function/
func init() {
	aliasToEmojiMap := emoji.Map()
	emojiToAliasMap = make(map[string]string)
	for alias, emoji := range aliasToEmojiMap {
		// if current emoji exists multiple times (with different aliases), prefer alphabetically prior one
		if existingAlias, ok := emojiToAliasMap[emoji]; ok {
			if strings.Compare(existingAlias, alias) == 1 {
				emojiToAliasMap[emoji] = alias
			}
		} else {
			emojiToAliasMap[emoji] = alias
		}
	}
}

func extractRuneData(str string) []RuneData {
	result := make([]RuneData, 0)
	for _, r := range str {
		result = append(result, RuneData{
			Rune:          r,
			RuneName:      runenames.Name(r),
			UnicodeEscape: fmt.Sprintf("\\U%08X", r),
		})
	}
	return result
}

func processEmoji(decodingContext *generationDecodingContext, r rune) string {
	if decodingContext.decodingFinished {
		decodingContext.decodingFinished = false
	}
	if unicode.IsGraphic(r) || r == zwj {
		decodingContext.waitingRunes += string(r)
	}
	decodingContext.waitingRunesExtraStr = ""
	if emojiParts, ok := searchMinimumEmojiCount(decodingContext.waitingRunes); ok {
		decodingContext.waitingRunesExtraStr = ""
		for _, emojiPart := range emojiParts {
			emojiAlias, ok := emojiToAliasMap[emojiPart]
			runeData := extractRuneData(emojiPart)
			if ok {
				compositeUnicodeEscape := ""
				for _, runeItem := range runeData {
					compositeUnicodeEscape += runeItem.UnicodeEscape
				}
				decodingContext.waitingRunesExtraStr += fmt.Sprintf("[%s%s]", emojiAlias, compositeUnicodeEscape)
			} else {
				runeData := extractRuneData(emojiPart)
				for _, runeItem := range runeData {
					decodingContext.waitingRunesExtraStr += fmt.Sprintf("[:%s:%s]", runeItem.RuneName, runeItem.UnicodeEscape)
				}
			}
		}
	} else {
		runeData := extractRuneData(decodingContext.waitingRunes)
		for _, runeItem := range runeData {
			decodingContext.waitingRunesExtraStr += fmt.Sprintf("[:%s:%s]", runeItem.RuneName, runeItem.UnicodeEscape)
		}
	}
	if !unicode.IsGraphic(r) && r != zwj {
		extraStr := decodingContext.waitingRunesExtraStr
		decodingContext.waitingRunes = ""
		decodingContext.waitingRunesExtraStr = ""
		return extraStr + string(r)
	}
	return string(r)
}

func searchMinimumEmojiCount(str string) ([]string, bool) {
	strOriginal := str
	//foundEmojiCount := 1
	//for foundEmojiCount > 0 {
	//	foundEmojiCount = 0
	for i := 0; i < len(str); {
		increment_i := true
		for j := len(str); j > i; {
			skip := false
			substr := str[i:j]
			if j > 0 && substr[len(substr)-1:] == ">" {
				openIdx := strings.LastIndex(substr, "<")
				if openIdx > -1 {
					j = openIdx - 1
					skip = true
				}
			}
			if substr[0:1] == "<" {
				closeIdx := strings.Index(substr, ">")
				if closeIdx > -1 {
					i = closeIdx + 1
					skip = true
				}
			}
			if skip {
				continue
			}

			if _, ok := emojiToAliasMap[substr]; ok {
				replaceText := fmt.Sprintf("<%d>", len(substr))
				str = str[:i] + replaceText + str[j:]
				i += len(replaceText)
				j += len(replaceText) - len(substr)
				//foundEmojiCount++
				increment_i = false
				continue
			}
			_, rsize := utf8.DecodeLastRuneInString(str[i:j])
			j -= rsize
		}
		if increment_i {
			_, rsize := utf8.DecodeRuneInString(str[i:])
			i += rsize
		}
	}
	//}

	runeSizes := make([]int, 0)
	for len(str) > 0 {
		matchLoc := emojiPartsRegexp.FindStringIndex(str)
		var rsize int
		var err error
		if matchLoc == nil || matchLoc[0] > 0 {
			_, rsize = utf8.DecodeRuneInString(str)
			runeSizes = append(runeSizes, rsize)
			str = str[rsize:]
		} else {
			match := emojiPartsRegexp.FindStringSubmatch(str)
			rsize, err = strconv.Atoi(match[1])
			if err != nil {
				return nil, false
			}
			runeSizes = append(runeSizes, rsize)
			str = str[len(match[0]):]
		}
	}
	result := make([]string, 0)
	idx := 0
	for runeSizeIdx := 0; runeSizeIdx < len(runeSizes); runeSizeIdx++ {
		result = append(result, strOriginal[idx:idx+runeSizes[runeSizeIdx]])
		idx += runeSizes[runeSizeIdx]
	}
	return result, len(result) > 0
}

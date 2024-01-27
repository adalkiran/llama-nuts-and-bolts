package inference

import (
	"fmt"

	"github.com/enescakir/emoji"
	"golang.org/x/text/unicode/runenames"
)

var emojiToAliasMap map[string]string

// See: https://tutorialedge.net/golang/the-go-init-function/
func init() {
	aliasToEmojiMap := emoji.Map()
	emojiToAliasMap = make(map[string]string)
	for alias, emoji := range aliasToEmojiMap {
		emojiToAliasMap[emoji] = alias
	}
}

func emojiToAlias(potentialEmojiRune rune, runeSize int, returnAliasWithEmoji bool) string {
	potentialEmojiStr := string(potentialEmojiRune)

	alias, ok := emojiToAliasMap[potentialEmojiStr]
	if !ok {
		if runeSize > 1 {
			unicodeEscape := fmt.Sprintf("\\U%08X", potentialEmojiRune)
			alias = fmt.Sprintf(":%s:", runenames.Name(potentialEmojiRune))
			return fmt.Sprintf("%s[%s%s]", potentialEmojiStr, alias, unicodeEscape)
		} else {
			return potentialEmojiStr
		}
	}
	if returnAliasWithEmoji {
		unicodeEscape := fmt.Sprintf("\\U%08X", potentialEmojiRune)
		return fmt.Sprintf("%s[%s%s]", potentialEmojiStr, alias, unicodeEscape)
	} else {
		return alias
	}
}

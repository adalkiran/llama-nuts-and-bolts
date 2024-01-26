package inference

import (
	"fmt"

	"github.com/enescakir/emoji"
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

func emojiToAlias(potentialEmoji string, returnAliasWithEmoji bool) string {
	alias, ok := emojiToAliasMap[potentialEmoji]
	if !ok {
		return potentialEmoji
	}
	if returnAliasWithEmoji {
		return fmt.Sprintf("%s[%s]", potentialEmoji, alias)
	} else {
		return alias
	}
}

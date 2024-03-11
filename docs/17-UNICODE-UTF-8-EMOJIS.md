# **17. UNICODE, UTF-8 and EMOJIS**

In computing, computers commonly process the information *digitally* as *bit*s. Then, *bit*s come together to form *byte*s, and *byte*s create various data types. Most of these types are different ways to represent a number, think of, *character*s in a *string* are represented with numbers, behind the curtains.

In our project, normal Unicode text characters are supported built-in via Go language platform. But, LLaMa models have an capability to generate *emoji*s as *sequence of byte tokens*. In this case, the built-in support might not be enough.

Also, this project supports to be compiled for and run on multiple operating systems such as Windows, Linux, and, MacOS. The application provides a CLI (command line interface). At this point, the Linux and MacOS terminals have Unicode emoji rendering capabilities. But in Windows, some terminals can't render completely, some can render partially.

Because we don't want the user (especially on Windows) to see only multiple "question marks" instead of emoji characters, we have added a capability to detect known emojis and code points. With this capability, our project can print known names of emojis and code points on the console. So, this provides the user to see the name texts if they can't see the emoji glyph.

## **17.1. UNICODE**

At the first times of digital computing, early 1960s, the [American Standard Code for Information Interchange](https://en.wikipedia.org/wiki/ASCII) standard was defined to represent commonly used characters of English as numbers between 0-255, as one byte.

Over time, computing and computers were becoming widespread and globalized. The need for support for languages other than English was emerging. Variety of characters was getting increased. This problem was partially solved with different encoding methods and character tables based on languages, which enables us to represent different characters in a single byte. They are also as [code page](https://en.wikipedia.org/wiki/Code_page).

As time passed, the ASCII standard and code pages were becoming unsatisfactory. Thus, some other character encoding method employing multi-byte representation was needed.

In late 1980s, a group of individuals from Xerox and Apple has started to investigate on creating an universal character set. Then, in early 1990s, the [Unicode Consortium - Unicode, Inc.](https://en.wikipedia.org/wiki/Unicode_Consortium) was founded and [The Unicode Standard](https://en.wikipedia.org/wiki/Unicode) was born.

>**From [Wikipedia](https://en.wikipedia.org/wiki/Unicode):** Unicode text is processed and stored as binary data using one of several encodings, which define how to translate the standard's abstracted codes for characters into sequences of bytes. The Unicode Standard itself defines three encodings: UTF-8, UTF-16, and UTF-32, though several others exist. Of these, UTF-8 is the most widely used by a large margin, in part due to its backwards-compatibility with ASCII.

In Unicode, characters are represented as [code points](https://pro.arcgis.com/en/pro-app/3.1/help/data/geodatabases/overview/a-quick-tour-of-unicode.htm) (In Go language, code points are called as [rune](https://www.geeksforgeeks.org/rune-in-golang/)s).

Sources:

* [Unicode.org website](https://home.unicode.org/)
* [Codepoints.net website](https://codepoints.net/)

## **17.2. UTF-8**

Now, with Unicode, we have a character set consisting of code points that may not fit in a single byte. We need an encoding method. There are [UTF-8](https://en.wikipedia.org/wiki/UTF-8), [UTF-16](https://en.wikipedia.org/wiki/UTF-16), [UTF-32](https://en.wikipedia.org/wiki/UTF-32), and others. Nowadays, [UTF-8](https://en.wikipedia.org/wiki/UTF-8) is the most commonly used one.

UTF-8 is a variable-length character encoding standard. It has the flexibility to represent some characters in single-byte, some in 2-bytes, 3-bytes, n-bytes, etc... The most common 128 English characters could be represented with 8 bits (single-byte), this ability enables backward compatibility with the ASCII standard.

You can watch the source videos below and have more information.

Sources:

* [Youtube - Computer Organization and Design Fundamentals - Ep 020: Unicode Code Points and UTF-8 Encoding](https://www.youtube.com/watch?v=tbdym9ZtepQ&list=PLxfrSxK7P38X7XfG4X8Y9cdOURvC7ObMF)
* [Youtube - Computer Organization and Design Fundamentals - Ep 021: UTF-8 Encoding Examples](https://www.youtube.com/watch?v=c_hfKgektt4&list=PLxfrSxK7P38X7XfG4X8Y9cdOURvC7ObMF)
* [Youtube - Unicode, in friendly terms: ASCII, UTF-8, code points, character encodings, and more](https://www.youtube.com/watch?v=ut74oHojxqo)
* [Strings, bytes, runes and characters in Go](https://go.dev/blog/strings)

## **17.3. Emojis**

Emojis are Unicode characters that are rendered as special pictograms, glyphs. Emojis are mostly represented with multiple bytes, also multiple emojis could form another "one" emoji by getting together with [Zero-width joiner](https://en.wikipedia.org/wiki/Zero-width_joiner).

Sources:

* [Unicode.org website](https://home.unicode.org/)
* [Unicode.org - Emoji Sequence Text Files](https://unicode.org/Public/emoji/15.1/)
* [Github Supported Emoji Sequence List JSON](https://raw.githubusercontent.com/github/gemoji/master/db/emoji.json)
* [Emoji dissector](https://emojidissector.com/)
* [Emoji - Go library lets you use emoji characters in strings](https://github.com/enescakir/emoji)

### **17.3.1. Emoji Samples**

As you can see in ```TestSimulatedEmojiOutputMultipleCompositeEmojis``` unit test function in [cmd/main_test.go](../cmd/main_test.go):

| Link  | Icon   |  Name | Unicode  |  Bytes |
|---|---|---|---|---|
|  |  "" | `ZWJ: ZERO WIDTH JOINER`   | `\U0000200D`  | (0xE2 0x80 0x8D)  |
| [iEmoji Link](https://www.iemoji.com/view/emoji/91/smileys-people/man) |  👨 | `:man:`   | `\U0001F468`  | (0xF0 0x9F 0x91 0xA8) |
| [iEmoji Link](https://www.iemoji.com/view/emoji/91/smileys-people/man) |  👨 | `:man: + :ZWJ:`   | `\U0001F468` + `\U0000200D`  | (0xF0 0x9F 0x91 0xA8), (0xE2 0x80 0x8D) |
| [iEmoji Link](https://www.iemoji.com/view/emoji/91/smileys-people/man) \| [iEmoji Link](https://www.iemoji.com/view/emoji/90/smileys-people/woman) |  👨‍👩<br> = <br> 👨 + `:ZWJ:` + 👩 | `:man: + :ZWJ: + :woman:`   | `\U0001F468` + `\U0000200D` + `\U0001F469`  | (0xF0 0x9F 0x91 0xA8), (0xE2 0x80 0x8D), (0xF0 0x9F 0x91 0xA9) |
| |  👨‍👩‍👧 <br> = <br> 👨 + `:ZWJ:` + 👩 + `:ZWJ:` + 👧 | `:family_man_woman_girl:`<br>=<br>`:man: + :ZWJ: + :woman: + :ZWJ: + :girl:`   | `\U0001F468` + `\U0000200D` + `\U0001F469` + `\U0000200D` + `\U0001F467`  | (0xF0 0x9F 0x91 0xA8), (0xE2 0x80 0x8D), (0xF0 0x9F 0x91 0xA9), (0xE2 0x80 0x8D), (0xF0 0x9F 0x91 0xA7) |
| [iEmoji Link](https://www.iemoji.com/view/emoji/1715/smileys-people/family-man-woman-girl-boy) |  👨‍👩‍👧‍👦 <br> = <br> 👨 + `:ZWJ:` + 👩 + `:ZWJ:` + 👧 + `:ZWJ:` + 👦  | `:family_man_woman_girl_boy:`<br>=<br>`:man: + :ZWJ: + :woman: + :ZWJ: + :girl: + :ZWJ: + :boy:`   | `\U0001F468` + `\U0000200D` + `\U0001F469` + `\U0000200D` + `\U0001F467` + `\U0000200D` + `\U0001F466`  | (0xF0 0x9F 0x91 0xA8), (0xE2 0x80 0x8D), (0xF0 0x9F 0x91 0xA9), (0xE2 0x80 0x8D), (0xF0 0x9F 0x91 0xA7), (0xE2 0x80 0x8D), (0xF0 0x9F 0x91 0xA6) |

### **17.3.2. LLaMa Emoji Generation**

Large Language Models like LLaMa and most of other NLP (natural language processing) systems use tokenization to represent words or word partitions. So, the generative ones generate new tokens. However, their tokenizers have a limited number of items in their vocabulary. Mostly these vocabularies don't include emojis.

LLaMa's tokenizer model supports emojis employing byte type tokens. For e.g., if the LLaMa model wants to generate the "👨" (:man:) emoji, it generates this emoji byte-by-byte.

In our example, the "👨" (:man:) emoji is encoded in UTF-8 encoding with 4 bytes: 0xF0, 0x9F, 0x91, 0xA8. The LLaMa model generates "<0xF0>" byte token at first, then generates "<0x9F>", "<0x91>", and "<0xA8>" respectively. After generation of each byte token, our project's [InferenceEngine.TokenToString(...)](../src/inference/tokenize.go) method checks if enough byte tokens are generated for representing an emoji, via ```utf8.Valid(...)``` as follows.

If the generated new token is a byte type token, it is added into ```decodingContext.waitingBytes``` array.

If ```utf8.Valid(...)``` returns true, this method converts the waiting bytes to a rune, then a string. If false, in other words, if the waiting bytes don't consist of a valid UTF-8 byte sequence (including an unfinished sequence that may be finished after upcoming new bytes), the application waits for the next token to generate.

Also, if the waiting byte sequence is a valid UTF-8 byte sequence, we also check if the sequence represents a known emoji or code point with a human readable name via [inference.processEmoji(...)](../src/inference/emoji.go) function.

This detection process seems simple at first sight, but emojis consisting of multiple emojis with [Zero-width joiner](https://en.wikipedia.org/wiki/Zero-width_joiner) make this process harder and more complex. But our project can handle these types of issues.

<sup>from [src/inference/tokenize.go](../src/inference/tokenize.go)</sup>

```go
func (ie *InferenceEngine) TokenToString(tokenId model.TokenId, decodingContext *generationDecodingContext) (token sentencepiece.SentencePiece, resultString string, addedToWaiting bool) {
    ...
    switch token.PieceType {
    ...
    case sentencepiece.BYTE:
        if decodingContext.waitingBytes == nil {
            decodingContext.waitingBytes = make([]byte, 0)
        }
        decodingContext.waitingBytes = append(decodingContext.waitingBytes, token.ByteFallback)
        if utf8.Valid(decodingContext.waitingBytes) {
            r, rsize := utf8.DecodeRune(decodingContext.waitingBytes)
            decodingContext.waitingBytes = decodingContext.waitingBytes[rsize:]
            resultString += processEmoji(decodingContext, r, rsize)
        } else {
            addedToWaiting = true
        }
        return
    ...
    }
    ...
}
```

<br>

---

<div align="right">

[&lt;&nbsp;&nbsp;Previous chapter: MAKING PREDICTION with LLAMA MODEL - 3](./16-MAKING-PREDICTION-WITH-LLAMA-MODEL-3.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Next chapter: CONCLUSION&nbsp;&nbsp;&gt;](./18-CONCLUSION.md)

</div>

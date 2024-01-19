package sentencepiece

import (
	"encoding/hex"
	"fmt"
	"regexp"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/protobuf"
)

// See: https://github.com/google/sentencepiece/blob/022f8c3fed4d2feb4e4c670949cf01cef477dcc4/src/sentencepiece_model.proto

type ModelProto struct {
	// Sentence pieces with scores.
	Pieces *[]SentencePiece // num: 1
	// Spec used to generate this model file.
	TrainerSpec *TrainerSpec // num: 2
	// Spec for text normalization.
	NormalizerSpec *NormalizerSpec //num: 3
}

var extractHexadecimalPiece = *regexp.MustCompile(`<0x(\w+)>`)

type SentencePiece struct {
	Piece     string // piece must not be empty.
	Score     float32
	PieceType Type //[default = NORMAL];

	ByteFallback byte
}

func newSentencePiece(piece string, score float32, pieceType Type) SentencePiece {
	result := SentencePiece{
		Piece:     piece,
		Score:     score,
		PieceType: pieceType,
	}
	if result.PieceType == BYTE {
		match := extractHexadecimalPiece.FindStringSubmatch(result.Piece)
		if len(match) >= 2 {
			byteValue, err := hex.DecodeString(match[1])
			if err == nil && len(byteValue) == 1 {
				result.ByteFallback = byteValue[0]
			}
		}
	}
	return result
}

func (sp SentencePiece) String() string {
	return fmt.Sprintf("\"%s\" score: %f, type: %s", sp.Piece, sp.Score, sp.PieceType)
}

type TrainerSpec struct {
}

type NormalizerSpec struct {
	// name of normalization rule.
	Name string //num:1

	// Pre-compiled normalization rule created by
	// Builder::GetPrecompiledCharsMap() or Builder::CompileCharsMap() method.
	// Usually this field is set by Builder::GetNormalizerSpec() method.
	PrecompiledCharsmap []byte //num: 2

	// Adds dummy whitespace at the beginning of text in order to
	// treat "world" in "world" and "hello world" in the same way.
	AddDummyPrefix bool //num: 3 [default = true];

	// Removes leading, trailing, and duplicate internal whitespace.
	RemoveExtraWhitespaces bool //num: 4 [default = true];

	// Replaces whitespace with meta symbol.
	// This field must be true to train sentence piece model.
	EscapeWhitespaces bool //num: 5 [default = true];

	// Custom normalization rule file in TSV format.
	// https://github.com/google/sentencepiece/blob/master/doc/normalization.md
	// This field is only used in SentencePieceTrainer::Train() method, which
	// compiles the rule into the binary rule stored in `precompiled_charsmap`.
	NormalizationRuleTsv string //num: 6;
}

type Type byte

const (
	NORMAL       Type = 1 // normal symbol
	UNKNOWN      Type = 2 // unknown symbol. only <unk> for now.
	CONTROL      Type = 3 // control symbols. </s>, <s>, <2ja> etc.
	USER_DEFINED Type = 4 // user defined symbols. Typical usage of USER_DEFINED symbol is placeholder.
	BYTE         Type = 6 // byte symbols. Used when `byte_fallback` is true.
	UNUSED       Type = 5 // this piece is not used.
)

func (t Type) String() string {
	switch t {
	case NORMAL:
		return "NORMAL"
	case UNKNOWN:
		return "UNKNOWN"
	case CONTROL:
		return "CONTROL"
	case USER_DEFINED:
		return "USER_DEFINED"
	case BYTE:
		return "BYTE"
	case UNUSED:
		return "UNUSED"
	default:
		return "?"
	}
}

var modelprotoDescriptor = protobuf.ProtoDescriptor{
	MainObjectConstructorFn: func() interface{} {
		result := ModelProto{}
		result.Pieces = new([]SentencePiece)
		return result
	},
	MessageProcessorFns: map[protobuf.Number]func(interface{}, protobuf.Message){
		1: func(mainObject interface{}, message protobuf.Message) {
			mo := mainObject.(ModelProto)
			props := message.Value.(map[protobuf.Number]interface{})
			pieceTypeVal, err := common.InterfaceToInt(props[3])
			if err != nil {
				pieceTypeVal = int(NORMAL)
			}
			item := newSentencePiece(props[1].(string), props[2].(float32), Type(pieceTypeVal))
			*mo.Pieces = append(*mo.Pieces, item)
		},
		2: func(mainObject interface{}, message protobuf.Message) {
			// Do nothing, we don't need TrainerSpec at this time.
		},
		3: func(mainObject interface{}, message protobuf.Message) {
			mo := mainObject.(ModelProto)
			props := message.Value.(map[protobuf.Number]interface{})
			ns := NormalizerSpec{}
			ns.Name = props[1].(string)
			ns.PrecompiledCharsmap = props[2].([]byte)

			ns.AddDummyPrefix = common.InterfaceToBool(props[3], true)
			ns.RemoveExtraWhitespaces = common.InterfaceToBool(props[4], true)
			ns.EscapeWhitespaces = common.InterfaceToBool(props[5], true)
			stringVal, ok := props[6].(string)
			if !ok {
				byteArrVal, ok := props[6].([]byte)
				if !ok {
					stringVal = ""
				} else {
					stringVal = string(byteArrVal)
				}
			}
			ns.NormalizationRuleTsv = stringVal
			mo.NormalizerSpec = &ns
			fmt.Println("NOTE TODO: MessageProcessorFns don't affect original object's fields")
		},
	},
}

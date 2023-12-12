package sentencepiece

// See: https://github.com/google/sentencepiece/blob/022f8c3fed4d2feb4e4c670949cf01cef477dcc4/src/sentencepiece_model.proto

type ModelProto struct {
	pieces          SentencePiece  // num: 1
	trainer_spec    TrainerSpec    // num: 2
	normalizer_spec NormalizerSpec //num: 3
}

type SentencePiece struct {
	piece     string // piece must not be empty.
	score     float32
	pieceType Type //[default = NORMAL];
}

type TrainerSpec struct {
}

type NormalizerSpec struct {
}

type Type = byte

const (
	NORMAL       Type = 1 // normal symbol
	UNKNOWN      Type = 2 // unknown symbol. only <unk> for now.
	CONTROL      Type = 3 // control symbols. </s>, <s>, <2ja> etc.
	USER_DEFINED Type = 4 // user defined symbols. Typical usage of USER_DEFINED symbol is placeholder.
	BYTE         Type = 6 // byte symbols. Used when `byte_fallback` is true.
	UNUSED       Type = 5 // this piece is not used.
)

/*
var modelProtoDefinition = map[protobuf.Number]func(){
	1: func() {
	},
}
*/

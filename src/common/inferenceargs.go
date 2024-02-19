package common

type InferenceArgs struct {
	SequenceLength int // text context, 0 = from model
}

func NewInferenceArgs() InferenceArgs {
	return InferenceArgs{
		SequenceLength: 0,
	}
}

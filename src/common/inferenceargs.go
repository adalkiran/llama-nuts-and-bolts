package common

type InferenceArgs struct {
	Seed           int64 // RNG (Random Number Generator) seed, -1 for random
	SequenceLength int   // text context, 0 = from model
}

func NewInferenceArgs() InferenceArgs {
	return InferenceArgs{
		Seed:           -1,
		SequenceLength: 0,
	}
}

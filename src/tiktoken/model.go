package tiktoken

type ModelData struct {
	MergeableRanks map[string]int
	SpecialTokens  map[string]int

	BeginOfSentenceId int
	EndOfSentenceId   int
	UnknownId         int
	PadId             int
	StopTokenIds      []int
}

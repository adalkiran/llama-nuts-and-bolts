package inference

import (
	"math"
	"math/rand"
	"time"

	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

/*
type Batch struct {
	tokenIds            []model.TokenId
	positions           []model.Position
	sequenceIds         [][]model.SequenceId
	includeOutputLogits []bool
}

func NewBatch() *Batch {
	return &Batch{
		tokenIds:            make([]model.TokenId, 0),
		positions:           make([]model.Position, 0),
		sequenceIds:         make([][]model.SequenceId, 0),
		includeOutputLogits: make([]bool, 0),
	}
}

func (b *Batch) Add(tokenId model.TokenId, position model.Position, sequenceIds []model.SequenceId, includeOutputLogits bool) {
	b.tokenIds = append(b.tokenIds, tokenId)
	b.positions = append(b.positions, position)
	b.sequenceIds = append(b.sequenceIds, sequenceIds)
	b.includeOutputLogits = append(b.includeOutputLogits, includeOutputLogits)
}
*/

type InferenceContextInitParams struct {
	Seed           int64 // RNG (Random Number Generator) seed, -1 for random
	SequenceLength int   // text context, 0 = from model
}

func NewInferenceContextInitParams() InferenceContextInitParams {
	return InferenceContextInitParams{
		Seed:           -1,
		SequenceLength: 0,
	}
}

type InferenceContextParams struct {
	SequenceLength int // context size used during inference
}

type InferenceContext struct {
	model         *model.Model
	contextParams *InferenceContextParams

	randomNumberGenerator *rand.Rand
}

func NewInferenceContext(model *model.Model, initParams InferenceContextInitParams) *InferenceContext {
	// See: https://github.com/ggerganov/llama.cpp/blob/a7aee47b98e45539d491071b25778b833b77e387/llama.cpp#L9304C14-L9304C14
	context := &InferenceContext{
		model:         model,
		contextParams: &InferenceContextParams{},
	}
	contextParams := context.contextParams
	if initParams.Seed == -1 {
		initParams.Seed = time.Now().UnixNano()
	}
	if initParams.SequenceLength > 0 {
		contextParams.SequenceLength = initParams.SequenceLength
	} else {
		contextParams.SequenceLength = model.ModelArgs.MaxSequenceLength
	}
	context.randomNumberGenerator = rand.New(rand.NewSource(initParams.Seed))

	kvCacheInit(context)

	return context
}

type InferenceEngine struct {
	context *InferenceContext
}

func NewInferenceEngine(context *InferenceContext) *InferenceEngine {
	return &InferenceEngine{
		context: context,
	}
}

func (ie *InferenceEngine) Generate(promptBatches [][]model.TokenId) ([][]model.TokenId, error) {
	sequence, minPromptLength, sequenceLength := ie.createTokenSequence(promptBatches)
	prevPos := 0
	for curPos := minPromptLength; curPos < sequenceLength; curPos++ {
		input := make([][]model.TokenId, len(sequence))
		for i := range input {
			input[i] = sequence[i][prevPos:curPos]
		}
		/*logits := */ err := ie.context.model.Transformer.Forward(input, prevPos)
		if err != nil {
			return nil, err
		}
	}
	return nil, nil
}

func (ie *InferenceEngine) createTokenSequence(promptTokenBatches [][]model.TokenId) (result [][]model.TokenId, minBatchLength int, sequenceLength int) {
	sequenceLength = ie.context.contextParams.SequenceLength
	result = make([][]model.TokenId, len(promptTokenBatches))
	minBatchLength = math.MaxInt
	for i, promptTokenBatch := range promptTokenBatches {
		minBatchLength = int(math.Min(float64(minBatchLength), float64(len(promptTokenBatch))))
		batch := createEmptyTokenBatch(sequenceLength, ie.context.model.Vocabulary.PadId)
		copy(batch[:len(promptTokenBatch)], promptTokenBatch)
		result[i] = batch
	}
	return
}

func createEmptyTokenBatch(sequenceLength int, padId model.TokenId) []model.TokenId {
	result := make([]model.TokenId, sequenceLength)
	for i := 0; i < sequenceLength; i++ {
		result[i] = padId
	}
	return result
}

func kvCacheInit(context *InferenceContext) {
	// See: https://github.com/ggerganov/llama.cpp/blob/a7aee47b98e45539d491071b25778b833b77e387/llama.cpp#L1555
	/*
		config := context.model.ModelArgs
		contextParams := context.contextParams
		n_embd := config.N_embd_grouped_query_attn()
		n_layer := config.N_layer
		n_ctx := contextParams.SequenceLength

		n_mem := n_layer * n_ctx
		n_elements := n_embd * n_mem
		n_elements = n_elements
	*/
	//panic("Not implemented")
}

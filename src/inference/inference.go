package inference

import (
	"math/rand"
	"time"

	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

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

type InferenceContextInitParams struct {
	Seed  int64 // RNG (Random Number Generator) seed, -1 for random
	N_ctx int   // text context, 0 = from model
}

func NewInferenceContextInitParams() InferenceContextInitParams {
	return InferenceContextInitParams{
		Seed:  -1,
		N_ctx: 0,
	}
}

type InferenceContextParams struct {
	N_ctx int // context size used during inference
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
	if initParams.N_ctx > 0 {
		contextParams.N_ctx = initParams.N_ctx
	} else {
		contextParams.N_ctx = model.Config.N_ctx
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

func kvCacheInit(context *InferenceContext) {
	// See: https://github.com/ggerganov/llama.cpp/blob/a7aee47b98e45539d491071b25778b833b77e387/llama.cpp#L1555
	config := context.model.Config
	contextParams := context.contextParams
	n_embd := config.N_embd_grouped_query_attn()
	n_layer := config.N_layer
	n_ctx := contextParams.N_ctx

	n_mem := n_layer * n_ctx
	n_elements := n_embd * n_mem
	n_elements = n_elements
}

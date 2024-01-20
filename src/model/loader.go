package model

import (
	"fmt"
	"path/filepath"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/ml"
	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
	"github.com/adalkiran/llama-nuts-and-bolts/src/torch"
)

const (
	BYTES_MEGABYTE = 1024 * 1024
	BYTES_GIGABYTE = 1024 * 1024 * 1024
)

func LoadModel(modelFilePath string) (*Model, error) {
	torchModelReader, err := torch.NewTorchModelReader(modelFilePath)
	if err != nil {
		return nil, err
	}
	defer torchModelReader.Close()
	common.GLogger.ConsolePrintf("Loading model file: \"%s\"...", modelFilePath)
	modelTensors, err := torchModelReader.Load()
	if err != nil {
		return nil, err
	}
	model := &Model{Tensors: modelTensors}
	err = loadModelArgs(modelFilePath, model)
	if err != nil {
		return nil, err
	}
	err = loadVocab(modelFilePath, model)
	if err != nil {
		return nil, err
	}

	common.GLogger.ConsolePrintf("Found %d tensors in the model.", len(model.Tensors.GetKeys()))

	err = checkModelArgs(model)
	if err != nil {
		return nil, err
	}

	model.ModelArchitecture = ModelArchitectureLlama
	switch model.ModelArgs.N_Layers {
	case 32:
		model.ModelType = ModelType7B
	}

	if model.Transformer, err = NewLlamaTransformer(model); err != nil {
		printMeta(model)
		return nil, err
	}

	printMeta(model)

	return model, nil
}

func loadModelArgs(modelFilePath string, model *Model) error {
	configFilePath := filepath.Dir(modelFilePath) + "/params.json"
	common.GLogger.ConsolePrintf("Loading model configuration file: \"%s\"...", configFilePath)
	modelArgs, err := loadModelArgsFromFile(configFilePath)
	if err != nil {
		return err
	}
	model.ModelArgs = modelArgs
	common.GLogger.ConsolePrintf("Model configuration: %v", *model.ModelArgs)
	return nil
}

func loadVocab(modelFilePath string, model *Model) error {
	vocabFilePath := filepath.Dir(modelFilePath) + "/tokenizer.model"
	common.GLogger.ConsolePrintf("Loading vocabulary/tokens file: \"%s\"...", vocabFilePath)
	vocabModelProto, err := sentencepiece.Load(vocabFilePath)
	if err != nil {
		return err
	}
	model.Vocabulary = NewVocabulary(vocabModelProto)
	common.GLogger.ConsolePrintf("Found %d tokens in the model.", len(model.Vocabulary.IdToToken))
	return nil
}

func checkModelArgs(model *Model) error {
	errList := make([]string, 0)
	modelArgs := model.ModelArgs

	// Compare VocabSize vs. model.Vocabulary.idToToken length
	if modelArgs.VocabSize < 1 {
		modelArgs.VocabSize = len(model.Vocabulary.IdToToken)
	} else {
		if modelArgs.VocabSize != len(model.Vocabulary.IdToToken) {
			errList = append(errList, fmt.Sprintf("VocabSize=%d and vocabulary model length=%d aren't equal", model.ModelArgs.VocabSize, len(model.Vocabulary.IdToToken)))
		}
	}

	if len(errList) == 0 {
		return nil
	} else {
		return fmt.Errorf("error while checking config and model: %s", errList)
	}
}

func printMeta(model *Model) {
	fmt.Print("\nTensors:\n")
	fmt.Print("=================================\n")
	for i, tensorName := range model.Tensors.GetKeys() {
		tensor, _ := model.Tensors.Get(tensorName)
		fmt.Printf("Tensor %4d: %-48s | %-6s | %v\n", i, tensorName, tensor.DataType.Name, tensor.Size)
	}

	fmt.Print("\nModel Metadata:\n")
	fmt.Print("=================================\n")

	fmt.Printf("Properties from model files:\n")
	fmt.Printf("%-60s = %s\n", "Format", "Torch model")
	fmt.Printf("%-60s = %s\n", "Architecture", model.ModelArchitecture.String())
	fmt.Printf("%-60s = %s\n", "Vocabulary type", "SPM (SentencePiece)")

	fmt.Printf("\nProperties from model configuration:\n")

	fmt.Printf("%-60s = %d\n", "VocabSize (tokenizer length)", model.ModelArgs.VocabSize)
	fmt.Printf("%-60s = %d\n", "MaxSequenceLength (max context length)", model.ModelArgs.MaxSequenceLength)
	fmt.Printf("%-60s = %d\n", "Dim (embedding dimension)", model.ModelArgs.Dim)
	fmt.Printf("%-60s = %d\n", "N_Heads (attention head count)", model.ModelArgs.N_Heads)
	n_KVHeadsDefaultStr := ""
	if model.ModelArgs.N_KVHeads == -1 {
		n_KVHeadsDefaultStr = " (set to default value of N_Heads)"
	}
	fmt.Printf("%-60s = %d%s\n", "N_KVHeads (attention head count KV)", model.ModelArgs.N_KVHeads, n_KVHeadsDefaultStr)
	fmt.Printf("%-60s = %d\n", "N_Layers (layer count)", model.ModelArgs.N_Layers)
	fmt.Printf("%-60s = %.1e\n", "NormEpsilon (attention layernorm epsilon)", model.ModelArgs.NormEpsilon)
	fmt.Printf("%-60s = %d\n", "MultipleOf (for feed forward SwiGLU alignment)", model.ModelArgs.MultipleOf)
	if model.ModelArgs.FFNDimMultiplier > -1 {
		fmt.Printf("%-60s = %.1e\n", "FFNDimMultiplier (custom multiplier for hidden dimension)", model.ModelArgs.FFNDimMultiplier)
	} else {
		fmt.Printf("%-60s = %s\n", "FFNDimMultiplier (custom multiplier for hidden dimension)", "None")
	}

	fmt.Printf("\nProperties by calculation:\n")

	headDim := -1
	if model.Transformer != nil && len(model.Transformer.Layers) > 0 && model.Transformer.Layers[0].attention != nil {
		headDim = model.Transformer.Layers[0].attention.HeadDim
	}
	fmt.Printf("%-60s = %d\n", "HeadDim (dimension of each attention head)", headDim)

	ffnHiddenDim := -1
	if model.Transformer != nil && len(model.Transformer.Layers) > 0 && model.Transformer.Layers[0].feedForward != nil {
		ffnHiddenDim = model.Transformer.Layers[0].feedForward.FFNHiddenDim
	}

	fmt.Printf("%-60s = %d\n", "FFNHiddenDim (feed forward network hidden layer dimension)", ffnHiddenDim)

	fmt.Printf("\nModel statistics:\n")

	fmt.Printf("%-60s = %s\n", "Model type", model.ModelType.String())
	elementCount := float64(model.GetElementCount())
	fmt.Printf("%-60s = %.2f B\n", "Model element count", elementCount*1e-9)
	bytesCount := float64(model.GetBytesCount())
	bitsPerElement := 8 * bytesCount / elementCount
	if bytesCount < BYTES_GIGABYTE {
		fmt.Printf("%-60s = %.2f MB (%.2f bits per element)\n", "model element count", bytesCount/(BYTES_MEGABYTE), bitsPerElement)
	} else {
		fmt.Printf("%-60s = %.2f GB (%.2f bits per element)\n", "model element count", bytesCount/(BYTES_GIGABYTE), bitsPerElement)
	}

}

func getTensor(model *Model, name string, expectedShape []int) (*ml.Tensor, error) {
	result, ok := model.Tensors.Get(name)
	if !ok {
		return nil, fmt.Errorf("tensor \"%s\" not found", name)
	}
	if fmt.Sprintf("%v", result.Size) != fmt.Sprintf("%v", expectedShape) {
		return nil, fmt.Errorf("tensor \"%s\" has incorrect shape; expected %v, got %v", name, expectedShape, result.Size)
	}
	return result, nil
}

func getLayerTensor(model *Model, nameFormat string, layerIndex int, expectedShape []int) (*ml.Tensor, error) {
	name := fmt.Sprintf(nameFormat, layerIndex)
	return getTensor(model, name, expectedShape)
}

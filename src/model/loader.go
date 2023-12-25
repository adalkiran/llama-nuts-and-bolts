package model

import (
	"fmt"
	"path/filepath"

	"github.com/adalkiran/llama-nuts-and-bolts/src/sentencepiece"
	"github.com/adalkiran/llama-nuts-and-bolts/src/torch"
)

const (
	BYTES_MEGABYTE = 1024 * 1024
	BYTES_GIGABYTE = 1024 * 1024 * 1024
)

func LoadModel(modelFilePath string) (*Model, error) {
	torchModelReader := torch.NewTorchModelReader(modelFilePath)
	defer torchModelReader.Close()
	fmt.Printf("Loading model file: \"%s\"...\n", modelFilePath)
	modelTensors, err := torchModelReader.Load()
	if err != nil {
		return nil, err
	}
	model := &Model{Tensors: modelTensors}
	err = loadConfig(modelFilePath, model)
	if err != nil {
		return nil, err
	}
	err = loadVocab(modelFilePath, model)
	if err != nil {
		return nil, err
	}

	fmt.Printf("Found %d tensors in the model.\n", len(model.Tensors.GetKeys()))

	err = checkConfig(model)
	if err != nil {
		return nil, err
	}

	model.ModelArchitecture = ModelArchitectureLlama
	switch model.Config.N_layer {
	case 32:
		model.ModelType = ModelType7B
	}

	printMeta(model)

	//err = defineArchitecture(model)
	if err != nil {
		return nil, err
	}

	err = loadTensors(torchModelReader, model)
	if err != nil {
		return nil, err
	}

	return model, nil
}

func loadConfig(modelFilePath string, model *Model) error {
	configFilePath := filepath.Dir(modelFilePath) + "/params.json"
	fmt.Printf("Loading model configuration file: \"%s\"...\n", configFilePath)
	config, err := loadConfigFromFile(configFilePath, model)
	if err != nil {
		return err
	}
	model.Config = config
	fmt.Printf("Model configuration: %v\n", *model.Config)
	return nil
}

func loadVocab(modelFilePath string, model *Model) error {
	vocabFilePath := filepath.Dir(modelFilePath) + "/tokenizer.model"
	fmt.Printf("Loading vocabulary/tokens file: \"%s\"...\n", vocabFilePath)
	vocabModelProto, err := sentencepiece.Load(vocabFilePath)
	if err != nil {
		return err
	}
	model.Vocabulary = NewVocabulary(vocabModelProto)
	fmt.Printf("Found %d tokens in the model.\n", len(model.Vocabulary.IdToToken))
	return nil
}

func checkConfig(model *Model) error {
	errList := make([]string, 0)

	// Compare n_vocab vs. model.Vocabulary.idToToken length
	if model.Config.N_vocab < 1 {
		model.Config.N_vocab = len(model.Vocabulary.IdToToken)
	} else {
		if model.Config.N_vocab != len(model.Vocabulary.IdToToken) {
			errList = append(errList, fmt.Sprintf("n_vocab=%d and vocabulary model length=%d aren't equal", model.Config.N_vocab, len(model.Vocabulary.IdToToken)))
		}
	}

	// Compare n_embd vs. "tok_embeddings.weight" tensor shape
	embeddingTensor, ok := model.Tensors.Get("tok_embeddings.weight")

	if ok {
		if embeddingTensor.GetShape()[0] != model.Config.N_vocab {
			errList = append(errList, fmt.Sprintf("n_vocab=%d and tensor \"tok_embeddings.weight\" shape[0]=%d  aren't equal.", model.Config.N_vocab, embeddingTensor.GetShape()[0]))
		}
		if embeddingTensor.GetShape()[1] != model.Config.N_embd {
			errList = append(errList, fmt.Sprintf("n_embd=%d and tensor \"tok_embeddings.weight\" shape[1]=%d  aren't equal.", model.Config.N_embd, embeddingTensor.GetShape()[1]))
		}
	} else {
		errList = append(errList, "tensor \"tok_embeddings.weight\" not found in the model.")
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
		fmt.Printf("Tensor %4d: %-48s | %-6s | %v\n", i, tensorName, tensor.GetDataType().GetName(), tensor.GetShape())
	}

	fmt.Print("\nModel Metadata:\n")
	fmt.Print("=================================\n")

	fmt.Printf("%-50s = %s\n", "format", "Torch model")
	fmt.Printf("%-50s = %s\n", "architecture", model.ModelArchitecture.String())
	fmt.Printf("%-50s = %s\n", "vocab type", "SPM (SentencePiece)")
	fmt.Printf("%-50s = %d\n", "n_vocab (tokenizer length)", model.Config.N_vocab)
	fmt.Printf("%-50s = %d\n", "n_ctx (context length)", model.Config.N_ctx)
	fmt.Printf("%-50s = %d\n", "n_embd (embedding length)", model.Config.N_embd)
	fmt.Printf("%-50s = %d\n", "n_head (attention head count)", model.Config.N_head)
	fmt.Printf("%-50s = %d\n", "n_head_kv (attention head count KV)", model.Config.N_head_kv)
	fmt.Printf("%-50s = %d\n", "n_layer (layer count)", model.Config.N_layer)
	fmt.Printf("%-50s = %.1e\n", "f_norm_eps (attention layernorm epsilon)", model.Config.F_norm_eps)
	fmt.Printf("%-50s = %.1e\n", "f_norm_rms_eps (attention layernorm rms epsilon)", model.Config.F_norm_rms_eps)
	fmt.Printf("%-50s = %d\n", "n_ff (feed forward length)", model.Config.N_ff)
	fmt.Printf("%-50s = %s\n", "model type", model.ModelType.String())
	elementCount := float64(model.GetElementCount())
	fmt.Printf("%-50s = %.2f B\n", "model element count", elementCount*1e-9)
	bytesCount := float64(model.GetBytesCount())
	bitsPerElement := 8 * bytesCount / elementCount
	if bytesCount < BYTES_GIGABYTE {
		fmt.Printf("%-50s = %.2f MB (%.2f bits per element)\n", "model element count", bytesCount/(BYTES_MEGABYTE), bitsPerElement)
	} else {
		fmt.Printf("%-50s = %.2f GB (%.2f bits per element)\n", "model element count", bytesCount/(BYTES_GIGABYTE), bitsPerElement)
	}
}

func defineArchitecture(model *Model) error {
	n_embd := model.Config.N_embd                                         // 4096
	n_vocab := model.Config.N_vocab                                       // 32000
	n_ff := model.Config.N_ff                                             // 11008
	n_embd_grouped_query_attn := model.Config.N_embd_grouped_query_attn() // 4096
	var err error
	if model.tok_embd, err = getTensor(model, "tok_embeddings.weight", []int{n_embd, n_vocab}); err != nil {
		return err
	}

	model.Layers = make([]*Layer, model.Config.N_layer)

	for i := 0; i < model.Config.N_layer; i++ {
		layer := &Layer{}

		// normalization
		if layer.attn_norm, err = getLayerTensor(model, "layers.%s.attention_norm.weight", i, []int{n_embd}); err != nil {
			return err
		}

		// attention
		if layer.attn_wq, err = getLayerTensor(model, "layers.%s.attention.wq.weight", i, []int{n_embd, n_embd}); err != nil {
			return err
		}
		if layer.attn_wk, err = getLayerTensor(model, "layers.%s.attention.wk.weight", i, []int{n_embd, n_embd_grouped_query_attn}); err != nil {
			return err
		}
		if layer.attn_wv, err = getLayerTensor(model, "layers.%s.attention.wv.weight", i, []int{n_embd, n_embd_grouped_query_attn}); err != nil {
			return err
		}
		if layer.attn_wo, err = getLayerTensor(model, "layers.%s.attention.wo.weight", i, []int{n_embd, n_embd}); err != nil {
			return err
		}
		// feed forward normalization
		if layer.ffn_norm, err = getLayerTensor(model, "layers.%s.ffn_norm.weight", i, []int{n_embd}); err != nil {
			return err
		}

		// feed forward
		if layer.ffn_gate, err = getLayerTensor(model, "layers.%s.feed_forward.w1.weight", i, []int{n_embd, n_ff}); err != nil {
			return err
		}
		if layer.ffn_down, err = getLayerTensor(model, "layers.%s.feed_forward.w2.weight", i, []int{n_ff, n_embd}); err != nil {
			return err
		}
		if layer.ffn_up, err = getLayerTensor(model, "layers.%s.feed_forward.w3.weight", i, []int{n_embd, n_ff}); err != nil {
			return err
		}

	}

	if model.output_norm, err = getTensor(model, "norm.weight", []int{n_embd}); err != nil {
		return err
	}

	if model.output, err = getTensor(model, "output.weight", []int{n_embd, n_vocab}); err != nil {
		return err
	}
	return nil
}

func getTensor(model *Model, name string, expectedShape []int) (*torch.TensorDescriptor, error) {
	result, ok := model.Tensors.Get(name)
	if !ok {
		return nil, fmt.Errorf("tensor \"%s\" not found", name)
	}
	if fmt.Sprintf("%v", result.GetShape()) != fmt.Sprintf("%v", expectedShape) {
		return nil, fmt.Errorf("tensor \"%s\" has incorrect shape; expected %v, got %v", name, expectedShape, result.GetShape())
	}
	return result, nil
}

func getLayerTensor(model *Model, nameFormat string, layerIndex int, expectedShape []int) (*torch.TensorDescriptor, error) {
	name := fmt.Sprintf(nameFormat, layerIndex)
	return getTensor(model, name, expectedShape)
}

func loadTensors(torchModelReader *torch.TorchModelReader, model *Model) error {
	tok_embeddings, _ := model.Tensors.Get(model.Tensors.GetKeys()[0])
	tok_embeddings.Load(torchModelReader)
	return nil
}

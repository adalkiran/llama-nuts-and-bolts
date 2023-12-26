package main

import (
	"fmt"
	"log"

	"github.com/adalkiran/llama-nuts-and-bolts/src/inference"
	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
)

func main() {
	fmt.Println("Welcome to Llama Nuts and Bolts!")
	fmt.Print("=================================\n\n\n")
	modelFilePath := "../models-original/7B/consolidated.00.pth"

	prompts := []string{
		"Hello my name is",
		"Je m'appelle",
	}

	//initialize()

	llamaModel, err := model.LoadModel(modelFilePath)
	if err != nil {
		log.Fatal(err)
	}

	contextInitParams := inference.NewInferenceContextInitParams()
	contextInitParams.Seed = 1234
	contextInitParams.SequenceLength = 8

	context := inference.NewInferenceContext(llamaModel, contextInitParams)

	engine := inference.NewInferenceEngine(context)

	tokenIdBatches, err := engine.TokenizeBatch(prompts, true)
	if err != nil {
		log.Fatal(err)
	}

	for iBatch, tokenIdBatch := range tokenIdBatches {
		fmt.Printf("\nBatch %d: ", iBatch)
		fmt.Print(engine.TokenBatchToString(tokenIdBatch))
		fmt.Println()
	}
	engine.Generate(tokenIdBatches)
}

/*

const GELU_COEF_A float32 = 0.044715
const GELU_QUICK_COEF float32 = -1.702
const SQRT_2_OVER_PI float32 = 0.79788456080286535587989211986876

var (
	table_gelu       [1 << 16]float32
	table_gelu_quick [1 << 16]float32
	table_silu       [1 << 16]float32
	table_exp        [1 << 16]float32
)

func gelu(x float32) float32 {
	//    return 0.5f  *  x  *  (  1.0f  +  tanhf(  SQRT_2_OVER_PI  *  x  *  (  1.0f  +  GELU_COEF_A * x * x )));
	fmt.Printf("x: %f\n", x)
	fmt.Printf("GELU_COEF_A: %f\n", GELU_COEF_A)
	fmt.Printf("(  1.0f  +  GELU_COEF_A * x * x )  %f\n", (1.0 + GELU_COEF_A*x*x))
	fmt.Printf("tanhf(  SQRT_2_OVER_PI  *  x  *  (  1.0f  +  GELU_COEF_A * x * x ))  %f\n", math.Tanh(float64(SQRT_2_OVER_PI*x*(1.0+GELU_COEF_A*x*x))))
	return float32(0.5 * x * (1.0 + float32(math.Tanh(float64(SQRT_2_OVER_PI*x*(1.0+GELU_COEF_A*x*x))))))
}

func gelu_quick(x float32) float32 {
	return x * (1.0 / (1.0 + float32(math.Exp(float64(GELU_QUICK_COEF*x)))))
}

func silu(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

func exp(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func initialize() {
	for i := 0; i < (1 << 16); i++ {
		ii := float32(i)
		table_gelu[i] = gelu(ii)
		table_gelu_quick[i] = gelu_quick(ii)
		table_silu[i] = silu(ii)
		table_exp[i] = exp(ii)
	}
}
*/

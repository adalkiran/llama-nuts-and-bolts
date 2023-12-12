package main

import (
	"fmt"
	"log"

	"github.com/adalkiran/llama-nuts-and-bolts/src/model"
	"github.com/adalkiran/llama-nuts-and-bolts/src/torch"
)

func main() {
	fmt.Println("Welcome to Llama Nuts and Bolts!")
	fmt.Println("=================================\n\n\n")
	modelFilePath := "../models-original/7B/consolidated.00.pth"

	model, err := model.LoadModel(modelFilePath)
	if err != nil {
		log.Fatal(err)
	}

	for _, layerName := range model.Layers.GetKeys() {
		item, _ := model.Layers.Get(layerName)
		layerTensor := item.(*torch.TensorDescriptor)
		println(fmt.Sprintf("%-48s | %-6s | %v", layerName, layerTensor.GetDataType().GetName(), layerTensor.GetShape()))
	}
}

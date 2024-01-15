package ml

import (
	"context"
	"runtime"
	"sync"
	"unsafe"

	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

type DstVal struct {
	offset int
	val    float32
}

type linearTransformation_wOutFn = func(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int, inputItemSize int,
	weightsPtr unsafe.Pointer, weightsWOutOffset int, weightsInputSize int,
	dstItemOffset int, dstValChan chan<- DstVal)

func linearTransformation_ProcessRowChan(dstF32Ptr unsafe.Pointer, dstValChan <-chan DstVal) {
	for dstVal := range dstValChan {
		// location: {rowIdx, wOutIdx}
		*(*float32)(unsafe.Add(dstF32Ptr, dstVal.offset)) = dstVal.val
	}
}

/*
	linearTransformation_BF16
*/

func linearTransformation_BF16_wOut(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int, inputItemSize int,
	weightsPtr unsafe.Pointer, weightsWOutOffset int, weightsInputSize int,
	dstItemOffset int, dstValChan chan<- DstVal) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}
	valDstF32 := float32(0)
	for wInIdx := 0; wInIdx < weightsInputSize; wInIdx++ {
		// Goal in Python manner: dst[rowIdx][wOutIdx] += input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]

		// Getting input[rowIdx][wInIdx]
		// location: {rowIdx, wInIdx}
		//val1F32 := input.GetItemByOffset_BF16(inputRowOffset + wInIdx*inputItemSize).Float32()
		val1F32 := dtype.BFloat16bitsToFloat32((*(*uint16)(unsafe.Add(inputPtr, inputRowOffset+wInIdx*inputItemSize))))

		// Getting weights[wOutIdx][wInIdx]
		// location: {wOutIdx, wInIdx}
		//val2F32 := weights.GetItemByOffset_BF16(weightsWOutOffset + wInIdx*inputItemSize).Float32()
		val2F32 := dtype.BFloat16bitsToFloat32(*(*uint16)(unsafe.Add(weightsPtr, weightsWOutOffset+wInIdx*inputItemSize)))

		//Calculating: input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]
		multiplicationValF32 := val1F32 * val2F32

		//Calculating:  dst[rowIdx][wOutIdx] += multiplicationValF32
		valDstF32 += multiplicationValF32
	}
	dstValChan <- DstVal{
		offset: dstItemOffset,
		val:    valDstF32,
	}
}

func linearTransformation_F32_wOut(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int, inputItemSize int,
	weightsPtr unsafe.Pointer, weightsWOutOffset int, weightsInputSize int,
	dstItemOffset int, dstValChan chan<- DstVal) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}
	valDstF32 := float32(0)
	for wInIdx := 0; wInIdx < weightsInputSize; wInIdx++ {
		// Goal in Python manner: dst[rowIdx][wOutIdx] += input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]

		// Getting input[rowIdx][wInIdx]
		// location: {rowIdx, wInIdx}
		val1F32 := *(*float32)(unsafe.Add(inputPtr, inputRowOffset+wInIdx*inputItemSize))

		// Getting weights[wOutIdx][wInIdx]
		// location: {wOutIdx, wInIdx}
		val2F32 := *(*float32)(unsafe.Add(weightsPtr, weightsWOutOffset+wInIdx*inputItemSize))

		//Calculating: input[rowIdx][wInIdx] * weights[wOutIdx][wInIdx]
		multiplicationValF32 := val1F32 * val2F32

		//Calculating:  dst[rowIdx][wOutIdx] += multiplicationValF32
		valDstF32 += multiplicationValF32
	}
	dstValChan <- DstVal{
		offset: dstItemOffset,
		val:    valDstF32,
	}
}

func linearTransformation_General_row(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int, inputItemSize int,
	weightsPtr unsafe.Pointer, weightsRowStride int, weightsOutputSize int, weightsInputSize int,
	dstF32Ptr unsafe.Pointer, dstRowOffset int, dstItemSize int, dstValChan chan<- DstVal, wOutFn linearTransformation_wOutFn) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}
	for wOutIdx := 0; wOutIdx < weightsOutputSize; wOutIdx++ {
		weightsWOutOffset := wOutIdx * weightsRowStride

		// location: {rowIdx, wOutIdx}
		dstItemOffset := dstRowOffset + wOutIdx*dstItemSize

		wg.Add(1)
		go wOutFn(wg, ctx, inputPtr, inputRowOffset, inputItemSize,
			weightsPtr, weightsWOutOffset, weightsInputSize, dstItemOffset, dstValChan)
	}
}

func linearTransformation_General(input *Tensor, weights *Tensor, wOutFn linearTransformation_wOutFn) (*Tensor, error) {
	// Create a cancellation context and wait group for synchronization
	ctx, _ := context.WithCancelCause(context.Background())
	var wg sync.WaitGroup

	rowsSize := input.Size[0]
	// Linear unit weights size: [out_features, in_features]
	weightsOutputSize := weights.Size[0]
	weightsInputSize := weights.Size[1]

	dstF32 := NewEmptyTensor([]int{rowsSize, weightsOutputSize}, DT_F32)

	inputItemSize := input.DataType.ItemSize()
	dstItemSize := dstF32.DataType.ItemSize()

	inputPtr := unsafe.Pointer(&input.RawData[0])
	weightsPtr := unsafe.Pointer(&weights.RawData[0])
	dstF32Ptr := unsafe.Pointer(&dstF32.RawData[0])

	inputRowStride := input.calculateByteOffset([]int{1, 0})
	weightsRowStride := weights.calculateByteOffset([]int{1, 0})
	dstRowStride := dstF32.calculateByteOffset([]int{1, 0})

	dstValChan := make(chan DstVal, 3000)

	for i := 0; i < 100; i++ {
		go linearTransformation_ProcessRowChan(dstF32Ptr, dstValChan)
	}

	for rowIdx := 0; rowIdx < rowsSize; rowIdx++ {
		inputRowOffset := rowIdx * inputRowStride
		dstRowOffset := rowIdx * dstRowStride

		wg.Add(1)
		go linearTransformation_General_row(&wg, ctx, inputPtr, inputRowOffset, inputItemSize,
			weightsPtr, weightsRowStride, weightsOutputSize, weightsInputSize,
			dstF32Ptr, dstRowOffset, dstItemSize, dstValChan, wOutFn)

	}

	select {
	case <-ctx.Done():
		close(dstValChan)
		// Cancellation signal received, one of the goroutines encountered an error
		return nil, context.Cause(ctx)
	case <-waitGroupDone(&wg):
		runtime.Gosched()
		close(dstValChan)
		return dstF32.ToBFloat16()
	}
}

func linearTransformation_BF16(input *Tensor, weights *Tensor) (*Tensor, error) {
	return linearTransformation_General(input, weights, linearTransformation_BF16_wOut)
}

func linearTransformation_F32(input *Tensor, weights *Tensor) (*Tensor, error) {
	return linearTransformation_General(input, weights, linearTransformation_F32_wOut)
}

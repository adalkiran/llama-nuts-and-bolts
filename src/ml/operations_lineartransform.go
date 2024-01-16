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

type DstRow struct {
	offset int
	val    []byte
}

type linearTransformation_wOutFn = func(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int,
	weightsPtr unsafe.Pointer, weightsWOutOffset int, weightsInputSize int,
	dstItemOffset int, dstValChan chan<- DstVal)

func linearTransformation_ProcessRowChan(dstF32 *Tensor, dstRowChan <-chan DstRow) {
	for dstRow := range dstRowChan {
		copy(dstF32.RawData[dstRow.offset:], dstRow.val)
	}
}

/*
	linearTransformation_BF16
*/

func linearTransformation_BF16_wOut(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int,
	weightsPtr unsafe.Pointer, weightsWOutOffset int, weightsInputSize int,
	dstItemOffset int, dstValChan chan<- DstVal) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}
	inputItemSize := DT_BF16.ItemSize()
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

func linearTransformation_F32_wOut(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int,
	weightsPtr unsafe.Pointer, weightsWOutOffset int, weightsInputSize int,
	dstItemOffset int, dstValChan chan<- DstVal) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}
	inputItemSize := DT_F32.ItemSize()
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

func linearTransformation_General_row(wg *sync.WaitGroup, ctx context.Context, inputPtr unsafe.Pointer, inputRowOffset int,
	weightsPtr unsafe.Pointer, weightsRowStride int, weightsOutputSize int, weightsInputSize int,
	dstRowOffset int, dstRowChan chan<- DstRow, wOutFn linearTransformation_wOutFn) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}

	wgRow := &sync.WaitGroup{}
	dstValRowChan := make(chan DstVal, weightsOutputSize)
	dstItemSize := DT_F32.ItemSize()
	dstRawLocal := make([]byte, weightsOutputSize*dstItemSize)
	dstRawLocalPtr := unsafe.Pointer(&dstRawLocal[0])
	for wOutIdx := 0; wOutIdx < weightsOutputSize; wOutIdx++ {
		weightsWOutOffset := wOutIdx * weightsRowStride

		// location: {rowIdx, wOutIdx}
		dstItemLocalOffset := wOutIdx * dstItemSize

		wgRow.Add(1)
		go wOutFn(wgRow, ctx, inputPtr, inputRowOffset,
			weightsPtr, weightsWOutOffset, weightsInputSize, dstItemLocalOffset, dstValRowChan)
	}

	runtime.Gosched()
	wgRow.Wait()
	close(dstValRowChan)

	for dstValRow := range dstValRowChan {
		*(*float32)(unsafe.Add(dstRawLocalPtr, dstValRow.offset)) = dstValRow.val
	}
	dstRowChan <- DstRow{
		offset: dstRowOffset,
		val:    dstRawLocal,
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

	inputPtr := unsafe.Pointer(&input.RawData[0])
	weightsPtr := unsafe.Pointer(&weights.RawData[0])
	//dstF32Ptr := unsafe.Pointer(&dstF32.RawData[0])

	inputRowStride := input.calculateByteOffset([]int{1, 0})
	weightsRowStride := weights.calculateByteOffset([]int{1, 0})
	dstRowStride := dstF32.calculateByteOffset([]int{1, 0})

	dstRowChan := make(chan DstRow, 3000)

	rowProcessorCount := dstF32.GetElementCount() / 100
	for i := 0; i < rowProcessorCount; i++ {
		go linearTransformation_ProcessRowChan(dstF32, dstRowChan)
	}

	if rowsSize > 1 {
		for rowIdx := 0; rowIdx < rowsSize; rowIdx++ {
			inputRowOffset := rowIdx * inputRowStride
			dstRowOffset := rowIdx * dstRowStride

			wg.Add(1)
			go linearTransformation_General_row(&wg, ctx, inputPtr, inputRowOffset,
				weightsPtr, weightsRowStride, weightsOutputSize, weightsInputSize,
				dstRowOffset, dstRowChan, wOutFn)

		}
	} else {
		inputRowOffset := 0
		dstRowOffset := 0

		wg.Add(1)
		linearTransformation_General_row(&wg, ctx, inputPtr, inputRowOffset,
			weightsPtr, weightsRowStride, weightsOutputSize, weightsInputSize,
			dstRowOffset, dstRowChan, wOutFn)
	}

	select {
	case <-ctx.Done():
		close(dstRowChan)
		// Cancellation signal received, one of the goroutines encountered an error
		return nil, context.Cause(ctx)
	case <-waitGroupDone(&wg):
		runtime.Gosched()
		close(dstRowChan)
		return dstF32.ToBFloat16()
	}
}

func linearTransformation_BF16(input *Tensor, weights *Tensor) (*Tensor, error) {
	return linearTransformation_General(input, weights, linearTransformation_BF16_wOut)
}

func linearTransformation_F32(input *Tensor, weights *Tensor) (*Tensor, error) {
	return linearTransformation_General(input, weights, linearTransformation_F32_wOut)
}

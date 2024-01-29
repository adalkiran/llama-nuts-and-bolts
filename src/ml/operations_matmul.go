package ml

import (
	"context"
	"runtime"
	"sync"
	"unsafe"

	"github.com/adalkiran/llama-nuts-and-bolts/src/common"
	"github.com/adalkiran/llama-nuts-and-bolts/src/dtype"
)

type matMul_otherColFn = func(wg *sync.WaitGroup, ctx context.Context,
	inputRowPtr unsafe.Pointer, inputColsSize int,
	otherColPtr unsafe.Pointer, otherColStride int,
	dstItemOffsetInRow int, dstValInRowChan chan<- DstVal)

func matMul_ProcessRowChan(dstF32 *Tensor, dstRowChan <-chan DstRow) {
	for dstRow := range dstRowChan {
		copy(dstF32.RawData[dstRow.rowOffset:], dstRow.val)
	}
}

func matMul_BF16_otherCol(wg *sync.WaitGroup, ctx context.Context,
	inputRowPtr unsafe.Pointer, inputColsSize int,
	otherColPtr unsafe.Pointer, otherColStride int,
	dstItemOffsetInRow int, dstValInRowChan chan<- DstVal) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}
	inputItemSize := DT_BF16.ItemSize

	valDstF32 := float32(0)

	for inputColIdx := 0; inputColIdx < inputColsSize; inputColIdx++ {
		// Goal in Python manner: dst[rowIdx][otherColIdx] += input[rowIdx][inputColIdx] * other[inputColIdx][otherColIdx]

		// location: {locFirstPart, inputColIdx, 0}

		// Getting input[rowIdx][inputColIdx], location: {locFirstPart, rowIdx, inputColIdx}
		//val1F32 := input.GetItemByOffset_BF16(inputRowOffset + inputColIdx*inputItemSize).Float32()
		val1F32 := dtype.BFloat16bitsToFloat32((*(*uint16)(unsafe.Add(inputRowPtr, inputColIdx*inputItemSize))))

		// Getting other[inputColIdx][otherColIdx], location: {locFirstPart, inputColIdx, otherColIdx}
		//val2F32 := other.GetItemByOffset_BF16(otherColOffset + inputColIdx*otherColStride).Float32()
		val2F32 := dtype.BFloat16bitsToFloat32(*(*uint16)(unsafe.Add(otherColPtr, inputColIdx*otherColStride)))

		//Calculating: input[rowIdx][inputColIdx] * other[inputColIdx][otherColIdx]
		multiplicationValF32 := val1F32 * val2F32

		//Calculating:  dst[rowIdx][otherColIdx] += multiplicationValF32
		valDstF32 += multiplicationValF32
	}
	dstValInRowChan <- DstVal{
		itemOffsetInRow: dstItemOffsetInRow,
		val:             valDstF32,
	}
}

func matMul_General_row(wg *sync.WaitGroup, ctx context.Context,
	inputRowPtr unsafe.Pointer, inputItemSize int, inputColsSize int,
	otherPtr unsafe.Pointer, otherColOffset int, otherColStride int, otherColsSize int,
	dstRowOffset int, dstRowChan chan<- DstRow, otherColFn matMul_otherColFn) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}

	wgRow := &sync.WaitGroup{}
	dstValInRowChan := make(chan DstVal, otherColsSize)
	dstItemSize := DT_F32.ItemSize
	dstRawLocal := make([]byte, otherColsSize*dstItemSize)
	dstRawLocalPtr := unsafe.Pointer(&dstRawLocal[0])
	for otherColIdx := 0; otherColIdx < otherColsSize; otherColIdx++ {
		// location: {locFirstPuart, rowIdx, otherColIdx}
		dstItemLocalOffset := otherColIdx * dstItemSize

		otherColDetailOffset := otherColOffset + otherColIdx*inputItemSize
		otherColPtr := unsafe.Add(otherPtr, otherColDetailOffset)

		wgRow.Add(1)
		go otherColFn(wgRow, ctx, inputRowPtr, inputColsSize,
			otherColPtr, otherColStride,
			dstItemLocalOffset, dstValInRowChan)
	}

	runtime.Gosched()
	wgRow.Wait()
	close(dstValInRowChan)

	for dstValInRow := range dstValInRowChan {
		*(*float32)(unsafe.Add(dstRawLocalPtr, dstValInRow.itemOffsetInRow)) = dstValInRow.val
	}
	dstRowChan <- DstRow{
		rowOffset: dstRowOffset,
		val:       dstRawLocal,
	}
}

func matMul_General_firstPart(wg *sync.WaitGroup, ctx context.Context,
	input *Tensor, locFirstPart []int,
	other *Tensor, otherColOffset int, otherColStride int, otherColsSize int,
	dstF32 *Tensor, dstRowChan chan<- DstRow, otherColFn matMul_otherColFn) {
	defer wg.Done()

	if ctx.Err() != nil {
		return
	}

	inputItemSize := input.DataType.ItemSize
	inputRowsSize := input.Size[len(input.Size)-2]
	inputColsSize := input.Size[len(input.Size)-1]

	inputPtr := unsafe.Pointer(&input.RawData[0])
	otherPtr := unsafe.Pointer(&other.RawData[0])

	for rowIdx := 0; rowIdx < inputRowsSize; rowIdx++ {
		// location: {locFirstPart, rowIdx, 0}
		rowLocStart := append(append([]int{}, locFirstPart...), []int{rowIdx, 0}...)
		inputRowOffset := input.calculateByteOffset(rowLocStart)
		dstRowOffset := dstF32.calculateByteOffset(rowLocStart)
		inputRowPtr := unsafe.Add(inputPtr, inputRowOffset)

		wg.Add(1)
		go matMul_General_row(wg, ctx,
			inputRowPtr, inputItemSize, inputColsSize,
			otherPtr, otherColOffset, otherColStride, otherColsSize,
			dstRowOffset, dstRowChan, otherColFn)
	}

}

func matMul_General(input *Tensor, other *Tensor, otherColFn matMul_otherColFn) (*Tensor, error) {
	// Create a cancellation context and wait group for synchronization
	ctx, _ := context.WithCancelCause(context.Background())
	var wg sync.WaitGroup

	inputRowsSize := input.Size[len(input.Size)-2]

	otherColsSize := other.Size[len(other.Size)-1]

	inputSizeFirstPart := input.Size[0 : len(input.Size)-2]
	dstF32 := NewEmptyTensor(append(append([]int{}, inputSizeFirstPart...), inputRowsSize, otherColsSize), DT_F32)

	firstPartElementCount := 1
	for dimension := 0; dimension < len(inputSizeFirstPart); dimension++ {
		firstPartElementCount *= inputSizeFirstPart[dimension]
	}
	dstRowChanSize := firstPartElementCount * inputRowsSize
	rowProcessorCount := firstPartElementCount * inputRowsSize

	dstRowChan := make(chan DstRow, int(dstRowChanSize))

	for i := 0; i < rowProcessorCount; i++ {
		go matMul_ProcessRowChan(dstF32, dstRowChan)
	}
	for iteratorFirstPart := IterateOverSize(inputSizeFirstPart, 0); iteratorFirstPart.HasNext(); {
		locFirstPart := iteratorFirstPart.Next()
		otherColOffset := other.calculateByteOffset(append(append([]int{}, locFirstPart...), []int{0, 0}...))
		otherColStride := other.calculateByteOffset(append(append([]int{}, locFirstPart...), []int{1, 0}...)) - otherColOffset
		wg.Add(1)
		go matMul_General_firstPart(&wg, ctx,
			input, locFirstPart,
			other, otherColOffset, otherColStride, otherColsSize,
			dstF32, dstRowChan, otherColFn)
	}

	runtime.Gosched()
	select {
	case <-ctx.Done():
		close(dstRowChan)
		// Cancellation signal received, one of the goroutines encountered an error
		return nil, context.Cause(ctx)
	case <-common.WaitGroupDone(&wg):
		runtime.Gosched()
		close(dstRowChan)
		return dstF32.ToBFloat16()
	}
}

func matMul_BF16(input *Tensor, weights *Tensor) (*Tensor, error) {
	return matMul_General(input, weights, matMul_BF16_otherCol)
}

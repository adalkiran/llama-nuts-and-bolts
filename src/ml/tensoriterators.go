package ml

type OneTensorIterator struct {
	loc  []int
	size []int

	ignoreTrailingDimensions int
	maxMeaningfulDim         int
	currentIndex             int64
	itemCount                int64
}

func (it *OneTensorIterator) Next() (loc []int) {
	for dimension := it.maxMeaningfulDim; dimension >= 0; dimension-- {
		if it.loc[dimension] < it.size[dimension]-1 {
			it.loc[dimension]++
			it.currentIndex++
			return it.loc
		} else if dimension > 0 {
			it.loc[dimension] = 0
		}
	}
	return it.loc
}

func (it *OneTensorIterator) HasNext() bool {
	return it.currentIndex < it.itemCount-1
}

type TwoTensorIterator struct {
	refTensorIterator *OneTensorIterator

	size                     []int
	ignoreTrailingDimensions int
}

func (it *TwoTensorIterator) Next() (loc1 []int, loc2 []int) {
	refLoc := it.refTensorIterator.Next()
	loc := make([]int, len(it.size))
	dimDiff := len(it.refTensorIterator.size) - len(it.size)
	for dimension := 0; dimension < len(it.size); dimension++ {
		loc[dimension] = refLoc[dimDiff+dimension] % it.size[dimension]
	}
	return refLoc, loc
}

func (it *TwoTensorIterator) HasNext() bool {
	return it.refTensorIterator.HasNext()
}

func calculateIteratorLimits(size []int) (maxMeaningfulDim int, itemCount int64) {
	maxMeaningfulDim = 0
	for dimension := len(size) - 1; dimension >= 0; dimension-- {
		if size[dimension] > 1 {
			maxMeaningfulDim = dimension
			break
		}
	}
	itemCount = int64(1)
	for _, sizeItem := range size {
		itemCount = itemCount * int64(sizeItem)
	}
	return
}

func IterateOver(tensor *Tensor, ignoreTrailingDimensions int) *OneTensorIterator {
	maxMeaningfulDim, itemCount := calculateIteratorLimits(tensor.Size[0 : len(tensor.Size)-ignoreTrailingDimensions])
	loc := make([]int, len(tensor.Size))
	loc[maxMeaningfulDim] = -1

	return &OneTensorIterator{
		loc:                      loc,
		size:                     tensor.Size,
		ignoreTrailingDimensions: ignoreTrailingDimensions,
		maxMeaningfulDim:         maxMeaningfulDim,
		currentIndex:             -1,
		itemCount:                itemCount,
	}
}

func IterateOverTwo(refTensor *Tensor, expandingTensor *Tensor, ignoreTrailingDimensions int) *TwoTensorIterator {
	return &TwoTensorIterator{
		refTensorIterator:        IterateOver(refTensor, ignoreTrailingDimensions),
		size:                     expandingTensor.Size,
		ignoreTrailingDimensions: ignoreTrailingDimensions,
	}
}

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
	if len(it.size) == 0 {
		it.currentIndex = it.itemCount
		return it.loc
	}
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

func IterateOverSize(size []int, ignoreTrailingDimensions int) *OneTensorIterator {
	maxMeaningfulDim, itemCount := calculateIteratorLimits(size[0 : len(size)-ignoreTrailingDimensions])
	loc := make([]int, len(size))
	if len(size) > 0 {
		loc[maxMeaningfulDim] = -1
	}

	return &OneTensorIterator{
		loc:                      loc,
		size:                     size,
		ignoreTrailingDimensions: ignoreTrailingDimensions,
		maxMeaningfulDim:         maxMeaningfulDim,
		currentIndex:             -1,
		itemCount:                itemCount,
	}
}

func IterateOver(tensor *Tensor, ignoreTrailingDimensions int) *OneTensorIterator {
	return IterateOverSize(tensor.Size, ignoreTrailingDimensions)
}

func IterateOverTwoSize(refSize []int, expandingSize []int, ignoreTrailingDimensions int) *TwoTensorIterator {
	return &TwoTensorIterator{
		refTensorIterator:        IterateOverSize(refSize, ignoreTrailingDimensions),
		size:                     expandingSize,
		ignoreTrailingDimensions: ignoreTrailingDimensions,
	}
}

func IterateOverTwo(refTensor *Tensor, expandingTensor *Tensor, ignoreTrailingDimensions int) *TwoTensorIterator {
	return IterateOverTwoSize(refTensor.Size, expandingTensor.Size, ignoreTrailingDimensions)
}

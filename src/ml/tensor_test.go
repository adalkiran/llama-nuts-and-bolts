package ml

import "testing"

func TestCheckBroadcastableOnce(t *testing.T) {
	expected := true
	actual := CheckBroadcastableOnce([]int{6, 3}, []int{2, 1})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = false
	actual = CheckBroadcastableOnce([]int{2, 1}, []int{6, 3})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = true
	actual = CheckBroadcastableOnce([]int{5, 2}, []int{5, 2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = false
	actual = CheckBroadcastableOnce([]int{5, 3}, []int{5, 2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = false
	actual = CheckBroadcastableOnce([]int{2}, []int{5, 2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}

	expected = true
	actual = CheckBroadcastableOnce([]int{5, 2}, []int{2})
	if actual != expected {
		t.Errorf("Expected %v, but got %v", expected, actual)
	}
}

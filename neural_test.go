package neural

import (
	"fmt"
	"testing"
)

func slicesEqual(a, b [][]float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, row := range a {
		for j := range row {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}

	return true
}

func TestMatrixMultiply(t *testing.T) {
	A := Initialize(3, 4)
	A.data = [][]float64{
		{1, 2, 3, 4},
		{2, 3, 4, 5},
		{3, 4, 5, 6},
	}

	B := Initialize(4, 2)
	B.data = [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
		{7, 8},
	}

	result, err := MatrixMultiply(A, B)
	if err != nil {
		t.Errorf("thinks the dimensions r off lol")
	}
	if slicesEqual(result.data, [][]float64{
		{50, 60}, {66, 80}, {82, 100},
	}) {
		t.Log("Basic MM test passed ...")

	} else {
		t.Errorf("u fucked up matrix multiply")
	}

	//t.Logf("Passed matrix multiplication")
}

func TestMatrixSlice(t *testing.T) {
	A := Initialize(3, 4)
	A.data = [][]float64{
		{1, 2, 3, 4},
		{2, 3, 4, 5},
		{3, 4, 5, 6},
	}

	subSet, err := A.slice(1, 1, 2, 2)
	expected := [][]float64{
		{3, 4},
		{4, 5},
	}

	if err != nil {
		t.Errorf("you fucked up slicing matricies")
	}

	if slicesEqual(subSet.data, expected) {
		t.Log("super thourough slice test passed...")
	} else {
		t.Errorf("womp womp slices dont work")
		fmt.Println((subSet.data))
	}
}

func TestReLU(t *testing.T) {
	A := Initialize(3, 4)
	A.data = [][]float64{
		{1, 2, 3, 4},
		{2, -1, 4, 5},
		{3, 4, -0.2, -2},
	}

	activated := [][]float64{
		{1, 2, 3, 4},
		{2, 0, 4, 5},
		{3, 4, 0, 0},
	}

	ReLU(A)

	if slicesEqual(A.data, activated) {
		t.Log("*extensive* activation layer function test passed ...")
	} else {
		t.Errorf("its ok u prolly dont need this function anyway")
	}
}

func TestMaxPool(t *testing.T) {
	A := Initialize(4, 4)
	A.data = [][]float64{
		{1, 2, 3, 4},
		{2, 1, 4, 5},
		{3, 4, 0, 0},
		{0, 0, 0, 3},
	}

	bluePrint := [][]float64{
		{2, 5},
		{4, 3},
	}

	pooled, err := MaxPool(A, 2)
	if err != nil {
		t.Errorf("max pooling just got dropped in a pool")
	}
	if slicesEqual(pooled.data, bluePrint) {
		t.Log("pooling worked a grand total of once ...")
	} else {
		t.Errorf("welp someone is gonna crack there skull")
	}
}

func TestConvolve(t *testing.T) {
	A := Initialize(4, 4)
	A.data = [][]float64{
		{1, 2, 3, 4},
		{2, 1, 4, 5},
		{3, 4, 0, 0},
		{0, 0, 0, 3},
	}

	B := Initialize(2, 2)
	B.data = [][]float64{
		{1, 1},
		{-1, -1},
	}

	convolved, err := Convolve(A, B, 2)
	if err != nil {
		t.Error("the main thang dont work lol")
	}

	t.Log("we dont even have a test for this one so we cant fail ...")

	//ok like i didnt wanna compute this by hand
	fmt.Println(convolved.data)
}

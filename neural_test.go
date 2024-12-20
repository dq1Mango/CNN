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
	(*A.Matrix) = [][]float64{
		{1, 2, 3, 4},
		{2, 3, 4, 5},
		{3, 4, 5, 6},
	}

	B := Initialize(3, 4)
	(*B.Matrix) = [][]float64{
		{1, 2},
		{3, 4},
		{5, 6},
		{7, 8},
	}

	result, err := MatrixMultiply(A, B)
	if err != nil {
		t.Errorf("thinks the dimensions r off lol")
	}
	if slicesEqual(*result.Matrix, [][]float64{
		{50, 60}, {66, 80}, {82, 100},
	}) {
		fmt.Println("Basic MM test passed ...")
	} else {
		t.Errorf("u fucked up matrix multiply")
	}

	t.Logf("Passed matrix multiplication")
}

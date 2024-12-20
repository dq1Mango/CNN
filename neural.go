package neural

import (
	"fmt"
	"math"
)

type Matrix struct {
	Matrix *[][]float64
	rows   int
	cols   int
}

func Initialize(rows, cols int) *Matrix {
	var matrix Matrix

	array := make([][]float64, rows)
	//might want to add this later idunno
	/*if len(initial) != 0 {
	if len(initial) != rows {
		return &matrix, fmt.Errorf("supplied matrix does not match row dimension")
	}

	for i, row := range initial {
		if len(row) != cols {
			return &matrix, fmt.Errorf("supplied matrix does match collumn dimension")
		}
		for j, col := range row {
			slice[i][j] = col
		}
	}*/

	//maybe theres a better way to do this idk
	for i := range rows {
		array[i] = make([]float64, cols)
	}

	matrix.Matrix = &array
	matrix.rows = rows
	matrix.cols = cols

	return &matrix
}

func MatrixMultiply(A, B *Matrix) (*Matrix, error) {
	C := Initialize(A.rows, B.cols)

	if (*A).cols != (*B).rows {
		fmt.Println("error lol")
		return C, fmt.Errorf("invalid dimensions")
	}

	for i := range (*A).rows {
		for j := range (*B).cols {
			sum := 0.0
			for k := range (*A).cols {
				//fmt.Println(i, j, k)
				sum += (*A.Matrix)[i][k] * (*B.Matrix)[k][j]
			}
			(*C.Matrix)[i][j] = sum
		}

	}

	return C, nil
}

func DotProduct(A, B *Matrix) (float64, error) {
	product := 0.0

	if (*A).rows != (*B).rows || (*A).cols != (*B).cols {
		fmt.Println("invalid dimensions")
		return product, fmt.Errorf("invalid dimensions")
	}

	for i := range (*A).rows {
		for j := range (*A).cols {
			product += (*A.Matrix)[i][j] * (*B.Matrix)[i][j]
		}
	}

	return product, nil
}

func ReLU(num *float64) float64 {
	return math.Max(0, *num)
}

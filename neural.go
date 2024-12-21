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

func (matrix *Matrix) slice(row, col, height, width int) (*Matrix, error) {

	slice := Initialize(height, width)

	if row < 0 || col < 0 || row+height > matrix.rows || col+width > matrix.cols {
		return slice, fmt.Errorf("Index out of range")
	}

	for i := range height {
		for j := range width {
			(*slice.Matrix)[i][j] = (*matrix.Matrix)[i+row][j+col]
		}
	}

	return slice, nil
}

func Flatten(matrix *Matrix) *[]float64 {

	earth := []float64{} //1 AM variable names go hard

	for i := range (*matrix).rows {
		for j := range (*matrix).cols {
			earth = append(earth, math.Max(0, (*matrix.Matrix)[i][j]))
		}
	}

	return &earth
}

func ReLU(matrix *Matrix) {
	for i := range (*matrix).rows {
		for j := range (*matrix).cols {
			(*matrix.Matrix)[i][j] = math.Max(0, (*matrix.Matrix)[i][j])
		}
	}
}

// TODO: add someway to pad the matrix with 0's
func MaxPool(matrix *Matrix, size int) (*Matrix, error) {

	pooled := Initialize((*matrix).rows/size, (*matrix).cols/size)

	for i := 0; i < (*matrix).rows; i += size {
		for j := 0; j < (*matrix).cols; j += size {
			//this assumes that you have already ReLU-ed it, if not u might be cooked
			pool, _ := (*matrix).slice(i, j, size, size)
			flatPool := Flatten(pool)
			max := 0.0

			for _, value := range *flatPool {
				if value > max {
					max = value
				}
			}

			//is this slow? idk im too tired
			(*pooled.Matrix)[i/size][j/size] = max

		}
	}

	return pooled, nil
}

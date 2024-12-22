package neural

import (
	"fmt"
	"math"
	"math/rand"
	"time"
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

func MatrixAdd(A, B *Matrix) (*Matrix, error) {
	result := Initialize((*A).rows, (*A).cols)

	if (*A).rows != (*B).rows || (*A).cols != (*B).cols {
		return result, fmt.Errorf("invalid dimensions")
	}

	for i := range (*A).rows {
		for j := range (*A).cols {
			(*result.Matrix)[i][j] = (*A.Matrix)[i][j] + (*B.Matrix)[i][j]
		}
	}

	return result, nil
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

func Flatten(matrix *Matrix) *Matrix {

	earth := Initialize(1, (*matrix).rows*(*matrix).cols) //1 AM variable names go hard

	for i := range (*matrix).rows {
		for j := range (*matrix).cols {
			(*earth.Matrix)[0][i*(*matrix).cols+j] = (*matrix.Matrix)[i][j]
		}
	}

	return earth
}

func ReLU(matrix *Matrix) {
	for i := range (*matrix).rows {
		for j := range (*matrix).cols {
			(*matrix.Matrix)[i][j] = math.Max(0, (*matrix.Matrix)[i][j])
		}
	}
}

func PadRow(matrix *Matrix, padding int) {
	for range padding {
		var zeros = make([]float64, (*matrix).cols, (*matrix).cols)
		for j := range (*matrix).cols {
			zeros[j] = 0
		}
		(*matrix.Matrix) = append((*matrix.Matrix), zeros)
	}

	(*matrix).rows += padding
}

func PadCol(matrix *Matrix, padding int) {
	for range padding {
		for i := range (*matrix).rows {
			(*matrix.Matrix)[i] = append((*matrix.Matrix)[i], 0)
		}
	}
	(*matrix).cols += padding
}

// TODO: add someway to pad the matrix with 0's ... WE TO-DID IT!!!
func MaxPool(matrix *Matrix, size int) (*Matrix, error) {

	PadRow(matrix, (*matrix).rows%size)
	PadCol(matrix, (*matrix).cols%size)

	pooled := Initialize((*matrix).rows/size, (*matrix).cols/size)

	for i := 0; i < (*matrix).rows; i += size {
		for j := 0; j < (*matrix).cols; j += size {
			//this assumes that you have already ReLU-ed it, if not u might be cooked
			pool, _ := (*matrix).slice(i, j, size, size)
			flatPool := Flatten(pool)
			maxx := 0.0

			for _, value := range (*flatPool.Matrix)[0] {
				if value > maxx {
					maxx = value
				}
			}
			//is this slow? idk im too tired
			(*pooled.Matrix)[i/size][j/size] = maxx

		}
	}

	return pooled, nil
}

func Convolve(input, kernel *Matrix, stride int) (*Matrix, error) {
	PadRow(input, (*input).rows%stride)
	PadCol(input, (*input).cols%stride)

	//what is this python (pt 2)
	output := Initialize(((*input).rows-(*kernel).rows)/stride+1, ((*input).cols-(*kernel).cols)/stride+1)

	for i := 0; i <= (*input).rows-(*kernel).rows; i += stride {
		for j := 0; j <= (*input).cols-(*kernel).cols; j += stride {
			//technically this does support non square kernels, tho i have never seen one before
			slice, _ := input.slice(i, j, (*kernel).rows, (*kernel).cols)

			dot, err := DotProduct(slice, kernel)
			if err != nil {
				fmt.Println("we fucked up bad")
				return output, fmt.Errorf("uggghhhhh")
			}

			(*output.Matrix)[i/stride][j/stride] = dot
		}
	}

	return output, nil
}

func Dense(input, weights, biases *Matrix) (*Matrix, error) {
	output, _ := MatrixMultiply(input, weights)

	return MatrixAdd(output, biases)
}

type Layer struct {
	Operation string
	Kernel    *Matrix //super consistent pointer usage
	Step      int
	Weights   *Matrix
	Biases    *Matrix

	//Operation *func()
}

func CreateReLU() *Layer {
	return &Layer{
		Operation: "ReLU",
	}
}

func CreateMaxPool(size int) *Layer {
	return &Layer{
		Operation: "maxPool",
		Step:      size,
	}
}

func CreateConvolution(width, height, stride int) *Layer {
	kernel := Initialize(height, width)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := range height {
		for j := range width {
			(*kernel.Matrix)[i][j] = r.NormFloat64()
		}
	}

	return &Layer{
		Operation: "convolve",
		Kernel:    kernel,
		Step:      stride,
	}
}

func CreateDense(inputSize, outputSize int) *Layer {

	weights := Initialize(inputSize, outputSize)
	biases := Initialize(1, outputSize)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := range inputSize {
		for j := range outputSize {
			(*weights.Matrix)[i][j] = r.NormFloat64()
			(*biases.Matrix)[0][j] = r.NormFloat64()
		}
	}

	return &Layer{
		Operation: "dense",
		Kernel:    weights,
		Biases:    biases,
	}
}

type Network struct {
	Layers []Layer
}

func CreateNetwork() *Network {
	return &Network{[]Layer{}}
}

func (network *Network) Add(layer *Layer) {
	(*network).Layers = append((*network).Layers, *layer)
}

func (network *Network) Compute(input Matrix) (*Matrix, error) {
	current := &input

	for _, layer := range (*network).Layers {
		//i probably should use annonomous funtions
		if layer.Operation == "ReLU" {
			ReLU(current)
		} else if layer.Operation == "maxPool" {
			current, _ = MaxPool(current, layer.Step)
		} else if layer.Operation == "convolve" {
			current, _ = Convolve(current, layer.Kernel, layer.Step)
		} else if layer.Operation == "flatten" {
			current = Flatten(current)
		} else if layer.Operation == "dense" {
			current, _ = Dense(current, layer.Kernel, layer.Biases)
		} else {
			panic("invalid operation key")
		}
	}

	return current, nil
}

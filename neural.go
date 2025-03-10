package neural

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime/debug"
	"strconv"
	"time"
)

type Matrix struct {
	data [][]float64
	rows int
	cols int
}

func Initialize(rows, cols int) *Matrix {
	var matrix Matrix

	array := make([][]float64, rows)
	for i := range rows {
		array[i] = make([]float64, cols)
	}

	for i := range rows {
		for j := range cols {
			array[i][j] = 0
		}
	}

	matrix.data = array
	matrix.rows = rows
	matrix.cols = cols

	return &matrix
}

func MatrixAdd(A, B *Matrix) *Matrix {
	result := Initialize((*A).rows, (*A).cols)

	if (*A).rows != (*B).rows || (*A).cols != (*B).cols {
		panic("invalid dimensions")
	}

	for i := range (*A).rows {
		for j := range (*A).cols {
			result.data[i][j] = A.data[i][j] + B.data[i][j]
		}
	}

	return result
}

func MatrixSubtract(A, B *Matrix) (*Matrix, error) {
	result := Initialize((*A).rows, (*A).cols)

	if (*A).rows != (*B).rows || (*A).cols != (*B).cols {
		return result, fmt.Errorf("invalid dimensions")
	}

	for i := range (*A).rows {
		for j := range (*A).cols {
			result.data[i][j] = A.data[i][j] - B.data[i][j]
		}
	}

	return result, nil
}

func MatrixMultiply(A, B *Matrix) (*Matrix, error) {
	C := Initialize(A.rows, B.cols)

	if A.cols != B.rows {
		fmt.Println("error lol")
		debug.PrintStack()
		return C, fmt.Errorf("invalid dimensions")
	}

	for i := range A.rows {
		for j := range B.cols {
			sum := 0.0
			for k := range A.cols {
				//fmt.Println(i, j, k)
				sum += A.data[i][k] * B.data[k][j]
			}
			C.data[i][j] = sum
		}

	}

	return C, nil
}

func Sum(matrix *Matrix) float64 {
	sum := 0.0
	for _, row := range matrix.data {
		for _, value := range row {
			sum += value
		}
	}

	return sum
}

func DotProduct(A, B *Matrix) (float64, error) {
	product := 0.0

	if A.rows != B.rows || A.cols != B.cols {
		fmt.Println("invalid dimensions")
		return product, fmt.Errorf("invalid dimensions")
	}

	for i := range A.rows {
		for j := range A.cols {
			product += A.data[i][j] * B.data[i][j]
		}
	}

	return product, nil
}

func (matrix *Matrix) slice(row, col, height, width int) (*Matrix, error) {

	slice := Initialize(height, width)

	if row < 0 || col < 0 || row+height > matrix.rows || col+width > matrix.cols {
		return slice, fmt.Errorf("index out of range")
	}

	for i := range height {
		for j := range width {
			slice.data[i][j] = matrix.data[i+row][j+col]
		}
	}

	return slice, nil
}

func (matrix *Matrix) scale(scaler float64) {
	for i := range matrix.rows {
		for j := range matrix.cols {
			matrix.data[i][j] *= scaler
		}
	}
}

func (matrix *Matrix) duplicate() *Matrix {
	copyy := Initialize(matrix.rows, matrix.cols)
	for i := range matrix.rows {
		for j := range matrix.cols {
			copyy.data[i][j] = matrix.data[i][j]
		}
	}
	return copyy
}

func Flatten(matricies []Matrix) *Matrix {
	earth := Initialize(1, len(matricies)*matricies[0].rows*matricies[0].cols) //1 AM variable names go hard
	for index, matrix := range matricies {
		for i := range matrix.rows {
			for j := range matrix.cols {
				earth.data[0][index*matrix.cols*matrix.rows+i*matrix.cols+j] = matrix.data[i][j]
			}
		}
	}

	return earth
}

func ReLU(matrix Matrix) *Matrix {
	activated := Initialize(matrix.rows, matrix.cols)
	for i := range matrix.rows {
		for j := range matrix.cols {
			//the leaky verison
			/*if matrix.data[i][j] < 0 {
				matrix.data[i][j] *= 0.05
			}*/
			activated.data[i][j] = math.Max(0, matrix.data[i][j])
		}
	}

	return activated
}

func ReLULayer(matricies []Matrix) []Matrix {
	activated := make([]Matrix, len(matricies))
	for index, matrix := range matricies {
		activated[index] = *ReLU(matrix)
	}

	return activated

}

func PadRow(matrix *Matrix, padding int) {
	for range padding {
		var zeros = make([]float64, matrix.cols)
		for j := range matrix.cols {
			zeros[j] = 0
		}
		matrix.data = append(matrix.data, zeros)
	}

	matrix.rows += padding
}

func PadCol(matrix *Matrix, padding int) {
	for range padding {
		for i := range matrix.rows {
			matrix.data[i] = append(matrix.data[i], 0)
		}
	}
	matrix.cols += padding
}

func MaxPool(matrix Matrix, size int) (*Matrix, error) {
	PadRow(&matrix, matrix.rows%size)
	PadCol(&matrix, matrix.cols%size)
	pooled := Initialize(matrix.rows/size, matrix.cols/size)

	for i := 0; i < matrix.rows; i += size {
		for j := 0; j < matrix.cols; j += size {
			//this assumes that you have already ReLU-ed it, if not u might be cooked
			pool, _ := matrix.slice(i, j, size, size)
			flatPool := Flatten([]Matrix{*pool})
			maxx := 0.0

			for _, value := range (flatPool.data)[0] {
				if value > maxx {
					maxx = value
				}
			}
			//is this slow? idk im too tired
			pooled.data[i/size][j/size] = maxx

		}
	}
	return pooled, nil
}

func MaxPoolLayer(matricies []Matrix, size int) ([]Matrix, error) {

	pooled := make([]Matrix, len(matricies))
	for index, matrix := range matricies {
		tmp, err := MaxPool(matrix, size)
		if err != nil {
			return pooled, err
		}
		pooled[index] = *tmp

	}

	return pooled, nil
}

func Convolve(input, kernel *Matrix, stride int) (*Matrix, error) {
	PadRow(input, input.rows%stride)
	PadCol(input, input.cols%stride)

	//what is this python (pt 2)
	output := Initialize((input.rows-kernel.rows)/stride+1, (input.cols-kernel.cols)/stride+1)

	for i := 0; i <= input.rows-kernel.rows; i += stride {
		for j := 0; j <= input.cols-kernel.cols; j += stride {
			//technically this does support non square kernels, tho i have never seen one before
			slice, _ := input.slice(i, j, kernel.rows, kernel.cols)

			dot, err := DotProduct(slice, kernel)
			if err != nil {
				fmt.Println("we fucked up bad")
				return output, fmt.Errorf("uggghhhhh")
			}

			output.data[i/stride][j/stride] = dot
		}
	}

	return output, nil
}

func ConvolveLayer(input []Matrix, filter Layer) []Matrix {
	output := make([]Matrix, len(input)*len(filter.Kernels))
	//this applies every kernel to every feature map which is not always what u want but in this case it is what i want :)
	for index, image := range input {
		for i, kernel := range filter.Kernels {
			featureMap, _ := Convolve(&image, &kernel, filter.Step)
			output[index*len(filter.Kernels)+i] = *featureMap
		}
	}

	return output
}

func Dense(input, weights, biases *Matrix) (*Matrix, error) {
	output, _ := MatrixMultiply(input, weights)
	output = MatrixAdd(output, biases)
	return output, nil
}

func DenseLayer(input []Matrix, dense Layer) ([]Matrix, error) {
	output, err := Dense(&input[0], dense.Weights, dense.Biases)
	return []Matrix{*output}, err
}

func softMax(input []Matrix) []Matrix {
	output := make([]Matrix, len(input))
	total := 0.0
	for _, matrix := range input {
		for i := range matrix.rows {
			for j := range matrix.cols {
				total += math.Pow(math.E, matrix.data[i][j])
			}
		}
	}

	for index, matrix := range input {
		output[index] = *Initialize(matrix.rows, matrix.cols)
		for i := range matrix.rows {
			for j := range matrix.cols {
				output[index].data[i][j] = math.Pow(math.E, matrix.data[i][j]) / total
			}
		}
	}

	/*for _, row := range input.data {
		for _, value := range row {
			total += math.Pow(math.E, value)
		}
	}

	output := Initialize(input.rows, input.cols)

	for i := range input.rows {
		for j := range input.cols {
			output.data[i][j] = math.Pow(math.E, input.data[i][j]) / total
	}
	}*/

	return output
}

type Layer struct {
	Operation string
	Step      int
	Popcorn   int
	Kernels   []Matrix //super consistent pointer usage
	Weights   *Matrix
	Biases    *Matrix

	//Operation *func()
}

func CreateReLU() *Layer {
	return &Layer{
		Operation: "ReLU",
	}
}

func CreateSoftMax() *Layer {
	return &Layer{
		Operation: "softMax",
	}
}

func CreateMaxPool(size int) *Layer {
	return &Layer{
		Operation: "maxPool",
		Step:      size,
	}
}

func CreateConvolution(width, height, stride, filters int) *Layer {
	kernels := make([]Matrix, filters)
	for f := range filters {
		kernel := Initialize(height, width)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))

		for i := range height {
			for j := range width {
				kernel.data[i][j] = r.Float64() - 0.5 //its not skuffed idk what u mean
			}
		}
		kernels[f] = *kernel
	}

	return &Layer{
		Operation: "convolve",
		Kernels:   kernels,
		Step:      stride,
		Popcorn:   filters,
	}
}

func CreateFlatten() *Layer {
	return &Layer{
		Operation: "flatten",
	}
}

func CreateDense(inputSize, outputSize int) *Layer {

	weights := Initialize(inputSize, outputSize)
	biases := Initialize(1, outputSize)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := range inputSize {
		for j := range outputSize {
			weights.data[i][j] = r.Float64() - 0.5
			biases.data[0][j] = r.Float64() - 0.5
		}
	}

	return &Layer{
		Operation: "dense",
		Weights:   weights,
		Biases:    biases,
	}
}

func CreateLeastSquares(inputSize int) *Layer {
	return &Layer{
		Operation: "leastSquares",
		Step:      inputSize,
	}
}

func CreateCrossEntropy(inputSize int) *Layer {
	return &Layer{
		Operation: "crossEntropy",
		Step:      inputSize,
	}
}

type Network []Layer

func CreateNetwork() *Network {
	net := make(Network, 0)
	return &net
}

func (network *Network) Add(layer *Layer) {
	*network = append(*network, *layer)
}

func truncateFloat(f float64, decimals int) float64 {
	shift := math.Pow10(decimals)
	return math.Trunc(f*shift) / shift
}

func (network Network) Compute(input Matrix) ([]Matrix, error) {
	current := []Matrix{input}

	for _, layer := range network[:len(network)-1] {
		//i probably should use annonomous funtions
		//fmt.Println(current)
		if layer.Operation == "maxPool" {
			current, _ = MaxPoolLayer(current, layer.Step)
		} else if layer.Operation == "convolve" {
			current = ConvolveLayer(current, layer)
		} else if layer.Operation == "flatten" {
			current = []Matrix{*Flatten(current)} // this is not a bad design choice i swear
		} else if layer.Operation == "dense" {
			current, _ = DenseLayer(current, layer)
		} else if layer.Operation == "ReLU" {
			current = ReLULayer(current)
		} else if layer.Operation == "softMax" {
			current = softMax(current)
		} else {
			panic("invalid operation key")
		}
	}

	almost := current[0]

	output := Initialize(almost.rows, almost.cols)

	for i := range almost.rows {
		for j := range almost.cols {
			output.data[i][j] = truncateFloat(almost.data[i][j], 5)
		}
	}

	//TODO: also return the appropriate loss here

	//return current, nil
	return []Matrix{*output}, nil
}

type image struct {
	Content *Matrix
	Label   int
}

func GetImage(path string) *Matrix {
	contents, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	pixels := Initialize(64, 64)
	for i := range 64 {
		pixels.data[i] = make([]float64, 64)
	}

	//lets hope your data looks exactly like mine
	for i := 0; i < int(math.Pow(64, 2)); i++ {
		pixel := string(contents[i*2])
		pixels.data[i/64][i%64], err = strconv.ParseFloat(pixel, 64)

		if err != nil {
			fmt.Println("could not find \"float\"")
		}
	}

	return pixels
}

// aka "normalize"
func round(b byte) float64 {
	if b < 128 {
		return 0
	} else {
		return 1
	}
}

// this serves little purpose beyond compartmenalization -- except now it servers as an idx parser, which im sure is all anyone really wants
func GetData(trainPath, labelPath string, size int) []image {
	data := make([]image, size)
	training, err := os.Open(trainPath)
	if err != nil {
		panic(err)
	}
	labeling, err := os.Open(labelPath)
	if err != nil {
		panic(err)
	}

	train := bufio.NewReader(training)
	label := bufio.NewReader(labeling)

	//buffers through the magic number
	for range 3 {
		_, err := train.ReadByte()
		if err != nil {
			fmt.Println(err)
		}
		_, err = label.ReadByte()
		if err != nil {
			fmt.Println(err)
		}
	}

	//dimensions of labels better be 1
	dimensions, err := train.ReadByte()
	_, err = label.ReadByte()

	buf := make([]byte, 4)
	_, err = io.ReadFull(train, buf)

	otherBuf := make([]byte, 4)
	_, err = io.ReadFull(label, otherBuf)

	fmt.Println(buf, otherBuf)
	length := binary.BigEndian.Uint32(buf)

	if length != binary.BigEndian.Uint32(otherBuf) {
		fmt.Errorf("training and label data have mismatched sizes")
	}

	rows, cols := 28, 28 //what a greate idx parser

	for range dimensions - 1 {
		buf := make([]byte, 4)
		_, err = io.ReadFull(train, buf)
		fmt.Println(binary.BigEndian.Uint32(buf))
	}

	for index := range size {

		if index >= size {
			break
		}

		instance := Initialize(rows, cols)
		for i := range rows {
			for j := range cols {
				pixel, err := train.ReadByte()
				if err != nil {
					fmt.Println(err)
				}

				instance.data[i][j] = round(pixel)
			}
		}

		label, err := label.ReadByte()
		if err != nil {
			fmt.Println(err)
		}

		data[index] = image{
			Content: instance,
			Label:   int(label),
		}

	}
	return data

}

func zeroNetwork(network Network) Network {
	copyy := *(CreateNetwork())

	for index, layer := range network {
		if layer.Operation == "convolve" {
			copyy.Add(CreateConvolution(layer.Kernels[0].rows, layer.Kernels[0].cols, layer.Step, layer.Popcorn))
			for kernel := range layer.Kernels {

				copyy[index].Kernels[kernel] = (*Initialize(layer.Kernels[0].rows, layer.Kernels[0].cols))
			}
		} else if layer.Operation == "dense" {
			copyy.Add(CreateDense(layer.Weights.rows, layer.Weights.cols))
			copyy[index].Weights = Initialize(layer.Weights.rows, layer.Weights.cols)
		} else if layer.Operation == "maxPool" {
			copyy.Add(CreateMaxPool(layer.Step))
		} else if layer.Operation == "flatten" {
			copyy.Add(CreateFlatten())
		} else if layer.Operation == "ReLU" {
			copyy.Add(CreateReLU())
		} else if layer.Operation == "softMax" {
			copyy.Add(CreateSoftMax())
		} else if layer.Operation == "leastSquares" {
			copyy.Add(CreateLeastSquares(layer.Step))
		} else if layer.Operation == "crossEntropy" {
			copyy.Add(CreateCrossEntropy(layer.Step))
		}
	}

	return copyy
}

/*func copyNetwork(network Network) Network {
	copyy := CreateNetwork()

	for _, layer := range network {
		copyy = append(copyy, &Layer{

		})
	}

	return copyy
}*/

func backPropogation(network, newNetwork Network, computed [][]Matrix, dLoss []Matrix, index int) {
	//fmt.Println("cant find it: ", dLoss)
	nextStep := make([]Matrix, len(computed[index]))
	for depth := range nextStep {

		nextStep[depth] = (*Initialize(computed[index][0].rows, computed[index][0].cols))
	}
	if network[index].Operation == "dense" {
		for i, node := range dLoss[0].data[0] {
			for weight := range network[index].Weights.rows {
				//if computed[index].data[0][weight] == 0 {

				//	nextStep.data[0][weight] += (network[index].Kernel.data[weight][i] * math.Min(node, 0))
				//} else {

				nextStep[0].data[0][weight] += (network[index].Weights.data[weight][i] * node)
				//}
				//fmt.Println("before: ", newNetwork[index].Kernel.data[weight][i])
				//fmt.Println((computed[index].data[0][weight] * node))
				newNetwork[index].Weights.data[weight][i] -= (computed[index][0].data[0][weight] * node)
				//fmt.Println("after: ", newNetwork[index].Kernel.data[weight][i])
			}
			newNetwork[index].Biases.data[0][i] -= node
		}
	} else if network[index].Operation == "convolve" {
		//if index == 2 {
		//	fmt.Println("cant find it: ", dLoss)
		//}

		//im running out of names for index variables
		for place, kernel := range network[index].Kernels { //yeah yeah ik i couuuuuuld combine these two loops into one but we will see if it actually has any performance impact
			for aff, affected := range dLoss[place*len(computed[index]) : (place+1)*len(computed[index])] {
				dontDivideByZero := 0
				if place == 0 {
					dontDivideByZero = aff
				} else {
					dontDivideByZero = aff % place
				}
				for ii := 0; ii <= computed[index][0].rows-kernel.rows; ii += network[index].Step {
					for jj := 0; jj <= computed[index][0].cols-kernel.cols; jj += network[index].Step {
						for i := range kernel.rows {
							for j := range kernel.cols {
								nextStep[dontDivideByZero].data[ii+i][jj+j] += kernel.data[i][j] * affected.data[ii/network[index].Step][jj/network[index].Step]
								//maybe this will work ¯\_(ツ)_/¯ who knows really
								newNetwork[index].Kernels[place].data[i][j] -= computed[index][dontDivideByZero].data[ii+i][jj+j] * affected.data[ii/network[index].Step][jj/network[index].Step]
							}
						}
						//(*output.data)[i/stride][j/stride] = dot
					}
				}
			}
		}

	} else if network[index].Operation == "flatten" {
		for index, value := range dLoss[0].data[0] {
			nextStep[index/(nextStep[0].rows*nextStep[0].cols)].data[index/nextStep[0].rows%nextStep[0].cols][index%nextStep[0].cols] = value
		}
	} else if network[index].Operation == "maxPool" {
		//not quite sure if this is actually what you are supposed to do but i can only think of one other way to do this so were gonna try it like this
		for whereWeAre, value := range dLoss {

			for i := range value.cols {
				for j := range value.cols {
					for ii := range network[index].Step {
						for jj := range network[index].Step {
							nextStep[whereWeAre].data[i*network[index].Step+ii][j*network[index].Step+jj] = value.data[i][j]
						}
					}
				}
			}
		}
	} else if network[index].Operation == "ReLU" {
		for outerIndex, matrix := range dLoss {
			for i := range computed[index+1][0].rows {
				for j := range computed[index+1][0].cols {
					if computed[index+1][outerIndex].data[i][j] == 0 {
						nextStep[outerIndex].data[i][j] = 0
					} else {
						nextStep[outerIndex].data[i][j] = matrix.data[i][j]

					}
				}
			}
		}
	} else if network[index].Operation == "softMax" {

		exponets := make([]Matrix, len(computed[index]))
		for outerIndex := range exponets {
			exponets[outerIndex] = (*Initialize(nextStep[outerIndex].rows, nextStep[outerIndex].cols))
		}
		total := 0.0
		for outerIndex, matrix := range computed[index] {

			for i := range matrix.rows {
				for j := range matrix.cols {
					exponets[outerIndex].data[i][j] = math.Pow(math.E, matrix.data[i][j])
				}
			}
			total += Sum(&exponets[outerIndex])
		}
		totalSquared := total * total
		for outerIndex, matrix := range exponets {
			for i := range matrix.rows {
				for j := range matrix.cols {
					for ii := range matrix.rows {
						for jj := range matrix.rows {
							if jj == ii {

								nextStep[outerIndex].data[i][j] += ((total)*matrix.data[i][j] - (matrix.data[i][j])*(matrix.data[i][j])) / (totalSquared)
							} else {
								nextStep[outerIndex].data[i][j] += (total - matrix.data[i][j]*matrix.data[ii][jj]) / totalSquared
							}
						}
					}

				}
			}
		}
	} else {
		panic("no operation matched: " + network[index].Operation)
	}
	if index == 0 {
		return
	}
	backPropogation(network, newNetwork, computed, nextStep, index-1)
}

/*
func MakeUpData() []image {
	data := make([]image, 1000)
	for i := range 10 {
		for j := range 100 {
			barelyMatrix := Initialize(1, 1)
			barelyMatrix.data[0][0] = float64(i)
			data[i*100+j] = image{
				content: barelyMatrix,
				label:   2*i + 10,
			}
		}
	}
	return data
}*/

func MakeUpTestData() *Matrix {
	test := Initialize(1, 1)
	test.data[0][0] = 101
	return test
}

func worker(network Network, length int, input chan image, propogated chan Network, die chan bool) {
	for {
		select {
		case instance := <-input:

			//first we have to compute what the network would evaluate each layer to be in its current state
			stepByStep := make([][]Matrix, length)
			stepByStep[0] = []Matrix{*instance.Content}

			for index, layer := range network[:length-1] {
				//fmt.Println(stepByStep[index].rows, stepByStep[index].cols)
				//i probably should use annonomous funtions
				if layer.Operation == "maxPool" {
					stepByStep[index+1], _ = MaxPoolLayer(stepByStep[index], layer.Step)
				} else if layer.Operation == "convolve" {
					stepByStep[index+1] = ConvolveLayer(stepByStep[index], layer)
				} else if layer.Operation == "flatten" {
					stepByStep[index+1] = []Matrix{*Flatten(stepByStep[index])}
				} else if layer.Operation == "dense" {
					stepByStep[index+1], _ = DenseLayer(stepByStep[index], layer)
				} else if layer.Operation == "ReLU" {
					stepByStep[index+1] = ReLULayer(stepByStep[index])
				} else if layer.Operation == "softMax" {
					stepByStep[index+1] = softMax(stepByStep[index])
				} else {
					panic("invalid operation key")
				}
			}

			//we are gonna do some (more) cheating i think
			//expected := Initialize(1, 10)
			//(*expected.data) = [][]float64{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}} //whats this Initialize function good for anyway
			//(*expected.data)[1][image.label] = 1

			//this part is poorly abstracted but idc, ... now it is (less) poorly abstracted
			/*
				input := stepByStep[length]
				total := 0.0
				for _, row := range input.data {
					for _, value := range row {
						total += math.Pow(math.E, value)
					}
				}

				output := Initialize(input.rows, input.cols)
				for i := range (*input).rows {
					for j := range (*input).cols {
						output.data[i][j] = math.Pow(math.E, input.data[i][j]) / total
					}
				}

				exponet := math.Pow(math.E, output.data[0][image.label])


				//dLoss := Initialize(1, 2)*/

			/*
				totalSquared := total * total
				for i := range dLoss.cols {
					if i == image.label {

						dLoss.data[0][i] = entropySlope * ((total)*exponet - (exponet)*(exponet)) / (totalSquared)
					} else {
						dLoss.data[0][i] = entropySlope * -1 * exponet / totalSquared
					}
					//if stepByStep[length].data[0][i] == 0 {
					//	dLoss.data[0][i] = math.Min(dLoss.data[0][i], 0)
					//}
				}*/

			//now only the loss function is implementeded badly

			dLoss := make([]Matrix, 1) //what a well name field
			dLoss[0] = (*Initialize(1, network[length-1].Step))

			//TODO: support more complex loss functions
			if network[length-1].Operation == "leastSquares" {
				dLoss[0].data[0][0] = 2 * (stepByStep[length-1][0].data[0][0] - float64(instance.Label))

			} else if network[length-1].Operation == "crossEntropy" {
				dLoss[0].data[0][instance.Label] = (-1.0 / stepByStep[length-1][0].data[0][instance.Label]) + 1 //i feel like this makes sense
			}

			//least sqaures

			//fmt.Println()
			//fmt.Println("input:", instance.Content.data[0][0])
			//fmt.Println("expected: ", instance.Label)
			//fmt.Println("output:", stepByStep[length][0].data[0][0])
			//fmt.Println("loss: ", dLoss[0])

			//cross entropy
			/*dLoss := make([]Matrix, 1)
			dLoss[0] = *Initialize(1, stepByStep[length][0].cols)

			entropySlope := (-1.0 / stepByStep[length][0].data[0][instance.Label])
			dLoss[0].data[0][instance.Label] = entropySlope*/

			changes := zeroNetwork(network)
			backPropogation(network, changes, stepByStep, dLoss, length-2)

			propogated <- changes

		case <-die:
			return //death...
		}
	}
}

func addNetworks(origin, changes Network, learningRate float64) Network {
	result := zeroNetwork(origin)

	//too lazy to compare slices lol
	/*if result != zeroNetwork(net2) {
		panic("networks of different dimensions, unadable")
	}*/

	for index, layer := range changes {
		if layer.Operation == "convolve" {
			for i, kernel := range layer.Kernels {
				kernel.scale(learningRate)
				result[index].Kernels[i] = *MatrixAdd(&origin[index].Kernels[i], &kernel)

			}
		}
		if layer.Operation == "dense" {

			changes[index].Weights.scale(learningRate)
			result[index].Weights = MatrixAdd(origin[index].Weights, layer.Weights)

			changes[index].Biases.scale(learningRate) //shhhhhh
			result[index].Biases = MatrixAdd(origin[index].Biases, layer.Biases)
		}
	}

	return result
}

func collector(changes Network, propogated, summed chan Network, size int) {

	for range size {
		change := <-propogated
		//fmt.Println(change[0].Weights)
		changes = addNetworks(changes, change, 1)
	}

	summed <- changes
}

func (network *Network) Train(data []image, batchSize int, learningRate float64) {
	fmt.Println("we ran", len(data))
	fmt.Println("og network", network)

	/*network[0].Kernel.data = [][]float64{
		{-1.0, -1.0, -1.0},
		{2.0, 2.0, 2.0},
		{-1.0, -1.0, -1.0},
	}
	network[2].Kernel.data = [][]float64{
		{(1.0 / 18)}, {1.0 / 18}, {1.0 / 18}, {1.0 / 18}, {1.0 / 18}, {1.0 / 18}, {1.0 / 18}, {1.0 / 18}, {1.0 / 18}, {1.0 / 18},
	}*/
	//														this makes me sad
	batches := make([][]image, int(math.Ceil(float64(len(data))/float64(batchSize))))

	//randomize the order
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	r.Shuffle(len(data), func(i, j int) {

		data[i], data[j] = data[j], data[i]
	})

	for index, value := range data {
		batches[index/batchSize] = append(batches[index/batchSize], value)
	}

	length := len(*network) //the boost in performance this will surely give us is monumental

	input := make(chan image)
	propogated := make(chan Network)
	summed := make(chan Network)
	die := make(chan bool)

	threads := 10

	for _, batch := range batches {

		changes := zeroNetwork(*network)

		go collector(changes, propogated, summed, batchSize)

		for range threads {
			go worker(*network, length, input, propogated, die)
		}

		for _, image := range batch {
			input <- image
			//time.Sleep(time.Second / 10)
			//fmt.Scanln()

		}

		//descend gradient
		changes = <-summed

		for range threads {
			die <- true
		}

		/*fmt.Println()
		fmt.Println("original Network:")
		fmt.Println((*network)[0].Weights)
		fmt.Println((*network)[0].Biases)
		fmt.Println("changes:")
		fmt.Println(changes[0].Weights)
		fmt.Println(changes[0].Biases)*/

		*network = addNetworks(*network, changes, learningRate)

		//fmt.Println("new network: ", (*network)[0].Weights, (*network)[0].Biases)

		/*fmt.Println("new network:")
		fmt.Println((*network)[0].Weights)
		fmt.Println((*network)[0].Biases)*/

	}

	fmt.Println()
	for index, layer := range *network {
		if layer.Operation == "convolve" { //|| layer.Operation == "dense" {
			fmt.Println(index, ": ", layer.Kernels)
		}

		if layer.Operation == "dense" {
			fmt.Println(index, ": ", layer.Weights)
			fmt.Println(index, ": ", layer.Biases)

		}
	}
}

// note that this is not used
func hasNaN(matrix *Matrix) bool {
	for i := range matrix.rows {
		for j := range matrix.cols {
			if math.IsNaN(matrix.data[i][j]) {
				return true
			}
		}
	}

	return false
}

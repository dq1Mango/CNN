package neural

import (
	"fmt"
	"log"
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

func MatrixAdd(A, B *Matrix) (*Matrix, error) {
	result := Initialize((*A).rows, (*A).cols)

	if (*A).rows != (*B).rows || (*A).cols != (*B).cols {
		panic("invalid dimensions")
	}

	for i := range (*A).rows {
		for j := range (*A).cols {
			result.data[i][j] = A.data[i][j] + B.data[i][j]
		}
	}

	return result, nil
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

func Flatten(matrix *Matrix) *Matrix {

	earth := Initialize(1, matrix.rows*matrix.cols) //1 AM variable names go hard

	for i := range matrix.rows {
		for j := range matrix.cols {
			earth.data[0][i*(*matrix).cols+j] = matrix.data[i][j]
		}
	}

	return earth
}

// Rectified Linear Unit
func ReLU(matrix *Matrix) {
	for i := range matrix.rows {
		for j := range matrix.cols {
			//the leaky verison
			/*if matrix.data[i][j] < 0 {
				matrix.data[i][j] *= 0.05
			}*/
			matrix.data[i][j] = math.Max(0, matrix.data[i][j])
		}
	}
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

// TODO: add someway to pad the matrix with 0's ... WE TO-DID IT!!!
func MaxPool(matrix *Matrix, size int) (*Matrix, error) {

	PadRow(matrix, matrix.rows%size)
	PadCol(matrix, matrix.cols%size)

	pooled := Initialize(matrix.rows/size, matrix.cols/size)

	for i := 0; i < matrix.rows; i += size {
		for j := 0; j < matrix.cols; j += size {
			//this assumes that you have already ReLU-ed it, if not u might be cooked
			pool, _ := matrix.slice(i, j, size, size)
			flatPool := Flatten(pool)
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

	//ReLU(output)

	return output, nil
}

func Dense(input, weights, biases *Matrix) (*Matrix, error) {
	output, _ := MatrixMultiply(input, weights)
	output, _ = MatrixAdd(output, biases)
	//ReLU(output)
	return output, nil
}

// note that this is not used for some reason
func softMax(input *Matrix) *Matrix {
	total := 0.0

	for _, row := range input.data {
		for _, value := range row {
			total += math.Pow(math.E, value)
		}
	}

	output := Initialize(input.rows, input.cols)

	for i := range input.rows {
		for j := range input.cols {
			output.data[i][j] = math.Pow(math.E, input.data[i][j]) / total
		}
	}

	return output
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
			kernel.data[i][j] = r.NormFloat64()
		}
	}

	return &Layer{
		Operation: "convolve",
		Kernel:    kernel,
		Step:      stride,
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
			weights.data[i][j] = r.NormFloat64()
			biases.data[0][j] = r.NormFloat64()
		}
	}

	return &Layer{
		Operation: "dense",
		Kernel:    weights,
		Biases:    biases,
	}
}

func CreateSoftMax() *Layer {
	return &Layer{
		Operation: "softMax",
	}
}

type Network []Layer

func CreateNetwork() Network {
	return make(Network, 0)
}

func (network *Network) Add(layer *Layer) {
	*network = append(*network, *layer)
}

func truncateFloat(f float64, decimals int) float64 {
	shift := math.Pow10(decimals)
	return math.Trunc(f*shift) / shift
}

func (network Network) Compute(input Matrix) (*Matrix, error) {
	current := &input

	for _, layer := range network {
		//i probably should use annonomous funtions
		//fmt.Println(current)
		if layer.Operation == "maxPool" {
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

	total := 0.0
	for _, row := range current.data {
		for _, value := range row {
			total += math.Pow(math.E, value)
		}
	}

	output := Initialize(current.rows, current.cols)

	for i := range current.rows {
		for j := range current.cols {
			output.data[i][j] = truncateFloat(math.Pow(math.E, current.data[i][j])/total, 5)
		}
	}

	return output, nil
}

type image struct {
	data  *Matrix
	label int
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

// this serves little purpose beyond compartmenalization
func GetData(path string) []image {
	data := make([]image, 0, 770)

	directories, err := os.ReadDir(path)

	if err != nil {
		fmt.Println("could not get ls to work")
		panic("could not get ls to work")
	}

	//more like "Get(very specifically sorted)Data"
	for _, dir := range directories {

		files, err := os.ReadDir(path + dir.Name() + "/")

		if err != nil {
			fmt.Println("could not get ls to work for the dirs", path)
			panic("could not get ls to work for the dirs")
		}

		for _, file := range files {
			contents, err := os.ReadFile(path + dir.Name() + "/" + file.Name())

			if err != nil { //wow i handle errors like 3 different ways if only i cared about errors
				log.Fatal(err)
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

			//converted, _ := strconv.Atoi(file.Name())
			label, _ := strconv.Atoi(dir.Name())
			data = append(data, image{
				data:  pixels,
				label: label,
			})
		}
	}

	return data

}

func zeroNetwork(network Network) Network {
	copyy := CreateNetwork()

	for index, layer := range network {
		if layer.Operation == "convolve" {
			copyy.Add(CreateConvolution(layer.Kernel.rows, layer.Kernel.cols, layer.Step))
			copyy[index].Kernel = Initialize(layer.Kernel.rows, layer.Kernel.cols)
		} else if layer.Operation == "dense" {
			copyy.Add(CreateDense(layer.Kernel.rows, layer.Kernel.cols))
			copyy[index].Kernel = Initialize(layer.Kernel.rows, layer.Kernel.cols)
		} else if layer.Operation == "maxPool" {
			copyy.Add(CreateMaxPool(layer.Step))
		} else if layer.Operation == "flatten" {
			copyy.Add(CreateFlatten())
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

// note that this is not used
func sign(n float64) float64 {
	if n < 0 {
		return -1
	} else {
		return 1
	}
}

func backPropogation(network, newNetwork Network, computed []*Matrix, dLoss *Matrix, index int) {
	fmt.Println("cant find it: ", dLoss)
	nextStep := Initialize(computed[index].rows, computed[index].cols)
	if network[index].Operation == "dense" {
		for i, node := range dLoss.data[0] {
			for weight := range network[index].Kernel.rows {
				//if computed[index].data[0][weight] == 0 {

				//	nextStep.data[0][weight] += (network[index].Kernel.data[weight][i] * math.Min(node, 0))
				//} else {

				nextStep.data[0][weight] += (network[index].Kernel.data[weight][i] * node)
				//}
				//fmt.Println("before: ", newNetwork[index].Kernel.data[weight][i])
				//fmt.Println((computed[index].data[0][weight] * node))
				newNetwork[index].Kernel.data[weight][i] -= (computed[index].data[0][weight] * node)
				//fmt.Println("after: ", newNetwork[index].Kernel.data[weight][i])
			}
			newNetwork[index].Biases.data[0][i] -= node
		}
	} else if network[index].Operation == "convolve" {
		//if index == 2 {
		//	fmt.Println("cant find it: ", dLoss)
		//}
		kernel := network[index].Kernel
		for ii := 0; ii <= computed[index].rows-kernel.rows; ii += network[index].Step {
			for jj := 0; jj <= computed[index].cols-kernel.cols; jj += network[index].Step {
				for i := range kernel.rows {
					for j := range kernel.cols {
						nextStep.data[ii+i][jj+j] += kernel.data[i][j] * dLoss.data[ii/network[index].Step][jj/network[index].Step]
						//maybe this will work ¯\_(ツ)_/¯ who knows really
						newNetwork[index].Kernel.data[i][j] -= computed[index].data[ii+i][jj+j] * dLoss.data[ii/network[index].Step][jj/network[index].Step]
					}
				}
				//(*output.data)[i/stride][j/stride] = dot
			}
		}
	} else if network[index].Operation == "flatten" {
		for index, value := range dLoss.data[0] {
			nextStep.data[index/nextStep.rows][index%nextStep.cols] = value
		}
	} else if network[index].Operation == "maxPool" {
		//not quite sure if this is actually what you are supposed to do but i can only think of one other way to do this so were gonna try it like this
		for i := range dLoss.rows {
			for j := range dLoss.cols {
				for ii := range network[index].Step {
					for jj := range network[index].Step {
						nextStep.data[i*network[index].Step+ii][j*network[index].Step+jj] = dLoss.data[i][j]
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

func MakeUpData() []image {
	examples := 100
	data := make([]image, 6*examples)
	for i := 0; i < 6*examples; i += 6 {
		for flip := range 2 {
			for j := range 3 {

				example := Initialize(5, 5)
				for k := range 5 {
					example.data[((1-flip)*(j+1))+(flip)*k][(1-flip)*(k)+(flip*(j+1))] = 1

				}
				data[i+3*flip+j] = image{
					data:  example,
					label: 1 - flip,
				}
			}

		}
	}
	return data
}

func MakeUpTestData() []Matrix {
	vertical := Initialize(5, 5)
	horizontal := Initialize(5, 5)

	(*vertical).data = [][]float64{
		{0, 0, 1, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 1, 0, 0},
		{0, 0, 1, 0, 0},
	}
	(*horizontal).data = [][]float64{
		{0, 0, 0, 0, 0},
		{1, 1, 1, 1, 1},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
	}

	return []Matrix{*vertical, *horizontal}
}

/*func MakeUpData() []image {
	data := make([]image, 1000)
	for i := range 10 {
		for j := range 100 {
			barelyMatrix := Initialize(1, 1)
			barelyMatrix.data[0][0] = float64(i)
			data[i*100+j] = image{
				data:  barelyMatrix,
				label: 2*i + 10,
			}
		}
	}
	return data
}

func MakeUpTestData() *Matrix {
	test := Initialize(1, 1)
	test.data[0][0] = 101
	return test
}*/

func (network Network) Train(data []image, batchSize int, learningRate float64) {
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

	length := len(network) //the boost in performance this will surely give us is monumental

	for batcDex, batch := range batches {
		if batcDex < 0 { //> 77 {
			break
		}

		fmt.Println("current network")
		for _, layer := range network {
			if layer.Operation == "convolve" {
				for _, row := range layer.Kernel.data {
					fmt.Println(row)
				}
			} else if layer.Operation == "dense" {
				fmt.Println(layer.Kernel.data)
			}
		}

		changes := zeroNetwork(network)

		for _, image := range batch {
			//first we have to compute what the network would evaluate each layer to be in its current state
			stepByStep := make([]*Matrix, length+1)
			stepByStep[0] = image.data

			for index, layer := range network {
				//fmt.Println(stepByStep[index].rows, stepByStep[index].cols)
				//i probably should use annonomous funtions
				if layer.Operation == "maxPool" {
					stepByStep[index+1], _ = MaxPool(stepByStep[index], layer.Step)
				} else if layer.Operation == "convolve" {
					stepByStep[index+1], _ = Convolve(stepByStep[index], layer.Kernel, layer.Step)
					//should really generalize this but everybody ues ReLU
					//ReLU(stepByStep[index+1])
				} else if layer.Operation == "flatten" {
					stepByStep[index+1] = Flatten(stepByStep[index])
				} else if layer.Operation == "dense" {
					stepByStep[index+1], _ = Dense(stepByStep[index], layer.Kernel, layer.Biases)
					//ReLU(stepByStep[index+1])
				} else {
					panic("invalid operation key")
				}
			}

			//we are gonna do some (more) cheating i think
			//expected := Initialize(1, 10)
			//(*expected.data) = [][]float64{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}} //whats this Initialize function good for anyway
			//(*expected.data)[1][image.label] = 1

			//this part is poorly abstracted but idc
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
				//dLoss := Initialize(1, 10)
				//dLoss.data[0] = []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

				dLoss := Initialize(1, 2)
				entropySlope := (-1.0 / output.data[0][image.label])
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
				}

				if math.IsNaN(dLoss.data[0][image.label]) {
					//fmt.Println("continueing")
					continue
				}*/

			for _, step := range stepByStep {
				for _, row := range step.data {
					fmt.Println(row)
				}
			}
			dLoss := Initialize(1, 1)
			dLoss.data[0][0] = 2 * (stepByStep[length].data[0][0] - float64(image.label))
			backPropogation(network, changes, stepByStep, dLoss, length-1)
			fmt.Println("things we want to change:")
			for _, row := range changes[0].Kernel.data {
				fmt.Println(row)
			}
			//time.Sleep(time.Second / 10)
			fmt.Scanln()

		}

		//descend gradient
		for index, layer := range network {
			if layer.Operation == "convolve" || layer.Operation == "dense" {
				changes[index].Kernel.scale(learningRate)
				network[index].Kernel, _ = MatrixAdd(layer.Kernel, changes[index].Kernel)
				if layer.Operation == "dense" {
					changes[index].Biases.scale(learningRate * 50)
					network[index].Biases, _ = MatrixAdd(layer.Biases, changes[index].Biases)
				}
			}
		}
		//fmt.Println("new network: ", network[0].Kernel, network[0].Biases)
	}
	fmt.Println()
	for index, layer := range network {
		if layer.Operation == "convolve" { //|| layer.Operation == "dense" {
			fmt.Println(index, ": ", layer.Kernel)
		}

		if layer.Operation == "dense" {
			//fmt.Println(index, ": ", layer.Kernel)
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

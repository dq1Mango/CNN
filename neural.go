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
		return slice, fmt.Errorf("Index out of range")
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

func ReLU(matricies []Matrix) []Matrix {
	activated := make([]Matrix, len(matricies))
	for index, matrix := range matricies {
		activated[index] = *Initialize(matrix.rows, matrix.cols)
		for i := range matrix.rows {
			for j := range matrix.cols {
				//the leaky verison
				/*if matrix.data[i][j] < 0 {
					matrix.data[i][j] *= 0.05
				}*/
				matricies[index].data[i][j] = math.Max(0, matrix.data[i][j])
			}
		}
	}
	return activated

}

func PadRow(matrix *Matrix, padding int) {
	for range padding {
		var zeros = make([]float64, matrix.cols, matrix.cols)
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
func MaxPool(matricies []Matrix, size int) ([]Matrix, error) {

	pooled := make([]Matrix, len(matricies))
	for index, matrix := range matricies {
		PadRow(&matrix, matrix.rows%size)
		PadCol(&matrix, matrix.cols%size)
		pooled[index] = (*Initialize(matrix.rows/size, matrix.cols/size))

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
				pooled[index].data[i/size][j/size] = maxx

			}
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
				output[index].data[i][j] = matrix.data[i][j] / total
				total += math.Pow(math.E, matrix.data[i][j])
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
	kernel := Initialize(height, width)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	for i := range height {
		for j := range width {
			kernel.data[i][j] = r.NormFloat64()
		}
	}

	return &Layer{
		Operation: "convolve",
		Kernels:   []Matrix{*kernel},
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
			weights.data[i][j] = r.NormFloat64()
			biases.data[0][j] = r.NormFloat64()
		}
	}

	return &Layer{
		Operation: "dense",
		Weights:   weights,
		Biases:    biases,
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

func (network Network) Compute(input Matrix) ([]Matrix, error) {
	current := []Matrix{input}

	for _, layer := range network {
		//i probably should use annonomous funtions
		//fmt.Println(current)
		if layer.Operation == "maxPool" {
			current, _ = MaxPool(current, layer.Step)
		} else if layer.Operation == "convolve" {
			current = ConvolveLayer(current, layer)
		} else if layer.Operation == "flatten" {
			current = []Matrix{*Flatten(current)} // this is not a bad design choice i swear
		} else if layer.Operation == "dense" {
			current, _ = DenseLayer(current, layer)
		} else if layer.Operation == "ReLU" {
			ReLU(current)
		} else if layer.Operation == "softMax" {
			current = softMax(current)
		} else {
			panic("invalid operation key")
		}
	}

	/*total := 0.0
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
	}*/

	return current, nil
	//return output, nil
}

type image struct {
	content *Matrix
	label   int
}

func GetImage(path string) *Matrix {
	contents, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	pixels := Initialize(64, 64)
	for i := range 64 {
		pixels.data[i] = make([]float64, 64, 64)
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
				pixels.data[i] = make([]float64, 64, 64)
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
				content: pixels,
				label:   label,
			})
		}
	}

	return data

}

func zeroNetwork(network Network) Network {
	copyy := CreateNetwork()

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

	fmt.Println("cant find it: ", dLoss)
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
			for _, affected := range dLoss[place*network[index].Popcorn : (place+1)*network[index].Popcorn] {
				for ii := 0; ii <= computed[index][0].rows-kernel.rows; ii += network[index].Step {
					for jj := 0; jj <= computed[index][0].cols-kernel.cols; jj += network[index].Step {
						for i := range kernel.rows {
							for j := range kernel.cols {
								nextStep[place].data[ii+i][jj+j] += kernel.data[i][j] * affected.data[ii/network[index].Step][jj/network[index].Step]
								//maybe this will work ¯\_(ツ)_/¯ who knows really
								newNetwork[index].Kernels[place].data[i][j] -= computed[index][place].data[ii+i][jj+j] * affected.data[ii/network[index].Step][jj/network[index].Step]
							}
						}
						//(*output.data)[i/stride][j/stride] = dot
					}
				}
			}
		}
	} else if network[index].Operation == "flatten" {
		for index, value := range dLoss[0].data[0] {
			nextStep[index/(nextStep[0].rows*nextStep[0].cols)].data[index/nextStep[0].rows][index%nextStep[0].cols] = value
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

		exponets := make([]Matrix, network[index].Popcorn)
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
					content: example,
					label:   flip,
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

	length := len(network) //the boost in performance this will surely give us is monumental

	for batcDex, batch := range batches {
		if batcDex < 0 { //> 77 {
			break
		}

		//fmt.Println("current network")
		changes := zeroNetwork(network)

		for _, image := range batch {
			//first we have to compute what the network would evaluate each layer to be in its current state
			stepByStep := make([][]Matrix, length+1)
			stepByStep[0] = []Matrix{*image.content}

			for index, layer := range network {
				//fmt.Println(stepByStep[index].rows, stepByStep[index].cols)
				//i probably should use annonomous funtions
				if layer.Operation == "maxPool" {
					stepByStep[index+1], _ = MaxPool(stepByStep[index], layer.Step)
				} else if layer.Operation == "convolve" {
					stepByStep[index+1] = ConvolveLayer(stepByStep[index], layer)
				} else if layer.Operation == "flatten" {
					stepByStep[index+1] = []Matrix{*Flatten(stepByStep[index])}
				} else if layer.Operation == "dense" {
					stepByStep[index+1], _ = DenseLayer(stepByStep[index], layer)
				} else if layer.Operation == "ReLU" {
					stepByStep[index+1] = ReLU(stepByStep[index+1])
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

				//dLoss := Initialize(1, 2)

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
				}*/

			//now only the loss function is implementeded badly
			//dLoss := Initialize(1, 10)
			//dLoss.data[0][image.label] = (-1.0 / stepByStep[length].data[0][image.label])
			dLoss := make([]Matrix, 1)
			dLoss[0] = (*Initialize(1, 1))
			dLoss[0].data[0][0] = 2 * (stepByStep[length][0].data[0][0] - float64(image.label))
			backPropogation(network, changes, stepByStep, dLoss, length-1)
			fmt.Println("things we want to change:")

			//time.Sleep(time.Second / 10)
			//fmt.Scanln()

		}

		//descend gradient
		for index, layer := range network {
			if layer.Operation == "convolve" {
				for i, kernel := range changes[index].Kernels {
					kernel.scale(learningRate)
					network[index].Kernels[i] = (*MatrixAdd(&layer.Kernels[i], &kernel))

				}
				if layer.Operation == "dense" {
					changes[index].Weights.scale(learningRate)
					network[index].Weights = MatrixAdd(layer.Weights, changes[index].Weights)

					changes[index].Biases.scale(learningRate * 50)
					network[index].Biases = MatrixAdd(layer.Biases, changes[index].Biases)
				}
			}
		}
		//fmt.Println("new network: ", network[0].Kernel, network[0].Biases)
	}
	fmt.Println()
	for index, layer := range network {
		if layer.Operation == "convolve" { //|| layer.Operation == "dense" {
			fmt.Println(index, ": ", layer.Kernels)
		}

		if layer.Operation == "dense" {
			fmt.Println(index, ": ", layer.Weights)
			fmt.Println(index, ": ", layer.Biases)

		}
	}

}

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

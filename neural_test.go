package neural

import (
	"fmt"
	"math"
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

func TestMatrixScale(t *testing.T) {
	A := Initialize(2, 3)
	A.data = [][]float64{
		{1, 2, 3},
		{2, 0, -2},
	}
	scaled := Initialize(2, 3)
	scaled.data = [][]float64{
		{2, 4, 6},
		{4, 0, -4},
	}

	A.scale(2)

	if slicesEqual(A.data, scaled.data) {
		fmt.Println("Passed scaling ")
	} else {
		t.Errorf("So we found our issue")
	}
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
		fmt.Println("womp womp")
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

	A = ReLU(*A)

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

	pooled, err := MaxPool(*A, 2)
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

func TestDense(t *testing.T) {
	network := *(CreateNetwork())
	network.Add(CreateDense(1, 1))
	network[0].Weights.data[0][0] = 2
	network[0].Biases.data[0][0] = 10

	test := makeUpLinearTestData()
	fmt.Println(network.Compute(*test))
}

func genLinearData() []image {
	data := make([]image, 1000)
	for i := range 10 {
		for j := range 100 {
			barelyMatrix := Initialize(1, 1)
			barelyMatrix.data[0][0] = float64(i)
			data[i*100+j] = image{
				Content: barelyMatrix,
				Label:   2*i + 10,
			}
		}
	}
	return data
}

func makeUpLinearTestData() *Matrix {
	test := Initialize(1, 1)
	test.data[0][0] = 11
	return test
}

func TestBasicLinearRegression(t *testing.T) {
	network := CreateNetwork()
	network.Add(CreateDense(1, 1))
	network.Add(CreateLeastSquares(1))
	data := genLinearData()
	test := makeUpLinearTestData()
	for range 10 {
		network.Train(data, 10, 0.001)
	}

	computed, err := network.Compute(*test)
	if err != nil {
		t.Error(err)
	}

	if math.Abs(32-computed[0].data[0][0]) < 1 {
		t.Logf("basic linear regression passed")
	} else {
		t.Errorf("couldnt even pass the most basic case :skull")
	}
}

func MakeUpConvolutionData() []image {
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
					Content: example,
					Label:   flip,
				}
			}

		}
	}
	return data
}

func MakeUpConvolutionTestData() []Matrix {
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
func TestBasicConvolutionLayer(t *testing.T) {
	network := CreateNetwork()
	network.Add(CreateConvolution(3, 3, 1, 1))
	network.Add(CreateFlatten())
	network.Add(CreateDense(9, 2))
	network.Add(CreateSoftMax())
	network.Add(CreateCrossEntropy(2))

	data := MakeUpConvolutionData()
	for range 10 {
		network.Train(data, 10, 0.001)
	}

	tests := MakeUpConvolutionTestData()

	fmt.Println(network.Compute(tests[0]))
	fmt.Println(network.Compute(tests[1]))

}

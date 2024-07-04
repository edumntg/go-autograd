package tensor

import (
	"math/rand"
)

type Tensor struct {
	value         [][]float64
	grad          [][]float64
	shape         []int
	prev          []Tensor
	_grad_func    func()
	requires_grad bool
}

func NewTensor(value [][]float64) *Tensor {
	return &Tensor{
		value: value,
		shape: []int{len(value), len(value[0])},
	}
}

func RandomTensor(shape []int) *Tensor {
	value_arr := make([][]float64, shape[0])
	for i := 0; i < shape[0]; i++ {
		value_arr[i] = make([]float64, shape[1])
		for j := 0; j < shape[1]; j++ {
			value_arr[i][j] = rand.Float64()
		}
	}

	return &Tensor{
		value: value_arr,
		shape: shape,
	}
}

func (t *Tensor) Get() [][]float64 {
	return t.value
}

func (t *Tensor) Set(row, column int, value float64) {
	t.value[row][column] = value
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Add(other *Tensor) *Tensor {

	out_arr := make([][]float64, t.shape[0])
	for i := 0; i < t.shape[0]; i++ {
		out_arr[i] = make([]float64, t.shape[1])
		for j := 0; j < t.shape[1]; j++ {
			out_arr[i][j] = t.value[i][j] + other.value[i][j]
		}
	}

	// Declare the gradient function. This is the function used to calculate the gradient for a + operation
	_grad_func := func() {
		for i := 0; i < t.shape[0]; i++ {
			for j := 0; j < t.shape[1]; j++ {
				t.grad[i][j] += out_arr[i][j]
				other.grad[i][j] += out_arr[i][j]
			}
		}
	}

	return &Tensor{
		value:      out_arr,
		_grad_func: _grad_func,
	}
}

func (t *Tensor) Mul(other *Tensor) *Tensor {

	out_arr := make([][]float64, t.shape[0])
	for i := 0; i < t.shape[0]; i++ {
		out_arr[i] = make([]float64, other.shape[1])
	}

	// APply multiplication
	total := float64(0.0)
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < other.shape[1]; j++ {
			for k := 0; k < t.shape[1]; k++ {
				total += t.value[i][k] * other.value[k][j]
			}

			out_arr[i][j] = total
			total = 0.0
		}
	}

	return &Tensor{
		value: out_arr,
	}
}

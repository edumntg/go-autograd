package tensor

import "math/rand"

type Tensor struct {
	value      float32
	grad       float32
	prev       []Tensor
	_grad_func func()
}

func NewTensor(value float32) *Tensor {
	return &Tensor{
		value: value,
	}
}

func RandomTensor() *Tensor {
	return &Tensor{
		value: rand.Float32(),
	}
}

func NewTensorArray(value []float32) []*Tensor {
	tensor_arr := make([]*Tensor, len(value))

	for i := 0; i < len(value); i++ {
		tensor_arr[i] = NewTensor(value[i])
	}

	return tensor_arr
}

func RandomTensorArray(size int) []*Tensor {
	tensor_arr := make([]*Tensor, size)

	for i := 0; i < size; i++ {
		tensor_arr[i] = RandomTensor()
	}

	return tensor_arr
}

func (t *Tensor) Get() float32 {
	return t.value
}

func (t *Tensor) Add(other *Tensor) *Tensor {
	out := &Tensor{value: (t.value + other.value), prev: []Tensor{*t, *other}}

	// Declare the gradient function. This is the function used to calculate the gradient for a + operation
	_grad_func := func() {
		t.grad += out.grad
		other.grad += out.grad
	}

	out._grad_func = _grad_func
	return out
}

func (t *Tensor) Mul(other *Tensor) *Tensor {
	out := &Tensor{value: (t.value * other.value), prev: []Tensor{*t, *other}}

	_grad_func := func() {
		t.grad = out.grad * other.value
		other.grad = out.grad * t.value
	}

	out._grad_func = _grad_func
	return out
}

func (t *Tensor) Div(other *Tensor) *Tensor {
	out := &Tensor{value: (t.value / other.value), prev: []Tensor{*t, *other}}

	_grad_func := func() {
		t.grad = out.grad / other.value
		other.grad = out.grad / t.value
	}

	out._grad_func = _grad_func

	return out
}

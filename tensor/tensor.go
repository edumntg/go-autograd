package tensor

type Tensor struct {
	value      float32
	grad       float32
	prev       []Tensor
	_grad_func func()
}

func NewTensor(value, grad float32) *Tensor {
	t := new(Tensor)
	t.value = value
	t.grad = grad
	t.prev = []Tensor{}
	return t
}

func (t *Tensor) add(other *Tensor) Tensor {
	out := Tensor{value: (t.value + other.value), prev: []Tensor{*t, *other}}

	// Declare the gradient function. This is the function used to calculate the gradient for a + operation
	_grad_func := func() {
		t.grad += out.grad
		other.grad += out.grad
	}

	out._grad_func = _grad_func
	return out
}

func (t *Tensor) mul(other *Tensor) Tensor {
	out := Tensor{value: (t.value * other.value), prev: []Tensor{*t, *other}}

	_grad_func := func() {
		t.grad = out.grad * other.value
		other.grad = out.grad * t.value
	}

	out._grad_func = _grad_func
	return out
}

func (t *Tensor) div(other *Tensor) Tensor {
	out := Tensor{value: (t.value / other.value), prev: []Tensor{*t, *other}}

	_grad_func := func() {
		t.grad = out.grad / other.value
		other.grad = out.grad / t.value
	}

	out._grad_func = _grad_func

	return out
}

package nn

import (
	. "go-nn/tensor"
	"math/rand"
)

type Linear struct {
	//N []*Neuron // a layer is full of neurons
	W *Tensor
	B *Tensor
}

// func NewLinear(in_size, out_size int) *Linear {
// 	n_array := make([]*Neuron, out_size)
// 	for i := 0; i < out_size; i++ {
// 		n_array[i] = NewNeuron(in_size)
// 	}

// 	return &Linear{N: n_array}
// }

func New1DRandomFloatArray(size int) []float64 {
	arr := make([]float64, size)
	for i := 0; i < size; i++ {
		arr[i] = rand.Float64()
	}

	return arr
}

func NewLinear(in_size, out_size int) *Linear {
	// Initialize weights
	var W *Tensor = RandomTensor([]int{in_size, out_size})
	var B *Tensor = RandomTensor([]int{1, out_size})

	return &Linear{W: W, B: B}
}

func (l *Linear) Forward(x *Tensor) *Tensor {
	return l.W.Mul(x).Add(l.B) // x@w + b
}

// func (l *Linear) Print() {
// 	for i := 0; i < len(l.W); i++ {
// 		l.W[i].Print()
// 	}
// }

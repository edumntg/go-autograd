package nn

import . "go-nn/tensor"

type Linear struct {
	N []*Neuron // a layer is full of neurons
}

func NewLinear(in_size, out_size int) *Linear {
	n_array := make([]*Neuron, out_size)
	for i := 0; i < out_size; i++ {
		n_array[i] = NewNeuron(in_size)
	}

	return &Linear{N: n_array}
}

func (l *Linear) Forward(x []*Tensor) []*Tensor {
	out_arr := make([]*Tensor, len(l.N))

	for i := 0; i < len(l.N); i++ {
		out_arr[i] = (l.N[i]).Forward(x)
	}

	return out_arr
}

func (l *Linear) Print() {
	for i := 0; i < len(l.N); i++ {
		l.N[i].Print()
	}
}

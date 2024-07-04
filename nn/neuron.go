package nn

// import (
// 	"fmt"
// 	. "go-nn/tensor"
// 	"math/rand"
// )

// type Neuron struct {
// 	W []*Tensor // weights
// 	B *Tensor   // bias
// }

// func NewNeuron(n_in int) *Neuron {
// 	// First initialize arrays to fill tensors
// 	weight_obj := make([]*Tensor, n_in)

// 	for i := 0; i < n_in; i++ {
// 		weight_obj[i] = NewTensor(rand.Float64())
// 	}

// 	bias_obj := NewTensor(rand.Float64())

// 	return &Neuron{W: weight_obj, B: bias_obj}
// }

// func (n *Neuron) Forward(x []*Tensor) *Tensor {
// 	t := NewTensor(0)
// 	for i := 0; i < len(x); i++ {
// 		t = t.Add(x[i].Mul(n.W[i]))
// 	}

// 	return t
// }

// func (n *Neuron) Print() {
// 	for i := 0; i < len(n.W); i++ {
// 		fmt.Println("[", i, "]: ", n.W[i])
// 	}
// }

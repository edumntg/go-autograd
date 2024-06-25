package nn

import . "go-nn/tensor"

type Neuron struct {
	shape []int
	W     []*Tensor // weights
	B     *Tensor   // bias
}

func NewNeuron(shape []int) *Neuron {
	n := new(Neuron)
	n.shape = shape
	n.initializeWeights()
	n.initializeBias()

	return n
}

func (n *Neuron) initializeWeights() {

}

func (n *Neuron) initializeBias() {

}

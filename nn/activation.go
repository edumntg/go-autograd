package nn

import (
	"fmt"
	. "go-nn/tensor"
	"math"
)

type ReLU struct{}
type Softmax struct{}

func (r *ReLU) Forward(x []*Tensor) []*Tensor {
	fmt.Println("Applying ReLU...")
	out_arr := make([]*Tensor, len(x))
	for i := 0; i < len(x); i++ {
		out_arr[i] = NewTensor(max(0, x[i].Get()))
	}

	return out_arr
}

func (r *ReLU) Print() {}

func (s *Softmax) Forward(x []*Tensor) []*Tensor {
	fmt.Println("Applying Softmax...")
	out_arr := make([]*Tensor, len(x))

	// Compute tensor sum
	exp_sum := 0.0
	for i := 0; i < len(x); i++ {
		exp_sum += math.Exp(x[i].Get())
	}

	for i := 0; i < len(x); i++ {
		val := math.Exp(x[i].Get()) / exp_sum
		out_arr[i] = NewTensor(val)
	}

	return out_arr
}

func (s *Softmax) Print() {}

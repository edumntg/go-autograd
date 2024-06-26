package nn

import (
	"fmt"
	. "go-nn/tensor"
	"math"
)

type ReluF struct{}
type SoftmaxF struct{}

func Relu() *ReluF {
	return &ReluF{}
}

func Softmax() *SoftmaxF {
	return &SoftmaxF{}
}

func (r *ReluF) Forward(x []*Tensor) []*Tensor {
	fmt.Println("Applying ReLU...")
	out_arr := make([]*Tensor, len(x))
	for i := 0; i < len(x); i++ {
		out_arr[i] = NewTensor(max(0, x[i].Get()))
	}

	return out_arr
}

func (r *ReluF) Print() {}

func (s *SoftmaxF) Forward(x []*Tensor) []*Tensor {
	fmt.Println("Applying Softmax...")
	out_arr := make([]*Tensor, len(x))

	// Get max value
	max_val := 0.0
	for i := 0; i < len(x); i++ {
		if x[i].Get() > max_val {
			max_val = x[i].Get()
		}
	}

	// Compute tensor sum
	exp_sum := 0.0
	for i := 0; i < len(x); i++ {
		exp_sum += math.Exp(x[i].Get() - max_val)
	}

	for i := 0; i < len(x); i++ {
		val := math.Exp(x[i].Get()-max_val) / exp_sum
		out_arr[i] = NewTensor(val)
	}

	return out_arr
}

func (s *SoftmaxF) Print() {}

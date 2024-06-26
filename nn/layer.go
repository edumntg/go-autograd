package nn

import . "go-nn/tensor"

type Layer interface {
	Forward(x []*Tensor) []*Tensor
	Print()
}

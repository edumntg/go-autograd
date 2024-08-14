package nn

import (
	"fmt"
	. "go-nn/engine"
)

type Layer struct {
	W      [][]*Value
	b      []*Value
	NonLin bool
}

func NewLayer(inSize, outSize int) *Layer {
	W := make([][]*Value, inSize)
	for i := 0; i < inSize; i++ {
		W[i] = make([]*Value, outSize)
		for j := 0; j < outSize; j++ {
			W[i][j] = RandomValue()
		}
	}
	b := make([]*Value, outSize)
	for i := 0; i < outSize; i++ {
		b[i] = NewValue(0.0)
	}

	return &Layer{
		W:      W,
		b:      b,
		NonLin: true,
	}
}

func (l *Layer) Forward(x [][]*Value) [][]*Value {
	// Apply matrix multiplication between x and W + b
	rows1 := len(x)
	//cols1 := len(x[0])
	rows2 := len(l.W)
	cols2 := len(l.W[0])

	out := make([][]*Value, rows1)

	for i := 0; i < rows1; i++ {
		out[i] = make([]*Value, cols2)
		for j := 0; j < cols2; j++ {
			out[i][j] = NewValue(0.0)
			for k := 0; k < rows2; k++ {
				out[i][j] = out[i][j].Add(x[i][k].Mul(l.W[k][j]))
			}

			// Add bias
			out[i][j] = out[i][j].Add(l.b[j])

			if l.NonLin {
				out[i][j] = out[i][j].Relu()
			}
		}
	}

	return out
}

func (l *Layer) Parameters() []*Value {
	out := make([]*Value, 0)
	for i := 0; i < len(l.W); i++ {
		for j := 0; j < len(l.W[i]); j++ {
			out = append(out, l.W[i][j])
		}
	}

	for i := 0; i < len(l.b); i++ {
		out = append(out, l.b[i])
	}

	return out
}

func (l *Layer) String() string {
	var layerType string = ""
	if l.NonLin {
		layerType = "ReLU"
	}
	return fmt.Sprintf("%sLayer(shape=(%d,%d))", layerType, len(l.W), len(l.b))
}

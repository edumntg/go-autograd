package nn

import (
	. "go-nn/engine"
)

type MLP struct {
	Module
	Layers []*Layer
}

func NewMLP() *MLP {
	return &MLP{
		Module: Module{},
		Layers: make([]*Layer, 0),
	}
}

func (mlp *MLP) Forward(x [][]*Value) []*Value {
	//fmt.Printf("Input size: (%d, %d)\n", len(x), len(x[0]))
	for _, layer := range mlp.Layers {
		x, _ = layer.Forward(x)
		//fmt.Printf("Size after layer %d: (%d, %d)\n", i, len(x), len(x[0]))
	}

	return Flatten(x)
}

func (mlp *MLP) Add(layer *Layer) {
	mlp.Layers = append(mlp.Layers, layer)
}

func (mlp *MLP) Parameters() []*Value {
	parameters := make([]*Value, 0)
	for _, layer := range mlp.Layers {
		for _, param := range layer.Parameters() {
			parameters = append(parameters, param)
		}
	}

	return parameters
}

package nn

import . "go-nn/engine"

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
	for _, layer := range mlp.Layers {
		x = layer.Forward(x)
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

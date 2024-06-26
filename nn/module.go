package nn

import . "go-nn/tensor"

type Module struct {
	layers []*Linear
}

func NewModule() *Module {
	layers := []*Linear{}
	return &Module{layers: layers}
}

func (m *Module) Add(layer *Linear) {
	layers := append(m.layers, layer)
	m.layers = layers
}

func (m *Module) Forward(x []*Tensor) []*Tensor {
	if len(m.layers) == 0 {
		return nil
	}

	output := m.layers[0].Forward(x)

	for i := 1; i < len(m.layers); i++ {
		output = m.layers[i].Forward(output)
	}

	return output
}

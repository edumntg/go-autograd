package nn

import . "go-nn/tensor"

type Module struct {
	layers []Layer
}

func NewModule() *Module {
	layers := []Layer{}
	return &Module{layers: layers}
}

func (m *Module) Add(layer Layer) {
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

func (m *Module) Print() {
	for i := 0; i < len(m.layers); i++ {
		m.layers[i].Print()
	}
}

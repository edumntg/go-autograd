package nn

import . "go-nn/engine"

type Module struct {
	Parameters []*Value
}

func (m *Module) GetParameters() []*Value {
	return m.Parameters
}

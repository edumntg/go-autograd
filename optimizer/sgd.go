package optimizer

import (
	. "go-nn/engine"
)

type Sgd struct {
	Optimizer
	momentum float64
	velocity []float64
}

func NewSgd(parameters []*Value, lr float64) *Sgd {
	return &Sgd{
		Optimizer: Optimizer{
			parameters:     parameters,
			lr:             lr,
			scheduler:      func() { return },
			stepsPerformed: 0,
		},
		momentum: 0.0,
	}
}

func NewSgdWithMomentum(parameters []*Value, lr float64, momentum float64) *Sgd {
	return &Sgd{
		Optimizer: Optimizer{
			parameters:     parameters,
			lr:             lr,
			scheduler:      func() { return },
			stepsPerformed: 0,
		},
		momentum: momentum,
	}
}

func (sgd *Sgd) step() {
	for i, param := range sgd.parameters {
		sgd.velocity[i] = sgd.momentum*sgd.velocity[i] - sgd.lr*param.Grad
		param.Data += sgd.velocity[i]
	}
}

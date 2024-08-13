package loss

import . "go-nn/engine"

type MSELoss struct {
	Loss
}

func (mse *MSELoss) forward(yTrue []*Value, yPred []*Value) *Value {
	N := len(yTrue)
	out := NewValue(0.0)
	for i := 0; i < N; i++ {
		out = out.Add(yTrue[i].Sub(yPred[i]).Pow(2))
	}

	out = out.DivFloat(float64(N))

	return out
}

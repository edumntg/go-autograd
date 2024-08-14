package main

import (
	"fmt"
	. "go-nn/engine"
	"go-nn/loss"
	"go-nn/nn"
	"go-nn/optimizer"
	"go-nn/scaler"
)

func main() {

	var N int = 100
	var EPOCHS int = 100
	var lr float64 = 1e-1

	var nFeatures int = 2
	var x [][]*Value = make([][]*Value, N)
	var y []*Value = make([]*Value, N)

	for i := 0; i < N; i++ {
		x[i] = make([]*Value, nFeatures)
		for j := 0; j < nFeatures; j++ {
			x[i][j] = RandomValue()
			x[i][j].RequiresGrad = false
		}
		y[i] = NewValue((x[i][0].MulFloat(0.5).Add(x[i][1].MulFloat(-7.5)).AddFloat(20.0)).Data)
		y[i].RequiresGrad = false

	}

	// Scale data
	sc := scaler.NewScaler()
	sc.FitTransformArray(x)
	sc.FitTransformSingle(y)

	mlp := nn.NewMLP()
	mlp.Add(nn.NewLayer(nFeatures, 32))
	mlp.Add(nn.NewLayer(32, 16))
	mlp.Add(nn.NewLayer(16, 1))

	criterion := loss.NewMSELoss()
	optim := optimizer.NewSgd(mlp.Parameters(), lr)

	for epoch := 0; epoch < EPOCHS; epoch++ {
		optim.ZeroGrad()

		scores := mlp.Forward(x)

		lossVal := criterion.Forward(scores, y)

		lossVal.Backward()

		// Update params
		optim.Step()

		fmt.Printf("Epoch %d, Loss %.4f\n", epoch, lossVal.Item())
	}
}

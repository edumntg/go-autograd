package main

import (
	"fmt"
	. "go-nn/engine"
	"go-nn/loss"
	"go-nn/nn"
	"go-nn/optimizer"
	"go-nn/scaler"
)

func TestValue() {
	a := NewValue(-4.0)
	b := NewValue(2.0)
	c := a.Add(b)                                 // c = a + b
	d := a.Mul(b).Add(b.Pow(3))                   // d = a * b + b^3
	c = c.Add(c.AddFloat(1))                      // c += c + 1
	c = c.Add(c.AddFloat(1).Add(a.Neg()))         // c += 1 + c + (-a)
	d = d.Add(d.MulFloat(2).Add(b.Add(a).Relu())) // d += 2*d + (b+a).relu()
	d = d.Add(d.MulFloat(3).Add(b.Sub(a).Relu())) // d += 3*d + (b-a).relu()
	e := c.Sub(d)                                 // e = c - d
	f := e.Pow(2)                                 // f = e^2
	g := f.DivFloat(2)                            // g = f / 2
	g = g.Add(f.Pow(-1).MulFloat(10))             // g = g + 10 / f

	fmt.Println(g.Data) // prints 24.7041

	g.Backward()

	fmt.Println(a.Grad) // 138.8338 or dg/da
	fmt.Println(b.Grad) // 645.5773 or dg/db
}

func TestMLP() {
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

		lossVal := criterion.Forward(y, scores)

		lossVal.Backward()

		// Update params
		optim.Step()

		fmt.Printf("Epoch %d, Loss %.10f\n", epoch, lossVal.Item())
	}
}

func main() {
	//TestValue()
	TestMLP()
}

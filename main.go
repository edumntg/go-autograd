package main

import . "go-nn/engine"

func main() {

	// Test value
	//a := NewValue(-4.0)
	//b := NewValue(2.0)
	//c := a.Add(b)
	//d := a.Mul(b).Add(b.Pow(3))
	//
	//c = c.Add(c.AddFloat(1.0))
	//c = c.Add(c.AddFloat(1.0).Add(a.Neg()))
	//d = d.Add(d.MulFloat(2.0).Add(b.Add(a).Relu()))
	//d = d.Add(d.MulFloat(3.0).Add(b.Sub(a).Relu()))
	//
	//e := c.Sub(d)
	//f := e.Pow(2)
	//g := f.DivFloat(2.0)
	//g = g.Add(f.Pow(-1).MulFloat(10))
	//fmt.Println(g.String()) // 24.5041
	//g.Backward()
	//fmt.Println(a.String()) // 138.8338
	//fmt.Println(b.String()) // 645.5773

	var N int = 100
	var EPOCHS int = 100
	var lr float64 = 1e-3

	var nFeatures int = 8
	var x [][]Value = make([][]Value, N)
	var y []Value = make([]Value, N)

	for i := 0; i < N; i++ {
		x[i] = make([]Value, nFeatures)
		for j := 0; j < nFeatures; j++ {
			x[i][j] = *RandomValue()
			x[i][j].RequiresGrad = false

			y[i] = *NewValue(x[i][0].MulFloat(0.5).Add(x[i][1].MulFloat(-7.5)).AddFloat(20.0).Data)
			y[i].RequiresGrad = false;
	}
}

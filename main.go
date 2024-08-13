package main

import (
	"fmt"
	. "go-nn/engine"
)

func main() {

	// Test value
	a := NewValue(-4.0)
	b := NewValue(2.0)
	c := a.Add(b)
	d := a.Mul(b).Add(b.Pow(3))

	c = c.Add(c.AddFloat(1.0))
	c = c.Add(c.AddFloat(1.0).Add(a.Neg()))
	d = d.Add(d.MulFloat(2.0).Add(b.Add(a).Relu()))
	d = d.Add(d.MulFloat(3.0).Add(b.Sub(a).Relu()))

	e := c.Sub(d)
	f := e.Pow(2)
	g := f.DivFloat(2.0)
	g = g.Add(f.Pow(-1).MulFloat(10))
	fmt.Println(g.String()) // 24.5041
	g.Backward()
	fmt.Println(a.String()) // 138.8338
	fmt.Println(b.String()) // 645.5773

}

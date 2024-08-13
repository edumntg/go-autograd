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
	d = d.Add(d.MulFloat(2.0).Add(b.Add(a).ReLU()))
	d = d.Add(d.MulFloat(3.0).Add(b.Sub(a).ReLU()))

	e := c.Sub(d)
	f := e.Pow(2)
	g := f.DivFloat(2.0)
	g = g.Add(f.Pow(-1).MulFloat(10))
	fmt.Println(g.ToString())
	g.Backward()
	fmt.Println(a.ToString())
	fmt.Println(b.ToString())

}

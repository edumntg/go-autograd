package engine

import (
	"fmt"
	"math"
	"math/rand"
)

type Value struct {
	data         float64
	grad         float64
	_backward    BackwardMethod
	_prev        map[*Value]bool
	requiresGrad bool
}

func NewValueWithChildren(data float64, children map[*Value]bool) *Value {
	return &Value{
		data: data,
		grad: 0.0,
		_backward: BackwardMethod{
			run: func() {
				return
			},
		},
		_prev:        children,
		requiresGrad: true,
	}
}

func NewValue(data float64) *Value {
	return NewValueWithChildren(data, map[*Value]bool{})
}

func (v *Value) Add(other *Value) *Value {
	out := NewValueWithChildren(v.data+other.data, map[*Value]bool{v: true, other: true})
	out._backward = BackwardMethod{
		run: func() {
			if !v.requiresGrad || !other.requiresGrad {
				return
			}

			v.grad += out.grad
			other.grad += out.grad
		},
	}

	return out
}

func (v *Value) AddFloat(other float64) *Value {
	return v.Add(NewValueWithChildren(other, map[*Value]bool{}))
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) SubFloat(other float64) *Value {
	return v.AddFloat(-other)
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValueWithChildren(v.data*other.data, map[*Value]bool{v: true, other: true})
	out._backward = BackwardMethod{
		run: func() {
			if !v.requiresGrad || !other.requiresGrad {
				return
			}

			v.grad += other.grad * out.grad
			other.grad += v.data * out.grad
		},
	}

	return out
}

func (v *Value) MulFloat(other float64) *Value {
	return v.Mul(NewValueWithChildren(other, map[*Value]bool{}))
}

func (v *Value) Pow(other float64) *Value {
	out := NewValueWithChildren(math.Pow(v.data, other), map[*Value]bool{v: true})
	out._backward = BackwardMethod{
		run: func() {
			if !v.requiresGrad {
				return
			}

			v.grad += other * math.Pow(v.data, other-1) * out.grad
		},
	}
	return out
}

func (v *Value) PowInt(other int64) *Value {
	return v.Pow(float64(other))
}

func (v *Value) Neg() *Value {
	return v.MulFloat(-1.0)
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.PowInt(-1))
}

func (v *Value) DivFloat(other float64) *Value {
	otherVal := NewValueWithChildren(other, map[*Value]bool{})
	return v.Div(otherVal)
}

func (v *Value) Inv() *Value {
	return v.Pow(-1.0)
}

func (v *Value) Exp() *Value {
	out := NewValueWithChildren(math.Exp(v.data), map[*Value]bool{v: true})
	out._backward = BackwardMethod{
		run: func() {
			if !v.requiresGrad {
				return
			}

			v.grad += math.Exp(v.data) * out.grad
		},
	}

	return out
}

func (v *Value) Abs() *Value {
	out := NewValueWithChildren(math.Abs(v.data), map[*Value]bool{v: true})
	out._backward = BackwardMethod{
		run: func() {
			if !v.requiresGrad {
				return
			}

			var value float64 = -1.0
			if v.data > 0 {
				value = 1.0
			}

			v.grad += value * out.grad
		},
	}

	return out
}

func (v *Value) Log(other *Value) *Value {
	out := NewValueWithChildren(math.Log(v.data), map[*Value]bool{v: true})
	out._backward = BackwardMethod{
		run: func() {
			if !v.requiresGrad {
				return
			}

			v.grad += v.Inv().data * out.grad
		},
	}

	return out
}

func (v *Value) ReLU() *Value {
	var current float64 = v.data
	if current < 0 {
		current = 0
	}

	out := NewValueWithChildren(current, map[*Value]bool{v: true})
	out._backward = BackwardMethod{
		run: func() {
			if !v.requiresGrad {
				return
			}

			var value float64 = 0.0
			if v.data > 0 {
				value = 1.0
			}

			v.grad += value * out.grad
		},
	}
	return out
}

func (v *Value) Sigmoid() *Value {
	return v.Neg().Exp().AddFloat(1.0).Inv()
}

func (v *Value) Backward() {
	var topo []Value
	visited := make(map[*Value]bool)

	buildTopo(v, topo, visited)
	v.grad = 1.0

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i]._backward.run()
	}
}

func buildTopo(v *Value, topo []Value, visited map[*Value]bool) {
	if !visited[v] {
		visited[v] = true
		for value, exists := range v._prev {
			if exists {
				buildTopo(value, topo, visited)
			}
		}
	}
}

func Random() *Value {
	return NewValue(rand.Float64())
}

func (v *Value) Item() float64 {
	return v.data
}

func (v *Value) ToString() string {
	return fmt.Sprintf("Value(data=%f, grad=%f, requires_grad=%t)", v.data, v.grad, v.requiresGrad)
}

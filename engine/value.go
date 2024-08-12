package engine

import "math"

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

func (v *Value) Div(other *Value) {

}

package engine

import (
	"fmt"
	"math"
	"math/rand"
)

type BackwardMethod func()

type Value struct {
	Data         float64
	Grad         float64
	_backward    BackwardMethod
	Prev         map[*Value]bool
	RequiresGrad bool
}

var rng = rand.New(rand.NewSource(0))

func NewValue(data float64, children ...*Value) *Value {
	v := &Value{
		Data:         data,
		Grad:         0,
		_backward:    func() {},
		Prev:         make(map[*Value]bool),
		RequiresGrad: true,
	}
	for _, child := range children {
		v.Prev[child] = true
	}
	return v
}

func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.Data+other.Data, v, other)
	out._backward = func() {
		if !v.RequiresGrad && !other.RequiresGrad {
			return
		}
		v.Grad += out.Grad
		other.Grad += out.Grad
	}
	return out
}

func (v *Value) AddFloat(other float64) *Value {
	return v.Add(NewValue(other))
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.Data*other.Data, v, other)
	out._backward = func() {
		if !v.RequiresGrad && !other.RequiresGrad {
			return
		}
		v.Grad += other.Data * out.Grad
		other.Grad += v.Data * out.Grad
	}
	return out
}

func (v *Value) MulFloat(other float64) *Value {
	return v.Mul(NewValue(other))
}

func (v *Value) Pow(other float64) *Value {
	out := NewValue(math.Pow(v.Data, other), v)
	out._backward = func() {
		if !v.RequiresGrad {
			return
		}
		v.Grad += other * math.Pow(v.Data, other-1) * out.Grad
	}
	return out
}

func (v *Value) Neg() *Value {
	return v.MulFloat(-1)
}

func (v *Value) Sub(other *Value) *Value {
	return v.Add(other.Neg())
}

func (v *Value) SubFloat(other float64) *Value {
	return v.AddFloat(-other)
}

func (v *Value) Div(other *Value) *Value {
	return v.Mul(other.Pow(-1))
}

func (v *Value) DivFloat(other float64) *Value {
	return v.Div(NewValue(other))
}

func (v *Value) Inv() *Value {
	return v.Pow(-1)
}

func (v *Value) Exp() *Value {
	out := NewValue(math.Exp(v.Data), v)
	out._backward = func() {
		if !v.RequiresGrad {
			return
		}
		v.Grad += math.Exp(v.Data) * out.Grad
	}
	return out
}

func (v *Value) Abs() *Value {
	out := NewValue(math.Abs(v.Data), v)
	out._backward = func() {
		if !v.RequiresGrad {
			return
		}
		value := -1.0
		if v.Data > 0 {
			value = 1.0
		}
		v.Grad += value * out.Grad
	}
	return out
}

func (v *Value) Log() *Value {
	out := NewValue(math.Log(v.Data), v)
	out._backward = func() {
		if !v.RequiresGrad {
			return
		}
		v.Grad += v.Inv().Data * out.Grad
	}
	return out
}

func (v *Value) Relu() *Value {
	current := v.Data
	if current < 0 {
		current = 0
	}
	out := NewValue(current, v)
	out._backward = func() {
		if !v.RequiresGrad {
			return
		}
		value := 0.0
		if out.Data > 0 {
			value = 1.0
		}
		v.Grad += value * out.Grad
	}
	return out
}

func (v *Value) Sigmoid() *Value {
	return v.Neg().Exp().AddFloat(1).Inv()
}

func (v *Value) Backward() {
	topo := []*Value{}
	visited := make(map[*Value]bool)
	inProgress := make(map[*Value]bool)

	var buildTopo func(*Value) error
	buildTopo = func(v *Value) error {
		if inProgress[v] {
			return fmt.Errorf("cycle detected in computation graph")
		}
		if !visited[v] {
			inProgress[v] = true
			for child := range v.Prev {
				if err := buildTopo(child); err != nil {
					return err
				}
			}
			delete(inProgress, v)
			visited[v] = true
			topo = append(topo, v)
		}
		return nil
	}

	if err := buildTopo(v); err != nil {
		panic(err) // or handle the error appropriately
	}
	v.Grad = 1

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i]._backward()
	}
}

func (v *Value) Optimize(lr float64) {
	v.Data -= lr * v.Grad
}

func Random() *Value {
	return NewValue(rng.Float64()*2 - 1)
}

func Flatten(arr [][]*Value) []*Value {
	out := make([]*Value, len(arr))
	for i := range arr {
		out[i] = arr[i][0]
	}
	return out
}

func (v *Value) Item() float64 {
	return v.Data
}

func (v *Value) Detach() *Value {
	out := NewValue(v.Data)
	out.RequiresGrad = false
	return out
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%v, grad=%v, requires_grad=%t)", v.Data, v.Grad, v.RequiresGrad)
}

package optimizer

import (
	"fmt"
	. "go-nn/engine"
	"math"
)

// Scheduler is a function type for learning rate scheduling
type Scheduler func()

// Optimizer implements the IOptimizer interface
type Optimizer struct {
	parameters     []*Value
	lr             float64
	scheduler      Scheduler
	stepsPerformed int
}

// NewOptimizer creates a new Optimizer instance
func NewOptimizer(parameters []*Value, learningRate float64) *Optimizer {
	if learningRate <= 0.0 {
		learningRate = 1e-3
	}
	return &Optimizer{
		parameters:     parameters,
		lr:             learningRate,
		scheduler:      func() {}, // No-op scheduler by default
		stepsPerformed: 0,
	}
}

// Step performs an optimization step
func (o *Optimizer) Step() {
	o.scheduler()
	o.stepsPerformed++
}

// Parameters returns the list of parameters
func (o *Optimizer) Parameters() []*Value {
	return o.parameters
}

// Print prints the parameters
func (o *Optimizer) Print() {
	for _, param := range o.parameters {
		fmt.Println(param)
	}
}

// ZeroGrad resets all gradients to zero
func (o *Optimizer) ZeroGrad() {
	for _, param := range o.parameters {
		param.Grad = 0.0
	}
}

// SetScheduler sets a new scheduler for the optimizer
func (o *Optimizer) SetScheduler(scheduler Scheduler) {
	o.scheduler = scheduler
}

// GetLearningRate returns the current learning rate
func (o *Optimizer) GetLearningRate() float64 {
	return o.lr
}

// SetLearningRate sets a new learning rate
func (o *Optimizer) SetLearningRate(lr float64) {
	o.lr = float64(math.Max(float64(lr), 0))
}

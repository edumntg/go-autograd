package main

import (
	"fmt"
	. "go-nn/nn"
	. "go-nn/tensor"
	"math/rand"
)

func main() {
	// Test tensors
	t1 := NewTensor(0.07)
	t2 := NewTensor(0.35)
	t3 := t1.Add(t2)
	t4 := t1.Mul(t2)
	t5 := t1.Div(t2)

	fmt.Println("Tensor add:", t3)
	fmt.Println("Tensor mul:", t4)
	fmt.Println("Tensor div:", t5)

	// Test Neurons
	fmt.Println("Printing Neuron...")
	n1 := NewNeuron(10)
	n1.Print()

	// Test linear layer
	fmt.Println("Printing linear layer...")
	l1 := NewLinear(8, 10)
	// Generate random input
	val_arr := make([]float32, 8)
	for i := 0; i < 8; i++ {
		val_arr[i] = rand.Float32()
	}
	x := NewTensorArray(val_arr)

	l_out := l1.Forward(x)

	for i := 0; i < len(l_out); i++ {
		fmt.Println(l_out[i].Get())
	}

}

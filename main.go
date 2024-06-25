package main

import (
	"fmt"
	. "go-nn/tensor"
)

func main() {
	t := NewTensor(0.0, 0.0)
	fmt.Println(t)
}

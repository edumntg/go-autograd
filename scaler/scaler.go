package scaler

import (
	. "go-nn/engine"
	"math"
)

type Scaler struct {
	singleMean float64
	singleStd  float64
	single     bool
	mean       []float64
	std        []float64
}

func NewScaler() *Scaler {
	return &Scaler{
		singleMean: 1.0,
		singleStd:  0.0,
		single:     false,
		mean:       make([]float64, 0),
		std:        make([]float64, 0),
	}
}

func (s *Scaler) FitSingle(x []*Value) {
	mean := 0.0
	std := 0.0

	// Compute means
	for _, v := range x {
		mean += v.Data
	}

	for _, v := range x {
		std += math.Pow(v.Item()-mean, 2)
	}

	mean = mean / float64(len(x))
	std = math.Sqrt(std / float64(len(x)))

	s.singleStd = std
	s.singleMean = mean
	s.single = true
}

func (s *Scaler) FitArray(x [][]*Value) {
	mean := make([]float64, len(x))
	std := make([]float64, len(x))

	s.mean = make([]float64, len(x))
	s.std = make([]float64, len(x))

	// Compute means
	for i := 0; i < len(x); i++ {
		mean[i] = 0
		std[i] = 0
		for j := 0; j < len(x[i]); j++ {
			mean[i] += x[i][j].Item()
		}
		mean[i] = mean[i] / float64(len(x[i]))

		for j := 0; j < len(x[i]); j++ {
			std[i] += math.Pow(x[i][j].Item()-mean[j], 2)
		}

		std[i] = math.Sqrt(std[i] / float64(len(x[i])))

		s.mean[i] = mean[i]
		s.std[i] = std[i]
	}
}

func (s *Scaler) FitTransformSingle(x []*Value) {
	s.FitSingle(x)

	// Now transform data
	for i := 0; i < len(x); i++ {
		x[i] = x[i].SubFloat(s.singleMean).DivFloat(s.singleStd)
	}
}

func (s *Scaler) FitTransformArray(x [][]*Value) {
	s.FitArray(x)

	out := make([][]Value, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = make([]Value, len(x[i]))
		for j := 0; j < len(x[i]); j++ {
			x[i][j] = x[i][j].SubFloat(s.mean[i]).DivFloat(s.std[i])
		}
	}
}

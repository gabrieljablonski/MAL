package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

type Vector1D []float64

func (v Vector1D) String() string {
	s := make([]string, len(v))

	for i, x := range v {
		s[i] = fmt.Sprintf("%.5f", x)
	}
	return fmt.Sprintf("[%s]", strings.Join(s, ", "))
}

const (
	D      = 2   // Vector size
	NP     = 100 // Population size
	CR     = 0.9 // Crossover rate
	F      = 0.5 // Weighing factor
	maxGen = 1000
)

var xRange = Vector1D{-5.12, 5.12}

func f(x Vector1D) float64 {
	y := float64(17 * D)
	for _, xi := range x {
		y += xi*xi - 10*math.Cos(2*math.Pi*xi + 1)
	}
	return y
}

func constrain(x, a, b float64) float64 {
	return math.Max(a, math.Min(x, b))
}

func vectorSum(x1, x2 Vector1D) Vector1D {
	for i := range x1 {
		x1[i] += x2[i]
	}
	return x1
}

func vectorDiff(x1, x2 Vector1D) Vector1D {
	for i := range x1{
		x1[i] -= x2[i]
	}
	return x1
}

func vectorMultConst(c float64, x Vector1D) Vector1D{
	for i := range x {
		x[i] *= c
	}
	return x
}

func mutate(i int, xs []Vector1D) Vector1D{
	r1, r2, r3 := rand.Intn(NP), rand.Intn(NP), rand.Intn(NP)
	// 'r1', 'r2', and 'r3' must be mutually distinct and different from 'i'
	for r1 != r2 && r1 != r3 && r2 != r3 && r1 != i && r2 != i && r3 != i{
		r1, r2, r3 = rand.Intn(NP), rand.Intn(NP), rand.Intn(NP)
	}
	// Compute 'v = x[r1] + F*(x[r3] - x[r2])'
	v := vectorSum(xs[r1], vectorMultConst(F, vectorDiff(xs[r3], xs[r2])))

	for i := range v {
		v[i] = constrain(v[i], xRange[0], xRange[1])
	}

	return v
}

func crossover(x, v Vector1D) Vector1D{
	u := make(Vector1D, D)
	L := rand.Intn(D)
	for j := range x {
		rj := rand.Float64()
		element := 0.0
		if rj <= CR || j == L{
			element = v[j]
		}
		if j != L && rj > CR{
			element = x[j]
		}
		u[j] = element
	}
	return u
}

func randX() Vector1D {
	x := make(Vector1D, D)
	for i := 0; i < D; i++ {
		x[i] = 2*xRange[1]*rand.Float64() + xRange[0]
	}
	return x
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func writeToFile(fileName string, xs []Vector1D) {
	file, err := os.Create(fileName)
	check(err)
	defer file.Close()

	w := bufio.NewWriter(file)
	for _, x := range xs{
		y := f(x)

		s := make([]string, len(x))
		for i, xi := range x {
			s[i] = fmt.Sprintf("%.5f", xi)
		}
		_, err = fmt.Fprintf(w, fmt.Sprintf("%s\t%.5f\n", strings.Join(s, "\t"), y))
		check(err)
	}
	w.Flush()
}

func init() {
	seed := time.Now().UnixNano()
	//seed := int64(0)
	rand.Seed(seed)
}

func main() {
	xs := make([]Vector1D, NP)

	for i := 0; i < NP; i++ {
		xs[i]= randX()
	}

	for gen := 0; gen < maxGen; gen++{
		fmt.Printf("Generation: %d\n", gen+1)
		newXs := make([]Vector1D, NP)
		for i, x := range xs {
			v := mutate(i, xs)
			u := crossover(x, v)
			selected := u
			if f(u) > f(x) {
				selected = x
			}
			newXs[i] = selected
		}
		copy(xs, newXs)
	}

	minValue := math.Inf(1)
	minValueIndex := 0
	for i, x := range xs {
		y := f(x)
		if y < minValue {
			minValue = y
			minValueIndex = i
		}
	}

	fmt.Printf("Function minimum: f(%s) = %.5f", xs[minValueIndex], minValue)

	//fileName := fmt.Sprintf("gen_%d.txt", maxGen)
	//writeToFile(fileName, xs)
}

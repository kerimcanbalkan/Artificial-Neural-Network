package neuralNet

import (
	"errors"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// sumAlongAxis sums a matrix along a particular dimension,
// preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)

	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
	default:
		return nil, errors.New("invaild axis, must be 0 or 1")
	}

	return output, nil
}

package neuralnet

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

// Makes a prediction based on a trained
func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {
	// Check if the neural network has been trained.
	if nn.WHidden == nil || nn.WOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.BHidden == nil || nn.BOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.WHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.BHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.WOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.BOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

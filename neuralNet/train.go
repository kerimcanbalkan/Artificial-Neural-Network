package neuralnet

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

// Trains the neural network using backpropagation.
func (nn *NeuralNet) Train(x, y *mat.Dense) error {
	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	WHidden := mat.NewDense(nn.Config.InputNeurons, nn.Config.HiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.Config.HiddenNeurons, nil)
	wOut := mat.NewDense(nn.Config.HiddenNeurons, nn.Config.OutputNeurons, nil)
	bOut := mat.NewDense(1, nn.Config.OutputNeurons, nil)

	WHiddenRaw := WHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		WHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Use backpropagation to adjust the weights and biases.
	if err := nn.Backpropagate(x, y, WHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define the trained neural network.
	nn.WHidden = WHidden
	nn.BHidden = bHidden
	nn.WOut = wOut
	nn.BOut = bOut

	return nil
}

func (nn *NeuralNet) Backpropagate(x, y, WHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	// Loop over the number of epochs.
	for i := 0; i < nn.Config.NumEpochs; i++ {

		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, WHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.Config.LearningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.Config.LearningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		WHiddenAdj := new(mat.Dense)
		WHiddenAdj.Mul(x.T(), dHiddenLayer)
		WHiddenAdj.Scale(nn.Config.LearningRate, WHiddenAdj)
		WHidden.Add(WHidden, WHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.Config.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}

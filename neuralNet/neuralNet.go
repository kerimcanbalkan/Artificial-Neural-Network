package neuralnet

import "gonum.org/v1/gonum/mat"

// neuralNet contains all of the information
// that defines a trained neural network.
type NeuralNet struct {
	Config  NeuralNetConfig
	WHidden *mat.Dense
	BHidden *mat.Dense
	WOut    *mat.Dense
	BOut    *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumEpochs     int
	LearningRate  float64
}

func NewNetwork(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{Config: config}
}

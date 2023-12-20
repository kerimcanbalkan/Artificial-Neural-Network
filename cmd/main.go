package main

import (
	"fmt"
	"log"

	"github.com/kerimcanbalkan/Artificial-Neural-Network/neuralnet"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// Form the training matrices.
	inputs, labels := neuralnet.MakeInputsAndLabels("data/train.csv")

	// Define the network architecture and learning parameters.
	config := neuralnet.NeuralNetConfig{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 3,
		NumEpochs:     5000,
		LearningRate:  0.3,
	}

	// Train the neural network.
	network := neuralnet.NewNetwork(config)
	if err := network.Train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := neuralnet.MakeInputsAndLabels("data/test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.Predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}

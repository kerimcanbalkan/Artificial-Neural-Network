package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// open the training dataset file.
	f, err := os.Open("data/train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold al the float values that will eventually be used to form our matrices.
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// inputsIndex will track the current index of inputs matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {
		// Skip the header row
		if idx == 0 {
			continue
		}
		// Loop over the float columns.
		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to  the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// Form the matrices.
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	// Define our nerwork architecture and learning parameters
	config := nn.neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}
}

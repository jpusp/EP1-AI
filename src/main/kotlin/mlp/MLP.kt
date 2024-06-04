package mlp

import functions.calculateMSE
import functions.loadWeightsFromFile
import functions.saveWeightsToFile

class MLP(
    val layers: List<Layer>,
    val onMSECalculated: (Int, Int, Double) -> Unit
) {
    fun forward(input: List<Double>): List<Double> {
        var currentOutput = input
        layers.forEach { layer ->
            currentOutput = layer.forward(currentOutput)
        }
        return currentOutput
    }

    fun train(
        inputs: List<List<Double>>,
        targets: List<List<Double>>,
        epochs: Int,
        epochsOffset: Int = 0,
        learningRate: Double,
        isEarlyStop: Boolean = false,
        patience: Int = 50,
        onEarlyStop: (Int) -> Unit
    ) {
        val minDelta = 0.5e-4
        var bestMSE = Double.MAX_VALUE
        var epochsWithoutImprovement = 0
        var bestWeightsHiddenLayer: List<Pair<List<Double>, Double>>? = null
        var bestWeightsOutputLayer: List<Pair<List<Double>, Double>>? = null

        for (epoch in 1..epochs) {
            var sumSquaredErrors = 0.0
            var totalCount = 0

            inputs.zip(targets).forEach { (input, target) ->
                val output = forward(input)
                val error = target.zip(output).map { (target, output) ->
                    val diff = target - output
                    sumSquaredErrors += diff * diff
                    diff
                }
                totalCount += target.size
                backpropagate(error, learningRate)
            }

            val mse = sumSquaredErrors / totalCount
            onMSECalculated(epoch, epochsOffset, mse)
            if (isEarlyStop) {
                if (bestMSE - mse > minDelta) {
                    bestMSE = mse
                    epochsWithoutImprovement = 0
                    bestWeightsHiddenLayer = getLayerWeights(0)
                    bestWeightsOutputLayer = getLayerWeights(1)
                } else {
                    epochsWithoutImprovement++
                    if (epochsWithoutImprovement >= patience) {
                        onEarlyStop(epoch)
                        break
                    }
                }
            }
        }

        if (isEarlyStop) {
            bestWeightsHiddenLayer?.let { saveWeightsToFile(it, "early_stopping_hidden_weights.txt") }
            bestWeightsOutputLayer?.let { saveWeightsToFile(it, "early_stopping_output_weights.txt") }
        } else {
            saveWeights(
                hiddenLayerPath = "normal_hidden_weights.txt",
                outputLayerPath = "normal_output_weights.txt"
            )
        }
    }

//    fun trainWithEarlyStopping(
//        inputs: List<List<Double>>,
//        targets: List<List<Double>>,
//        validationInputs: List<List<Double>>,
//        validationTargets: List<List<Double>>,
//        epochs: Int,
//        learningRate: Double,
//        patience: Int,
//        onEarlyStop: (Int) -> Unit
//    ) {
//        val minDelta = 0.5e-4
//        var bestMSE = Double.MAX_VALUE
//        var epochsWithoutImprovement = 0
//        var bestWeightsHiddenLayer: List<Pair<List<Double>, Double>>? = null
//        var bestWeightsOutputLayer: List<Pair<List<Double>, Double>>? = null
//
//        for (epoch in 1..epochs) {
//            train(
//                inputs = inputs,
//                targets = targets,
//                epochs = 1,
//                learningRate = learningRate,
//                shouldUpdateMse = false,
//                shouldSaveWeights = false
//            )
//            val mse = test(validationInputs, validationTargets)
//            onMSECalculated(epoch, 0, mse)
//
//            if (bestMSE - mse > minDelta) {
//                bestMSE = mse
//                epochsWithoutImprovement = 0
//                bestWeightsHiddenLayer = getLayerWeights(0)
//                bestWeightsOutputLayer = getLayerWeights(1)
//            } else {
//                epochsWithoutImprovement++
//                if (epochsWithoutImprovement >= patience) {
//                    onEarlyStop(epoch)
//                    break
//                }
//            }
//        }
//
//        bestWeightsHiddenLayer?.let { saveWeightsToFile(it, "early_stopping_hidden_weights.txt") }
//        bestWeightsOutputLayer?.let { saveWeightsToFile(it, "early_stopping_output_weights.txt") }
//    }

    private fun backpropagate(error: List<Double>, learningRate: Double) {
        var currentError = error
        for (layer in layers.reversed()) {
            currentError = layer.backpropagate(currentError, learningRate)
        }
    }

    fun test(inputs: List<List<Double>>, targets: List<List<Double>>): Double {
        val predictions = inputs.map { forward(it) }
        return calculateMSE(targets, predictions)
    }

    fun predict(inputs: List<List<Double>>): List<List<Double>> {
        return inputs.map { input ->
            forward(input)
        }
    }

    fun loadWeights(hiddenLayerPath: String, outputLayerPath: String) {
        val hiddenLayerWeights = loadWeightsFromFile(hiddenLayerPath)
        val outputLayerWeights = loadWeightsFromFile(outputLayerPath)

        layers[0].neurons.forEachIndexed { index, neuron ->
            neuron.weights = hiddenLayerWeights[index].first.toMutableList()
            neuron.bias = hiddenLayerWeights[index].second
        }

        layers[1].neurons.forEachIndexed { index, neuron ->
            neuron.weights = outputLayerWeights[index].first.toMutableList()
            neuron.bias = outputLayerWeights[index].second
        }
    }

    fun getLayerWeights(layerIndex: Int): List<Pair<List<Double>, Double>> {
        return layers[layerIndex].neurons.map { neuron ->
            Pair(neuron.weights.toList(), neuron.bias)
        }
    }

    private fun saveWeights(hiddenLayerPath: String, outputLayerPath: String) {
        saveWeightsToFile(getLayerWeights(0), hiddenLayerPath)
        saveWeightsToFile(getLayerWeights(1), outputLayerPath)
    }
}
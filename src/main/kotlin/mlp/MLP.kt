package mlp

import functions.calculateMSE
import functions.loadWeightsFromFile
import functions.saveWeightsToFile
import java.io.File

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
        learningRate: Double
    ) {
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
        }

        saveWeights(
            hiddenLayerPath = "normal_hidden_weights.txt",
            outputLayerPath = "normal_output_weights.txt"
        )
    }

    fun trainWithEarlyStopping(
        inputs: List<List<Double>>,
        targets: List<List<Double>>,
        validationInputs: List<List<Double>>,
        validationTargets: List<List<Double>>,
        epochs: Int,
        learningRate: Double,
        patience: Int,

    ) {
        val minDelta: Double = 1e-4
        var bestMSE = Double.MAX_VALUE
        var epochsWithoutImprovement = 0
        var bestWeightsHiddenLayer: List<Pair<List<Double>, Double>>? = null
        var bestWeightsOutputLayer: List<Pair<List<Double>, Double>>? = null

        for (epoch in 1..epochs) {
            print("Época: $epoch - ")
            train(
                inputs = inputs,
                targets = targets,
                epochs = 1,
                learningRate = learningRate
            )
            val mse = test(validationInputs, validationTargets)

            if (bestMSE - mse > minDelta) {
                bestMSE = mse
                epochsWithoutImprovement = 0
                bestWeightsHiddenLayer = getLayerWeights(0)
                bestWeightsOutputLayer = getLayerWeights(1)
            } else {
                epochsWithoutImprovement++
                if (epochsWithoutImprovement >= patience) {
                    println("Parada antecipada na epoca $epoch")
                    break
                }
            }
        }

        bestWeightsHiddenLayer?.let { saveWeightsToFile(it, "early_stopping_hidden_weights.txt") }
        bestWeightsOutputLayer?.let { saveWeightsToFile(it, "early_stopping_output_weights.txt") }
    }

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

    fun predict(inputs: List<List<Double>>): List<Int> {
        return inputs.map { input ->
            val output = forward(input)
            output.indexOf(output.maxOrNull() ?: 0.0)
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
//        File(hiddenLayerPath).bufferedWriter().use { out ->
//            layers[0].neurons.forEach { neuron ->
//                out.write(neuron.weights.joinToString(",") + "\n")
//            }
//            out.write(layers[0].neurons.map { it.bias }.joinToString(",") + "\n")
//        }
//
//        File(outputLayerPath).bufferedWriter().use { out ->
//            layers[1].neurons.forEach { neuron ->
//                out.write(neuron.weights.joinToString(",") + "\n")
//            }
//            out.write(layers[1].neurons.map { it.bias }.joinToString(",") + "\n")
//        }
    }

//    private fun saveWeights(preffix: String) {
//        layers.forEachIndexed { index, layer ->
//            val layerId = if (index == 0) "hidden" else "output"
//            File("$preffix${layerId}_weights.txt").bufferedWriter().use { out ->
//                layer.neurons.forEach { neuron ->
//                    out.write(neuron.weights.joinToString(",") + "\n")
//                }
//                out.write(layer.neurons.map { it.bias }.joinToString(",") + "\n")
//            }
//        }
//    }

}
package mlp

import functions.calculateMSE
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

        saveWeights()
    }

    fun trainWithEarlyStopping(
        inputs: List<List<Double>>,
        targets: List<List<Double>>,
        validationInputs: List<List<Double>>,
        validationTargets: List<List<Double>>,
        epochs: Int,
        learningRate: Double,
        patience: Int
    ) {
        var bestMSE = Double.MAX_VALUE
        var epochsWithoutImprovement = 0

        for (epoch in 1..epochs) {
            train(
                inputs = inputs,
                targets = targets,
                epochs = 1,
                learningRate = learningRate
            )
            val mse = test(validationInputs, validationTargets)

            if (mse < bestMSE) {
                bestMSE = mse
                epochsWithoutImprovement = 0
            } else {
                epochsWithoutImprovement++
                if (epochsWithoutImprovement >= patience) {
                    println("Parada antecipada na epoca $epoch")
                    break
                }
            }
        }
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

    private fun saveWeights() {
        layers.forEachIndexed { index, layer ->
            File("layer${index+1}_weights.txt").bufferedWriter().use { out ->
                layer.neurons.forEach { neuron ->
                    out.write(neuron.weights.joinToString(",") + "\n")
                }
                out.write(layer.neurons.map { it.bias }.joinToString(",") + "\n")
            }
        }
    }

}
package mlp

import functions.calculateMSE
import java.io.File

class MLP(
    val layers: List<Layer>,
    val onMSECalculated: (Int, Double) -> Unit
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
            onMSECalculated(epoch, mse)
        }

        saveWeights()
    }

    fun crossValidate(
        inputs: List<List<Double>>,
        targets: List<List<Double>>,
        k: Int,
        epochs: Int,
        learningRate: Double
    ): Double {
        val foldSize = inputs.size / k
        var totalMSE = 0.0

        for (i in 0 until k) {
            val validationStart = i * foldSize
            val validationEnd = validationStart + foldSize

            val trainingInputs = inputs.take(validationStart) + inputs.drop(validationEnd)
            val trainingTargets = targets.take(validationStart) + targets.drop(validationEnd)

            val validationInputs = inputs.subList(validationStart, validationEnd)
            val validationTargets = targets.subList(validationStart, validationEnd)

            val mlp = createMLP() // Certifique-se de ter um método para criar uma nova instância de MLP
            mlp.train(trainingInputs, trainingTargets, epochs, learningRate)
            totalMSE += mlp.test(validationInputs, validationTargets)
        }

        return totalMSE / k
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
            train(inputs, targets, 1, learningRate)
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
package mlp

import model.ActivationFunction

class Neuron(
    var weights: MutableList<Double>,
    var bias: Double,
    val activationFunction: ActivationFunction
) {
    var lastActivation: Double = 0.0
    var lastInputs: List<Double> = emptyList()

    fun activate(inputs: List<Double>): Double {
        lastInputs = inputs
        val totalSum = inputs
            .zip(weights) { input, weight -> input * weight }
            .sum()
            .plus(bias)

        lastActivation = activationFunction.function(totalSum)
        return lastActivation
    }
}
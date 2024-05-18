package functions

import kotlin.math.max

fun relu(x: Double): Double = max(0.0, x)

fun sigmoid(x: Double) = 1.0 / (1.0 + Math.exp(-x))

fun sigmoidDerivative(output: Double): Double = output * (1 - output)

fun softmax(inputs: List<Double>): List<Double> {
    val maxInput = inputs.maxOrNull() ?: 0.0
    val expInputs = inputs.map {
        kotlin.math.exp(it - maxInput)
    }

    val sumOfExpInputs = expInputs.sum()

    return expInputs.map {
        it / sumOfExpInputs
    }
}

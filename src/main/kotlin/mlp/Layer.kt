package mlp

class Layer(
    val neurons: List<Neuron>,
    private val inputSize: Int
) {
    fun forward(inputs: List<Double>): List<Double> {
        return neurons.map { neuron ->
            neuron.activate(inputs)
        }
    }

    fun backpropagate(errors: List<Double>, learningRate: Double): List<Double> {
        val nextErrors = MutableList(inputSize) { 0.0 }
        for ((index, neuron) in neurons.withIndex()) {
            val delta = errors[index] * neuron.activationFunction.derivative(neuron.lastActivation)
            neuron.weights.indices.forEach { i ->
                neuron.weights[i] += learningRate * delta * neuron.lastInputs[i]
                nextErrors[i] += neuron.weights[i] * delta
            }
            neuron.bias += learningRate * delta
        }

        return nextErrors
    }
}
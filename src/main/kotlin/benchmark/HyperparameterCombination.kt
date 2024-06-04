package benchmark

import model.ActivationFunction

data class HyperparameterCombination(
    val learningRate: Double,
    val hiddenNeurons: Int,
    val activationFunction: ActivationFunction,
    val epochs: Int,
    var accuracy: Double = 0.0
) {
    override fun toString(): String {
        return "épocas:$epochs - " +
                "taxaAprendizado:$learningRate - " +
                "funçãoAtivação:${activationFunction.name} - " +
                "neuroniosCamadaEscondida:$hiddenNeurons"
    }
}
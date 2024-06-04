package model

data class ActivationFunction(
    val name: String,
    val function: (Double) -> Double,
    val derivative: (Double) -> Double
)

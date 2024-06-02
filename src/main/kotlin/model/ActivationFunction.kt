package model

data class ActivationFunction(
    val function: (Double) -> Double,
    val derivative: (Double) -> Double
)

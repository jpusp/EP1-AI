package functions

import model.ActivationFunction
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.tanh

fun relu(x: Double): Double = max(0.0, x)
fun reluDerivative(x: Double): Double = if (x > 0) 1.0 else 0.0

fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))
fun sigmoidDerivative(output: Double): Double = output * (1 - output)

fun tanhDerivative(output: Double): Double = 1 - output * output

fun swish(x: Double): Double {
    val clippedX = x.coerceIn(-20.0, 20.0)
    return clippedX / (1 + exp(-clippedX))
}

fun swishDerivative(x: Double): Double {
    val clippedX = x.coerceIn(-20.0, 20.0)
    val sigmoid = 1 / (1 + exp(-clippedX))
    return sigmoid + clippedX * sigmoid * (1 - sigmoid)
}

fun softplus(x: Double): Double {
    val clippedX = x.coerceIn(-20.0, 20.0)
    return ln(1 + exp(clippedX))
}

fun softplusDerivative(x: Double): Double {
    val clippedX = x.coerceIn(-20.0, 20.0)
    return 1 / (1 + exp(-clippedX))
}

val reluActivation = ActivationFunction("ReLU", ::relu, ::reluDerivative)
val sigmoidActivation = ActivationFunction("Sigmoid", ::sigmoid, ::sigmoidDerivative)
val tanhActivation = ActivationFunction("TanH", ::tanh, ::tanhDerivative)
val swishActivation = ActivationFunction("Swish",::swish, ::swishDerivative)
val softplusActivation = ActivationFunction("SoftPlus", ::softplus, ::softplusDerivative)

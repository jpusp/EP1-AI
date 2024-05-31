import kotlin.math.pow
import kotlin.math.sqrt

fun calculateConfusionMatrix(
    trueLabels: List<Int>,
    predictedLabels: List<Int>,
    numClasses: Int
): Array<IntArray> {
    val confusionMatrix = Array(numClasses) { IntArray(numClasses) }
    for ((trueLabel, predictedLabel) in trueLabels.zip(predictedLabels)) {
        confusionMatrix[trueLabel][predictedLabel]++
    }
    return confusionMatrix
}

fun calculateAccuracy(confusionMatrix: Array<IntArray>): Double {
    val correctPredictions = confusionMatrix.indices.sumBy { confusionMatrix[it][it] }
    val totalPredictions = confusionMatrix.sumBy { it.sum() }
    return correctPredictions.toDouble() / totalPredictions
}

fun calculatePredictionError(trueLabels: List<Int>, predictedLabels: List<Int>): List<Double> {
    return trueLabels.zip(predictedLabels).map { (trueLabel, predictedLabel) ->
        (trueLabel - predictedLabel).toDouble()
    }
}

fun calculateStandardDeviation(data: List<Double>): Double {
    if (data.isEmpty()) return 0.0
    val mean = data.average()
    val variance = data.sumOf { (it - mean).pow(2) } / data.size
    return sqrt(variance)
}
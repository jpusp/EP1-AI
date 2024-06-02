import kotlin.math.pow
import kotlin.math.sqrt

fun calculateConfusionMatrix(
    trueLabels: List<Char>,
    predictedLabels: List<Char>,
    alphabet: List<Char>
): Array<IntArray> {
    val numClasses = alphabet.size
    val confusionMatrix = Array(numClasses) { IntArray(numClasses) }
    val labelToIndex = alphabet.withIndex().associate { it.value to it.index }

    for ((trueLabel, predictedLabel) in trueLabels.zip(predictedLabels)) {
        val trueIndex = labelToIndex[trueLabel] ?: continue
        val predictedIndex = labelToIndex[predictedLabel] ?: continue
        confusionMatrix[trueIndex][predictedIndex]++
    }
    return confusionMatrix
}


fun calculateAccuracy(confusionMatrix: Array<IntArray>): Double {
    val correctPredictions = confusionMatrix.indices.sumOf { confusionMatrix[it][it] }
    val totalPredictions = confusionMatrix.sumOf { it.sum() }
    return correctPredictions.toDouble() / totalPredictions
}

fun calculatePrecision(confusionMatrix: Array<IntArray>): Double {
    val numClasses = confusionMatrix.size
    var sumPrecision = 0.0
    var count = 0

    for (i in 0 until numClasses) {
        val tp = confusionMatrix[i][i].toDouble()
        val fp = (0 until numClasses).sumOf { confusionMatrix[it][i].toDouble() } - tp
        if (tp + fp > 0) {
            sumPrecision += tp / (tp + fp)
            count++
        }
    }

    return if (count == 0) 0.0 else sumPrecision / count
}

fun calculateRecall(confusionMatrix: Array<IntArray>): Double {
    val numClasses = confusionMatrix.size
    var sumRecall = 0.0
    var count = 0

    for (i in 0 until numClasses) {
        val tp = confusionMatrix[i][i].toDouble()
        val fn = (0 until numClasses).sumOf { confusionMatrix[i][it].toDouble() } - tp
        if (tp + fn > 0) {
            sumRecall += tp / (tp + fn)
            count++
        }
    }

    return if (count == 0) 0.0 else sumRecall / count
}

fun calculateF1Score(confusionMatrix: Array<IntArray>): Double {
    val precision = calculatePrecision(confusionMatrix)
    val recall = calculateRecall(confusionMatrix)

    return if (precision + recall == 0.0) 0.0 else 2 * (precision * recall) / (precision + recall)
}

fun calculateStandardDeviation(data: List<Double>): Double {
    if (data.isEmpty()) return 0.0
    val mean = data.average()
    val variance = data.sumOf { (it - mean).pow(2) } / data.size
    return sqrt(variance)
}
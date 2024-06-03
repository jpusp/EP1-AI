package functions

import model.TrainingSets

fun splitData(
    inputs: List<List<Double>>,
    targets: List<List<Double>>,
    validationSplit: Double = 0.1
): TrainingSets {
    val totalSize = inputs.size
    val validationSize = (totalSize * validationSplit).toInt()

    return TrainingSets(
        trainingInput = inputs.dropLast(validationSize),
        trainingTarget = targets.dropLast(validationSize),
        validationInput = inputs.takeLast(validationSize),
        validationTarget = targets.takeLast(validationSize)
    )
}

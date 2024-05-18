package functions

fun calculateMSE(targets: List<List<Double>>, predictions: List<List<Double>>): Double {
    if (targets.size != predictions.size) {
        throw IllegalArgumentException("Targets e predictions tem tamanhos diferentes.")
    }

    var sumSquaredErrors = 0.0
    var totalCount = 0

    for ((target, prediction) in targets.zip(predictions)) {
        if (target.size != prediction.size) {
            throw IllegalArgumentException("Cada par de target e prediction tem que ter o mesmo tamanho")
        }
        target.zip(prediction).forEach { (target, prediction) ->
            sumSquaredErrors += (target - prediction) * (target - prediction)
        }
        totalCount += target.size
    }

    return sumSquaredErrors / totalCount
}
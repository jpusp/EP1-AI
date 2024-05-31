package model

data class TrainingSets(
    val trainingInput: List<List<Double>>,
    val trainingTarget: List<List<Double>>,
    val validationInput: List<List<Double>>,
    val validationTarget: List<List<Double>>
)

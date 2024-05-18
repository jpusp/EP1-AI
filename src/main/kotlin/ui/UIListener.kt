package ui

interface UIListener {

    fun onTrainButtonClicked(epochs: Int, learningRate: Double)

    fun onTestButtonClicked()
}
package ui

import config.Config
import mlp.Layer
import mlp.MLP
import mlp.Neuron
import functions.sigmoid
import java.io.File
import javax.swing.SwingUtilities

fun main() {
    Main()
}

class Main : UIListener {
    private val config = Config()
    private val ui = NeuralNetworkUI(
        uiListener = this,
        config = config
    )

    init {
        SwingUtilities.invokeLater { ui }
    }

    override fun onTrainButtonClick(config: Config) {
        val trainingPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile()
        )

        val inputs = trainingPair.first
        val targets = trainingPair.second

        val mlp = createMLP(inputs.first().size)

        mlp.train(
            inputs = inputs,
            targets = targets,
            epochs = config.epochs(),
            learningRate = config.learningRate()
        )
    }

    override fun onCrossValidationButtonClick(config: Config) {
        val trainingPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile()
        )

        crossValidate(
            inputs = trainingPair.first,
            targets = trainingPair.second,
            k = config.k(),
            epochs = config.epochs(),
            learningRate = config.learningRate()
        )
    }

    override fun onEarlyStopButtonClick(config: Config) {
        val trainingPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile()
        )

        val inputs = trainingPair.first
        val targets = trainingPair.second

        val mlp = createMLP(inputs.first().size)

//        mlp.trainWithEarlyStopping(
//            inputs = inputs,
//            targets = targets,
//            epochs = config.epochs(),
//            learningRate = config.learningRate()
//        )
    }

    override fun onTestButtonClicked() {
//        val trainingPair = readFiles()
//        val mlp = createMLP()
//
//        val mse = mlp.test(trainingPair.first, trainingPair.second)
//        println("MSE: $mse")
    }

    fun crossValidate(
        inputs: List<List<Double>>,
        targets: List<List<Double>>,
        k: Int,
        epochs: Int,
        learningRate: Double
    ): Double {
        val foldSize = inputs.size / k
        var totalMSE = 0.0

        for (i in 0 until k) {
            val validationStart = i * foldSize
            val validationEnd = validationStart + foldSize

            val trainingInputs = inputs.take(validationStart) + inputs.drop(validationEnd)
            val trainingTargets = targets.take(validationStart) + targets.drop(validationEnd)

            val validationInputs = inputs.subList(validationStart, validationEnd)
            val validationTargets = targets.subList(validationStart, validationEnd)

            val mlp = createMLP(inputs.first().size)
            mlp.train(
                trainingInputs,
                trainingTargets,
                epochs,
                epochsOffset = epochs * i,
                learningRate
            )
            totalMSE += mlp.test(validationInputs, validationTargets)
        }

        return totalMSE / k
    }

    private fun updateMSE(
        epoch: Int,
        epochOffset: Int = 0,
        mse: Double
    ) {
        println("${epoch + epochOffset}; $mse")
        ui.updateMSE(epoch + epochOffset, mse)
    }

    private fun createMLP(
        initialSize: Int
    ): MLP {
        val hiddenLayer = createLayer(
            numInputs = initialSize,
            numNeurons = config.hiddenLayerCount(),
            activationFunction = ::sigmoid
        )

        val outputLayer = createLayer(
            numInputs = config.hiddenLayerCount(),
            numNeurons = config.outputLayerCount(),
            activationFunction = ::sigmoid
        )

        return MLP(
            layers = listOf(
                hiddenLayer,
                outputLayer
            ),
            onMSECalculated = { epoch, epochOffset, mse ->
                updateMSE(epoch, epochOffset, mse)
            }
        )
    }

    private fun createLayer(
        numInputs: Int,
        numNeurons: Int,
        activationFunction: (Double) -> Double
    ): Layer {
        val neurons = (1..numNeurons).map {
            Neuron(
                weights = MutableList(numInputs) { createRandomNumber() },
                bias = createRandomNumber(),
                activationFunction = activationFunction
            )
        }

        return Layer(neurons, numInputs)
    }

    private fun readFiles(
        trainingFile: File?,
        targetFile: File?,
    ): Pair<List<List<Double>>, List<List<Double>>> {
        val trainChars = mutableListOf<List<Double>>()
        val targets = mutableListOf<List<Double>>()

        trainingFile?.forEachLine { line ->
            val cleanedLine = line.cleanInput()
            val values = cleanedLine.split(",").map { it.toInt().toDouble() }.normalize()
            trainChars.add(values)
        }

        targetFile?.forEachLine { line ->
            val cleanedLine = line.cleanInput()
            val values = cleanedLine.split(",").map { it.toInt().toDouble() }.normalize()
            targets.add(values)
        }

        return Pair(trainChars, targets)
    }

    fun List<Double>.normalize(): List<Double> {
        return map { if (it == -1.0) 0.0 else 1.0 }
    }

    fun String.cleanInput(): String = this.replace("\uFEFF", "")

    private fun createRandomNumber(): Double = Math.random() * 0.1 - 0.05
}
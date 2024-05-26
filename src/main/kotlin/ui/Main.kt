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

    override fun onTrainButtonClicked(config: Config) {
        val trainingPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile()
        )
        val mlp = createMLP()

        mlp.train(
            inputs = trainingPair.first,
            targets = trainingPair.second,
            epochs = config.epochs(),
            learningRate = config.learningRate()
        )
    }

    override fun onTestButtonClicked() {
//        val trainingPair = readFiles()
//        val mlp = createMLP()
//
//        val mse = mlp.test(trainingPair.first, trainingPair.second)
//        println("MSE: $mse")
    }

    private fun updateMSE(epoch: Int, mse: Double) {
        println("$epoch; $mse")
        ui.updateMSE(epoch, mse)
    }

    fun createMLP(): MLP {
        val width = 7
        val height = 9
        val numNeuronsHiddenLayer = 35
        val numNeuronsOutputLayer = 7

        val hiddenLayer = createLayer(
            numInputs = width * height,
            numNeurons = numNeuronsHiddenLayer,
            activationFunction = ::sigmoid
        )

        val outputLayer = createLayer(
            numInputs = numNeuronsHiddenLayer,
            numNeurons = numNeuronsOutputLayer,
            activationFunction = ::sigmoid
        )

        return MLP(
            layers = listOf(
                hiddenLayer,
                outputLayer
            ),
            onMSECalculated = ::updateMSE
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
package ui

import calculateAccuracy
import calculateConfusionMatrix
import calculatePredictionError
import calculateStandardDeviation
import config.Config
import functions.saveWeightsToFile
import mlp.Layer
import mlp.MLP
import mlp.Neuron
import functions.sigmoid
import functions.splitData
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

        val trainingSets = splitData(inputs, targets)

        val mlp = createMLP(inputs.first().size)

        mlp.trainWithEarlyStopping(
            inputs = trainingSets.trainingInput,
            targets = trainingSets.trainingTarget,
            validationInputs = trainingSets.validationInput,
            validationTargets = trainingSets.validationTarget,
            epochs = config.epochs(),
            learningRate = config.learningRate(),
            patience = 50
        )
    }

    override fun onTestButtonClicked(
        hiddenWeightsPath: String,
        outputWeightsPath: String
    ) {
        val testPair = readFiles(
            trainingFile = config.testFile(),
            targetFile = config.targetFile()
        )

        val inputs = testPair.first
        val trueLabels = testPair.second.map { it.indexOf(it.maxOrNull() ?: 0.0) }

        val mlp = createMLP(
            initialSize = inputs.firstOrNull()?.size ?: 0,
            loadWeights = true,
            hiddenWeightsPath = hiddenWeightsPath,
            outputWeightsPath = outputWeightsPath,
        )

        val predictedLabels = mlp.predict(inputs)
        val predictionErrors = calculatePredictionError(trueLabels, predictedLabels)

        val confusionMatrix = calculateConfusionMatrix(trueLabels, predictedLabels, config.outputLayerCount())
        val accuracy = calculateAccuracy(confusionMatrix)
        val stdDeviation = calculateStandardDeviation(predictionErrors)

        println("Matriz de Confusão: ${confusionMatrix.contentDeepToString()}")
        println("Acurácia: $accuracy")
        println("Desvio Padrão: $stdDeviation")
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
        var bestMSE = Double.MAX_VALUE
        var bestWeightsHiddenLayer: List<Pair<List<Double>, Double>>? = null
        var bestWeightsOutputLayer: List<Pair<List<Double>, Double>>? = null

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

            val mse = mlp.test(validationInputs, validationTargets)
            totalMSE += mse

            if (mse < bestMSE) {
                bestMSE = mse
                bestWeightsHiddenLayer = mlp.getLayerWeights(0)
                bestWeightsOutputLayer = mlp.getLayerWeights(1)
            }
        }

        bestWeightsHiddenLayer?.let { saveWeightsToFile(it, "cross_validation_hidden_weights.txt") }
        bestWeightsOutputLayer?.let { saveWeightsToFile(it, "cross_validation_output_weights.txt") }

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
        initialSize: Int,
        loadWeights: Boolean = false,
        hiddenWeightsPath: String = "hidden_weights.txt",
        outputWeightsPath: String = "output_weights.txt"
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

        val mlp =  MLP(
            layers = listOf(
                hiddenLayer,
                outputLayer
            ),
            onMSECalculated = { epoch, epochOffset, mse ->
                updateMSE(epoch, epochOffset, mse)
            }
        )

        if (loadWeights) {
            mlp.loadWeights(hiddenWeightsPath, outputWeightsPath)
        }

        return mlp
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
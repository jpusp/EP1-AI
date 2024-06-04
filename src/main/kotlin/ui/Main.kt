package ui

import benchmark.HyperparameterCombination
import config.Config
import functions.*
import mlp.Layer
import mlp.MLP
import mlp.Neuron
import model.ActivationFunction
import java.util.*
import javax.swing.SwingUtilities
import kotlin.math.sqrt

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
            targetFile = config.targetFile(),
            action = { it.dropLast(config.testLinesCount()) }
        )

        val inputs = trainingPair.first
        val targets = trainingPair.second

        val mlp = createMLP(inputs.first().size)

        mlp.train(
            inputs = inputs,
            targets = targets,
            epochs = config.epochs(),
            learningRate = config.learningRate(),
            onEarlyStop = {}
        )
    }

    override fun onCrossValidationButtonClick(config: Config) {
        val trainingPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile(),
            action = { it.dropLast(config.testLinesCount()) }
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
            targetFile = config.targetFile(),
            action = { it.dropLast(config.testLinesCount()) }
        )

        val inputs = trainingPair.first
        val targets = trainingPair.second

        val mlp = createMLP(inputs.first().size)

        mlp.train(
            inputs = inputs,
            targets = targets,
            epochs = config.epochs(),
            epochsOffset = 0,
            learningRate = config.learningRate(),
            patience = config.patience(),
            isEarlyStop = true,
            onEarlyStop = { epoch ->
                ui.appendLog("Parada antecipada na epoca $epoch")
            }
        )
    }

    override fun onTestButtonClicked(
        hiddenWeightsPath: String,
        outputWeightsPath: String
    ) {
        val alphabet = ('A'..'Z').toList()
        val testPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile(),
            action = { it.takeLast(config.testLinesCount()) }
        )

        val testInputs = testPair.first
        val trueLabels = testPair.second
            .takeLast(config.testLinesCount())
            .map { oneHotToChar(it) }

        val mlp = createMLP(
            initialSize = testInputs.firstOrNull()?.size ?: 0,
            loadWeights = true,
            hiddenWeightsPath = hiddenWeightsPath,
            outputWeightsPath = outputWeightsPath,
        )

        val predictedLabels = mlp.predict(testInputs).map { oneHotToChar(it) }

        val confusionMatrix = calculateConfusionMatrix(trueLabels, predictedLabels, alphabet)
        val stdDeviation = calculateStandardDeviation(predictedLabels.map { it.toDouble() })

        println("Desvio Padrão: $stdDeviation")

        displayConfusionMatrixWithJFreeChart(confusionMatrix, alphabet)
    }

    private fun crossValidate(
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
                inputs = trainingInputs,
                targets = trainingTargets,
                epochs = epochs,
                epochsOffset = epochs * i,
                learningRate = learningRate,
                onEarlyStop = {}
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

    override fun onHyperParamsButtonClick() {
        val hyperparameters = listOf(
            HyperparameterCombination(0.1, 60, sigmoidActivation, 100),
            HyperparameterCombination(0.2, 60, sigmoidActivation, 100),
            HyperparameterCombination(0.5, 60, sigmoidActivation, 100),
            HyperparameterCombination(0.6, 60, sigmoidActivation, 100),
            HyperparameterCombination(0.2, 30, sigmoidActivation, 100),
            HyperparameterCombination(0.5, 70, sigmoidActivation, 120),
            HyperparameterCombination(0.4, 100, sigmoidActivation, 150),
            HyperparameterCombination(0.4, 70, sigmoidActivation, 300)
        )

        val results = hyperparameters.map {
            ui.logHyperParams(it)
            trainAndEvaluate(it)
        }

        displayAccuracyGraph(results)
    }

    fun trainAndEvaluate(params: HyperparameterCombination): HyperparameterCombination {
        val trainingPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile(),
            action = { it.dropLast(config.testLinesCount()) }
        )

        val testPair = readFiles(
            trainingFile = config.trainFile(),
            targetFile = config.targetFile(),
            action = { it.takeLast(config.testLinesCount()) }
        )

        val inputs = trainingPair.first
        val targets = trainingPair.second
        val testInputs = testPair.first
        val testTargets = testPair.second

        val mlp = createMLP(
            initialSize = inputs.firstOrNull()?.size ?: 0,
            activationFunction = params.activationFunction,
            hiddenLayerCount = params.hiddenNeurons
        )

        mlp.train(
            inputs = inputs,
            targets =targets,
            epochs = params.epochs,
            learningRate = params.learningRate,
            onEarlyStop = {}
        )

        val trueLabels = testTargets
            .takeLast(config.testLinesCount())
            .map { oneHotToChar(it) }

        val predictedLabels = mlp.predict(testInputs).map { oneHotToChar(it) }

        val confusionMatrix = calculateConfusionMatrix(trueLabels, predictedLabels, alphabet)
        val accuracy = calculateAccuracy(confusionMatrix)
        params.accuracy = accuracy
        return params
    }


    private fun updateMSE(
        epoch: Int,
        epochOffset: Int = 0,
        mse: Double
    ) {
        ui.updateMSE(epoch + epochOffset, mse)
    }

    // Função para criar uma MLP com as camadas ocultas e de saída
    private fun createMLP(
        initialSize: Int,
        loadWeights: Boolean = false,
        activationFunction: ActivationFunction = ui.getSelectedActivationFunction(),
        hiddenLayerCount: Int = config.hiddenLayerCount(),
        outputLayerCount: Int = config.outputLayerCount(),
        hiddenWeightsPath: String = "hidden_weights.txt",
        outputWeightsPath: String = "output_weights.txt"
    ): MLP {
        val hiddenLayer = createLayer(
            numInputs = initialSize,
            numNeurons = hiddenLayerCount,
            activationFunction = activationFunction
        )

        val outputLayer = createLayer(
            numInputs = hiddenLayerCount,
            numNeurons = outputLayerCount,
            activationFunction = activationFunction
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

    // Função para criar uma camada com neurônios
    private fun createLayer(
        numInputs: Int,
        numNeurons: Int,
        activationFunction: ActivationFunction
    ): Layer {
        val neurons = (1..numNeurons).map {
            val weights = if (activationFunction == reluActivation) {
                // cria valores aleatorios diferentes quando a funcao de ativacao é ReLU
                MutableList(numInputs) { createRandomNumberReLU(numInputs) }
            } else {
                MutableList(numInputs) { createRandomNumber() }
            }

            Neuron(
                weights = weights,
                bias = createRandomNumber(),
                activationFunction = activationFunction
            )
        }

        return Layer(neurons, numInputs)
    }

    private fun createRandomNumber(): Double = kotlin.random.Random.nextDouble() * 0.1 - 0.05

    private fun createRandomNumberReLU(numInputs: Int): Double {
        val random = Random()
        return random.nextGaussian() * sqrt(2.0 / numInputs)
    }
}
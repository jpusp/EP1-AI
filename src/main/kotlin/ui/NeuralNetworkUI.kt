package ui

import config.Config
import functions.*
import model.ActivationFunction
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.axis.*
import org.jfree.chart.labels.StandardXYToolTipGenerator
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import ui.components.chooseFile
import ui.components.createColumn
import ui.components.createInput
import ui.components.showError
import java.awt.GridLayout
import java.io.File
import javax.swing.*


const val NoFileSelected = "Nenhum arquivo selecionado"

class NeuralNetworkUI(
    private val uiListener: UIListener,
    private val config: Config
) : JFrame("Neural Network Configuration") {
    private var trainingFile: File? = null
    private var targetFile: File? = null
    private var testFile: File? = null
    private val trainingFileLabel = JLabel(NoFileSelected)
    private val targetFileLabel = JLabel(NoFileSelected)
    private val mseSeries = XYSeries("MSE")
    private val epochsTextField = JTextField("", 10)
    private val learningRateField = JTextField("0.2", 10)
    private val kFoldsField = JTextField("10", 10)
    private val hiddenLayersTextField = JTextField("35", 10)
    private val outputLayersTextField = JTextField("7", 10)
    private val textLinesTextField = JTextField("130", 10)
    private val activationButtonGroup = ButtonGroup()

    private val activationFunctions = listOf(
        "Sigmoid" to sigmoidActivation,
        "ReLU" to reluActivation,
        "TanH" to tanhActivation,
        "Swish" to swishActivation,
        "SoftPlus" to softplusActivation
    )

    private val activationRadioButtons = activationFunctions.map { (name, _) ->
        JRadioButton(name).apply {
            activationButtonGroup.add(this)
        }
    }

    init {
        setSize(1000, 600)
        defaultCloseOperation = EXIT_ON_CLOSE
        layout = BoxLayout(contentPane, BoxLayout.Y_AXIS)
        setLocationRelativeTo(null)

        add(createTrainingSection())
        add(createTrainingButtons())
        add(createTestingButtons())
        add(createGraph())

        isVisible = true
    }

    private fun createTrainingSection(): JPanel {
        val row = JPanel(GridLayout(1, 2, 10, 10))

        row.add(
            createColumn().apply {
                add(JButton("Selecionar Training File").apply {
                    config.trainFile()?.run {
                        trainingFile = this
                        trainingFileLabel.text = this.path
                    }
                    addActionListener {
                        trainingFile = chooseFile(trainingFileLabel, this@NeuralNetworkUI)
                    }
                })
                add(trainingFileLabel)

                add(JButton("Selecionar Target File").apply {
                    config.targetFile()?.run {
                        targetFile = this
                        targetFileLabel.text = this.path
                    }
                    addActionListener {
                        targetFile = chooseFile(targetFileLabel, this@NeuralNetworkUI)
                    }
                })
                add(targetFileLabel)

                add(
                    createInput(
                        label = "Épocas: ",
                        textField = epochsTextField,
                        initialText = config.epochs().toString()
                    )
                )

                add(
                    createInput(
                        label = "Taxa de aprendizado: ",
                        textField = learningRateField,
                        initialText = config.learningRate().toString()
                    )
                )

                add(
                    createInput(
                        label = "Neurônios na camada escondida: ",
                        textField = hiddenLayersTextField,
                        initialText = config.hiddenLayerCount().toString()
                    )
                )

                add(
                    createInput(
                        label = "Neurônios na camada saída: ",
                        textField = outputLayersTextField,
                        initialText = config.outputLayerCount().toString()
                    )
                )

                add(
                    createActivationFunctionSelection()
                )

                add(
                    createInput(
                        label = "k: ",
                        textField = kFoldsField,
                        initialText = config.k().toString()
                    )
                )
            }
        )

        row.add(
            createColumn().apply {
                add(JLabel("Quantas linhas do final do arquivo devem ser usadas para teste?"))
                add(
                    createInput(
                        label = "Linhas: ",
                        textField = textLinesTextField,
                        initialText = config.testLinesCount().toString()
                    )
                )
            }
        )

        return row
    }

    private fun createTrainingButtons(): JPanel {
        return JPanel(GridLayout(1, 3, 10, 10)).also { panel ->

            listOf(
                JButton("Treinar MLP").apply {
                    addActionListener {
                        trainMLP()
                    }
                },
                JButton("Treinar com cross-validation").apply {
                    addActionListener {
                        trainCrossValidation()
                    }
                },
                JButton("Treinar com parada antecipada").apply {
                    addActionListener {
                        trainEarlyStopping()
                    }
                }
            ).forEach { panel.add(it) }

        }
    }

    private fun createTestingButtons(): JPanel {
        return JPanel(GridLayout(1, 3, 10, 10)).also { panel ->
            listOf(
                JButton("Testar MLP").apply {
                    addActionListener {
                        testNetwork(
                            hiddenWeightsPath = "normal_hidden_weights.txt",
                            outputWeightsPath = "normal_output_weights.txt"
                        )
                    }
                },
                JButton("Testar MLP Validação Cruzada").apply {
                    addActionListener {
                        testNetwork(
                            hiddenWeightsPath = "cross_validation_hidden_weights.txt",
                            outputWeightsPath = "cross_validation_output_weights.txt"
                        )
                    }
                },
                JButton("Testar MLP Parada Antecipada").apply {
                    addActionListener {
                        testNetwork(
                            hiddenWeightsPath = "early_stopping_hidden_weights.txt",
                            outputWeightsPath = "early_stopping_output_weights.txt"
                        )
                    }
                }
            ).forEach { panel.add(it) }
        }
    }

    private fun createGraph(): ChartPanel {
        val dataset = XYSeriesCollection(mseSeries)
        val chart = ChartFactory.createXYLineChart(
            "Erro quadrático médio",
            "Época",
            "MSE",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        )

        val plot = chart.xyPlot
        val toolTipGenerator = StandardXYToolTipGenerator()
        plot.renderer.defaultToolTipGenerator = toolTipGenerator

        val logAxis = LogarithmicAxis("MSE")
        plot.rangeAxis = logAxis

        return ChartPanel(chart)
    }

    private fun createActivationFunctionSelection(): JPanel {
        val panel = JPanel()
        panel.layout = BoxLayout(panel, BoxLayout.Y_AXIS)
        panel.add(JLabel("Função de Ativação: "))

        activationRadioButtons.forEach { radioButton ->
            panel.add(radioButton)
        }

        activationRadioButtons.first().isSelected = true

        return panel
    }

    fun updateMSE(epoch: Int, mse: Double) {
        mseSeries.add(epoch.toDouble(), mse)
    }

    fun clearGraph() {
        mseSeries.clear()
    }

    private fun trainMLP() {
        trainNetwork { config ->
            uiListener.onTrainButtonClick(config)
        }
    }

    private fun trainCrossValidation() {
        trainNetwork { config ->
            uiListener.onCrossValidationButtonClick(config)
        }
    }

    private fun trainEarlyStopping() {
        trainNetwork { config ->
            uiListener.onEarlyStopButtonClick(config)
        }
    }

    private fun trainNetwork(
        action: (Config) -> Unit
    ) {
        clearGraph()
        val config = updateConfig()
        val isInputValid = validateInputs(config)

        if (isInputValid) {
            trainingFile?.let {
                targetFile?.let {
                    action(config)
                } ?: showError("Não foi selecionado o arquivo target")
            } ?: showError("Não foi selecionado o arquivo de treinamento")
        }
    }

    private fun testNetwork(
        hiddenWeightsPath: String,
        outputWeightsPath: String
    ) {
        updateConfig()
        uiListener.onTestButtonClicked(
            hiddenWeightsPath = hiddenWeightsPath,
            outputWeightsPath = outputWeightsPath
        )
    }

    private fun updateConfig(): Config {
        val epochs = epochsTextField.text.toIntOrNull() ?: 500
        val learningRate = learningRateField.text.toDoubleOrNull() ?: 0.0
        val hiddenLayerCount = hiddenLayersTextField.text.toIntOrNull() ?: 0
        val outputLayerCount = outputLayersTextField.text.toIntOrNull() ?: 0
        val k = kFoldsField.text.toIntOrNull() ?: 0
        val testLinesCount = textLinesTextField.text.toIntOrNull() ?: 0

        config.saveConfigs(
            trainFile = trainingFile,
            targetFile = targetFile,
            testLinesCount = testLinesCount,
            epochs = epochs,
            learningRate = learningRate,
            k = k,
            hiddenLayerCount = hiddenLayerCount,
            outputLayerCount = outputLayerCount
        )

        return config
    }

    private fun validateInputs(config: Config): Boolean {
        if (config.epochs() <= 0) {
            showError("O número de épocas deve ser maior que zero")
            return false
        }

        if (config.learningRate() <= 0.0) {
            showError("A taxa de aprendizado deve ser maior que zero")
            return false
        }

        if (config.hiddenLayerCount() <= 0) {
            showError("O número de neurônios na camada escondida deve ser maior que zero")
            return false
        }

        if (config.outputLayerCount() <= 0) {
            showError("O número de neurônios na camada de saída deve ser maior que zero")
            return false
        }

        return true
    }

    fun getSelectedActivationFunction(): ActivationFunction {
        return activationFunctions.first { (name, _) ->
            activationRadioButtons.first { it.text == name }.isSelected
        }.second
    }


}
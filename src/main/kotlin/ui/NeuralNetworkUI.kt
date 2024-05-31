package ui

import config.Config
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.axis.LogarithmicAxis
import org.jfree.chart.labels.StandardXYToolTipGenerator
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import java.awt.GridLayout
import java.io.File
import javax.swing.*


const val NoFileSelected = "Nenhum arquivo selecionado"

class NeuralNetworkUI(
    private val uiListener: UIListener,
    private val config: Config
) : JFrame("Neural Network Configuration") {
    private val fileChooser = JFileChooser()
    private var trainingFile: File? = null
    private var targetFile: File? = null
    private var testFile: File? = null
    private val trainingFileLabel = JLabel(NoFileSelected)
    private val targetFileLabel = JLabel(NoFileSelected)
    private val testFileLabel = JLabel(NoFileSelected)
    private val mseSeries = XYSeries("MSE")
    private val epochsTextField = JTextField("", 10)
    private val learningRateField = JTextField("0.2", 10)
    private val kFoldsField = JTextField("10", 10)
    private val hiddenLayersTextField = JTextField("35", 10)
    private val outputLayersTextField = JTextField("7", 10)

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
        val row = JPanel(GridLayout(1, 3, 10, 10))

        row.add(
            createColumn().apply {
                add(JButton("Selecionar Training File").apply {
                    config.trainFile()?.run {
                        trainingFile = this
                        trainingFileLabel.text = this.path
                    }
                    addActionListener {
                        trainingFile = chooseFile(trainingFileLabel)
                    }
                })
                add(trainingFileLabel)

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
                add(JButton("Selecionar Target File").apply {
                    config.targetFile()?.run {
                        targetFile = this
                        targetFileLabel.text = this.path
                    }
                    addActionListener {
                        targetFile = chooseFile(targetFileLabel)
                    }
                })
                add(targetFileLabel)
            }
        )

        row.add(
            createColumn().apply {
                add(JButton("Selecionar Test File").apply {
                    config.testFile()?.run {
                        testFile = this
                        testFileLabel.text = this.path
                    }
                    addActionListener {
                        testFile = chooseFile(testFileLabel)
                    }
                })
                add(testFileLabel)
            }
        )

        return row
    }

    private fun createTrainingButtons(): JPanel {
        return JPanel(GridLayout(1, 4, 10, 10)).also { panel ->

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

        config.saveConfigs(
            trainFile = trainingFile,
            targetFile = targetFile,
            testFile = testFile,
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

    private fun chooseFile(fileLabel: JLabel): File? {
        val returnValue = fileChooser.showOpenDialog(this)
        return if (returnValue == JFileChooser.APPROVE_OPTION) {
            fileLabel.text = fileChooser.selectedFile.path
            fileChooser.selectedFile
        } else null
    }

    private fun createRow() = JPanel().apply {
        layout = BoxLayout(this, BoxLayout.X_AXIS)
        alignmentX = LEFT_ALIGNMENT
    }

    private fun createInput(
        label: String,
        textField: JTextField,
        initialText: String
    ): JPanel {
        return createRow().apply {
            add(JLabel(label))
            add(textField)
            textField.text = initialText
        }
    }

    private fun createColumn() = JPanel().apply {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        alignmentY = TOP_ALIGNMENT
    }

    private fun showError(message: String) {
        JOptionPane.showMessageDialog(null, message, "Erro", JOptionPane.ERROR_MESSAGE)

    }
}
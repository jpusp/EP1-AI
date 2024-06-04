package ui

import benchmark.HyperparameterCombination
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
import ui.components.*
import java.awt.BorderLayout
import java.awt.Dimension
import java.awt.GridLayout
import java.io.File
import javax.swing.*
import javax.swing.border.EmptyBorder
import kotlin.concurrent.thread


const val NoFileSelected = "Nenhum arquivo selecionado"

class NeuralNetworkUI(
    private val uiListener: UIListener,
    private val config: Config
) : JFrame("Neural Network Configuration") {
    private var trainingFile: File? = null
    private var targetFile: File? = null
    private val trainingFileLabel = JLabel(NoFileSelected)
    private val targetFileLabel = JLabel(NoFileSelected)
    private val mseSeries = XYSeries("ErroQM")
    private val epochsTextField = JTextField("", 10)
    private val learningRateField = JTextField("0.2", 10)
    private val kFoldsField = JTextField("10", 10)
    private val hiddenLayersTextField = JTextField("35", 10)
    private val outputLayersTextField = JTextField("7", 10)
    private val textLinesTextField = JTextField("130", 10)
    private val activationButtonGroup = ButtonGroup()
    private val patienceTextField = JTextField("7", 10)
    private val hyperParametersLabel = JLabel().apply {
        alignmentX = LEFT_ALIGNMENT
    }

    private val logTextArea = JTextArea(10, 50)

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
        setSize(1000, 700)
        defaultCloseOperation = EXIT_ON_CLOSE
        layout = BoxLayout(contentPane, BoxLayout.Y_AXIS)

        setLocationRelativeTo(null)

        rootPane.border = EmptyBorder(20, 20, 20, 20)

        add(createFileSection())
        add(Box.createRigidArea(Dimension(0, 10)))
        add(createTrainingSection())
        add(Box.createRigidArea(Dimension(0, 10)))
        add(createTrainingButtons())
        add(createTestingButtons())
        add(createHyperParamsButton())
        add(createGraph())
        add(Box.createRigidArea(Dimension(0, 10)))

        val hyperParamsPanel = JPanel()
        hyperParamsPanel.layout = BorderLayout()
        hyperParamsPanel.add(hyperParametersLabel, BorderLayout.WEST)
        add(hyperParamsPanel)
        add(Box.createRigidArea(Dimension(0, 10)))
        add(createLogArea())

        isVisible = true
    }

    // Cria a seção de seleção de arquivos de treinamento e alvo
    private fun createFileSection(): JPanel {
        val row = JPanel(GridLayout(2, 1, 10, 0))
        return row.apply {
            add(
                createRow().apply {
                    add(
                        JButton("Selecionar Training File").apply {
                            config.trainFile()?.run {
                                trainingFile = this
                                trainingFileLabel.text = this.path
                            }
                            addActionListener {
                                trainingFile = chooseFile(trainingFileLabel, this@NeuralNetworkUI)
                            }
                        }
                    )

                    add(trainingFileLabel)
                }
            )

            add(
                createRow().apply {
                    add(
                        JButton("Selecionar Target File").apply {
                            config.targetFile()?.run {
                                targetFile = this
                                targetFileLabel.text = this.path
                            }
                            addActionListener {
                                targetFile = chooseFile(targetFileLabel, this@NeuralNetworkUI)
                            }
                        }
                    )

                    add(targetFileLabel)
                }
            )
        }
    }

    // Cria a seção de treinamento com os campos de entrada necessários
    private fun createTrainingSection(): JPanel {
        val row = JPanel(GridLayout(1, 2, 10, 10))

        row.add(
            createColumn().apply {
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

        row.add(
            createColumn().apply {
                add(createActivationFunctionSelection())

            }
        )

        return row
    }

    // Cria os botões de treinamento
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

    private fun createHyperParamsButton(): JPanel {
        return JPanel(GridLayout(1, 1, 10, 10)).also { panel ->
            panel.add(
                JButton("Testar múltiplos hiperparâmetros").apply {
                    addActionListener {
                        runInBackground {
                            uiListener.onHyperParamsButtonClick()
                        }
                    }
                }
            )
        }
    }

    // Cria o gráfico para exibir o Erro Quadrático Médio (EQM) ao longo das épocas
    private fun createGraph(): ChartPanel {
        val dataset = XYSeriesCollection(mseSeries)
        val chart = ChartFactory.createXYLineChart(
            "Erro quadrático médio",
            "Época",
            "ErroQM",
            dataset,
            PlotOrientation.VERTICAL,
            true,
            true,
            false
        )

        val plot = chart.xyPlot
        val toolTipGenerator = StandardXYToolTipGenerator()
        plot.renderer.defaultToolTipGenerator = toolTipGenerator

        val logAxis = LogarithmicAxis("ErroQM")
        plot.rangeAxis = logAxis

        return ChartPanel(chart)
    }

    private fun createActivationFunctionSelection(): JPanel {
        val panel = JPanel()
        panel.layout = BoxLayout(panel, BoxLayout.Y_AXIS)

        panel.add(
            createInput("Paciência:", patienceTextField, "50")
        )

        panel.add(JLabel("Função de Ativação: "))

        activationRadioButtons.forEach { radioButton ->
            panel.add(radioButton)
        }

        activationRadioButtons.first().isSelected = true

        return panel
    }

    // Cria a área de log para exibir as mensagens durante o treinamento e teste
    private fun createLogArea(): JScrollPane {
        logTextArea.isEditable = false
        return JScrollPane(logTextArea)
    }

    // Atualiza o gráfico com o valor do EQM para a época fornecida
    fun updateMSE(epoch: Int, mse: Double) {
        SwingUtilities.invokeLater {
            mseSeries.add(epoch.toDouble(), mse)
        }
        appendLog("Época: $epoch, EQM: $mse")
    }

    fun clearExecutionLogs() {
        SwingUtilities.invokeLater {
            hyperParametersLabel.text = ""
            mseSeries.clear()
            logTextArea.text = ""
        }
    }

    fun appendLog(message: String) {
        SwingUtilities.invokeLater {
            logTextArea.append("$message\n")
            logTextArea.caretPosition = logTextArea.document.length
        }
    }

    // Exibe os hiperparâmetros no log e na interface gráfica
    fun logHyperParams(params: HyperparameterCombination) {
        appendLog("Testando Hiperparâmetro: $params")
        hyperParametersLabel.text = params.toString()
    }

    // Inicia o treinamento da rede neural MLP
    private fun trainMLP() {
        runInBackground {
            trainNetwork { config ->
                uiListener.onTrainButtonClick(config)
            }
        }
    }

    // Inicia o treinamento da rede neural MLP com validação cruzada
    private fun trainCrossValidation() {
        runInBackground {
            trainNetwork { config ->
                uiListener.onCrossValidationButtonClick(config)
            }
        }
    }

    // Inicia o treinamento da rede neural MLP com parada antecipada
    private fun trainEarlyStopping() {
        runInBackground {
            trainNetwork { config ->
                uiListener.onEarlyStopButtonClick(config)
            }
        }
    }

    // Função genérica para realizar o treinamento da rede neural
    private fun trainNetwork(
        action: (Config) -> Unit
    ) {
        clearExecutionLogs()
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

    // Testa a rede neural MLP com o conjunto de dados de teste
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

    // Atualiza a configuração com os valores inseridos pelo usuário
    private fun updateConfig(): Config {
        val epochs = epochsTextField.text.toIntOrNull() ?: 500
        val learningRate = learningRateField.text.toDoubleOrNull() ?: 0.0
        val hiddenLayerCount = hiddenLayersTextField.text.toIntOrNull() ?: 0
        val outputLayerCount = outputLayersTextField.text.toIntOrNull() ?: 0
        val k = kFoldsField.text.toIntOrNull() ?: 0
        val testLinesCount = textLinesTextField.text.toIntOrNull() ?: 0
        val patience = patienceTextField.text.toIntOrNull() ?: 0

        config.saveConfigs(
            trainFile = trainingFile,
            targetFile = targetFile,
            testLinesCount = testLinesCount,
            epochs = epochs,
            learningRate = learningRate,
            k = k,
            hiddenLayerCount = hiddenLayerCount,
            outputLayerCount = outputLayerCount,
            patience = patience
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

    // Executa uma ação em background em uma nova thread
    private fun runInBackground(action: () -> Unit) {
        thread {
            try {
                action()
            } catch (e: Exception) {
                appendLog("Erro: ${e.message}")
            }
        }
    }


}
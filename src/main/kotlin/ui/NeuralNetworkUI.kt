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
    private val hiddenLayersTextField = JTextField("35", 10)
    private val outputLayersTextField = JTextField("7", 10)

    //private var epochs = 0
    private var learningRate = 0.2

    init {
        setSize(1000, 600)
        defaultCloseOperation = EXIT_ON_CLOSE
        layout = BoxLayout(contentPane, BoxLayout.Y_AXIS)
        setLocationRelativeTo(null)

        setupTrainingSection()
        setupGraph()
        setupTrainButton()
        setupTestButton()

        isVisible = true
    }

    private fun setupTrainingSection() {
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
                    addActionListener { chooseFile(testFileLabel) }
                })
                add(testFileLabel)
            }
        )

        add(row)
    }

    private fun setupTrainButton() {
        val trainButton = JButton("Train MLP").apply {
            addActionListener {
                trainNetwork()
            }
        }
        add(trainButton)
    }

    private fun setupTestButton() {
        val testButton = JButton("Test MLP").apply {
            addActionListener { testNetwork() }
        }
        add(testButton)
    }

    private fun setupGraph() {
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

        val chartPanel = ChartPanel(chart)
        add(chartPanel)
    }

    fun updateMSE(epoch: Int, mse: Double) {
        mseSeries.add(epoch.toDouble(), mse)
    }

    fun clearGraph() {
        mseSeries.clear()
    }

    private fun trainNetwork() {
        clearGraph()
        val epochs = epochsTextField.text.toIntOrNull() ?: 500
        val learningRate = learningRateField.text.toDoubleOrNull() ?: 0.0
        val hiddenLayerCount = hiddenLayersTextField.text.toIntOrNull() ?: 0
        val outputLayerCount = outputLayersTextField.text.toIntOrNull() ?: 0

        config.saveConfigs(
            trainFile = trainingFile,
            targetFile = targetFile,
            testFile = testFile,
            epochs = epochs,
            learningRate = learningRate,
            hiddenLayerCount = hiddenLayerCount,
            outputLayerCount = outputLayerCount
        )

        if (epochs <= 0) {
            showError("O número de épocas deve ser maior que zero")
            return
        }

        if (learningRate <= 0.0) {
            showError("A taxa de aprendizado deve ser maior que zero")
            return
        }

        if (hiddenLayerCount <= 0) {
            showError("O número de neurônios na camada escondida deve ser maior que zero")
            return
        }

        if (outputLayerCount <= 0) {
            showError("O número de neurônios na camada de saída deve ser maior que zero")
            return
        }

        trainingFile?.let {
            targetFile?.let {
                uiListener.onTrainButtonClicked(config)
            } ?: showError("Não foi selecionado o arquivo target")
        } ?: showError("Não foi selecionado o arquivo de treinamento")
    }

    private fun testNetwork() {
        //uiListener.onTestButtonClicked()
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
            add (JLabel(label))
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
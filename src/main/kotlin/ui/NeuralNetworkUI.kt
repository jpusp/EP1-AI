package ui

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.labels.StandardXYToolTipGenerator
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import java.awt.GridLayout
import javax.swing.*


const val NoFileSelected = "Nenhum arquivo selecionado"

class NeuralNetworkUI(
    private val uiListener: UIListener
) : JFrame("Neural Network Configuration") {
    private val fileChooser = JFileChooser()
    private val trainingFileLabel = JLabel(NoFileSelected)
    private val targetFileLabel = JLabel(NoFileSelected)
    private val testFileLabel = JLabel(NoFileSelected)
    private val mseSeries = XYSeries("MSE")
    private val epochsTextField = JTextField("500", 10)
    private val hiddenLayersTextField = JTextField("35", 10)
    private val outputLayersTextField = JTextField("7", 10)

    private var epochs = 500
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
                    addActionListener { chooseFile(trainingFileLabel) }
                })
                add(trainingFileLabel)

                add(
                    createRow().apply {
                        add (JLabel("Épocas"))
                        add(epochsTextField)
                    }
                )
            }
        )

        row.add(
            createColumn().apply {
                add(JButton("Selecionar Target File").apply {
                    addActionListener { chooseFile(targetFileLabel) }
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

        // Adicionando a linha ao contentPane
        add(row)
    }

    private fun setupTrainButton() {
        val trainButton = JButton("Train MLP").apply {
            addActionListener { trainNetwork() }
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

        val chartPanel = ChartPanel(chart)
        add(chartPanel)
    }

    fun updateMSE(epoch: Int, mse: Double) {
        mseSeries.add(epoch.toDouble(), mse)
    }

    private fun trainNetwork() {
        uiListener.onTrainButtonClicked(epochs, learningRate)
    }

    private fun testNetwork() {
        //uiListener.onTestButtonClicked()
    }

    private fun chooseFile(fileLabel: JLabel) {
        val returnValue = fileChooser.showOpenDialog(this)
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            fileLabel.text = fileChooser.selectedFile.path
        }
    }

    private fun createRow() = JPanel().apply {
        layout = BoxLayout(this, BoxLayout.X_AXIS)
        alignmentX = LEFT_ALIGNMENT
    }

    private fun createColumn() = JPanel().apply {
        layout = BoxLayout(this, BoxLayout.Y_AXIS)
        alignmentY = TOP_ALIGNMENT
    }
}
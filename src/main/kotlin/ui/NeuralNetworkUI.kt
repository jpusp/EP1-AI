package ui

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.labels.StandardXYToolTipGenerator
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
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

    private var epochs = 500
    private var learningRate = 0.2

    init {
        setSize(1000, 600)
        defaultCloseOperation = EXIT_ON_CLOSE
        layout = BoxLayout(contentPane, BoxLayout.Y_AXIS)
        setLocationRelativeTo(null)

        setupFileChoosers()
        setupGraph()
        setupTrainButton()
        setupTestButton()

        isVisible = true
    }

    private fun setupNorthPanel() {
        add(createRow().apply {
            add(JButton("Selecionar Training File").apply {
                addActionListener { chooseFile(trainingFileLabel) }
            })
            add(trainingFileLabel)
        })

        add(createRow().apply {
            add(JButton("Selecionar Target File").apply {
                addActionListener { chooseFile(targetFileLabel) }
            })
            add(targetFileLabel)
        })
    }

    private fun setupFileChoosers() {
        setupNorthPanel()

        val testButton = JButton("Selecionar Test File").apply {
            addActionListener { chooseFile(testFileLabel) }
        }
        add(testButton)
        add(testFileLabel)
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
}
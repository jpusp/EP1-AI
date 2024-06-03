package ui

import functions.calculateAccuracy
import functions.calculatePrecision
import functions.calculateRecall
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.axis.SymbolAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.plot.XYPlot
import org.jfree.chart.renderer.PaintScale
import org.jfree.chart.renderer.xy.XYBlockRenderer
import org.jfree.chart.title.PaintScaleLegend
import org.jfree.chart.ui.RectangleEdge
import org.jfree.data.xy.DefaultXYZDataset
import org.jfree.data.xy.XYZDataset
import ui.components.createColumn
import java.awt.BorderLayout
import java.awt.Color
import java.awt.Paint
import javax.swing.JFrame
import javax.swing.JFrame.DISPOSE_ON_CLOSE
import javax.swing.JLabel
import javax.swing.SwingUtilities
import javax.swing.border.EmptyBorder

fun displayConfusionMatrixWithJFreeChart(confusionMatrix: Array<IntArray>, alphabet: List<Char>) {
    val dataset = DefaultXYZDataset()

    val data = Array(3) { DoubleArray(confusionMatrix.size * confusionMatrix[0].size) }
    var index = 0
    for (i in confusionMatrix.indices) {
        for (j in confusionMatrix[i].indices) {
            data[0][index] = j.toDouble()
            data[1][index] = (confusionMatrix.size - 1 - i).toDouble()
            data[2][index] = confusionMatrix[i][j].toDouble()
            index++
        }
    }
    dataset.addSeries("Matriz de confusão", data)

    val alphabetArray = alphabet.map { it.toString() }.toTypedArray()

    val xAxis = SymbolAxis("Previsão", alphabetArray)
    val yAxis = SymbolAxis("Atual", alphabetArray.reversedArray())

    xAxis.standardTickUnits = NumberAxis.createIntegerTickUnits()
    yAxis.standardTickUnits = NumberAxis.createIntegerTickUnits()

    val renderer = XYBlockRenderer()
    val paintScale = createPaintScale(data[2])
    renderer.paintScale = paintScale

    val plot = XYPlot(dataset as XYZDataset, xAxis, yAxis, renderer)
    plot.orientation = PlotOrientation.VERTICAL

    val legend = PaintScaleLegend(paintScale, NumberAxis())
    legend.position = RectangleEdge.RIGHT

    val chart = JFreeChart("Matriz de confusão", JFreeChart.DEFAULT_TITLE_FONT, plot, false)
    chart.addSubtitle(legend)

    SwingUtilities.invokeLater {
        val frame = JFrame("Resultado do Treinamento")
        frame.layout = BorderLayout()

        frame.rootPane.border = EmptyBorder(20, 20, 20, 20)

        val chartPanel = ChartPanel(chart)
        frame.add(chartPanel, BorderLayout.CENTER)

        val metricsPanel = createColumn().apply {
            val accuracy = calculateAccuracy(confusionMatrix)
            val precision = calculatePrecision(confusionMatrix)
            val recall = calculateRecall(confusionMatrix)

            add(JLabel("Acurácia: $accuracy"))
            add(JLabel("Precisão: $precision"))
            add(JLabel("Sensibilidade: $recall"))
        }

        frame.add(metricsPanel, BorderLayout.SOUTH)

        frame.setSize(800, 600)
        frame.defaultCloseOperation = DISPOSE_ON_CLOSE
        frame.isVisible = true
    }
}

private fun createPaintScale(data: DoubleArray): ColorPaintScale {
    val min = data.minOrNull() ?: 0.0
    val max = data.maxOrNull() ?: 1.0
    return ColorPaintScale(min, max)
}

private class ColorPaintScale(private val min: Double, private val max: Double) : PaintScale {
    override fun getLowerBound(): Double = min
    override fun getUpperBound(): Double = max
    override fun getPaint(value: Double): Paint {
        val normalized = (value - min) / (max - min)
        return Color(1.0f, (1.0 - normalized).toFloat(), (1.0 - normalized).toFloat())
    }
}
package ui

import benchmark.HyperparameterCombination
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.labels.StandardCategoryToolTipGenerator
import org.jfree.chart.plot.CategoryPlot
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.renderer.category.BarRenderer
import org.jfree.data.category.DefaultCategoryDataset
import javax.swing.JFrame

fun displayAccuracyGraph(results: List<HyperparameterCombination>) {
    val dataset = DefaultCategoryDataset()

    for (result in results) {
        dataset.addValue(
            result.accuracy,
            "Acurácia",
            "${result.learningRate}, ${result.hiddenNeurons}, ${result.activationFunction.name}, ${result.epochs}")
    }

    val chart: JFreeChart = ChartFactory.createBarChart(
        "Resultados de teste de hiperparâmetros",
        "Hiperparâmetros",
        "Acurácia",
        dataset,
        PlotOrientation.VERTICAL,
        true, true, false
    )

    val plot: CategoryPlot = chart.categoryPlot
    val renderer = BarRenderer()
    renderer.defaultToolTipGenerator = StandardCategoryToolTipGenerator("{0}: {1} = {2}", java.text.NumberFormat.getInstance())
    plot.renderer = renderer

    val chartPanel = ChartPanel(chart)
    chartPanel.setDisplayToolTips(true)
    val frame = JFrame("Resultado")
    frame.contentPane.add(chartPanel)
    frame.pack()
    frame.setLocationRelativeTo(null)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.DISPOSE_ON_CLOSE
}

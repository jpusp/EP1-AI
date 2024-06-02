package ui.components

import java.awt.Component
import java.io.File
import javax.swing.*
import javax.swing.JFrame.LEFT_ALIGNMENT
import javax.swing.JFrame.TOP_ALIGNMENT

fun createColumn() = JPanel().apply {
    layout = BoxLayout(this, BoxLayout.Y_AXIS)
    alignmentY = TOP_ALIGNMENT
}

private fun createRow() = JPanel().apply {
    layout = BoxLayout(this, BoxLayout.X_AXIS)
    alignmentX = LEFT_ALIGNMENT
}

fun createInput(
    label: String? = null,
    textField: JTextField,
    initialText: String
): JPanel {
    return createRow().apply {
        label?.let { add(JLabel(it)) }
        textField.text = initialText
        textField.maximumSize = textField.preferredSize
        add(textField)
    }
}

fun chooseFile(fileLabel: JLabel, parent: Component): File? {
    val fileChooser = JFileChooser()
    val returnValue = fileChooser.showOpenDialog(parent)
    return if (returnValue == JFileChooser.APPROVE_OPTION) {
        fileLabel.text = fileChooser.selectedFile.path
        fileChooser.selectedFile
    } else null
}

fun showError(message: String) {
    JOptionPane.showMessageDialog(null, message, "Erro", JOptionPane.ERROR_MESSAGE)
}
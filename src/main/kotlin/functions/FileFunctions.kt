package functions

import java.io.File

fun readFiles(
    trainingFile: File?,
    targetFile: File?,
    action: (List<List<Double>>) -> List<List<Double>>
): Pair<List<List<Double>>, List<List<Double>>> {
    val trainChars = mutableListOf<List<Double>>()
    val targets = mutableListOf<List<Double>>()

    trainingFile?.forEachLine { line ->
        val cleanedLine = line.cleanInput()
        val values = cleanedLine
            .split(",")
            .filter { it.isNotEmpty() }
            .map { it.toInt().toDouble() }
        trainChars.add(values)
    }

    targetFile?.forEachLine { line ->
        val cleanedLine = line.cleanInput()
        val char = cleanedLine.firstOrNull()
        if (char != null) {
            val oneHot = charToOneHot(char)
            targets.add(oneHot)
        }
    }

    return Pair(action(trainChars), action(targets))
}

private fun String.cleanInput(): String = this.replace("\uFEFF", "").replace(" ", "")
package functions

import java.io.File

fun loadWeightsFromFile(path: String): List<Pair<List<Double>, Double>> {
    val lines = File(path).readLines()
    val weights = mutableListOf<Pair<List<Double>, Double>>()

    for (i in lines.indices step 2) {
        val w = lines[i].split(",").map(String::toDouble)
        val b = lines[i + 1].toDouble()
        weights.add(Pair(w, b))
    }

    return weights
}

fun saveWeightsToFile(weights: List<Pair<List<Double>, Double>>, filePath: String) {
    File(filePath).bufferedWriter().use { out ->
        weights.forEach { (w, b) ->
            out.write(w.joinToString(",") + "\n")
            out.write(b.toString() + "\n")
        }
    }
}
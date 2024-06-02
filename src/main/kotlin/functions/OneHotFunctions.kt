package functions

val alphabet = ('A'..'Z').toList()

fun charToOneHot(char: Char): List<Double> {
    val oneHot = MutableList(alphabet.size) { 0.0 }
    val index = alphabet.indexOf(char)
    if (index != -1) {
        oneHot[index] = 1.0
    }
    return oneHot
}

fun oneHotToChar(oneHot: List<Double>): Char {
    val index = oneHot.indexOf(oneHot.maxOrNull() ?: 0.0)
    return if (index != -1 && index < alphabet.size) {
        alphabet[index]
    } else {
        '?'
    }
}

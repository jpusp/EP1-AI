package config

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File

private const val TrainFilePathKey = "trainFilePath"
private const val TargetFilePathKey = "targetFilePath"
private const val TestLineCountKey = "testLineCountPath"
private const val EpochsKey = "epochs"
private const val LearningRateKey = "learningRate"
private const val KFoldsKey = "kFolds"
private const val HiddenLayerCountKey = "hiddenLayer"
private const val OutputLayerCountKey = "outputLayer"
class Config {
    private val configMap = mutableMapOf<String, Any>()
    private val gson = Gson()
    private val configFile = File("mlp_config")

    init {
        loadConfigs()
    }

    fun trainFile() = loadFilePath(TrainFilePathKey)

    fun targetFile() = loadFilePath(TargetFilePathKey)

    fun testLinesCount(): Int = loadValue<Number>(TestLineCountKey)?.toInt() ?: 0

    fun epochs(): Int = loadValue<Number>(EpochsKey)?.toInt() ?: 0

    fun learningRate(): Double = loadValue(LearningRateKey) ?: 0.0

    fun k(): Int = loadValue<Number>(KFoldsKey)?.toInt() ?: 0

    fun hiddenLayerCount() = loadValue<Number>(HiddenLayerCountKey)?.toInt() ?: 0

    fun outputLayerCount() = loadValue<Number>(OutputLayerCountKey)?.toInt() ?: 0

    fun saveConfigs(
        trainFile: File? = null,
        targetFile: File? = null,
        testLinesCount: Int = 0,
        epochs: Int = 0,
        learningRate: Double = 0.0,
        k: Int = 0,
        hiddenLayerCount: Int = 0,
        outputLayerCount: Int = 0
    ) {
        with(configMap) {
            putIfNotNull(TrainFilePathKey, trainFile?.path)
            putIfNotNull(TargetFilePathKey, targetFile?.path)
            put(TestLineCountKey, testLinesCount)
            put(EpochsKey, epochs)
            put(LearningRateKey, learningRate)
            put(KFoldsKey, k)
            put(HiddenLayerCountKey, hiddenLayerCount)
            put(OutputLayerCountKey, outputLayerCount)
        }

        storeConfigs()
    }

    private fun loadFilePath(key: String): File? {
        return loadValue<String>(key)?.let { File(it) }
    }

    private inline fun <reified T> loadValue(key: String): T?  {
        return configMap[key]?.let {
            if (it is T) it else null
        }
    }

    private fun storeConfigs() {
        val jsonString = gson.toJson(configMap)
        configFile.writeText(jsonString)
    }

    private fun loadConfigs() {
        if (configFile.exists()) {
            val jsonString = configFile.readText()
            val type = object : TypeToken<MutableMap<String, Any>>() {}.type
            val loadedMap: MutableMap<String, Any> = gson.fromJson(jsonString, type)
            configMap.clear()
            configMap.putAll(loadedMap)
        }
    }

    private fun MutableMap<String, Any>.putIfNotNull(key: String, value: Any?) {
        value?.let { this.put(key, value) }
    }
}
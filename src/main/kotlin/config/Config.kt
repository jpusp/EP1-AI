package config

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File

private const val TrainFilePathKey = "trainFilePath"
private const val TargetFilePathKey = "targetFilePath"
private const val TestFilePathKey = "testFilePath"
private const val EpochsKey = "epochs"
private const val LearningRateKey = "learningRate"
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

    fun saveTrainFilePath(path: File) = saveValue(TrainFilePathKey, path.path)

    fun targetFile() = loadFilePath(TargetFilePathKey)

    fun saveTargetFilePath(path: File) = saveValue(TargetFilePathKey, path.path)

    fun testFile() = loadFilePath(TestFilePathKey)

    fun saveTestFilePath(path: File) = saveValue(TestFilePathKey, path.path)

    fun epochs(): Int = loadValue<Number>(EpochsKey)?.toInt() ?: 0

    fun saveEpochs(epochs: Int) = saveValue(EpochsKey, epochs)

    fun learningRate(): Double = loadValue(LearningRateKey) ?: 0.0

    fun saveLearningRate(learningRate: Double) = saveValue(LearningRateKey, learningRate)

    fun hiddenLayerCount() = loadValue<Number>(HiddenLayerCountKey)?.toInt() ?: 0

    fun saveHiddenLayerCount(count: Int) = saveValue(HiddenLayerCountKey, count)

    fun outputLayerCount() = loadValue<Number>(OutputLayerCountKey)?.toInt() ?: 0

    fun saveOutputLayerCount(count: Int) = saveValue(OutputLayerCountKey, count)

    fun saveConfigs(
        trainFile: File? = null,
        targetFile: File? = null,
        testFile: File? = null,
        epochs: Int = 0,
        learningRate: Double = 0.0,
        hiddenLayerCount: Int = 0,
        outputLayerCount: Int = 0
    ) {
        with(configMap) {
            putIfNotNull(TrainFilePathKey, trainFile?.path)
            putIfNotNull(TargetFilePathKey, targetFile?.path)
            putIfNotNull(TestFilePathKey, testFile?.path)
            put(EpochsKey, epochs)
            put(LearningRateKey, learningRate)
            put(HiddenLayerCountKey, hiddenLayerCount)
            put(OutputLayerCountKey, outputLayerCount)
        }

        storeConfigs()
    }

    private fun loadFilePath(key: String): File? {
        return loadValue<String>(key)?.let { File(it) }
    }

    private fun saveValue(key: String, obj: Any) {
        configMap[key] = obj
        storeConfigs()
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
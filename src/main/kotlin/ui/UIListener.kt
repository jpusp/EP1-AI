package ui

import config.Config

interface UIListener {

    fun onTrainButtonClick(config: Config)

    fun onCrossValidationButtonClick(config: Config)

    fun onEarlyStopButtonClick(config: Config)

    fun onTestButtonClicked(hiddenWeightsPath: String, outputWeightsPath: String)
}
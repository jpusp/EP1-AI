package ui

import config.Config

interface UIListener {

    fun onTrainButtonClicked(config: Config)

    fun onTestButtonClicked()
}
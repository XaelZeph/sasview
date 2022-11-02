from sas.system import config

from .PreferencesWidget import PreferencesWidget


class PlottingPreferencesWidget(PreferencesWidget):
    def __init__(self):
        super(PlottingPreferencesWidget, self).__init__("Plotting Options")
        self.config_params = ['PLOTTING_RESIDUALS_AUTO',
                              'PLOTTING_RESIDUALS_BELOW_MAIN',
                              'FITTING_PLOT_FULL_WIDTH_LEGENDS',
                              'FITTING_PLOT_LEGEND_TRUNCATE',
                              'FITTING_PLOT_LEGEND_MAX_LINE_LENGTH']

    def _addAllWidgets(self):
        self.autoPlotResidual = self.addCheckBox(
            title="Automatically plot residuals?",
            checked=config.PLOTTING_RESIDUALS_AUTO)
        self.autoPlotResidual.clicked.connect(
            lambda: self._stageChange('PLOTTING_RESIDUALS_AUTO', self.autoPlotResidual.isChecked()))
        self.plotResidualsAsSubPlot = self.addCheckBox(
            title="Plot residuals in same window as main plot?",
            checked=config.PLOTTING_RESIDUALS_BELOW_MAIN)
        self.plotResidualsAsSubPlot.clicked.connect(
            lambda: self._stageChange('PLOTTING_RESIDUALS_BELOW_MAIN', self.plotResidualsAsSubPlot.isChecked()))
        self.legendFullWidth = self.addCheckBox(
            title="Use full-width plot legends (most compatible)?",
            checked=config.FITTING_PLOT_FULL_WIDTH_LEGENDS)
        self.legendFullWidth.clicked.connect(
            lambda: self._stageChange('FITTING_PLOT_FULL_WIDTH_LEGENDS', self.legendFullWidth.isChecked()))
        self.legendTruncate = self.addCheckBox(
            title="Use truncated legend entries?",
            checked=config.FITTING_PLOT_LEGEND_TRUNCATE)
        self.legendTruncate.clicked.connect(
            lambda: self._stageChange('FITTING_PLOT_LEGEND_TRUNCATE', self.legendTruncate.isChecked()))
        self.legendLineLength = self.addIntegerInput(
            title="Legend entry line length",
            default_number=config.FITTING_PLOT_LEGEND_MAX_LINE_LENGTH)
        self.legendLineLength.textChanged.connect(
            lambda: self._stageChange('FITTING_PLOT_LEGEND_MAX_LINE_LENGTH', int(self.legendLineLength.text())))

    def _toggleBlockAllSignaling(self, toggle):
        self.autoPlotResidual.blockSignals(toggle)
        self.plotResidualsAsSubPlot.blockSignals(toggle)
        self.legendFullWidth.blockSignals(toggle)
        self.legendTruncate.blockSignals(toggle)
        self.legendLineLength.blockSignals(toggle)

    def _restoreFromConfig(self):
        self.autoPlotResidual.setChecked(config.PLOTTING_RESIDUALS_AUTO)
        self.plotResidualsAsSubPlot.setChecked(config.PLOTTING_RESIDUALS_BELOW_MAIN)
        self.legendFullWidth.setChecked(config.FITTING_PLOT_FULL_WIDTH_LEGENDS)
        self.legendTruncate.setChecked(config.FITTING_PLOT_LEGEND_TRUNCATE)
        self.legendLineLength.setText(str(config.FITTING_PLOT_LEGEND_MAX_LINE_LENGTH))

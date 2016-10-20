#!/usr/bin/env python

from __future__ import print_function
import os
import copy
import logging
import matplotlib
import numpy as np
import scipy
import sys
import signal
from io import BytesIO
import six

matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'

from getdist.gui import SyntaxHighlight
from getdist import plots, IniFile
from getdist.mcsamples import GetChainRootFiles, SettingError, ParamError

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

try:
    import PySide
    from PySide.QtCore import Qt, SIGNAL, QSize, QSettings, QPoint, QCoreApplication
    from PySide.QtGui import *

    os.environ['QT_API'] = 'pyside'
    try:
        import getdist.gui.Resources_pyside
    except ImportError:
        print("Missing Resources_pyside.py: Run script update_resources.sh")
except ImportError:
    print("Can't import PySide modules, install PySide with 'pip install PySide'.")
    sys.exit()

from paramgrid import batchjob, gridconfig
# ==============================================================================

class GuiSelectionError(Exception):
    pass


class ParamListWidget(QListWidget):
    def __init__(self, widget, owner):
        QListWidget.__init__(self, widget)
        self.setDragDropMode(self.InternalMove)
        self.setMaximumSize(QSize(16777215, 120))
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.owner = owner
        list_model = self.model()
        list_model.layoutChanged.connect(owner._updateParameters)


class MainWindow(QMainWindow):
    def __init__(self, app, ini=None, base_dir=None):
        """
        Initialize of GUI components.
        """
        super(MainWindow, self).__init__()

        if base_dir is None: base_dir = batchjob.getCodeRootPath()
        os.chdir(base_dir)
        self.updating = False
        self.app = app
        self.base_dir = base_dir

        self.orig_rc = matplotlib.rcParams.copy()
        self.plot_module = 'getdist.plots'
        self.script_plot_module = self.plot_module

        # GUI setup
        self.createActions()
        self.createMenus()
        self._createWidgets()
        self.createStatusBar()
        self.settingDlg = None
        self.ConfigDlg = None
        self.plotSettingDlg = None
        self.custom_plot_settings = {}

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("GetDist GUI")

        # Allow to shutdown the GUI with Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Path for .ini file
        self.iniFile = ini or getdist.default_getdist_settings

        # Path of root directory
        self.rootdirname = None
        self.plotter = None
        self.root_infos = {}

        self._resetGridData()
        self._resetPlotData()

        Dirs = self.getSettings().value('directoryList')
        lastDir = self.getSettings().value('lastSearchDirectory')

        if Dirs is None and lastDir:
            Dirs = [lastDir]
        elif isinstance(Dirs, six.string_types):
            Dirs = [Dirs]  # Qsettings doesn't save single item lists reliably
        if Dirs is not None:
            Dirs = [x for x in Dirs if os.path.exists(x)]
            if lastDir is not None and not lastDir in Dirs and os.path.exists(lastDir):
                Dirs.insert(0, lastDir)
            self.listDirectories.addItems(Dirs)
            if lastDir is not None and os.path.exists(lastDir):
                self.listDirectories.setCurrentIndex(Dirs.index(lastDir))
                self.openDirectory(lastDir)
            else:
                self.listDirectories.setCurrentIndex(-1)

    def createActions(self):
        """
        Create Qt actions used in GUI.
        """
        self.exportAct = QAction(QIcon(":/images/file_export.png"),
                                 "&Export as PDF/Image", self,
                                 statusTip="Export image as PDF, PNG, JPG",
                                 triggered=self.export)

        self.scriptAct = QAction(QIcon(":/images/file_save.png"),
                                 "Save script", self,
                                 statusTip="Export commands to script",
                                 triggered=self.saveScript)

        self.reLoadAct = QAction("Re-load files", self,
                                 statusTip="Re-scan directory",
                                 triggered=self.reLoad)

        self.exitAct = QAction(QIcon(":/images/application_exit.png"),
                               "E&xit", self,
                               shortcut="Ctrl+Q",
                               statusTip="Exit application",
                               triggered=self.close)

        self.statsAct = QAction(QIcon(":/images/view_text.png"),
                                "Marge Stats", self,
                                shortcut="",
                                statusTip="Show Marge Stats",
                                triggered=self.showMargeStats)

        self.likeStatsAct = QAction(QIcon(":/images/view_text.png"),
                                    "Like Stats", self,
                                    shortcut="",
                                    statusTip="Show Likelihood (N-D) Stats",
                                    triggered=self.showLikeStats)

        self.convergeAct = QAction(QIcon(":/images/view_text.png"),
                                   "Converge Stats", self,
                                   shortcut="",
                                   statusTip="Show Convergence Stats",
                                   triggered=self.showConvergeStats)

        self.PCAAct = QAction(QIcon(":/images/view_text.png"),
                              "Parameter PCA", self,
                              shortcut="",
                              statusTip="Do PCA of selected parameters",
                              triggered=self.showPCA)

        self.paramTableAct = QAction(QIcon(""),
                                     "Parameter table (latex)", self,
                                     shortcut="",
                                     statusTip="View parameter table",
                                     triggered=self.showParamTable)

        self.optionsAct = QAction(QIcon(""),
                                  "Analysis settings", self,
                                  shortcut="",
                                  statusTip="Show settings for getdist and plot densities",
                                  triggered=self.showSettings)

        self.plotOptionsAct = QAction(QIcon(""),
                                      "Plot settings", self,
                                      shortcut="",
                                      statusTip="Show settings for plot display",
                                      triggered=self.showPlotSettings)

        self.configOptionsAct = QAction(QIcon(""),
                                        "Plot module config ", self,
                                        shortcut="",
                                        statusTip="Configure plot module",
                                        triggered=self.showConfigSettings)

        self.aboutAct = QAction(QIcon(":/images/help_about.png"),
                                "&About", self,
                                statusTip="Show About box",
                                triggered=self.about)

    def createMenus(self):
        """
        Create Qt menus.
        """
        self.fileMenu = self.menuBar().addMenu("&File")
        self.fileMenu.addAction(self.exportAct)
        self.fileMenu.addAction(self.scriptAct)
        self.separatorAct = self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.reLoadAct)
        self.fileMenu.addAction(self.exitAct)

        self.menuBar().addSeparator()
        self.dataMenu = self.menuBar().addMenu("&Data")
        self.dataMenu.addAction(self.statsAct)
        self.dataMenu.addAction(self.likeStatsAct)
        self.dataMenu.addAction(self.convergeAct)
        self.dataMenu.addSeparator()
        self.dataMenu.addAction(self.PCAAct)
        self.dataMenu.addAction(self.paramTableAct)

        self.menuBar().addSeparator()
        self.optionMenu = self.menuBar().addMenu("&Options")
        self.optionMenu.addAction(self.optionsAct)
        self.optionMenu.addAction(self.plotOptionsAct)
        self.optionMenu.addAction(self.configOptionsAct)

        self.menuBar().addSeparator()

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.aboutAct)

    def createStatusBar(self):
        """
        Create Qt status bar.
        """
        self.statusBar().showMessage("Ready", 2000)

    def showMessage(self, msg=''):
        self.statusBar().showMessage(msg)
        if msg:
            QCoreApplication.processEvents()

    def _createWidgets(self):
        """
        Create widgets.
        """
        self.tabWidget = QTabWidget(self)
        self.tabWidget.setTabPosition(QTabWidget.East)
        self.tabWidget.setTabPosition(QTabWidget.South)
        self.connect(self.tabWidget, SIGNAL("currentChanged(int)"), self.tabChanged)
        self.setCentralWidget(self.tabWidget)

        # First tab: Gui Selection
        self.firstWidget = QWidget(self)
        self.tabWidget.addTab(self.firstWidget, "Gui Selection")

        self.selectWidget = QWidget(self.firstWidget)

        self.listDirectories = QComboBox(self.selectWidget)
        self.connect(self.listDirectories,
                     SIGNAL("activated(const QString&)"),
                     self.openDirectory)

        self.pushButtonSelect = QPushButton(QIcon(":/images/file_add.png"),
                                            "", self.selectWidget)
        self.pushButtonSelect.setToolTip("Choose root directory")
        self.connect(self.pushButtonSelect, SIGNAL("clicked()"),
                     self.selectRootDirName)
        shortcut = QShortcut(QKeySequence(self.tr("Ctrl+O")), self)
        self.connect(shortcut, SIGNAL("activated()"), self.selectRootDirName)

        self.listRoots = ParamListWidget(self.selectWidget, self)

        self.connect(self.listRoots,
                     SIGNAL("itemChanged(QListWidgetItem *)"),
                     self.updateListRoots)

        self.pushButtonRemove = QPushButton(QIcon(":/images/file_remove.png"),
                                            "", self.selectWidget)
        self.pushButtonRemove.setToolTip("Remove a chain root")
        self.connect(self.pushButtonRemove, SIGNAL("clicked()"),
                     self.removeRoot)

        self.comboBoxParamTag = QComboBox(self.selectWidget)
        self.comboBoxParamTag.clear()
        self.connect(self.comboBoxParamTag,
                     SIGNAL("activated(const QString&)"), self.setParamTag)

        self.comboBoxDataTag = QComboBox(self.selectWidget)
        self.comboBoxDataTag.clear()
        self.connect(self.comboBoxDataTag,
                     SIGNAL("activated(const QString&)"), self.setDataTag)

        self.comboBoxRootname = QComboBox(self.selectWidget)
        self.comboBoxRootname.clear()
        self.connect(self.comboBoxRootname,
                     SIGNAL("activated(const QString&)"), self.setRootname)

        self.listParametersX = QListWidget(self.selectWidget)
        self.listParametersX.clear()

        self.listParametersY = QListWidget(self.selectWidget)
        self.listParametersY.clear()

        self.selectAllX = QCheckBox("Select All", self.selectWidget)
        self.selectAllX.setCheckState(Qt.Unchecked)
        self.connect(self.selectAllX, SIGNAL("clicked()"),
                     self.statusSelectAllX)

        self.selectAllY = QCheckBox("Select All", self.selectWidget)
        self.selectAllY.setCheckState(Qt.Unchecked)
        self.connect(self.selectAllY, SIGNAL("clicked()"),
                     self.statusSelectAllY)

        self.toggleFilled = QRadioButton("Filled")
        self.toggleLine = QRadioButton("Line")
        self.connect(self.toggleLine, SIGNAL("toggled(bool)"),
                     self.statusPlotType)

        self.checkShade = QCheckBox("Shaded", self.selectWidget)
        self.checkShade.setEnabled(False)

        self.toggleColor = QRadioButton("Color by:")
        self.connect(self.toggleColor, SIGNAL("toggled(bool)"),
                     self.statusPlotType)

        self.comboBoxColor = QComboBox(self)
        self.comboBoxColor.clear()
        self.comboBoxColor.setEnabled(False)

        self.toggleFilled.setChecked(True)

        self.trianglePlot = QCheckBox("Triangle Plot", self.selectWidget)
        self.trianglePlot.setCheckState(Qt.Unchecked)

        self.pushButtonPlot = QPushButton("Make plot", self.selectWidget)
        self.connect(self.pushButtonPlot, SIGNAL("clicked()"), self.plotData)

        # Graphic Layout
        layoutTop = QGridLayout()
        layoutTop.setSpacing(5)
        layoutTop.addWidget(self.listDirectories, 0, 0, 1, 3)
        layoutTop.addWidget(self.pushButtonSelect, 0, 3, 1, 1)

        layoutTop.addWidget(self.comboBoxRootname, 1, 0, 1, 3)
        layoutTop.addWidget(self.comboBoxParamTag, 1, 0, 1, 4)
        layoutTop.addWidget(self.comboBoxDataTag, 2, 0, 1, 4)
        layoutTop.addWidget(self.listRoots, 3, 0, 2, 3)
        layoutTop.addWidget(self.pushButtonRemove, 3, 3, 1, 1)

        layoutTop.addWidget(self.selectAllX, 5, 0, 1, 2)
        layoutTop.addWidget(self.selectAllY, 5, 2, 1, 2)
        layoutTop.addWidget(self.listParametersX, 6, 0, 5, 2)
        layoutTop.addWidget(self.listParametersY, 6, 2, 1, 2)
        layoutTop.addWidget(self.toggleFilled, 7, 2, 1, 2)
        layoutTop.addWidget(self.toggleLine, 8, 2, 1, 1)
        layoutTop.addWidget(self.checkShade, 8, 3, 1, 1)

        layoutTop.addWidget(self.toggleColor, 9, 2, 1, 1)
        layoutTop.addWidget(self.comboBoxColor, 9, 3, 1, 1)
        layoutTop.addWidget(self.trianglePlot, 10, 2, 1, 2)
        layoutTop.addWidget(self.pushButtonPlot, 12, 0, 1, 4)
        self.selectWidget.setLayout(layoutTop)

        self.listRoots.hide()
        self.pushButtonRemove.hide()
        self.comboBoxRootname.hide()
        self.comboBoxParamTag.hide()
        self.comboBoxDataTag.hide()

        self.plotWidget = QWidget(self.firstWidget)
        layout = QVBoxLayout(self.plotWidget)
        self.plotWidget.setLayout(layout)

        splitter = QSplitter(self.firstWidget)
        splitter.addWidget(self.selectWidget)
        splitter.addWidget(self.plotWidget)
        w = self.width()
        splitter.setSizes([w / 4., 3 * w / 4.])

        hbox = QHBoxLayout()
        hbox.addWidget(splitter)
        self.firstWidget.setLayout(hbox)

        # Second tab: Script
        self.secondWidget = QWidget(self)
        self.tabWidget.addTab(self.secondWidget, "Script Preview")

        self.editWidget = QWidget(self.secondWidget)
        self.script_edit = ""
        self.plotter_script = None

        self.toolBar = QToolBar()
        openAct = QAction(QIcon(":/images/file_open.png"),
                          "open script", self.toolBar,
                          statusTip="Open script",
                          triggered=self.openScript)
        saveAct = QAction(QIcon(":/images/file_save.png"),
                          "Save script", self.toolBar,
                          statusTip="Save script",
                          triggered=self.saveScript)
        clearAct = QAction(QIcon(":/images/view_clear.png"),
                           "Clear", self.toolBar,
                           statusTip="Clear",
                           triggered=self.clearScript)
        self.toolBar.addAction(openAct)
        self.toolBar.addAction(saveAct)
        self.toolBar.addAction(clearAct)

        self.textWidget = QPlainTextEdit(self.editWidget)
        textfont = QFont("Monospace")
        textfont.setStyleHint(QFont.TypeWriter)
        self.textWidget.setWordWrapMode(QTextOption.NoWrap)
        self.textWidget.setFont(textfont)
        SyntaxHighlight.PythonHighlighter(self.textWidget.document())

        self.pushButtonPlot2 = QPushButton("Make plot", self.editWidget)
        self.connect(self.pushButtonPlot2, SIGNAL("clicked()"), self.plotData2)

        layoutEdit = QVBoxLayout()
        layoutEdit.addWidget(self.toolBar)
        layoutEdit.addWidget(self.textWidget)
        layoutEdit.addWidget(self.pushButtonPlot2)
        self.editWidget.setLayout(layoutEdit)

        self.plotWidget2 = QWidget(self.secondWidget)
        layout2 = QVBoxLayout(self.plotWidget2)
        self.plotWidget2.setLayout(layout2)
        self.scrollArea = QScrollArea(self.plotWidget2)
        self.plotWidget2.layout().addWidget(self.scrollArea)

        splitter2 = QSplitter(self.secondWidget)
        splitter2.addWidget(self.editWidget)
        splitter2.addWidget(self.plotWidget2)
        w = self.width()
        splitter2.setSizes([w / 2., w / 2.])

        hbox2 = QHBoxLayout()
        hbox2.addWidget(splitter2)
        self.secondWidget.setLayout(hbox2)

        self.canvas = None
        self.readSettings()

    def closeEvent(self, event):
        self.writeSettings()
        event.accept()

    def getSettings(self):
        return QSettings('getdist', 'gui')

    def readSettings(self):
        settings = self.getSettings()
        h = min(QApplication.desktop().screenGeometry().height() * 4 / 5., 700)
        size = QSize(min(QApplication.desktop().screenGeometry().width() * 4 / 5., 900), h)
        pos = settings.value("pos", QPoint(100, 100))
        size = settings.value("size", size)
        self.resize(size)
        self.move(pos)
        self.plot_module = settings.value("plot_module", self.plot_module)
        self.script_plot_module = settings.value("script_plot_module", self.script_plot_module)

    def writeSettings(self):
        settings = self.getSettings()
        settings.setValue("pos", self.pos())
        settings.setValue("size", self.size())
        settings.setValue('plot_module', self.plot_module)
        settings.setValue('script_plot_module', self.script_plot_module)

    def export(self):
        """
        Callback for action 'Export as PDF/Image'.
        """
        index = self.tabWidget.currentIndex()
        if index == 0:
            plotter = self.plotter
        else:
            plotter = self.plotter_script

        if plotter:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Choose a file name", '.', "PDF (*.pdf);; Image (*.png *.jpg)")
            if not filename: return
            filename = str(filename)
            plotter.export(filename)
        else:
            QMessageBox.warning(self, "Export", "No plotter data to export")

    def saveScript(self):
        """
        Callback for action 'Save script'.
        """
        index = self.tabWidget.currentIndex()
        if index == 0:
            script = self.script
        else:
            self.script_edit = self.textWidget.toPlainText()
            script = self.script_edit

        if script == '':
            QMessageBox.warning(self, "Script", "No script to save")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Choose a file name", '.', "Python (*.py)")
        if not filename: return
        filename = str(filename)
        logging.debug("Export script to %s" % filename)
        with open(filename, 'w') as f:
            f.write(script)

    def reLoad(self):
        adir = self.getSettings().value('lastSearchDirectory')
        if adir is not None:
            batchjob.resetGrid(adir)
            self.openDirectory(adir)
        if self.plotter:
            self.plotter.sampleAnalyser.reset(self.iniFile)

    def getRootname(self):
        rootname = None
        item = self.listRoots.currentItem()
        if not item and self.listRoots.count(): item = self.listRoots.item(0)
        if item is not None:
            rootname = str(item.text())
        if rootname is None:
            QMessageBox.warning(self, "Chain Stats", "Select a root name first. ")
        return rootname

    def showConvergeStats(self):
        """
        Callback for action 'Show Converge Stats'.
        """
        rootname = self.getRootname()
        if rootname is None: return
        try:
            self.showMessage("Calculating convergence stats....")
            samples = self.getSamples(rootname)
            stats = samples.getConvergeTests(samples.converge_test_limit)
            summary = samples.getNumSampleSummaryText()
            if getattr(samples, 'GelmanRubin', None):
                summary += "var(mean)/mean(var), remaining chains, worst e-value: R-1 = %13.5F" % samples.GelmanRubin
            dlg = DialogConvergeStats(self, stats, summary, rootname)
            dlg.show()
            dlg.activateWindow()
        except Exception as e:
            self.errorReport(e, caption="Convergence stats")
        finally:
            self.showMessage()

    def showPCA(self):
        """
        Callback for action 'Show PCA'.
        """
        rootname = self.getRootname()
        if rootname is None: return
        try:
            samples = self.getSamples(rootname)
            pars = self.getXParams()
            if len(pars) == 1: pars += self.getYParams()
            if len(pars) < 2: raise GuiSelectionError('Select two or more parameters first')
            self.showMessage("Calculating PCA....")
            PCA = samples.PCA(pars)
            dlg = DialogPCA(self, PCA, rootname)
            dlg.show()
        except Exception as e:
            self.errorReport(e, caption="Parameter PCA")
        finally:
            self.showMessage()

    def showMargeStats(self):
        """
        Callback for action 'Show Marge Stats'.
        """
        rootname = self.getRootname()
        if rootname is None: return
        try:
            self.showMessage("Calculating margestats....")
            QCoreApplication.processEvents()
            samples = self.getSamples(rootname)
            stats = samples.getMargeStats()
            dlg = DialogMargeStats(self, stats, rootname)
            dlg.show()
        except Exception as e:
            self.errorReport(e, caption="Marge stats")
        finally:
            self.showMessage()

    def showParamTable(self):
        """
        Callback for action 'Show Parameter Table'.
        """
        rootname = self.getRootname()
        if rootname is None: return
        try:
            samples = self.getSamples(rootname)
            pars = self.getXParams()
            if len(pars) < 1:
                pars = self.getXParams(fulllist=True)
            if len(pars) < 1:
                raise GuiSelectionError('Select one or more parameters first')
            self.showMessage("Generating table....")
            cols = len(pars) // 20 + 1
            tables = [samples.getTable(columns=cols, limit=lim + 1, paramList=pars) for lim in
                      range(len(samples.contours))]
            dlg = DialogParamTables(self, tables, rootname)
            dlg.show()
        except Exception as e:
            self.errorReport(e, caption="Parameter tables")
        finally:
            self.showMessage()

    def showLikeStats(self):
        rootname = self.getRootname()
        if rootname is None: return
        samples = self.getSamples(rootname)
        stats = samples.getLikeStats()
        if stats is None:
            QMessageBox.warning(self, "Like stats", "Samples do not likelihoods")
            return
        dlg = DialogLikeStats(self, stats, rootname)
        dlg.show()

    def showSettings(self):
        """
        Callback for action 'Show settings'
        """
        if not self.plotter:
            QMessageBox.warning(self, "Settings", "Open chains first ")
            return
        if isinstance(self.iniFile, six.string_types): self.iniFile = IniFile(self.iniFile)
        self.settingDlg = self.settingDlg or DialogSettings(self, self.iniFile)
        self.settingDlg.show()
        self.settingDlg.activateWindow()

    def settingsChanged(self):
        if self.plotter:
            self.plotter.sampleAnalyser.reset(self.iniFile)
            if self.plotter.fig:
                self.plotData()

    def showPlotSettings(self):
        """
        Callback for action 'Show plot settings'
        """
        self.getPlotter()
        settings = self.default_plot_settings
        pars = ['plot_meanlikes', 'shade_meanlikes', 'prob_label', 'norm_prob_label', 'prob_y_ticks', 'lineM',
                'plot_args', 'solid_colors', 'default_dash_styles', 'line_labels', 'x_label_rotation',
                'num_shades', 'shade_level_scale', 'tight_layout', 'no_triangle_axis_labels', 'colormap',
                'colormap_scatter', 'colorbar_rotation', 'colorbar_label_pad', 'colorbar_label_rotation',
                'tick_prune', 'tight_gap_fraction', 'legend_loc', 'figure_legend_loc',
                'legend_frame', 'figure_legend_frame', 'figure_legend_ncol', 'legend_rect_border',
                'legend_frac_subplot_margin', 'legend_frac_subplot_line', 'num_plot_contours',
                'solid_contour_palefactor', 'alpha_filled_add', 'alpha_factor_contour_lines', 'axis_marker_color',
                'axis_marker_ls', 'axis_marker_lw']
        pars.sort()
        ini = IniFile()
        for par in pars:
            ini.getAttr(settings, par)
        ini.params.update(self.custom_plot_settings)
        self.plotSettingIni = ini

        self.plotSettingDlg = self.plotSettingDlg or DialogPlotSettings(self, ini, pars, title='Plot Settings',
                                                                        width=450)
        self.plotSettingDlg.show()
        self.plotSettingDlg.activateWindow()

    def plotSettingsChanged(self, vals):
        try:
            settings = self.default_plot_settings
            self.custom_plot_settings = {}
            for key, value in six.iteritems(vals):
                current = getattr(settings, key)
                if str(current) != value and len(value):
                    if isinstance(current, six.string_types):
                        self.custom_plot_settings[key] = value
                    elif current is None:
                        try:
                            self.custom_plot_settings[key] = eval(value)
                        except:
                            self.custom_plot_settings[key] = value
                    else:
                        self.custom_plot_settings[key] = eval(value)
                else:
                    self.custom_plot_settings.pop(key, None)
        except Exception as e:
            self.errorReport(e, caption="Plot settings")
        self.plotData()

    def showConfigSettings(self):
        """
        Callback for action 'Show config settings'
        """
        ini = IniFile()
        ini.params['plot_module'] = self.plot_module
        ini.params['script_plot_module'] = self.script_plot_module
        ini.comments['plot_module'] = ["module used by the GUI (e.g. change to planckStyle)"]
        ini.comments['script_plot_module'] = ["module used by saved plot scripts  (e.g. change to planckStyle)"]
        self.ConfigDlg = self.ConfigDlg or DialogConfigSettings(self, ini, list(ini.params.keys()), title='Plot Config')
        self.ConfigDlg.show()
        self.ConfigDlg.activateWindow()

    def configSettingsChanged(self, vals):
        scriptmod = vals.get('script_plot_module', self.script_plot_module)
        self.script = ''
        mod = vals.get('plot_module', self.plot_module)
        if mod != self.plot_module or scriptmod != self.script_plot_module:
            try:
                matplotlib.rcParams.clear()
                matplotlib.rcParams.update(self.orig_rc)
                __import__(mod)  # test for error
                logging.debug('Loaded module %s', mod)
                self.plot_module = mod
                self.script_plot_module = scriptmod
                if self.plotSettingDlg:
                    self.plotSettingDlg.close()
                    self.plotSettingDlg = None
                if self.plotter:
                    hasPlot = self.plotter.fig
                    self.closePlots()
                    self.getPlotter(loadNew=True)
                    self.custom_plot_settings = {}
                    if hasPlot: self.plotData()
            except Exception as e:
                self.errorReport(e, "plot_module")

    def about(self):
        """
        Callback for action 'About'.
        """
        QMessageBox.about(
            self, "About GetDist GUI",
            "GetDist GUI v " + getdist.__version__ + "\nhttps://github.com/cmbant/getdist/\n" +
            "\nPython: " + sys.version +
            "\nMatplotlib: " + matplotlib.__version__ +
            "\nSciPy: " + scipy.__version__ +
            "\nNumpy: " + np.__version__ +
            "\nPySide: " + PySide.__version__)

    def getDirectories(self):
        return [self.listDirectories.itemText(i) for i in range(self.listDirectories.count())]

    def saveDirectories(self):
        dirs = self.getDirectories()
        if self.rootdirname:
            dirs = [self.rootdirname] + [x for x in dirs if not x == self.rootdirname]
        if len(dirs) > 10: dirs = dirs[:10]
        settings = self.getSettings()
        settings.setValue('directoryList', dirs)
        if self.rootdirname:
            settings.setValue('lastSearchDirectory', self.rootdirname)

    def selectRootDirName(self):
        """
        Slot function called when pushButtonSelect is pressed.
        """
        settings = self.getSettings()
        last_dir = settings.value('lastSearchDirectory')
        if not last_dir: last_dir = os.getcwd()

        title = self.tr("Choose an existing chains grid or chains folder")
        dirName = QFileDialog.getExistingDirectory(
            self, title, last_dir,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        dirName = str(dirName)
        logging.debug("dirName: %s" % dirName)
        if dirName is None or dirName == '':
            return  # No directory selected

        if self.openDirectory(dirName, save=False):
            items = self.getDirectories()
            if dirName in items:
                self.listDirectories.setCurrentIndex(items.index(dirName))
            else:
                self.listDirectories.insertItem(0, dirName)
                self.listDirectories.setCurrentIndex(0)
            self.saveDirectories()

    def openDirectory(self, dirName, save=True):
        try:
            if gridconfig.pathIsGrid(dirName):
                self.rootdirname = dirName
                self._readGridChains(self.rootdirname)
                if save: self.saveDirectories()
                return True

            if self.is_grid:
                self._resetGridData()

            root_list = GetChainRootFiles(dirName)
            if not len(root_list):
                QMessageBox.critical(self, "Open chains", "No chains or grid found in that directory")
                cur_dirs = self.getDirectories()
                if dirName in cur_dirs:
                    self.listDirectories.removeItem(cur_dirs.index(dirName))
                    self.saveDirectories()
                return False
            self.rootdirname = dirName

            self.getPlotter(chain_dir=dirName)

            self._updateComboBoxRootname(root_list)
            if save: self.saveDirectories()
        except Exception as e:
            self.errorReport(e, caption="Open chains", capture=True)
            return False
        return True

    def getPlotter(self, chain_dir=None, loadNew=False):
        try:
            if self.plotter is None or chain_dir or loadNew:
                module = __import__(self.plot_module, fromlist=['dummy'])
                if self.plotter and not loadNew:
                    samps = self.plotter.sampleAnalyser.mcsamples
                else:
                    samps = None
                # set path of grids, so that any custom grid settings get propagated
                chain_dirs = []
                if chain_dir:
                    chain_dirs.append(chain_dir)
                for root in self.root_infos:
                    info = self.root_infos[root]
                    if info.batch:
                        if not info.batch in chain_dirs:
                            chain_dirs.append(info.batch)

                self.plotter = module.getPlotter(mcsamples=True, chain_dir=chain_dirs, analysis_settings=self.iniFile)
                if samps:
                    self.plotter.sampleAnalyser.mcsamples = samps
                self.default_plot_settings = copy.copy(self.plotter.settings)

        except Exception as e:
            self.errorReport(e, caption="Make plotter")
        return self.plotter

    def getSamples(self, root):
        return self.plotter.sampleAnalyser.addRoot(self.root_infos[root])

    def _updateParameters(self):
        roots = self.checkedRootNames()
        if not len(roots):
            return
        paramNames = self.getSamples(roots[0]).paramNames.list()

        self._updateListParameters(paramNames, self.listParametersX, self.getXParams())
        self._updateListParameters(paramNames, self.listParametersY, self.getYParams())
        self._updateComboBoxColor(paramNames)

    def _resetPlotData(self):
        # Script
        self.script = ""

    def _resetGridData(self):
        # Grid chains parameters
        self.is_grid = False
        self.batch = None
        self.grid_paramtag_jobItems = {}
        self.paramTag = ""
        self.dataTag = ""
        self.data2chains = {}
        # self.listRoots.clear()
        # self.listParametersX.clear()
        # self.listParametersY.clear()

    def _readGridChains(self, batchPath):
        """
        Setup of a grid chain.
        """
        # Reset data
        self._resetPlotData()
        self._resetGridData()
        self.is_grid = True
        logging.debug("Read grid chain in %s" % batchPath)
        batch = batchjob.readobject(batchPath)
        self.batch = batch
        items = dict()
        for jobItem in batch.items(True, True):
            if jobItem.chainExists():
                if jobItem.paramtag not in items: items[jobItem.paramtag] = []
                items[jobItem.paramtag].append(jobItem)
        logging.debug("Found %i names for grid" % len(list(items.keys())))
        self.grid_paramtag_jobItems = items

        self.getPlotter(chain_dir=batch)

        self.comboBoxRootname.hide()
        self.listRoots.show()
        self.pushButtonRemove.show()
        self.comboBoxParamTag.clear()
        self.comboBoxParamTag.addItems(sorted(self.grid_paramtag_jobItems.keys()))
        self.setParamTag(self.comboBoxParamTag.itemText(0))
        self.comboBoxParamTag.show()
        self.comboBoxDataTag.show()

    def _updateComboBoxRootname(self, listOfRoots):
        self.comboBoxParamTag.hide()
        self.comboBoxDataTag.hide()
        self.comboBoxRootname.show()
        self.comboBoxRootname.clear()
        self.listRoots.show()
        self.pushButtonRemove.show()
        baseRoots = [os.path.basename(root) for root in listOfRoots]
        self.comboBoxRootname.addItems(baseRoots)
        if len(baseRoots) > 1:
            self.comboBoxRootname.setCurrentIndex(-1)
        elif len(baseRoots):
            self.comboBoxRootname.setCurrentIndex(0)
            self.setRootname(self.comboBoxRootname.itemText(0))

    def newRootItem(self, root):

        for i in range(self.listRoots.count()):
            item = self.listRoots.item(i)
            if str(item.text()) == root:
                item.setCheckState(Qt.Checked)
                self._updateParameters()
                return

        self.updating = True
        item = QListWidgetItem(self.listRoots)
        item.setText('Loading... ' + root)
        self.listRoots.addItem(item)
        self.listRoots.repaint()
        QCoreApplication.processEvents()
        try:
            plotter = self.getPlotter()

            if self.batch:
                path = self.batch.resolveRoot(root).chainPath
            else:
                path = self.rootdirname
            info = plots.RootInfo(root, path, self.batch)
            plotter.sampleAnalyser.addRoot(info)

            self.root_infos[root] = info
            item.setCheckState(Qt.Checked)
            item.setText(root)
            self._updateParameters()
        except Exception as e:
            self.errorReport(e)
            self.listRoots.takeItem(self.listRoots.count() - 1)
            raise
        finally:
            self.updating = False

    def setRootname(self, strParamName):
        """
        Slot function called on change of comboBoxRootname.
        """
        self.newRootItem(str(strParamName))

    def updateListRoots(self, item):
        if self.updating: return
        self._updateParameters()

    def removeRoot(self):
        logging.debug("Remove root")
        self.updating = True
        try:
            for i in range(self.listRoots.count()):
                item = self.listRoots.item(i)
                if item and item.isSelected():
                    root = str(item.text())
                    logging.debug("Remove root %s" % root)
                    self.plotter.sampleAnalyser.removeRoot(root)
                    self.root_infos.pop(root, None)
                    self.listRoots.takeItem(i)
        finally:
            self._updateParameters()
            self.updating = False

    def setParamTag(self, strParamTag):
        """
        Slot function called on change of comboBoxParamTag.
        """
        self.paramTag = str(strParamTag)
        logging.debug("Param: %s" % self.paramTag)

        self.comboBoxDataTag.clear()
        self.comboBoxDataTag.addItems([jobItem.datatag for jobItem in self.grid_paramtag_jobItems[self.paramTag]])
        self.comboBoxDataTag.setCurrentIndex(-1)
        self.comboBoxDataTag.show()

    def setDataTag(self, strDataTag):
        """
        Slot function called on change of comboBoxDataTag.
        """
        self.dataTag = str(strDataTag)
        logging.debug("Data: %s" % strDataTag)
        self.newRootItem(self.paramTag + '_' + self.dataTag)

    def _updateListParameters(self, items, listParameters, items_old=None):
        listParameters.clear()
        for item in items:
            listItem = QListWidgetItem()
            listItem.setText(item)
            listItem.setFlags(listItem.flags() | Qt.ItemIsUserCheckable)
            listItem.setCheckState(Qt.Unchecked)
            listParameters.addItem(listItem)

        if items_old:
            for item in items_old:
                match_items = listParameters.findItems(item, Qt.MatchExactly)
                if match_items:
                    match_items[0].setCheckState(Qt.Checked)

    def getCheckedParams(self, checklist, fulllist=False):
        return [checklist.item(i).text() for i in range(checklist.count()) if
                fulllist or checklist.item(i).checkState() == Qt.Checked]

    def getXParams(self, fulllist=False):
        return self.getCheckedParams(self.listParametersX, fulllist)

    def getYParams(self):
        return self.getCheckedParams(self.listParametersY)

    def statusSelectAllX(self):
        """
        Slot function called when selectAllX is modified.
        """
        if self.selectAllX.isChecked():
            state = Qt.Checked
        else:
            state = Qt.Unchecked
        for i in range(self.listParametersX.count()):
            self.listParametersX.item(i).setCheckState(state)

    def statusSelectAllY(self):
        """
        Slot function called when selectAllY is modified.
        """
        if self.selectAllY.isChecked():
            state = Qt.Checked
        else:
            state = Qt.Unchecked
        for i in range(self.listParametersY.count()):
            self.listParametersY.item(i).setCheckState(state)

    def statusPlotType(self, checked):
        # radio buttons changed
        self.checkShade.setEnabled(self.toggleLine.isChecked())
        self.comboBoxColor.setEnabled(self.toggleColor.isChecked())

    def _updateComboBoxColor(self, listOfParams):
        if self.rootdirname and os.path.isdir(self.rootdirname):
            param_old = str(self.comboBoxColor.currentText())
            self.comboBoxColor.clear()
            self.comboBoxColor.addItems(listOfParams)
            idx = self.comboBoxColor.findText(param_old, Qt.MatchExactly)
            if idx != -1:
                self.comboBoxColor.setCurrentIndex(idx)

    def checkedRootNames(self):
        items = []
        for i in range(self.listRoots.count()):
            item = self.listRoots.item(i)
            if item.checkState() == Qt.Checked:
                items.append(str(item.text()))
        return items

    def errorReport(self, e, caption="Error", msg="", capture=False):
        if isinstance(e, SettingError):
            QMessageBox.critical(self, 'Setting error', str(e))
        elif isinstance(e, ParamError):
            QMessageBox.critical(self, 'Param error', str(e))
        elif isinstance(e, IOError):
            QMessageBox.critical(self, 'File error', str(e))
        elif isinstance(e, (GuiSelectionError, plots.GetDistPlotError)):
            QMessageBox.critical(self, caption, str(e))
        else:
            if not msg:
                import traceback

                msg = "\n".join(traceback.format_tb(sys.exc_info()[2])[-5:])
            QMessageBox.critical(self, caption, str(e) + "\n\n" + msg)
            del msg

        if not isinstance(e, GuiSelectionError) and not capture: raise

    def closePlots(self):
        if self.plotter.fig is not None:
            self.plotter.fig.clf()
        plt.close('all')

    def plotData(self):
        """
        Slot function called when pushButtonPlot is pressed.
        """
        if self.updating: return
        self.showMessage("Generating plot....")
        actionText = "plot"
        try:
            # Ensure at least 1 root name specified
            os.chdir(self.base_dir)

            roots = self.checkedRootNames()
            if not len(roots):
                logging.warning("No rootname selected")
                QMessageBox.warning(self, "Plot data", "No root selected")
                return

            if self.plotter is None:
                QMessageBox.warning(self, "Plot data", "No GetDistPlotter instance")
                return
            self.closePlots()

            # X and Y items
            items_x = self.getXParams()
            items_y = self.getYParams()
            self.plotter.settings = copy.copy(self.default_plot_settings)
            self.plotter.settings.setWithSubplotSize(3.5)
            self.plotter.settings.legend_position_config = 2
            self.plotter.settings.legend_frac_subplot_margin = 0.05
            self.plotter.settings.__dict__.update(self.custom_plot_settings)

            script = "import %s as gplot\nimport os\n\n" % self.script_plot_module
            if isinstance(self.iniFile, IniFile):
                script += 'analysis_settings = %s\n' % self.iniFile.params
            if len(items_x) > 1 or len(items_y) > 1:
                plot_func = 'getSubplotPlotter'
            else:
                plot_func = 'getSinglePlotter'

            for root in roots:
                self.plotter.sampleAnalyser.addRoot(self.root_infos[root])

            chain_dirs = []
            for root in roots:
                info = self.root_infos[root]
                if info.batch:
                    path = info.batch.batchPath
                else:
                    path = info.path
                if not path in chain_dirs:
                    chain_dirs.append(path)
            if len(chain_dirs) == 1:
                chain_dirs = "r'%s'" % chain_dirs[0].rstrip('\\').rstrip('/')

            if isinstance(self.iniFile, six.string_types) and self.iniFile != getdist.default_getdist_settings:
                script += "g=gplot.%s(chain_dir=%s, analysis_settings=r'%s')\n" % (plot_func, chain_dirs, self.iniFile)
            elif isinstance(self.iniFile, IniFile):
                script += "g=gplot.%s(chain_dir=%s,analysis_settings=analysis_settings)\n" % (plot_func, chain_dirs)
            else:
                script += "g=gplot.%s(chain_dir=%s)\n" % (plot_func, chain_dirs)

            if self.custom_plot_settings:
                for key, value in six.iteritems(self.custom_plot_settings):
                    if isinstance(value, six.string_types):
                        value = '"' + value + '"'
                    script += 'g.settings.%s = %s\n' % (key, value)

            if len(roots) < 3:
                script += 'roots = %s\n' % roots
            else:
                script += "roots = []\n"
                for root in roots:
                    script += "roots.append('%s')\n" % root

            logging.debug("Plotting with roots = %s" % str(roots))

            height = self.plotWidget.height() * 0.75
            width = self.plotWidget.width() * 0.75

            def setSizeQT(sz):
                self.plotter.settings.setWithSubplotSize(max(2.0, sz / 80.))

            def setSizeForN(n):
                setSizeQT(min(height, width) / max(n, 2))

            # Plot parameters
            filled = self.toggleFilled.isChecked()
            shaded = not filled and self.checkShade.isChecked()
            line = self.toggleLine.isChecked()
            color = self.toggleColor.isChecked()
            color_param = str(self.comboBoxColor.currentText())

            # Check type of plot
            if self.trianglePlot.isChecked():
                # Triangle plot
                actionText = "Triangle plot"
                if len(items_x) > 1:
                    params = items_x
                    logging.debug("Triangle plot with params = %s" % str(params))
                    script += "params = %s\n" % str(params)
                    if color:
                        param_3d = color_param
                        script += "param_3d = '%s'\n" % str(color_param)
                    else:
                        param_3d = None
                        script += "param_3d = None\n"
                    setSizeForN(len(params))
                    self.plotter.triangle_plot(roots, params, plot_3d_with_param=param_3d, filled=filled,
                                               shaded=shaded)
                    self.updatePlot()
                    script += "g.triangle_plot(roots, params, plot_3d_with_param=param_3d, filled=%s, shaded=%s)\n" % (
                        filled, shaded)
                else:
                    raise GuiSelectionError("Select more than 1 x parameter for triangle plot")

            elif len(items_x) > 0 and len(items_y) == 0:
                # 1D plot
                actionText = "1D plot"
                params = items_x
                logging.debug("1D plot with params = %s" % str(params))
                script += "params=%s\n" % str(params)
                setSizeForN(round(np.sqrt(len(params) / 1.4)))
                if len(roots) > 3:
                    ncol = 2
                else:
                    ncol = None
                self.plotter.plots_1d(roots, params=params, legend_ncol=ncol)
                self.updatePlot()
                script += "g.plots_1d(roots, params=params)\n"

            elif len(items_x) > 0 and len(items_y) > 0:
                if len(items_x) > 1 and len(items_y) > 1:
                    # Rectangle plot
                    actionText = 'Rectangle plot'
                    script += "xparams = %s\n" % str(items_x)
                    script += "yparams = %s\n" % str(items_y)
                    script += "filled=%s\n" % filled
                    logging.debug("Rectangle plot with xparams=%s and yparams=%s" % (str(items_x), str(items_y)))

                    setSizeQT(min(height / len(items_y), width / len(items_x)))
                    self.plotter.rectangle_plot(items_x, items_y, roots=roots, filled=filled)
                    self.updatePlot()
                    script += "g.rectangle_plot(xparams, yparams, roots=roots,filled=filled)\n"

                else:
                    # 2D plot
                    if len(items_x) == 1 and len(items_y) == 1:
                        pairs = [[items_x[0], items_y[0]]]
                        setSizeQT(min(height, width))
                    elif len(items_x) == 1 and len(items_y) > 1:
                        item_x = items_x[0]
                        pairs = list(zip([item_x] * len(items_y), items_y))
                        setSizeForN(round(np.sqrt(len(pairs) / 1.4)))
                    elif len(items_x) > 1 and len(items_y) == 1:
                        item_y = items_y[0]
                        pairs = list(zip(items_x, [item_y] * len(items_x)))
                        setSizeForN(round(np.sqrt(len(pairs) / 1.4)))
                    else:
                        pairs = []
                    if filled or line:
                        actionText = '2D plot'
                        script += "pairs = %s\n" % pairs
                        logging.debug("2D plot with pairs = %s" % str(pairs))
                        self.plotter.plots_2d(roots, param_pairs=pairs, filled=filled, shaded=shaded)
                        self.updatePlot()
                        script += "g.plots_2d(roots, param_pairs=pairs, filled=%s, shaded=%s)\n" % (
                            str(filled), str(shaded))
                    elif color:
                        # 3D plot
                        sets = [list(pair) + [color_param] for pair in pairs]
                        logging.debug("3D plot with sets = %s" % str(sets))
                        actionText = '3D plot'
                        triplets = ["['%s', '%s', '%s']" % tuple(trip) for trip in sets]
                        if len(sets) == 1:
                            script += "g.plot_3d(roots, %s)\n" % triplets[0]
                            self.plotter.settings.scatter_size = 6
                            self.plotter.make_figure(1, ystretch=0.75)
                            self.plotter.plot_3d(roots, sets[0])
                        else:
                            script += "sets = [" + ",".join(triplets) + "]\n"
                            script += "g.plots_3d(roots, sets)\n"
                            self.plotter.plots_3d(roots, sets)
                        self.updatePlot()
            else:
                text = ""
                text += "Wrong parameters selection. Specify parameters such as:\n"
                text += "\n"
                text += "Triangle plot: Click on 'Triangle plot' and select more than 1 x parameters\n"
                text += "\n"
                text += "1D plot: Select x parameter(s)\n"
                text += "\n"
                text += "2D plot: Select x parameter(s), y parameter(s) and select 'Filled' or 'Line'\n"
                text += "\n"
                text += "3D plot: Select x parameter, y parameter and 'Color by' parameter\n"
                text += "\n"
                QMessageBox.warning(self, "Plot usage", text)
                return

            script += "g.export()\n"
            self.script = script
        except Exception as e:
            self.errorReport(e, caption=actionText)
        finally:
            self.showMessage()

    def updatePlot(self):
        if self.plotter.fig is None:
            self.canvas = None
        else:
            i = 0
            while True:
                item = self.plotWidget.layout().takeAt(i)
                if item is None: break
                del item
            if hasattr(self, "canvas"): del self.canvas
            if hasattr(self, "toolbar"): del self.toolbar
            self.canvas = FigureCanvas(self.plotter.fig)
            if sys.platform != "darwin":
                # for some reason the toolbar crashes out on a Mac; just don't show it
                self.toolbar = NavigationToolbar(self.canvas, self)
                self.plotWidget.layout().addWidget(self.toolbar)
            self.plotWidget.layout().addWidget(self.canvas)
            self.plotWidget.layout()
            self.canvas.draw()
            self.plotWidget.show()

    # Edit script

    def tabChanged(self, index):
        """
        Update script text editor when entering 'gui' tab.
        """

        # Enable menu options for edition only
        self.reLoadAct.setEnabled(index == 0)
        self.dataMenu.setEnabled(index == 0)
        self.optionMenu.setEnabled(index == 0)

        if index == 1 and self.script:
            self.script_edit = self.textWidget.toPlainText()
            if self.script_edit and self.script_edit != self.script:
                reply = QMessageBox.question(
                    self, "Overwrite script",
                    "Script is not empty. Overwrite current script?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No: return

            self.script_edit = self.script
            self.textWidget.setPlainText(self.script_edit)

    def openScript(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose a file name", '.', "Python (*.py)")
        if not filename: return
        filename = str(filename)
        logging.debug("Open file %s" % filename)
        with open(filename, 'r') as f:
            self.script_edit = f.read()
        self.textWidget.setPlainText(self.script_edit)

    def clearScript(self):
        self.textWidget.clear()
        self.script_edit = ''

    def plotData2(self):
        """
        Slot function called when pushButtonPlot2 is pressed.
        """
        self.script_edit = self.textWidget.toPlainText()
        oldset = plots.defaultSettings
        oldrc = matplotlib.rcParams.copy()
        plots.defaultSettings = plots.GetDistPlotSettings()
        matplotlib.rcParams.clear()
        matplotlib.rcParams.update(self.orig_rc)
        self.showMessage("Rendering plot....")
        try:
            script_exec = self.script_edit
            if "g.export()" in script_exec:
                # Comment line which produces export to PDF
                script_exec = script_exec.replace("g.export", "#g.export")

            globaldic = {}
            localdic = {}
            exec (script_exec, globaldic, localdic)

            for v in six.itervalues(localdic):
                if isinstance(v, plots.GetDistPlotter):
                    self.updateScriptPreview(v)
                    break
        except Exception as e:
            self.errorReport(e, caption="Plot script")
        finally:
            plots.defaultSettings = oldset
            matplotlib.rcParams.clear()
            matplotlib.rcParams.update(oldrc)
            self.showMessage()

    def updateScriptPreview(self, plotter):
        if plotter.fig is None:
            return

        self.plotter_script = plotter

        i = 0
        while True:
            item = self.plotWidget2.layout().takeAt(i)
            if item is None: break
            if hasattr(item, "widget"):
                child = item.widget()
                del child
            del item

        # Save in PNG format, and display it in a QLabel
        buf = BytesIO()

        plotter.fig.savefig(
            buf,
            format='png',
            edgecolor='w',
            facecolor='w',
            dpi=100,
            bbox_extra_artists=plotter.extra_artists,
            bbox_inches='tight')
        buf.seek(0)

        image = QImage.fromData(buf.getvalue())

        pixmap = QPixmap.fromImage(image)
        label = QLabel(self.scrollArea)
        label.setPixmap(pixmap)

        self.scrollArea = QScrollArea(self.plotWidget2)
        self.scrollArea.setWidget(label)
        self.scrollArea.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        # self.scrollArea.setStyleSheet("background-color: rgb(255,255,255)")

        self.plotWidget2.layout().addWidget(self.scrollArea)
        self.plotWidget2.layout()
        self.plotWidget2.show()


# ==============================================================================


class DialogTextOutput(QDialog):
    def __init__(self, parent, text=None):
        QDialog.__init__(self, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.textfont = QFont("Monospace")
        self.textfont.setStyleHint(QFont.TypeWriter)
        if text:
            self.text = self.getTextBox(text)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def getTextBox(self, text):
        box = QTextEdit(self)
        box.setWordWrapMode(QTextOption.NoWrap)
        box.setFont(self.textfont)
        box.setText(text)
        return box


# ==============================================================================

class DialogLikeStats(DialogTextOutput):
    def __init__(self, parent, stats, root):
        DialogTextOutput.__init__(self, parent, stats.likeSummary())

        self.label = QLabel(self)
        self.table = QTableWidget(self)
        self.table.verticalHeader().hide()

        layout = QGridLayout()
        layout.addWidget(self.table, 1, 0)
        self.setLayout(layout)
        self.setWindowTitle(self.tr('Sample likelihood constraints: ' + root))

        self.text.setMaximumHeight(80)
        layout.addWidget(self.text, 0, 0)

        if stats:
            headers = stats.headerLine().strip().split() + ['label']
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels([h.replace('_', ' ') for h in headers])
            self.table.verticalHeader().setVisible(False)
            self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.setRowCount(stats.numParams())
            for irow, par in enumerate(stats.names):
                vals = [par.name, "%5g" % par.bestfit_sample]
                for lim in [0, 1]:
                    vals += ["%5g" % par.ND_limit_bot[lim], "%5g" % par.ND_limit_top[lim]]
                vals += [par.label]
                for icol, value in enumerate(vals):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                    if icol == 0 and not par.isDerived:
                        font = QFont()
                        font.setBold(True)
                        item.setFont(font)
                    self.table.setItem(irow, icol, item)

            self.table.resizeRowsToContents()
            self.table.resizeColumnsToContents()

            w = self.table.horizontalHeader().length() + 40
            h = self.table.verticalHeader().length() + 40
            h = min(QApplication.desktop().screenGeometry().height() * 4 / 5, h)
            self.resize(w, h)


# ==============================================================================

class DialogMargeStats(QDialog):
    def __init__(self, parent=None, stats="", root=''):
        QDialog.__init__(self, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

        self.label = QLabel(self)
        self.table = QTableWidget(self)
        self.table.verticalHeader().hide()

        layout = QGridLayout()
        layout.addWidget(self.table, 0, 0)
        self.setLayout(layout)

        self.setWindowTitle(self.tr('Marginalized constraints: ' + root + ".margestats"))
        self.setAttribute(Qt.WA_DeleteOnClose)

        if stats:

            headers = stats.headerLine(inc_limits=True)[0].split() + ['label']
            self.table.setColumnCount(len(headers))
            self.table.setHorizontalHeaderLabels([h.replace('_', ' ') for h in headers])
            self.table.verticalHeader().setVisible(False)
            # self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
            self.table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.table.setRowCount(stats.numParams())
            for irow, par in enumerate(stats.names):
                vals = [par.name, "%5g" % par.mean, "%5g" % par.err]
                for lim in par.limits:
                    vals += ["%5g" % lim.lower, "%5g" % lim.upper, lim.limitTag()]
                vals += [par.label]
                for icol, value in enumerate(vals):
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                    if icol == 0 and not par.isDerived:
                        font = QFont()
                        font.setBold(True)
                        item.setFont(font)
                    self.table.setItem(irow, icol, item)

            self.table.resizeRowsToContents()
            self.table.resizeColumnsToContents()

            w = self.table.horizontalHeader().length() + 40
            h = self.table.verticalHeader().length() + 40
            h = min(QApplication.desktop().screenGeometry().height() * 4 / 5, h)
            self.resize(w, h)


# ==============================================================================

class DialogConvergeStats(DialogTextOutput):
    def __init__(self, parent, stats, summary, root):
        DialogTextOutput.__init__(self, parent, stats)
        layout = QGridLayout()
        layout.addWidget(self.text, 1, 0)
        if summary:
            self.text2 = self.getTextBox(summary)
            self.text2.setMaximumHeight(100)
            layout.addWidget(self.text2, 0, 0)

        self.setLayout(layout)
        self.setWindowTitle(self.tr('Convergence stats: ' + root))
        h = min(QApplication.desktop().screenGeometry().height() * 4 / 5, 1200)
        self.resize(700, h)


# ==============================================================================

class DialogPCA(DialogTextOutput):
    def __init__(self, parent, PCA_text, root):
        DialogTextOutput.__init__(self, parent, PCA_text)
        layout = QGridLayout()
        layout.addWidget(self.text, 0, 0)
        self.setLayout(layout)
        self.setWindowTitle(self.tr('PCA constraints for: ' + root))
        h = min(QApplication.desktop().screenGeometry().height() * 4 / 5, 800)
        self.resize(500, h)


# ==============================================================================

class DialogParamTables(DialogTextOutput):
    def __init__(self, parent, tables, root):
        DialogTextOutput.__init__(self, parent)
        self.tables = tables
        self.root = root
        self.tabWidget = QTabWidget(self)
        self.tabWidget.setTabPosition(QTabWidget.North)
        self.connect(self.tabWidget, SIGNAL("currentChanged(int)"), self.tabChanged)
        layout = QGridLayout()
        layout.addWidget(self.tabWidget, 1, 0, 1, 2)
        self.copyButton = QPushButton(QIcon(""), "Copy latex")
        self.saveButton = QPushButton(QIcon(""), "Save latex")
        self.connect(self.copyButton, SIGNAL("clicked()"), self.copyLatex)
        self.connect(self.saveButton, SIGNAL("clicked()"), self.saveLatex)

        layout.addWidget(self.copyButton, 2, 0)
        layout.addWidget(self.saveButton, 2, 1)

        self.setLayout(layout)
        self.tabs = [QWidget(self) for _ in range(len(tables))]
        self.generated = [None] * len(tables)
        for table, tab in zip(tables, self.tabs):
            self.tabWidget.addTab(tab, table.results[0].limitText(table.limit) + '%')
        self.tabChanged(0)

        self.setWindowTitle(self.tr('Parameter tables for: ' + root))
        # h = min(QApplication.desktop().screenGeometry().height() * 4 / 5, 800)
        # self.resize(500, h)
        self.adjustSize()

    def tabChanged(self, index):
        if not self.generated[index]:
            viewWidget = QWidget(self.tabs[index])
            buf = self.tables[index].tablePNG(bytesIO=True)
            pixmap = QPixmap.fromImage(QImage.fromData(buf.getvalue()))
            label = QLabel(viewWidget)
            label.setPixmap(pixmap)
            layout = QGridLayout()
            layout.addWidget(label, 1, 0)
            self.tabs[index].setLayout(layout)
            self.generated[index] = True

    def copyLatex(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.tables[self.tabWidget.currentIndex()].tableTex())

    def saveLatex(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Choose a file name", '.', "Latex (*.tex)")
        if not filename: return
        self.tables[self.tabWidget.currentIndex()].write(str(filename))


# ==============================================================================

class DialogSettings(QDialog):
    def __init__(self, parent, ini, items=None, title='Analysis Settings', width=300, update=None):
        QDialog.__init__(self, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)

        self.update = update
        self.table = QTableWidget(self)
        self.table.verticalHeader().hide()

        layout = QGridLayout()
        layout.addWidget(self.table, 0, 0)
        button = QPushButton("Update")
        self.connect(button, SIGNAL("clicked()"), self.doUpdate)

        layout.addWidget(button, 1, 0)
        self.setLayout(layout)

        self.setWindowTitle(self.tr(title))

        headers = ['parameter', 'value']
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)

        if items is None:
            names = IniFile(getdist.default_getdist_settings)
            items = []
            self.ini = ini
            for key in ini.readOrder:
                if key in names.params:
                    items.append(key)

            for key in ini.params:
                if not key in items and key in names.params:
                    items.append(key)
        else:
            names = ini

        nblank = 1
        self.rows = len(items) + nblank
        self.table.setRowCount(self.rows)
        for irow, key in enumerate(items):
            item = QTableWidgetItem(str(key))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(irow, 0, item)
            item = QTableWidgetItem(ini.string(key))
            hint = names.comments.get(key, None)
            if hint: item.setToolTip("\n".join(hint))
            self.table.setItem(irow, 1, item)
        for i in range(nblank):
            item = QTableWidgetItem(str(""))
            irow = len(items) + i
            self.table.setItem(irow, 0, item)
            item = QTableWidgetItem(str(""))
            self.table.setItem(irow, 1, item)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)

        h = self.table.verticalHeader().length() + 40
        h = min(QApplication.desktop().screenGeometry().height() * 4 / 5, h)
        self.resize(width, h)

    def getDict(self):
        vals = {}
        for row in range(self.rows):
            key = self.table.item(row, 0).text().strip()
            if key:
                vals[key] = self.table.item(row, 1).text().strip()
        return vals

    def doUpdate(self):
        self.ini.params.update(self.getDict())
        self.parent().settingsChanged()


class DialogPlotSettings(DialogSettings):
    def doUpdate(self):
        self.parent().plotSettingsChanged(self.getDict())


class DialogConfigSettings(DialogSettings):
    def doUpdate(self):
        self.parent().configSettingsChanged(self.getDict())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow(app)
    mainWin.show()
    sys.exit(app.exec_())

# ==============================================================================

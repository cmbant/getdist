#!/usr/bin/env python

import os
import copy
import logging
import matplotlib
import matplotlib.colors
import numpy as np
import scipy
import sys
import signal
import warnings
from io import BytesIO

try:
    from PySide2 import QtCore
except ImportError as _e:
    if 'DLL load failed' in str(_e):
        print('DLL load failed attempting to load PySide2: problem with your python configuration')
    else:
        print(_e)
        print("Can't import PySide2 modules, you need to install Pyside2")
    if not os.path.exists(os.path.join(sys.prefix, 'conda-meta')):
        print('Using Anaconda is probably the most reliable method')
    print("E.g. make and use a new environment using conda-forge")
    print('conda create -n py37forge -c conda-forge python=3.7 scipy pandas matplotlib PySide2')

    sys.exit(-1)

matplotlib.use('Qt5Agg')

import getdist
from getdist import plots, IniFile
from getdist.chain_grid import ChainDirGrid, file_root_to_root, get_chain_root_files, load_supported_grid
from getdist.mcsamples import SettingError, ParamError

from getdist.gui.SyntaxHighlight import PythonHighlighter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as QNavigationToolbar

import PySide2
from PySide2.QtGui import QIcon, QKeySequence, QFont, QTextOption, QPixmap, QImage
from PySide2.QtCore import Qt, SIGNAL, QSize, QSettings, QCoreApplication
from PySide2.QtWidgets import QListWidget, QMainWindow, QDialog, QApplication, QAbstractItemView, QAction, \
    QTabWidget, QWidget, QComboBox, QPushButton, QShortcut, QCheckBox, QRadioButton, QGridLayout, QVBoxLayout, \
    QSplitter, QHBoxLayout, QToolBar, QPlainTextEdit, QScrollArea, QFileDialog, QMessageBox, QTableWidgetItem, \
    QLabel, QTableWidget, QListWidgetItem, QTextEdit

os.environ['QT_API'] = 'pyside2'

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)  # DPI support
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

try:
    # If cosmomc is configured
    from paramgrid import batchjob
except ImportError:
    batchjob = None


# noinspection PyCallByClass
class NavigationToolbar(QNavigationToolbar):
    def sizeHint(self):
        return QToolBar.sizeHint(self)


class GuiSelectionError(Exception):
    pass


class QStatusLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__(level=logging.WARNING)
        self.widget = parent

    def emit(self, record):
        msg = self.format(record)
        self.widget.showMessage(msg, color='red')

    def write(self, m):
        pass


class RootListWidget(QListWidget):
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

    def dropEvent(self, event):
        super().dropEvent(event)
        self.owner._updateParameters()


# noinspection PyCallByClass,PyArgumentList
class MainWindow(QMainWindow):
    def __init__(self, app, ini=None, base_dir=None):
        """
        Initialize of GUI components.
        """
        super().__init__()

        self.setWindowTitle("GetDist GUI")
        self.setWindowIcon(self._icon('Icon', False))

        if base_dir is None and batchjob:
            base_dir = batchjob.getCodeRootPath()
        if base_dir:
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
        if os.name == 'nt':
            # This is needed to display the app icon on the taskbar on Windows 7+
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('GetDist.Gui.1.0.0')

        # Allow to shutdown the GUI with Ctrl+C
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Path for .ini file
        self.default_settings = IniFile(getdist.default_getdist_settings)
        self.iniFile = ini
        if ini:
            self.base_settings = IniFile(ini)
        else:
            self.base_settings = self.default_settings
        self.current_settings = copy.deepcopy(self.base_settings)

        # Path of root directory
        self.rootdirname = None
        self.plotter = None
        self.root_infos = {}

        self._resetGridData()
        self._resetPlotData()

        self.log_handler = QStatusLogger(self)
        logging.getLogger().addHandler(self.log_handler)
        self._last_color = None

        dirs = self.getSettings().value('directoryList')
        last_dir = self.getSettings().value('lastSearchDirectory')

        if dirs is None and last_dir:
            dirs = [last_dir]
        elif isinstance(dirs, str):
            dirs = [dirs]  # QSettings doesn't save single item lists reliably
        if dirs is not None:
            dirs = [x for x in dirs if os.path.exists(x)]
            if last_dir is not None and last_dir not in dirs and os.path.exists(last_dir):
                dirs.insert(0, last_dir)
            self.listDirectories.addItems(dirs)
            if last_dir is not None and os.path.exists(last_dir):
                self.listDirectories.setCurrentIndex(dirs.index(last_dir))
                self.openDirectory(last_dir)
            else:
                self.listDirectories.setCurrentIndex(-1)

    def createActions(self):
        """
        Create Qt actions used in GUI.
        """
        self.openChainsAct = QAction("&Open folder...", self,
                                     shortcut="Ctrl+O",
                                     statusTip="Open a directory containing chain (sample files) to use",
                                     triggered=self.selectRootDirName)

        self.exportAct = QAction("&Export as PDF/Image...", self,
                                 statusTip="Export image as PDF, PNG, JPG",
                                 triggered=self.export)
        self.exportAct.setEnabled(False)

        self.scriptAct = QAction("Save script...", self,
                                 shortcut="Ctrl+S",
                                 statusTip="Export commands to script",
                                 triggered=self.saveScript)

        self.reLoadAct = QAction("Re-load files", self,
                                 statusTip="Re-scan directory",
                                 triggered=self.reLoad)

        self.exitAct = QAction("Exit", self,
                               shortcut="Ctrl+Q",
                               statusTip="Exit application",
                               triggered=self.close)

        self.statsAct = QAction("Marge Stats", self,
                                shortcut="",
                                statusTip="Show Marge Stats",
                                triggered=self.showMargeStats)

        self.likeStatsAct = QAction("Like Stats", self,
                                    shortcut="",
                                    statusTip="Show Likelihood (N-D) Stats",
                                    triggered=self.showLikeStats)

        self.convergeAct = QAction("Converge Stats", self,
                                   statusTip="Show Convergence Stats",
                                   triggered=self.showConvergeStats)

        self.PCAAct = QAction("Parameter PCA", self,
                              statusTip="Do PCA of selected parameters",
                              triggered=self.showPCA)

        self.paramTableAct = QAction("Parameter table (latex)", self,
                                     statusTip="View parameter table",
                                     triggered=self.showParamTable)

        self.optionsAct = QAction("Analysis settings", self,
                                  statusTip="Show settings for getdist and plot densities",
                                  triggered=self.showSettings)

        self.plotOptionsAct = QAction("Plot settings", self,
                                      statusTip="Show settings for plot display",
                                      triggered=self.showPlotSettings)

        self.resetPlotOptionsAct = QAction("Reset plot settings", self,
                                           statusTip="Reset settings for plot display",
                                           triggered=self.resetPlotSettings)

        self.resetAnalysisSettingsAct = QAction("Reset analysis settings", self,
                                                statusTip="Reset settings for sample analysis",
                                                triggered=self.resetAnalysisSettings)

        self.configOptionsAct = QAction("Plot style module", self,
                                        statusTip="Configure plot module that sets default settings",
                                        triggered=self.showConfigSettings)

        self.helpAct = QAction("GetDist documentation", self,
                               statusTip="Show getdist readthedocs",
                               triggered=self.openHelpDocs)

        self.githubAct = QAction("GetDist on GitHub", self,
                                 statusTip="Show getdist source",
                                 triggered=self.openGitHub)

        self.planckAct = QAction("Download Planck chains", self,
                                 statusTip="Download sample chain files",
                                 triggered=self.openPlanck)

        self.aboutAct = QAction("About", self,
                                statusTip="Show About box",
                                triggered=self.about)

    def createMenus(self):
        """
        Create Qt menus.
        """
        menu = self.menuBar()
        self.menu = menu
        self.fileMenu = menu.addMenu("&File")
        self.fileMenu.addAction(self.openChainsAct)
        self.fileMenu.addAction(self.exportAct)
        self.fileMenu.addAction(self.scriptAct)
        self.separatorAct = self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.reLoadAct)
        self.fileMenu.addAction(self.exitAct)

        menu.addSeparator()
        self.dataMenu = menu.addMenu("&Data")
        self.dataMenu.addAction(self.statsAct)
        self.dataMenu.addAction(self.likeStatsAct)
        self.dataMenu.addAction(self.convergeAct)
        self.dataMenu.addSeparator()
        self.dataMenu.addAction(self.PCAAct)
        self.dataMenu.addAction(self.paramTableAct)

        menu.addSeparator()
        self.optionMenu = menu.addMenu("&Options")
        self.optionMenu.addAction(self.optionsAct)
        self.optionMenu.addAction(self.plotOptionsAct)
        self.optionMenu.addAction(self.configOptionsAct)
        self.optionMenu.addSeparator()
        self.optionMenu.addAction(self.resetAnalysisSettingsAct)
        self.optionMenu.addAction(self.resetPlotOptionsAct)

        menu.addSeparator()

        self.helpMenu = menu.addMenu("&Help")
        self.helpMenu.addAction(self.helpAct)
        self.helpMenu.addAction(self.githubAct)
        self.helpMenu.addSeparator()
        self.helpMenu.addAction(self.planckAct)
        self.helpMenu.addSeparator()
        self.helpMenu.addAction(self.aboutAct)

    def createStatusBar(self):
        """
        Create Qt status bar.
        """
        self.statusBar().setStyleSheet("height:1em")
        self.statusBar().showMessage("Ready", 2000)

    def showMessage(self, msg='', color=None):
        if not msg and not color and self._last_color:
            return
        self._last_color = color
        bar = self.statusBar()
        bar.showMessage(msg)
        bar.setStyleSheet("color: %s" % (color or "black"))
        if msg:
            self.statusBar().repaint()
            QCoreApplication.processEvents()

    def _image_file(self, name):
        return os.path.join(os.path.dirname(__file__), 'images', name)

    def _icon(self, name, large=True):
        if large:
            name = name + '_large'
        pm = QPixmap(self._image_file('%s.png' % name))
        if hasattr(pm, 'setDevicePixelRatio'):
            pm.setDevicePixelRatio(self.devicePixelRatio())
        return QIcon(pm)

    def _createWidgets(self):
        """
        Create widgets.
        """
        if sys.platform in ["darwin", "Windows"]:
            self.setStyleSheet("* {font-size:9pt} QComboBox,QPushButton {height:1.3em}")
        else:
            self.setStyleSheet("* {font-size:12px} QComboBox,QPushButton {height:1.3em}")

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
        self.connect(self.listDirectories, SIGNAL("activated(const QString&)"), self.openDirectory)

        self.pushButtonSelect = QPushButton(self._icon("open"), "", self.selectWidget)
        self.pushButtonSelect.setMaximumWidth(30)
        self.pushButtonSelect.setToolTip("Open chain file root directory")
        self.connect(self.pushButtonSelect, SIGNAL("clicked()"), self.selectRootDirName)

        self.listRoots = RootListWidget(self.selectWidget, self)
        self.connect(self.listRoots, SIGNAL("itemChanged(QListWidgetItem *)"), self.updateListRoots)
        self.connect(self.listRoots, SIGNAL("itemSelectionChanged()"), self.selListRoots)

        self.pushButtonRemove = QPushButton(self._icon('remove'), "", self.selectWidget)
        self.pushButtonRemove.setToolTip("Remove a chain root")
        self.pushButtonRemove.setEnabled(False)
        self.pushButtonRemove.setMaximumWidth(30)
        self.connect(self.pushButtonRemove, SIGNAL("clicked()"), self.removeRoot)

        self.comboBoxParamTag = QComboBox(self.selectWidget)
        self.comboBoxParamTag.clear()
        self.connect(self.comboBoxParamTag, SIGNAL("activated(const QString&)"), self.setParamTag)

        self.comboBoxDataTag = QComboBox(self.selectWidget)
        self.comboBoxDataTag.clear()
        self.connect(self.comboBoxDataTag, SIGNAL("activated(const QString&)"), self.setDataTag)

        self.comboBoxRootname = QComboBox(self.selectWidget)
        self.comboBoxRootname.clear()
        self.connect(self.comboBoxRootname, SIGNAL("activated(const QString&)"), self.setRootname)

        self.listParametersX = QListWidget(self.selectWidget)
        self.listParametersX.clear()
        self.connect(self.listParametersX, SIGNAL("itemChanged(QListWidgetItem *)"), self.itemCheckChange)

        self.listParametersY = QListWidget(self.selectWidget)
        self.listParametersY.clear()
        self.connect(self.listParametersY, SIGNAL("itemChanged(QListWidgetItem *)"), self.itemCheckChange)

        self.selectAllX = QCheckBox("Select All", self.selectWidget)
        self.selectAllX.setCheckState(Qt.Unchecked)
        self.connect(self.selectAllX, SIGNAL("clicked()"), self.statusSelectAllX)

        self.selectAllY = QCheckBox("Select All", self.selectWidget)
        self.selectAllY.setCheckState(Qt.Unchecked)
        self.connect(self.selectAllY, SIGNAL("clicked()"), self.statusSelectAllY)

        self.toggleFilled = QRadioButton("Filled")
        self.toggleLine = QRadioButton("Line")
        self.connect(self.toggleLine, SIGNAL("toggled(bool)"), self.statusPlotType)

        self.checkShade = QCheckBox("Shaded", self.selectWidget)
        self.checkShade.setEnabled(False)

        self.checkInsideLegend = QCheckBox("Axis legend", self.selectWidget)
        self.checkInsideLegend.setCheckState(Qt.Unchecked)
        self.checkInsideLegend.setVisible(False)

        self.toggleColor = QRadioButton("Color by:")
        self.connect(self.toggleColor, SIGNAL("toggled(bool)"), self.statusPlotType)

        self.comboBoxColor = QComboBox(self)
        self.comboBoxColor.clear()
        self.comboBoxColor.setEnabled(False)

        self.toggleFilled.setChecked(True)

        self.trianglePlot = QCheckBox("Triangle Plot", self.selectWidget)
        self.trianglePlot.setCheckState(Qt.Unchecked)
        self.connect(self.trianglePlot, SIGNAL("toggled(bool)"), self.statusTriangle)

        self.pushButtonPlot = QPushButton("Make plot", self.selectWidget)
        self.connect(self.pushButtonPlot, SIGNAL("clicked()"), self.plotData)

        def h_stack(*items):
            widget = QWidget(self.selectWidget)
            lay = QHBoxLayout(widget)
            lay.setContentsMargins(0, 0, 0, 0)
            for item in items:
                lay.addWidget(item)
            widget.setLayout(lay)
            lay.setAlignment(items[1], Qt.AlignTop)
            return widget

        # Graphic Layout
        leftLayout = QGridLayout()
        leftLayout.setSpacing(5)
        leftLayout.addWidget(h_stack(self.listDirectories, self.pushButtonSelect), 0, 0, 1, 4)

        leftLayout.addWidget(self.comboBoxRootname, 1, 0, 1, 4)
        leftLayout.addWidget(self.comboBoxParamTag, 1, 0, 1, 4)
        leftLayout.addWidget(self.comboBoxDataTag, 2, 0, 1, 4)
        leftLayout.addWidget(h_stack(self.listRoots, self.pushButtonRemove), 3, 0, 2, 4)

        leftLayout.addWidget(self.selectAllX, 5, 0, 1, 2)
        leftLayout.addWidget(self.selectAllY, 5, 2, 1, 2)
        leftLayout.addWidget(self.listParametersX, 6, 0, 5, 2)
        leftLayout.addWidget(self.listParametersY, 6, 2, 1, 2)
        leftLayout.addWidget(self.toggleFilled, 7, 2, 1, 1)
        leftLayout.addWidget(self.checkInsideLegend, 7, 3, 1, 1)
        leftLayout.addWidget(self.toggleLine, 8, 2, 1, 1)
        leftLayout.addWidget(self.checkShade, 8, 3, 1, 1)
        leftLayout.addWidget(self.toggleColor, 9, 2, 1, 1)
        leftLayout.addWidget(self.comboBoxColor, 9, 3, 1, 1)
        leftLayout.addWidget(self.trianglePlot, 10, 2, 1, 2)

        leftLayout.addWidget(self.pushButtonPlot, 12, 0, 1, 4)

        self.selectWidget.setLayout(leftLayout)

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
        self.splitter = splitter
        hbox = QHBoxLayout()
        hbox.addWidget(splitter)
        hbox.setContentsMargins(0, 0, 0, 0)
        self.firstWidget.setLayout(hbox)

        # Second tab: Script
        self.secondWidget = QWidget(self)
        self.tabWidget.addTab(self.secondWidget, "Script Preview")

        self.editWidget = QWidget(self.secondWidget)
        self.script_edit = ""
        self.plotter_script = None

        self.toolBar = QToolBar()
        self.toolBar.setIconSize(QSize(22, 22))

        openAct = QAction(self._icon("open"),
                          "open script", self.toolBar,
                          statusTip="Open script",
                          triggered=self.openScript)
        saveAct = QAction(self._icon("save"),
                          "Save script", self.toolBar,
                          statusTip="Save script",
                          triggered=self.saveScript)
        clearAct = QAction(self._icon("delete"),
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
        PythonHighlighter(self.textWidget.document())

        self.pushButtonPlot2 = QPushButton("Make plot", self.editWidget)
        self.pushButtonPlot2.setToolTip("Ctrl+Return")
        self.connect(self.pushButtonPlot2, SIGNAL("clicked()"), self.plotData2)
        shortcut = QShortcut(QKeySequence(self.tr("Ctrl+Return")), self)
        self.connect(shortcut, SIGNAL("activated()"), self.plotData2)

        layoutEdit = QVBoxLayout()
        layoutEdit.addWidget(self.toolBar)
        layoutEdit.addWidget(self.textWidget)
        layoutEdit.addWidget(self.pushButtonPlot2)
        layoutEdit.setContentsMargins(0, 0, 0, 0)
        layoutEdit.setSpacing(0)
        self.editWidget.setLayout(layoutEdit)

        self.plotWidget2 = QWidget(self.secondWidget)
        layout2 = QVBoxLayout(self.plotWidget2)
        layout2.setContentsMargins(0, 0, 0, 0)
        self.plotWidget2.setLayout(layout2)
        self.scrollArea = QScrollArea(self.plotWidget2)
        layout2.addWidget(self.scrollArea)

        splitter2 = QSplitter(self.secondWidget)
        splitter2.addWidget(self.editWidget)
        splitter2.addWidget(self.plotWidget2)
        w = self.width()
        splitter2.setSizes([w / 2., w / 2.])

        hbox2 = QHBoxLayout()
        hbox2.addWidget(splitter2)
        hbox2.setContentsMargins(0, 0, 0, 0)

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
        screen = QApplication.desktop().screenGeometry()
        h = min(screen.height() * 4 / 5., 700)
        size = QSize(min(screen.width() * 4 / 5., 900), h)
        pos = settings.value("pos", None)
        savesize = settings.value("size", size)
        if savesize.width() > screen.width():
            savesize.setWidth(size.width())
        if savesize.height() > screen.height():
            savesize.setHeight(size.height())
        self.resize(savesize)
        if pos is None or pos.x() + savesize.width() > screen.width() or pos.y() + savesize.height() > screen.height():
            self.move(screen.center() - self.rect().center())
        else:
            self.move(pos)
        self.plot_module = settings.value("plot_module", self.plot_module)
        self.script_plot_module = settings.value("script_plot_module", self.script_plot_module)
        splitter_settings = settings.value("splitter_settings")
        if splitter_settings:
            self.splitter.restoreState(splitter_settings)

    def writeSettings(self):
        settings = self.getSettings()
        settings.setValue("pos", self.pos())
        settings.setValue("size", self.size())
        settings.setValue('plot_module', self.plot_module)
        settings.setValue('script_plot_module', self.script_plot_module)
        splitter_settings = self.splitter.saveState()
        if splitter_settings:
            settings.setValue("splitter_settings", splitter_settings)

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
            if not filename:
                return
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

        filename, _ = QFileDialog.getSaveFileName(self, "Choose a file name", '.', "Python (*.py)")
        if not filename:
            return
        filename = str(filename)
        logging.debug("Export script to %s" % filename)
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(script)

    def reLoad(self):
        adir = self.getSettings().value('lastSearchDirectory')
        if adir is not None:
            if batchjob:
                batchjob.resetGrid(adir)
            self.openDirectory(adir)
        if self.plotter:
            self.plotter.sample_analyser.reset(self.current_settings)

    def getRootname(self):
        rootname = None
        item = self.listRoots.currentItem()
        if not item and self.listRoots.count():
            item = self.listRoots.item(0)
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
        if rootname is None:
            return
        try:
            self.showMessage("Calculating convergence stats....")
            QApplication.setOverrideCursor(Qt.WaitCursor)
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
            QApplication.restoreOverrideCursor()
            self.showMessage()

    def showPCA(self):
        """
        Callback for action 'Show PCA'.
        """
        rootname = self.getRootname()
        if rootname is None:
            return
        try:
            samples = self.getSamples(rootname)
            pars = self.getXParams()
            if len(pars) == 1:
                pars += self.getYParams()
            if len(pars) < 2:
                raise GuiSelectionError('Select two or more parameters first')
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
        if rootname is None:
            return
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
        if rootname is None:
            return
        try:
            samples = self.getSamples(rootname)
            pars = self.getXParams()
            ignore_unknown = False
            if len(pars) < 1:
                pars = self.getXParams(fulllist=True)
                # If no parameters selected, it shouldn't fail if some sample is missing
                # parameters present in the first one
                ignore_unknown = True
            if len(pars) < 1:
                raise GuiSelectionError('Select one or more parameters first')
            # Add renames to match parameter across samples
            renames = self.paramNames.getRenames(keep_empty=True)
            pars = [getattr(samples.paramNames.parWithName(
                p, error=not ignore_unknown, renames=renames), "name", None)
                for p in pars]
            while None in pars:
                pars.remove(None)
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
        if rootname is None:
            return
        samples = self.getSamples(rootname)
        stats = samples.getLikeStats()
        if stats is None:
            QMessageBox.warning(self, "Like stats", "Samples do not likelihoods")
            return
        dlg = DialogLikeStats(self, stats, rootname)
        dlg.show()

    def changed_settings(self):
        changed = {}
        for key, value in self.current_settings.params.items():
            if self.default_settings.params[key] != value:
                changed[key] = value
        return changed

    def showSettings(self):
        """
        Callback for action 'Show settings'
        """
        if not self.plotter:
            QMessageBox.warning(self, "Settings", "Open chains first ")
            return
        self.settingDlg = self.settingDlg or DialogSettings(self, self.current_settings)
        self.settingDlg.show()
        self.settingDlg.activateWindow()

    def settingsChanged(self):
        if self.plotter:
            self.plotter.sample_analyser.reset(self.current_settings, chain_settings_have_priority=False)
        if self.tabWidget.currentIndex() == 0:
            if self.plotter and self.plotter.fig:
                self.plotData()
        else:
            script = self.textWidget.toPlainText().split("\n")
            tag = 'analysis_settings ='
            changed_settings = self.changed_settings()
            newline = tag + (' %s' % changed_settings).replace(', ', ",\n" + " " * 21)
            last = 0
            for i, line in enumerate(script):
                if line.startswith(tag):
                    for j in range(i, len(script)):
                        if '}' in script[j]:
                            del script[i:j + 1]
                            break
                    script.insert(i, newline)
                    last = None
                    break
                elif not last and line.startswith('g='):
                    last = i
            if last is not None:
                script.insert(last, newline)
            for i, line in enumerate(script):
                if line.startswith('g=plots.') and 'analysis_settings' not in line:
                    script[i] = line.strip()[:-1] + ', analysis_settings=analysis_settings)'
                    break
            self.textWidget.setPlainText("\n".join(script))
            self.plotData2()

    def showPlotSettings(self):
        """
        Callback for action 'Show plot settings'
        """
        self.getPlotter()
        settings = self.default_plot_settings
        pars = []
        skips = ['param_names_for_labels', 'progress']
        comments = {}
        for line in settings.__doc__.split("\n"):
            if 'ivar' in line:
                items = line.split(':', 2)
                par = items[1].split('ivar ')[1]
                if par not in skips:
                    pars.append(par)
                    comments[par] = items[2].strip()
        pars.sort()
        ini = IniFile()
        for par in pars:
            ini.getAttr(settings, par, comment=[comments.get(par, None)])
            if isinstance(ini.params[par], matplotlib.colors.Colormap):
                ini.params[par] = ini.params[par].name
        ini.params.update(self.custom_plot_settings)
        self.plotSettingIni = ini

        self.plotSettingDlg = self.plotSettingDlg or DialogPlotSettings(self, ini, pars, title='Plot Settings',
                                                                        width=420)
        self.plotSettingDlg.show()
        self.plotSettingDlg.activateWindow()

    def resetPlotSettings(self):
        self.custom_plot_settings = {}
        if self.plotSettingDlg:
            self.plotSettingDlg.close()
            self.plotSettingDlg = None

    def resetAnalysisSettings(self):
        self.current_settings = copy.deepcopy(self.base_settings)
        if self.settingDlg:
            self.settingDlg.close()
            self.settingDlg = None

    def plotSettingsChanged(self, vals):
        deleted = []
        try:
            settings = self.default_plot_settings
            self.custom_plot_settings = {}
            for key, value in vals.items():
                current = getattr(settings, key)
                if str(current) != value and len(value):
                    if isinstance(current, str):
                        self.custom_plot_settings[key] = value
                    else:
                        try:
                            self.custom_plot_settings[key] = eval(value)
                        except:
                            import re
                            if current is None or re.match(r'^[\w]+$', value):
                                self.custom_plot_settings[key] = value
                            else:
                                raise
                else:
                    deleted.append(key)
                    self.custom_plot_settings.pop(key, None)
        except Exception as e:
            self.errorReport(e, caption="Plot settings")
        if self.tabWidget.currentIndex() == 0:
            self.plotData()
        else:
            # Try to update current plot script text
            script = self.textWidget.toPlainText().split("\n")
            if self.custom_plot_settings:
                for key, value in self.custom_plot_settings.items():
                    if isinstance(value, str):
                        value = '"' + value + '"'
                    script_set = 'g.settings.%s =' % key
                    script_line = '%s %s' % (script_set, value)
                    last = None
                    for i, line in enumerate(script):
                        if line.startswith(script_set):
                            script[i] = script_line
                            script_line = None
                            break
                        elif line.startswith('g.settings.') or last is None and line.startswith('g='):
                            last = i
                    if script_line and last:
                        script.insert(last + 1, script_line)
            for key in deleted:
                script_set = 'g.settings.%s =' % key
                for i, line in enumerate(script):
                    if line.startswith(script_set):
                        del script[i]
                        break

            self.textWidget.setPlainText("\n".join(script))
            self.plotData2()

    def showConfigSettings(self):
        """
        Callback for action 'Show config settings'
        """
        ini = IniFile()
        ini.params['plot_module'] = self.plot_module
        ini.params['script_plot_module'] = self.script_plot_module
        ini.comments['plot_module'] = [
            "stylw module used by the GUI (e.g. change to getdist.styles.planck, getdist.styles.tab10)"]
        ini.comments['script_plot_module'] = ["module used by saved plot scripts  (e.g. getdist.styles.planck)"]
        self.ConfigDlg = self.ConfigDlg or DialogConfigSettings(self, ini, list(ini.params.keys()), title='Plot Config')
        self.ConfigDlg.show()
        self.ConfigDlg.activateWindow()

    def configSettingsChanged(self, vals):
        scriptmod = vals.get('script_plot_module', self.script_plot_module)
        self.script = ''
        mod = vals.get('plot_module', self.plot_module)
        if mod != self.plot_module or scriptmod != self.script_plot_module:
            try:
                self._set_rc(self.orig_rc)
                __import__(mod)  # test for error
                logging.debug('Loaded module %s', mod)
                self.plot_module = mod
                self.script_plot_module = scriptmod
                if self.plotSettingDlg:
                    self.plotSettingDlg.close()
                    self.plotSettingDlg = None
                if self.plotter:
                    has_plot = self.plotter.fig
                    self.closePlots()
                    self.getPlotter(loadNew=True)
                    self.custom_plot_settings = {}
                    if has_plot:
                        self.plotData()
            except Exception as e:
                self.errorReport(e, "plot_module")

    def openHelpDocs(self):
        import webbrowser
        webbrowser.open("https://getdist.readthedocs.org/")

    def openGitHub(self):
        import webbrowser
        webbrowser.open("https://github.com/cmbant/getdist/")

    def openPlanck(self):
        import webbrowser
        webbrowser.open("http://pla.esac.esa.int/pla/#cosmology")

    def about(self):
        """
        Callback for action 'About'.
        """
        QMessageBox.about(self, "About GetDist GUI",
                          "GetDist GUI v " + getdist.__version__ +
                          "\nAntony Lewis (University of Sussex) and contributors" +
                          "\nhttps://github.com/cmbant/getdist/\n" +
                          "\nPython: " + sys.version +
                          "\nMatplotlib: " + matplotlib.__version__ +
                          "\nSciPy: " + scipy.__version__ +
                          "\nNumpy: " + np.__version__ +
                          "\nPySide2: " + PySide2.__version__ +
                          "\nQt (PySide): " + PySide2.QtCore.__version__ +
                          "\n\nPix ratio: %s; Logical dpi: %s, %s" % (
                              self.devicePixelRatio(), self.logicalDpiX(), self.logicalDpiY()) +
                          '\nUsing getdist at: %s' % os.path.dirname(getdist.__file__))

    def getDirectories(self):
        return [self.listDirectories.itemText(i) for i in range(self.listDirectories.count())]

    def saveDirectories(self):
        dirs = self.getDirectories()
        if self.rootdirname:
            dirs = [self.rootdirname] + [x for x in dirs if not x == self.rootdirname]
        if len(dirs) > 10:
            dirs = dirs[:10]
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
        if not last_dir:
            last_dir = os.getcwd()

        title = self.tr("Choose an existing chains grid or chains folder")
        dir_name = QFileDialog.getExistingDirectory(self, title, last_dir,
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        dir_name = os.path.abspath(str(dir_name))
        logging.debug("dirName: %s" % dir_name)
        if dir_name is None or dir_name == '':
            return  # No directory selected

        if self.openDirectory(dir_name, save=False):
            items = self.getDirectories()
            if dir_name in items:
                self.listDirectories.setCurrentIndex(items.index(dir_name))
            else:
                self.listDirectories.insertItem(0, dir_name)
                self.listDirectories.setCurrentIndex(0)
            self.saveDirectories()

    def openDirectory(self, dirName, save=True):
        try:
            batch = load_supported_grid(dirName)
            if batch:
                self.rootdirname = dirName
                self._readGridChains(batch)
                if save:
                    self.saveDirectories()
                return True

            self._resetGridData()

            root_list = get_chain_root_files(dirName)
            if not len(root_list):
                if self._readChainsSubdirectories(dirName):
                    if save:
                        self.saveDirectories()
                    return True

                QMessageBox.critical(self, "Open chains", "No chains or grid found in that directory")
                cur_dirs = self.getDirectories()
                if dirName in cur_dirs:
                    self.listDirectories.removeItem(cur_dirs.index(dirName))
                    self.saveDirectories()
                return False

            self.rootdirname = dirName

            self.getPlotter(chain_dir=dirName)

            self._updateComboBoxRootname(root_list)
            if save:
                self.saveDirectories()
        except Exception as e:
            self.errorReport(e, caption="Open chains", capture=True)
            return False
        return True

    def getPlotter(self, chain_dir=None, loadNew=False):
        try:
            if self.plotter is None or chain_dir or loadNew:
                module = __import__(self.plot_module, fromlist=['dummy'])
                if hasattr(module, "style_name"):
                    plots.set_active_style(module.style_name)
                if self.plotter and not loadNew:
                    samps = self.plotter.sample_analyser.mcsamples
                else:
                    samps = None
                # set path of grids, so that any custom grid settings get propagated
                chain_dirs = []
                if chain_dir:
                    chain_dirs.append(chain_dir)
                for root in self.root_infos:
                    info = self.root_infos[root]
                    if info.batch:
                        if info.batch not in chain_dirs:
                            chain_dirs.append(info.batch)

                self.plotter = plots.get_subplot_plotter(chain_dir=chain_dirs, analysis_settings=self.current_settings)
                if samps:
                    self.plotter.sample_analyser.mcsamples = samps
                self.default_plot_settings = copy.copy(self.plotter.settings)

        except Exception as e:
            self.errorReport(e, caption="Make plotter")
        return self.plotter

    def getSamples(self, root):
        return self.plotter.sample_analyser.add_root(self.root_infos[root])

    def _updateParameters(self):
        roots = self.checkedRootNames()
        if not len(roots):
            return
        # Get previous selection (with its renames) before we overwrite the list of tags
        old_selection = {"x": [], "y": []}
        for x_, getter in zip(old_selection, [self.getXParams, self.getYParams]):
            if not hasattr(self, "paramNames"):
                break
            old_selection[x_] = [
                [name] + list(self.paramNames.getRenames().get(name, []))
                for name in getter()]
        # Copy paramNames (we don't want to change the original info)
        self.paramNames = (
            lambda x: x.filteredCopy(x))(self.getSamples(roots[0]).paramNames)
        # Add renames from all roots
        for r in roots[1:]:
            self.paramNames.updateRenames(self.getSamples(r).getRenames())

        # Update old selection to new names
        def find_new_name(old_names):
            for old_name in old_names:
                new_name = self.paramNames.parWithName(old_name)
                if new_name:
                    return new_name.name
            return None

        for x_ in old_selection:
            old_selection[x_] = [
                find_new_name(p) for p in old_selection[x_]]
        # Create tags for list widget
        renames = self.paramNames.getRenames(keep_empty=True)
        renames_list_func = lambda x: (" (" + ", ".join(x) + ")") if x else ""
        self.paramNamesTags = {p + renames_list_func(r): p for p, r in renames.items()}
        self._updateListParameters(list(self.paramNamesTags), self.listParametersX)
        self._updateListParameters(list(self.paramNamesTags), self.listParametersY)
        # Update selection in both boxes (needs to be done after *both* boxes have been
        # updated because of some internal checks
        self._updateListParametersSelection(old_selection["x"], self.listParametersX)
        self._updateListParametersSelection(old_selection["y"], self.listParametersY)

        self._updateComboBoxColor(self.paramNames.list())

    def _resetPlotData(self):
        # Script
        self.script = ""

    def _resetGridData(self):
        # Grid chains parameters
        self.batch = None
        self.grid_paramtag_jobItems = {}
        self.paramTag = ""

    def _readChainsSubdirectories(self, path):
        self.batch = None
        for root, info in self.root_infos.items():
            if info.batch and os.path.normpath(info.batch.batchPath) == os.path.normpath(path):
                self.batch = info.batch

        self.batch = self.batch or ChainDirGrid(path)

        if self.batch.base_dir_names:
            self.rootdirname = path
            self.getPlotter(chain_dir=self.batch)
            self.comboBoxRootname.hide()
            self.listRoots.show()
            self.pushButtonRemove.show()
            self.comboBoxParamTag.clear()
            self.comboBoxParamTag.addItems(sorted(self.batch.base_dir_names))
            self.setParamTag(self.comboBoxParamTag.itemText(0))
            self.comboBoxParamTag.show()
            self.comboBoxDataTag.show()
            return True

        return False

    def _readGridChains(self, batch):
        """
        Setup of a grid of chain results.
        """
        # Reset data
        self._resetPlotData()
        self._resetGridData()
        logging.debug("Read grid chain in %s" % batch.batchPath)
        self.batch = batch
        items = dict()
        for jobItem in batch.items(True, True):
            if jobItem.chainExists():
                if jobItem.paramtag not in items:
                    items[jobItem.paramtag] = []
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
        baseRoots = [file_root_to_root(root) for root in listOfRoots]
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
                if hasattr(self.batch, 'resolve_root'):
                    path = self.batch.resolve_root(root).chainPath
                else:
                    path = self.batch.resolveRoot(root).chainPath
            else:
                path = self.rootdirname
            # new style, if the prefix is just a folder
            if root[-1] in (os.sep, "/"):
                path = os.sep.join(path.replace("/", os.sep).split(os.sep)[:-1])
            info = plots.RootInfo(root, path, self.batch)
            plotter.sample_analyser.add_root(info)

            self.root_infos[root] = info
            item.setCheckState(Qt.Checked)
            item.setText(root)
            self._updateParameters()
        except Exception as e:
            self.errorReport(e, capture=True)
            self.listRoots.takeItem(self.listRoots.count() - 1)
        finally:
            self.updating = False
            self.selListRoots()

    def setRootname(self, strParamName):
        """
        Slot function called on change of comboBoxRootname.
        """
        self.newRootItem(str(strParamName))

    def updateListRoots(self, item):
        if self.updating:
            return
        self._updateParameters()

    def selListRoots(self):
        self.pushButtonRemove.setEnabled(len(self.listRoots.selectedItems())
                                         or self.listRoots.count() == 1)

    def removeRoot(self):
        logging.debug("Remove root")
        self.updating = True
        count = self.listRoots.count()
        try:
            for i in range(count):
                item = self.listRoots.item(i)
                if item and (count == 1 or item.isSelected()):
                    root = str(item.text())
                    logging.debug("Remove root %s" % root)
                    self.plotter.sample_analyser.remove_root(root)
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
        if isinstance(self.batch, ChainDirGrid):
            self.comboBoxDataTag.addItems(self.batch.roots_for_dir(self.paramTag))
        else:
            self.comboBoxDataTag.addItems([jobItem.datatag for jobItem in self.grid_paramtag_jobItems[self.paramTag]])

        self.comboBoxDataTag.setCurrentIndex(-1)
        self.comboBoxDataTag.show()

    def setDataTag(self, strDataTag):
        """
        Slot function called on change of comboBoxDataTag.
        """
        if isinstance(self.batch, ChainDirGrid):
            self.newRootItem(str(strDataTag))
        else:
            logging.debug("Data: %s" % strDataTag)
            self.newRootItem(self.paramTag + '_' + str(strDataTag))

    def _updateListParameters(self, items, listParameters):
        listParameters.clear()
        for item in items:
            listItem = QListWidgetItem()
            listItem.setText(item)
            listItem.setFlags(listItem.flags() | Qt.ItemIsUserCheckable)
            listItem.setCheckState(Qt.Unchecked)
            listParameters.addItem(listItem)

    def _updateListParametersSelection(self, oldItems, listParameters):
        if not oldItems:
            return
        for item in oldItems:
            try:
                # Inverse dict search in new name tags
                itemtag = next(tag for tag, name in self.paramNamesTags.items()
                               if name == item)
                match_items = listParameters.findItems(itemtag, Qt.MatchExactly)
            except StopIteration:
                match_items = None
            if match_items:
                match_items[0].setCheckState(Qt.Checked)

    def getCheckedParams(self, checklist, fulllist=False):
        return [checklist.item(i).text() for i in range(checklist.count()) if
                fulllist or checklist.item(i).checkState() == Qt.Checked]

    def getXParams(self, fulllist=False, error=True):
        """
        Returns a list of selected parameter names (not tags) in the X-axis box.

        If `fulllist=True` (default: `False`), returns all of them.
        """
        return [self.paramNamesTags[tag]
                for tag in self.getCheckedParams(self.listParametersX, fulllist)]

    def getYParams(self):
        """
        Returns a list of selected parameter names (not tags) in the X-axis box.
        """
        return [self.paramNamesTags[tag]
                for tag in self.getCheckedParams(self.listParametersY)]

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

    def statusTriangle(self, checked):
        self.checkInsideLegend.setVisible(
            not checked and len(self.getXParams()) == 1 and len(self.getYParams()) == 1)
        self.checkInsideLegend.setEnabled(self.checkInsideLegend.isVisible())

    def itemCheckChange(self, item):
        self.checkInsideLegend.setVisible(
            len(self.getXParams()) == 1 and len(self.getYParams()) == 1 and
            self.trianglePlot.checkState() != Qt.Checked)
        self.checkInsideLegend.setEnabled(self.checkInsideLegend.isVisible())

    def _updateComboBoxColor(self, listOfParams):
        if self.rootdirname and os.path.isdir(self.rootdirname):
            param_old = str(self.comboBoxColor.currentText())
            param_old_new_name = getattr(
                self.paramNames.parWithName(param_old), "name", None)
            self.comboBoxColor.clear()
            self.comboBoxColor.addItems(listOfParams)
            idx = self.comboBoxColor.findText(param_old_new_name, Qt.MatchExactly)
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
            QMessageBox.critical(self, caption, type(e).__name__ + ': ' + str(e) + "\n\n" + msg)
            del msg

            if not isinstance(e, GuiSelectionError) and not capture:
                raise

    def closePlots(self):
        if self.plotter.fig is not None:
            self.plotter.fig.clf()
        plt.close('all')

    def plotData(self):
        """
        Slot function called when pushButtonPlot is pressed.
        """
        if self.updating:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.showMessage("Generating plot....")
        actionText = "plot"
        try:
            # Ensure at least 1 root name specified
            if self.base_dir:
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
            self.plotter.settings.__dict__.update(self.custom_plot_settings)

            script = "from getdist import plots\n"
            if self.script_plot_module != 'getdist.plots':
                script += "from %s import style_name\nplots.set_active_style(style_name)\n" % self.script_plot_module
            script += "\n"
            override_setting = self.changed_settings()
            if override_setting:
                script += ('analysis_settings = %s\n' % override_setting).replace(', ', ",\n" + " " * 21)
            if len(items_x) > 1 or len(items_y) > 1:
                plot_func = 'get_subplot_plotter('
                if not self.plotter.settings.fig_width_inch and len(items_y) and \
                        not (len(items_x) > 1 and len(items_y) > 1) and not self.trianglePlot.isChecked():
                    plot_func += 'subplot_size=3.5, '

            else:
                plot_func = 'get_single_plotter('

            for root in roots:
                self.plotter.sample_analyser.add_root(self.root_infos[root])

            chain_dirs = []
            for root in roots:
                info = self.root_infos[root]
                if info.batch:
                    path = info.batch.batchPath
                else:
                    path = info.path
                if path not in chain_dirs:
                    chain_dirs.append(path)
            if len(chain_dirs) == 1:
                chain_dirs = "r'%s'" % chain_dirs[0].rstrip('\\').rstrip('/')

            if override_setting:
                script += "g=plots.%schain_dir=%s,analysis_settings=analysis_settings)\n" % (plot_func, chain_dirs)
            elif self.iniFile:
                script += "g=plots.%schain_dir=%s, analysis_settings=r'%s')\n" % (plot_func, chain_dirs, self.iniFile)
            else:
                script += "g=plots.%schain_dir=%s)\n" % (plot_func, chain_dirs)

            if self.custom_plot_settings:
                for key, value in self.custom_plot_settings.items():
                    if isinstance(value, str):
                        value = '"' + value + '"'
                    script += 'g.settings.%s = %s\n' % (key, value)

            if len(roots) < 3:
                script += 'roots = %s\n' % roots
            else:
                script += "roots = []\n"
                for root in roots:
                    script += "roots.append('%s')\n" % root

            logging.debug("Plotting with roots = %s" % str(roots))

            # fudge factor of 0.8 seems to help with overlapping labels on retina Mac.
            height = self.plotWidget.height() / self.logicalDpiX() * 0.8
            width = self.plotWidget.width() / self.logicalDpiX() * 0.8

            def setSizeForN(cols, rows):
                if self.plotter.settings.fig_width_inch is not None:
                    self.plotter.settings.fig_width_inch = min(self.plotter.settings.fig_width_inch, width)
                else:
                    self.plotter.settings.fig_width_inch = width
                if self.plotter.settings.subplot_size_ratio:
                    self.plotter.settings.fig_width_inch = min(self.plotter.settings.fig_width_inch,
                                                               height * cols / rows /
                                                               self.plotter.settings.subplot_size_ratio)
                else:
                    self.plotter.settings.subplot_size_ratio = min(1.5, height * cols /
                                                                   (self.plotter.settings.fig_width_inch * rows))

            def make_space_for_legend():
                if len(roots) > 1 and not self.plotter.settings.constrained_layout:
                    self.plotter._tight_layout(rect=(0, 0, 1, 1 - min(0.7, len(roots) * 0.05)))

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
                    else:
                        param_3d = None
                    setSizeForN(len(params), len(params))
                    self.plotter.triangle_plot(roots, params, plot_3d_with_param=param_3d, filled=filled,
                                               shaded=shaded)
                    self.updatePlot()
                    script += "g.triangle_plot(roots, params, filled=%s" % filled
                    if shaded:
                        script += ", shaded=True"
                    if param_3d:
                        script += ", plot_3d_with_param='%s'" % color_param
                    script += ")\n"
                else:
                    raise GuiSelectionError("Select more than 1 x parameter for triangle plot")

            elif len(items_x) > 0 and len(items_y) == 0:
                # 1D plot
                actionText = "1D plot"
                params = items_x
                logging.debug("1D plot with params = %s" % str(params))
                script += "params=%s\n" % str(params)
                setSizeForN(*self.plotter.default_col_row(len(params)))
                if len(roots) > 3:
                    ncol = 2
                else:
                    ncol = None
                self.plotter.plots_1d(roots, params=params, legend_ncol=ncol)
                make_space_for_legend()
                self.updatePlot()
                script += "g.plots_1d(roots, params=params)\n"

            elif len(items_x) > 0 and len(items_y) > 0:
                if len(items_x) > 1 and len(items_y) > 1:
                    # Rectangle plot
                    actionText = 'Rectangle plot'
                    script += "xparams = %s\n" % str(items_x)
                    script += "yparams = %s\n" % str(items_y)
                    logging.debug("Rectangle plot with xparams=%s and yparams=%s" % (str(items_x), str(items_y)))

                    setSizeForN(len(items_x), len(items_y))
                    self.plotter.rectangle_plot(items_x, items_y, roots=roots, filled=filled)
                    make_space_for_legend()
                    self.updatePlot()
                    script += "g.rectangle_plot(xparams, yparams, roots=roots, filled=%s)\n" % filled

                else:
                    # 2D plot
                    single = False
                    if len(items_x) == 1 and len(items_y) == 1:
                        pairs = [[items_x[0], items_y[0]]]
                        setSizeForN(1, 1)
                        single = self.checkInsideLegend.checkState() == Qt.Checked
                    elif len(items_x) == 1 and len(items_y) > 1:
                        item_x = items_x[0]
                        pairs = list(zip([item_x] * len(items_y), items_y))
                        setSizeForN(*self.plotter.default_col_row(len(pairs)))
                    elif len(items_x) > 1 and len(items_y) == 1:
                        item_y = items_y[0]
                        pairs = list(zip(items_x, [item_y] * len(items_x)))
                        setSizeForN(*self.plotter.default_col_row(len(pairs)))
                    else:
                        pairs = []
                    if filled or line:
                        actionText = '2D plot'
                        logging.debug("2D plot with pairs = %s" % str(pairs))
                        if single:
                            self.plotter.make_figure(1)
                            self.plotter.plot_2d(roots, pairs[0], filled=filled, shaded=shaded)
                            script += "g.plot_2d(roots, %s, filled=%s, shaded=%s)\n" % (
                                pairs[0], str(filled), str(shaded))
                            labels = self.plotter._default_legend_labels(None, roots)
                            self.plotter.add_legend(labels)
                            script += 'g.add_legend(%s)\n' % labels
                        else:
                            script += "pairs = %s\n" % pairs
                            self.plotter.plots_2d(roots, param_pairs=pairs, filled=filled, shaded=shaded)
                            make_space_for_legend()
                            script += "g.plots_2d(roots, param_pairs=pairs, filled=%s, shaded=%s)\n" % (
                                str(filled), str(shaded))
                        self.updatePlot()
                    elif color:
                        # 3D plot
                        sets = [list(pair) + [color_param] for pair in pairs]
                        logging.debug("3D plot with sets = %s" % str(sets))
                        actionText = '3D plot'
                        triplets = ["['%s', '%s', '%s']" % tuple(trip) for trip in sets]
                        if len(sets) == 1:
                            script += "g.plot_3d(roots, %s)\n" % triplets[0]
                            self.plotter.settings.scatter_size = 6
                            self.plotter.make_figure()
                            self.plotter.plot_3d(roots, sets[0])
                        else:
                            script += "sets = [" + ",".join(triplets) + "]\n"
                            script += "g.plots_3d(roots, sets)\n"
                            self.plotter.plots_3d(roots, sets)
                            make_space_for_legend()
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
            self.exportAct.setEnabled(True)
        except Exception as e:
            self.errorReport(e, caption=actionText)
        finally:
            self.showMessage()
            QApplication.restoreOverrideCursor()

    def updatePlot(self):
        if self.plotter.fig is None:
            self.canvas = None
        else:
            i = 0
            while True:
                item = self.plotWidget.layout().takeAt(i)
                if item is None:
                    break
                del item
            if hasattr(self, "canvas"):
                del self.canvas
            if hasattr(self, "toolbar"):
                del self.toolbar
            self.canvas = FigureCanvas(self.plotter.fig)
            self.toolbar = NavigationToolbar(self.canvas, self)
            self.plotWidget.layout().addWidget(self.toolbar)
            self.plotWidget.layout().addWidget(self.canvas)
            self.plotWidget.show()

    def tabChanged(self, index):
        """
        Update script text editor when entering 'gui' tab.
        """

        # Enable menu options for main tab only
        self.reLoadAct.setEnabled(index == 0)
        self.openChainsAct.setEnabled(index == 0)
        self.dataMenu.setEnabled(index == 0)
        self.dataMenu.menuAction().setVisible(index == 0)

        if index == 1 and self.script:
            self.script_edit = self.textWidget.toPlainText()
            if self.script_edit and self.script_edit != self.script:
                reply = QMessageBox.question(
                    self, "Overwrite script",
                    "Script is not empty. Overwrite current script?",
                    QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    return

            self.script_edit = self.script
            self.textWidget.setPlainText(self.script_edit)
        if index == 1:
            self.textWidget.setFocus()

    def openScript(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose a file name", '.', "Python (*.py)")
        if not filename:
            return
        filename = str(filename)
        logging.debug("Open file %s" % filename)
        with open(filename, 'r', encoding='utf-8-sig') as f:
            self.script_edit = f.read()
        self.textWidget.setPlainText(self.script_edit)

    def clearScript(self):
        self.textWidget.clear()
        self.script_edit = ''

    def _set_rc(self, opts):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matplotlib.rcParams.clear()
            matplotlib.rcParams.update(opts)

    def plotData2(self):
        """
        Slot function called when pushButtonPlot2 is pressed.
        """
        if self.tabWidget.currentIndex() == 0:
            self.plotData()
            return

        self.script_edit = self.textWidget.toPlainText()
        oldset = plots.default_settings
        old_style = plots.set_active_style()
        oldrc = matplotlib.rcParams.copy()
        plots.default_settings = plots.GetDistPlotSettings()
        self._set_rc(self.orig_rc)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.showMessage("Rendering plot....")
        try:
            script_exec = self.script_edit

            if "g.export()" in script_exec:
                # Comment line which produces export to PDF
                script_exec = script_exec.replace("g.export", "#g.export")

            globaldic = {}
            localdic = {}
            exec(script_exec, globaldic, localdic)

            for v in localdic.values():
                if isinstance(v, plots.GetDistPlotter):
                    self.updateScriptPreview(v)
                    break

            self.exportAct.setEnabled(True)

        except SyntaxError as e:
            QMessageBox.critical(self, "Plot script", type(e).__name__ + ': %s\n %s' % (e, e.text))
        except Exception as e:
            self.errorReport(e, caption="Plot script", capture=True)
        finally:
            QApplication.restoreOverrideCursor()
            plots.default_settings = oldset
            plots.set_active_style(old_style)
            self._set_rc(oldrc)
            self.showMessage()

    def updateScriptPreview(self, plotter):
        if plotter.fig is None:
            return

        self.plotter_script = plotter

        i = 0
        while True:
            item = self.plotWidget2.layout().takeAt(i)
            if item is None:
                break
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
            dpi=self.logicalDpiX() * self.devicePixelRatio(),
            bbox_extra_artists=plotter.extra_artists,
            bbox_inches='tight')
        buf.seek(0)

        # noinspection PyTypeChecker
        image = QImage.fromData(buf.getvalue())

        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(self.devicePixelRatio())
        label = QLabel(self.scrollArea)
        label.setPixmap(pixmap)

        self.scrollArea = QScrollArea(self.plotWidget2)
        self.scrollArea.setWidget(label)
        self.scrollArea.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

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

# noinspection PyArgumentList
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

# noinspection PyArgumentList
class DialogMargeStats(QDialog):
    def __init__(self, parent=None, stats=None, root=''):
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
        # noinspection PyArgumentList
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
        # noinspection PyArgumentList
        h = min(QApplication.desktop().screenGeometry().height() * 4 / 5, 800)
        self.resize(500, h)


# ==============================================================================

# noinspection PyCallByClass
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
        self.generated = [False] * len(tables)
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
            dpi = self.logicalDpiX() * self.devicePixelRatio()
            buf = self.tables[index].tablePNG(bytesIO=True, dpi=dpi)
            pixmap = QPixmap.fromImage(QImage.fromData(buf.getvalue()))
            pixmap.setDevicePixelRatio(self.devicePixelRatio())
            label = QLabel(viewWidget)
            label.setPixmap(pixmap)
            layout = QGridLayout()
            layout.addWidget(label, 1, 0)
            self.tabs[index].setLayout(layout)
            self.generated[index] = True

    def copyLatex(self):
        # noinspection PyArgumentList
        clipboard = QApplication.clipboard()
        clipboard.setText(self.tables[self.tabWidget.currentIndex()].tableTex())

    def saveLatex(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Choose a file name", '.', "Latex (*.tex)")
        if not filename:
            return
        self.tables[self.tabWidget.currentIndex()].write(str(filename))


# ==============================================================================

# noinspection PyArgumentList
class DialogSettings(QDialog):
    def __init__(self, parent, ini, items=None, title='Analysis Settings', width=320, update=None):
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
                if key not in items and key in names.params:
                    items.append(key)
        else:
            names = ini

        nblank = 1
        self.rows = len(items) + nblank
        self.table.setRowCount(self.rows)
        for irow, key in enumerate(items):
            value = ini.string(key)
            item = QTableWidgetItem(str(key))
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(irow, 0, item)
            is_bool = value in ['False', 'True']
            item = QTableWidgetItem("" if is_bool else value)
            if is_bool:
                item.setCheckState(Qt.Checked if ini.bool(key) else Qt.Unchecked)
                item.setFlags(item.flags() ^ Qt.ItemIsEditable | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            else:
                item.setFlags(item.flags() ^ Qt.ItemIsUserCheckable)

            hint = names.comments.get(key, None)
            if hint:
                item.setToolTip("\n".join(hint))
            self.table.setItem(irow, 1, item)
        for i in range(nblank):
            item = QTableWidgetItem(str(""))
            irow = len(items) + i
            self.table.setItem(irow, 0, item)
            item = QTableWidgetItem(str(""))
            self.table.setItem(irow, 1, item)

        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)

        self.table.resizeColumnsToContents()
        maxh = min(parent.rect().height(),
                   (QApplication.desktop().screenGeometry().height() - parent.rect().top()) * 4 / 5)
        self.resize(width, maxh)
        self.table.setColumnWidth(1, self.table.width() - self.table.columnWidth(0))
        self.table.resizeRowsToContents()

        h = self.table.verticalHeader().length() + self.table.horizontalHeader().height() * 4
        h = min(maxh, h)
        self.resize(width, h)
        pos = parent.pos()
        if pos.x() - width > 0:
            pos.setX(pos.x() - width - 1)
            self.move(pos)
        elif parent.frameGeometry().right() + width < QApplication.desktop().screenGeometry().width():
            pos.setX(parent.frameGeometry().right() + 1)
            self.move(pos)

    def getDict(self):
        vals = {}
        for row in range(self.rows):
            key = self.table.item(row, 0).text().strip()
            if key:
                item = self.table.item(row, 1)
                if item.flags() & Qt.ItemIsUserCheckable:
                    vals[key] = str(item.checkState() == Qt.Checked)
                else:
                    vals[key] = item.text().strip()
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


def run_gui():
    import argparse

    parser = argparse.ArgumentParser(description='GetDist GUI')
    parser.add_argument('-v', '--verbose', help='verbose', action="store_true")
    parser.add_argument('--ini', help='Path to .ini file', default=None)
    parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + getdist.__version__)
    args = parser.parse_args()

    # Configure the logging
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    form = '%(asctime).19s [%(levelname)s]\t[%(filename)s:%(lineno)d]\t\t%(message)s'
    logging.basicConfig(level=level, format=form)
    logging.captureWarnings(True)

    sys.argv[0] = 'GetDist GUI'
    app = QApplication(sys.argv)
    app.setApplicationName("GetDist GUI")
    mainWin = MainWindow(app, ini=args.ini)
    mainWin.show()
    mainWin.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()

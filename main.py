import os
import random
import logging
import pandas as pd
import chardet
import csv
from PySide6 import QtCore, QtWidgets, QtGui
from ml_sampling.isolation_forest import isolation_forest_sampling
from ml_sampling.lof import lof_sampling
from ml_sampling.kmeans import kmeans_sampling
from ml_sampling.autoencoder import autoencoder_sampling
from ml_sampling.hdbscan import hdbscan_sampling
from statistical_sampling.random import random_sampling
from statistical_sampling.systematic import systematic_sampling
from statistical_sampling.stratified import stratified_sampling
from statistical_sampling.monetary_unit import monetary_unit_sampling
from utils.visualization import (
    create_strata_chart,
    create_cumulative_chart,
    create_umap_projection,
    visualize_optuna_results,
)
from utils.preprocessing import preprocess_data
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import glob
import time

try:
    from dbfread import DBF
except ImportError:
    DBF = None

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Worker(QtCore.QObject):
    # Define signals
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)  # Emit progress percentage
    result_ready = QtCore.Signal(object)

    def __init__(self, sampling_function, params, choice, method_info, file_path):
        super().__init__()
        self.sampling_function = sampling_function
        self.params = params
        self.choice = choice
        self.method_info = method_info
        self.file_path = file_path

    @QtCore.Slot()
    def run(self):
        try:
            choice = self.choice
            sampling_function = self.sampling_function
            params = self.params

            # Progress callback
            def progress_callback(value):
                self.progress.emit(value)

            if choice in (7, 9):  # K-Means and HDBSCAN
                params['progress_callback'] = progress_callback

            if choice in (5, 6, 7, 8, 9):
                # Remove used parameters from params
                features = params.pop('features')
                random_seed = params.pop('random_seed')
                data = params.pop('data')
                data_preprocessed = params.pop('data_preprocessed')
                sample_size = params.pop('sample_size')
                result = sampling_function(
                    data, data_preprocessed, sample_size, features, random_seed, **params)
            else:
                data = params.pop('data')
                sample_size = params.pop('sample_size')
                # Remove 'sample_size' from params to avoid passing it twice
                params.pop('sample_size', None)
                result = sampling_function(data, sample_size, **params)

            self.result_ready.emit((result, self.method_info,
                                    self.file_path, self.choice))
        except Exception as e:
            logger.exception("Error occurred in worker thread")
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class VisualizationWorker(QtCore.QObject):
    # Define signals
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, visualization_function, args, kwargs):
        super().__init__()
        self.visualization_function = visualization_function
        self.args = args
        self.kwargs = kwargs

    @QtCore.Slot()
    def run(self):
        try:
            self.visualization_function(*self.args, **self.kwargs)
            self.finished.emit()
        except Exception as e:
            logger.exception("Error occurred in visualization worker thread")
            self.error.emit(str(e))


class SamplingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Створення аудиторських вибірок")
        self.setMinimumSize(800, 800)
        self.data = None
        self.data_preprocessed = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.use_threshold = False
        self.use_stratify = False
        self.preprocessing_method_description = ""
        self.widgets = {}
        self.language = 'ua'  # 'ua' for Ukrainian, 'en' for English

        self.sampling_methods = {
            1: {
                "name_ua": "Випадкова вибірка",
                "name_en": "Random Sampling",
                "description_ua": "кожен елемент генеральної сукупності має рівну ймовірність потрапити у вибірку.",
                "description_en": "each element of the population has an equal chance of being selected.",
            },
            2: {
                "name_ua": "Систематична вибірка",
                "name_en": "Systematic Sampling",
                "description_ua": "елементи вибираються з генеральної сукупності через рівні інтервали.",
                "description_en": "elements are selected from the population at regular intervals.",
            },
            3: {
                "name_ua": "Стратифікована вибірка",
                "name_en": "Stratified Sampling",
                "description_ua": "генеральна сукупність ділиться на страти (групи), і з кожної страти формується випадкова вибірка.",
                "description_en": "the population is divided into strata, and random samples are taken from each stratum.",
            },
            4: {
                "name_ua": "Метод грошової одиниці",
                "name_en": "Monetary Unit Sampling",
                "description_ua": "ймовірність вибору елемента пропорційна його грошовій величині. Використовується для оцінки сумарної величини помилок.",
                "description_en": "the probability of selecting an item is proportional to its monetary value.",
            },
            5: {
                "name_ua": "Isolation Forest",
                "name_en": "Isolation Forest",
                "description_ua": "алгоритм для виявлення аномалій на основі випадкових лісів.",
                "description_en": "an algorithm for anomaly detection based on random forests.",
            },
            6: {
                "name_ua": "Local Outlier Factor",
                "name_en": "Local Outlier Factor",
                "description_ua": "метод для виявлення локальних аномалій у даних.",
                "description_en": "a method for detecting local anomalies in data.",
            },
            7: {
                "name_ua": "Кластеризація K-Means",
                "name_en": "K-Means Clustering",
                "description_ua": "групування даних за схожістю для виявлення незвичайних точок.",
                "description_en": "grouping data by similarity to detect unusual points.",
            },
            8: {
                "name_ua": "Автоенкодер",
                "name_en": "Autoencoder",
                "description_ua": "зменшення розмірності даних для виявлення відхилень через аналіз помилки відновлення.",
                "description_en": "reducing data dimensionality to detect deviations by analyzing reconstruction error.",
            },
            9: {
                "name_ua": "HDBSCAN",
                "name_en": "HDBSCAN",
                "description_ua": "знаходження аномалій, класифікуючи точки як шум, спираючись на їхню щільність і відстань до інших точок, що дозволяє виокремлювати викиди в даних.",
                "description_en": "finding anomalies by classifying points as noise based on their density and distance to other points.",
            },
        }

        self.init_ui()

    def init_ui(self):
        # Stylesheet for the dark theme and widget appearance
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLineEdit, QComboBox {
                background-color: #3c3f41;
                border: 1px solid #6c6c6c;
                padding: 4px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #444444;
                color: #ffffff;
                border: none;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QGroupBox {
                border: 1px solid #6c6c6c;
                margin-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QRadioButton {
                font-size: 14px;
            }
            QCheckBox {
                font-size: 14px;
            }
            QLabel {
                font-size: 14px;
            }
            QProgressBar {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #6c6c6c;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #00aa00;
            }
        """)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # Language Switch Button
        self.language_button = QtWidgets.QPushButton("EN")
        self.language_button.setFixedSize(40, 40)
        self.language_button.clicked.connect(self.switch_language)
        self.language_button.setStyleSheet(
            "background-color: #444444; color: #ffffff; border: none;")
        language_layout = QtWidgets.QHBoxLayout()
        language_layout.addStretch()
        language_layout.addWidget(self.language_button)
        main_layout.addLayout(language_layout)

        # Sampling Methods
        self.method_group = QtWidgets.QGroupBox()
        self.method_layout = QtWidgets.QVBoxLayout()
        self.method_group.setLayout(self.method_layout)
        self.method_group.setTitle(self.tr("Оберіть тип вибірки:"))
        self.method_buttons = []
        self.method_button_group = QtWidgets.QButtonGroup()
        for key, method in self.sampling_methods.items():
            radio_button = QtWidgets.QRadioButton()
            radio_button.setChecked(key == 1)
            self.method_button_group.addButton(radio_button, key)
            self.method_buttons.append(radio_button)
            self.method_layout.addWidget(radio_button)
        self.method_button_group.buttonClicked.connect(
            self.on_method_change)
        main_layout.addWidget(self.method_group)

        # Options
        self.options_group = QtWidgets.QGroupBox()
        self.options_layout = QtWidgets.QFormLayout()
        self.options_group.setLayout(self.options_layout)

        # File selection
        self.file_button = QtWidgets.QPushButton(self.tr("Огляд"))
        self.file_button.clicked.connect(self.browse_file)
        self.file_label = QtWidgets.QLabel(
            self.tr("Файл з генеральною сукупністю:"))
        self.file_path = QtWidgets.QLineEdit()
        self.file_shape_label = QtWidgets.QLabel()
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        self.options_layout.addRow(file_layout)
        self.options_layout.addRow(self.file_path)
        self.options_layout.addRow(self.file_shape_label)

        # Sample size
        self.sample_size_label = QtWidgets.QLabel(self.tr("Розмір вибірки:"))
        self.sample_size_input = QtWidgets.QLineEdit()
        self.options_layout.addRow(
            self.sample_size_label, self.sample_size_input)

        # Strata column
        self.strata_label = QtWidgets.QLabel(
            self.tr("Стовпець для стратифікації:"))
        self.strata_combo = QtWidgets.QComboBox()
        self.options_layout.addRow(self.strata_label, self.strata_combo)

        # Value column
        self.value_label = QtWidgets.QLabel(
            self.tr("Стовпець зі значеннями грошових одиниць:"))
        self.value_combo = QtWidgets.QComboBox()
        self.options_layout.addRow(self.value_label, self.value_combo)

        # Use threshold
        self.use_threshold_checkbox = QtWidgets.QCheckBox(
            self.tr("Використовувати порогове значення?"))
        self.use_threshold_checkbox.toggled.connect(
            self.toggle_threshold_input)
        self.threshold_label = QtWidgets.QLabel(self.tr("Порогове значення:"))
        self.threshold_input = QtWidgets.QLineEdit()
        self.threshold_label.setVisible(False)
        self.threshold_input.setVisible(False)
        self.options_layout.addRow(self.use_threshold_checkbox)
        self.options_layout.addRow(self.threshold_label, self.threshold_input)

        # Use stratify
        self.use_stratify_checkbox = QtWidgets.QCheckBox(
            self.tr("Використовувати стратифікацію?"))
        self.use_stratify_checkbox.toggled.connect(
            self.toggle_stratify_input)
        self.mus_strata_label = QtWidgets.QLabel(
            self.tr("Стовпець для стратифікації:"))
        self.mus_strata_combo = QtWidgets.QComboBox()
        self.mus_strata_label.setVisible(False)
        self.mus_strata_combo.setVisible(False)
        self.options_layout.addRow(self.use_stratify_checkbox)
        self.options_layout.addRow(
            self.mus_strata_label, self.mus_strata_combo)

        # Column types button
        self.column_types_button = QtWidgets.QPushButton(
            self.tr("Вказати типи колонок"))
        self.column_types_button.clicked.connect(self.define_column_types)
        self.options_layout.addRow(self.column_types_button)

        self.preprocess_label = QtWidgets.QLabel(
            self.tr("Передобробка даних виконана."))
        self.preprocess_label.setStyleSheet("color: green;")
        self.preprocess_label.setVisible(False)
        self.options_layout.addRow(self.preprocess_label)

        # Spacer to prevent layout shrinking
        self.options_layout.addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        main_layout.addWidget(self.options_group)

        # Create Sample Button
        self.create_button = QtWidgets.QPushButton(
            self.tr("Створити вибірку"))
        self.create_button.clicked.connect(self.create_sample)
        self.create_button.setStyleSheet(
            "background-color: #444444; color: #ffffff;")
        main_layout.addWidget(self.create_button)

        # Status Label
        self.status_label = QtWidgets.QLabel()
        main_layout.addWidget(self.status_label)

        # Result Label
        self.result_label = QtWidgets.QLabel()
        self.result_label.setStyleSheet("color: green;")
        main_layout.addWidget(self.result_label)

        # Progress Bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.update_ui_language()
        self.on_method_change()

        # Bring window to front
        self.raise_()
        self.activateWindow()

    def switch_language(self):
        if self.language == 'ua':
            self.language = 'en'
            self.language_button.setText("UA")
        else:
            self.language = 'ua'
            self.language_button.setText("EN")
        self.update_ui_language()

    def update_ui_language(self):
        # Update method descriptions
        for idx, radio_button in enumerate(self.method_buttons):
            key = idx + 1
            method = self.sampling_methods[key]
            if self.language == 'ua':
                text = f"{method['name_ua']}: {method['description_ua']}"
            else:
                text = f"{method['name_en']}: {method['description_en']}"
            radio_button.setText(text)

        # Update labels and buttons
        if self.language == 'ua':
            self.method_group.setTitle("Оберіть тип вибірки:")
            self.file_label.setText("Файл з генеральною сукупністю:")
            self.file_button.setText("Огляд")
            self.sample_size_label.setText("Розмір вибірки:")
            self.strata_label.setText("Стовпець для стратифікації:")
            self.value_label.setText(
                "Стовпець зі значеннями грошових одиниць:")
            self.use_threshold_checkbox.setText(
                "Використовувати порогове значення?")
            self.threshold_label.setText("Порогове значення:")
            self.use_stratify_checkbox.setText(
                "Використовувати стратифікацію?")
            self.mus_strata_label.setText("Стовпець для стратифікації:")
            self.column_types_button.setText("Вказати типи колонок")
            self.preprocess_label.setText("Передобробка даних виконана.")
            self.create_button.setText("Створити вибірку")
            self.status_label.setText("")
        else:
            self.method_group.setTitle("Select sampling method:")
            self.file_label.setText("File with population data:")
            self.file_button.setText("Browse")
            self.sample_size_label.setText("Sample size:")
            self.strata_label.setText("Strata column:")
            self.value_label.setText("Value column:")
            self.use_threshold_checkbox.setText("Use threshold value?")
            self.threshold_label.setText("Threshold value:")
            self.use_stratify_checkbox.setText("Use stratification?")
            self.mus_strata_label.setText("Strata column:")
            self.column_types_button.setText("Define column types")
            self.preprocess_label.setText("Data preprocessing completed.")
            self.create_button.setText("Create sample")
            self.status_label.setText("")

    def browse_file(self):
        file_dialog = QtWidgets.QFileDialog()
        file_types = "All Files (*);;CSV Files (*.csv);;Excel Files (*.xls *.xlsx);;DBF Files (*.dbf);;JSON Files (*.json);;Parquet Files (*.parquet)"
        file_path, _ = file_dialog.getOpenFileName(
            self, self.tr("Виберіть файл з генеральною сукупністю"), "", file_types)
        if file_path:
            self.file_path.setText(file_path)
            self.status_label.setText(self.tr("Завантаження файлу..."))
            QtCore.QCoreApplication.processEvents()
            try:
                # Determine file type by extension
                _, file_extension = os.path.splitext(file_path)
                file_extension = file_extension.lower()
                if file_extension in ['.csv']:
                    self.data = self.read_csv_file(file_path)
                elif file_extension in ['.xls', '.xlsx']:
                    self.data = pd.read_excel(file_path)
                elif file_extension in ['.dbf']:
                    self.data = self.read_dbf_file(file_path)
                elif file_extension == '.json':
                    self.data = pd.read_json(file_path)
                elif file_extension == '.parquet':
                    self.data = pd.read_parquet(file_path)
                else:
                    # Try to read as CSV with auto-detection
                    self.data = self.read_csv_file(file_path)
                self.populate_column_dropdowns()
                shape_text = f"{self.tr('Розмір файлу')}: {self.data.shape}"
                self.file_shape_label.setText(shape_text)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, self.tr("Помилка"), f"{self.tr('Помилка при читанні файлу')}: {e}")
            finally:
                self.status_label.setText("")

    def read_csv_file(self, file_path):
        # Detect encoding
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(100000))  # Read first 100,000 bytes
        encoding = result['encoding']
        # Detect delimiter
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            sample = f.read(1024)
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','
        # Read CSV with detected encoding and delimiter
        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
        return df

    def read_dbf_file(self, file_path):
        if DBF is None:
            QtWidgets.QMessageBox.critical(
                self, self.tr("Помилка"), self.tr("Модуль 'dbfread' не встановлено. Будь ласка, встановіть його, щоб відкрити DBF файли."))
            return pd.DataFrame()
        # Detect encoding for DBF
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(100000))
        encoding = result['encoding']
        # Read DBF file
        table = DBF(file_path, encoding=encoding)
        df = pd.DataFrame(iter(table))
        return df

    def populate_column_dropdowns(self):
        try:
            numerical_columns = [
                col for col in self.data.columns if pd.api.types.is_numeric_dtype(self.data[col])]
            self.value_combo.clear()
            self.value_combo.addItems(numerical_columns)
            columns = list(self.data.columns)
            self.strata_combo.clear()
            self.strata_combo.addItems(columns)
            self.mus_strata_combo.clear()
            self.mus_strata_combo.addItems(columns)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, self.tr("Помилка"), f"{self.tr('Помилка при заповненні списку колонок')}: {e}")

    def on_method_change(self):
        choice = self.method_button_group.checkedId()
        self.sample_size_label.setVisible(True)
        self.sample_size_input.setVisible(True)
        self.strata_label.setVisible(False)
        self.strata_combo.setVisible(False)
        self.value_label.setVisible(False)
        self.value_combo.setVisible(False)
        self.use_threshold_checkbox.setVisible(False)
        self.threshold_label.setVisible(False)
        self.threshold_input.setVisible(False)
        self.use_stratify_checkbox.setVisible(False)
        self.mus_strata_label.setVisible(False)
        self.mus_strata_combo.setVisible(False)
        self.column_types_button.setVisible(False)
        self.preprocess_label.setVisible(False)

        if choice == 3:
            self.strata_label.setVisible(True)
            self.strata_combo.setVisible(True)
        elif choice == 4:
            self.value_label.setVisible(True)
            self.value_combo.setVisible(True)
            self.use_threshold_checkbox.setVisible(True)
            self.threshold_label.setVisible(
                self.use_threshold_checkbox.isChecked())
            self.threshold_input.setVisible(
                self.use_threshold_checkbox.isChecked())
            self.use_stratify_checkbox.setVisible(True)
            self.mus_strata_label.setVisible(
                self.use_stratify_checkbox.isChecked())
            self.mus_strata_combo.setVisible(
                self.use_stratify_checkbox.isChecked())
        elif choice in (5, 6, 7, 8, 9):
            self.column_types_button.setVisible(True)
            if self.data_preprocessed is not None:
                self.preprocess_label.setVisible(True)

    def toggle_threshold_input(self):
        self.use_threshold = self.use_threshold_checkbox.isChecked()
        self.threshold_label.setVisible(self.use_threshold)
        self.threshold_input.setVisible(self.use_threshold)

    def toggle_stratify_input(self):
        self.use_stratify = self.use_stratify_checkbox.isChecked()
        self.mus_strata_label.setVisible(self.use_stratify)
        self.mus_strata_combo.setVisible(self.use_stratify)

    def define_column_types(self):
        if self.data is None:
            QtWidgets.QMessageBox.critical(
                self, self.tr("Помилка"), self.tr("Спочатку завантажте дані."))
            return

        def select_columns(column_type):
            columns = self.data.columns.tolist()
            if column_type == "numerical":
                columns = [col for col in self.data.columns if pd.api.types.is_numeric_dtype(
                    self.data[col])]
            elif column_type == "categorical":
                columns = [
                    col for col in self.data.columns if col not in self.numerical_columns]

            if not columns:
                QtWidgets.QMessageBox.critical(
                    self, self.tr("Помилка"), f"{self.tr('У датафреймі немає')} {self.tr(column_type)} {self.tr('колонок.')}")
                return

            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(
                f"{self.tr('Вибір')} {self.tr(column_type)} {self.tr('колонок')}")
            dialog_layout = QtWidgets.QVBoxLayout()
            dialog.setLayout(dialog_layout)

            label = QtWidgets.QLabel(
                f"{self.tr('Виберіть')} {self.tr(column_type)} {self.tr('колонки')}:")
            dialog_layout.addWidget(label)

            list_widget = QtWidgets.QListWidget()
            list_widget.setSelectionMode(
                QtWidgets.QAbstractItemView.MultiSelection)
            list_widget.addItems(columns)
            dialog_layout.addWidget(list_widget)

            button = QtWidgets.QPushButton(self.tr("Далі"))
            button.clicked.connect(dialog.accept)
            dialog_layout.addWidget(button)

            if dialog.exec():
                selected_columns = [item.text()
                                    for item in list_widget.selectedItems()]
                if column_type == "numerical":
                    self.numerical_columns = selected_columns
                elif column_type == "categorical":
                    self.categorical_columns = selected_columns
                if not selected_columns:
                    QtWidgets.QMessageBox.warning(
                        self, self.tr("Попередження"), f"{self.tr('Не обрано жодної')} {self.tr(column_type)} {self.tr('колонки.')}")
                if column_type == "numerical":
                    select_columns("categorical")
                else:
                    (
                        self.data_preprocessed,
                        self.preprocessing_method_description,
                    ) = preprocess_data(
                        self.data, self.numerical_columns, self.categorical_columns
                    )
                    self.preprocess_label.setVisible(True)

        select_columns("numerical")

    def create_sample(self):
        self.create_button.setEnabled(False)
        self.status_label.setText(self.tr("Створення вибірки..."))
        self.result_label.setText("")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QtCore.QCoreApplication.processEvents()

        choice = self.method_button_group.checkedId()
        file_path = self.file_path.text()
        sample_size = int(self.sample_size_input.text())
        population = self.data.copy()
        self.data = population

        dataset_size = population.shape[0]
        if choice == 9 and dataset_size > 300000:
            QtWidgets.QMessageBox.warning(self, self.tr("Помилка"), self.tr(
                "HDBSCAN не підтримує вибірки більше 300,000 рядків."))
            self.create_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        if choice == 6 and dataset_size > 1000000:
            QtWidgets.QMessageBox.warning(self, self.tr("Помилка"), self.tr(
                "Local Outlier Factor не підтримує вибірки більше 1,000,000 рядків."))
            self.create_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return

        method_info = self.sampling_methods.get(choice)
        sampling_function = self.get_sampling_function(choice)
        kwargs = self.get_sampling_parameters(choice)

        # Create worker and thread
        self.worker = Worker(sampling_function, kwargs, choice,
                             method_info, file_path)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Connect signals
        self.worker.error.connect(self.handle_worker_error)
        self.worker.progress.connect(self.update_progress_bar)
        self.worker.result_ready.connect(self.process_sampling_result)
        self.thread.start()

    def get_sampling_function(self, choice):
        sampling_functions = {
            1: random_sampling,
            2: systematic_sampling,
            3: stratified_sampling,
            4: monetary_unit_sampling,
            5: isolation_forest_sampling,
            6: lof_sampling,
            7: kmeans_sampling,
            8: autoencoder_sampling,
            9: hdbscan_sampling,
        }
        return sampling_functions.get(choice)

    def get_sampling_parameters(self, choice):
        params = {}
        params["random_seed"] = random.randint(1, 10000)
        params["sample_size"] = int(self.sample_size_input.text())
        params["data"] = self.data  # Include data here

        if choice == 3:
            params["strata_column"] = self.strata_combo.currentText()
        elif choice == 4:
            params["value_column"] = self.value_combo.currentText()
            params["threshold"] = (
                float(self.threshold_input.text())
                if self.use_threshold
                else None
            )
            if self.use_stratify:
                params["strata_column"] = self.mus_strata_combo.currentText()
            else:
                params["strata_column"] = None
        elif choice in (5, 6, 7, 8, 9):
            if not self.numerical_columns and not self.categorical_columns:
                raise ValueError(
                    self.tr("Вкажіть типи колонок для передобробки даних.")
                )
            if not hasattr(self, "data_preprocessed"):
                raise ValueError(
                    self.tr(
                        "Передобробка даних не виконана. Натисніть 'Вказати типи колонок' та збережіть вибір.")
                )
            params["data_preprocessed"] = self.data_preprocessed
            params["features"] = self.numerical_columns + \
                self.categorical_columns
        return params

    @QtCore.Slot(int)
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    @QtCore.Slot(str)
    def handle_worker_error(self, error_message):
        self.create_button.setEnabled(True)
        self.status_label.setText("")
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(
            self, self.tr("Помилка"), error_message)

    @QtCore.Slot(object)
    def process_sampling_result(self, result_tuple):
        result, method_info, file_path, choice = result_tuple
        self.create_button.setEnabled(True)
        self.status_label.setText("")
        self.progress_bar.setVisible(False)

        try:
            # Unpack result based on sampling method
            if choice in (1, 2, 3, 4):
                population_with_results, sample, sampling_method_description = result
            elif choice in (5, 6, 7, 8, 9):
                (
                    population_with_results,
                    population_for_chart,
                    sample,
                    sampling_method_description,
                ) = result[:4]
                best_study = result[4] if len(result) > 4 else None

            if sample is None or sample.empty:
                raise ValueError(
                    f"{self.tr('Не вдалося сформувати вибірку або вибірка порожня.')}\n{sampling_method_description}"
                )

            file_name, file_ext = os.path.splitext(file_path)
            sample_type = method_info["name_en"].lower().replace(" ", "_")
            output_path_en = f"{file_name}_{sample_type}.pdf"

            sample_output_path = f"{file_name}_{sample_type}_sample.csv"
            population_output_path = f"{file_name}_{sample_type}_population.csv"

            population_with_results.to_csv(
                population_output_path, index=False)
            sample.to_csv(sample_output_path, index=False)

            chart_paths = []

            # Retrieve parameters used during sampling
            strata_column = None
            value_column = None
            threshold = None

            if choice == 3:
                strata_column = self.strata_combo.currentText()
            elif choice == 4:
                value_column = self.value_combo.currentText()
                threshold = (
                    float(self.threshold_input.text())
                    if self.use_threshold else None
                )
                strata_column = (
                    self.mus_strata_combo.currentText()
                    if self.use_stratify else None
                )

            if choice in (3, 4) and strata_column:
                strata_chart_path = f"{file_name}_{sample_type}_strata_chart.png"
                create_strata_chart(
                    population_with_results,
                    sample,
                    strata_column,
                    strata_chart_path,
                    threshold=threshold,
                    value_column=value_column,
                )
                chart_paths.append(strata_chart_path)

            if choice == 4:
                base_cumulative_chart_path = (
                    f"{file_name}_{sample_type}_cumulative_chart"
                )
                create_cumulative_chart(
                    population_with_results,
                    value_column,
                    strata_column if strata_column else None,
                    base_cumulative_chart_path,
                    threshold=threshold,
                )
                pattern = f"{file_name}_{sample_type}_cumulative_chart*.png"
                cumulative_chart_files = glob.glob(pattern)
                chart_paths.extend(cumulative_chart_files)

            if choice in (5, 6, 7, 8, 9):
                # Start UMAP visualization in a separate thread
                self.status_label.setText(
                    self.tr("Створення UMAP візуалізації..."))
                self.create_button.setEnabled(False)
                visualization_args = [population_for_chart]
                visualization_kwargs = {
                    "label_column": "is_sample",
                    "features": population_for_chart.columns.drop(['is_sample', 'cluster'], errors='ignore'),
                    "output_path": f"{file_name}_{sample_type}_umap_projection.png",
                    "cluster_column": 'cluster' if 'cluster' in population_for_chart.columns else None
                }
                self.visualization_worker = VisualizationWorker(
                    create_umap_projection, visualization_args, visualization_kwargs)
                self.visualization_thread = QtCore.QThread()
                self.visualization_worker.moveToThread(
                    self.visualization_thread)
                self.visualization_thread.started.connect(
                    self.visualization_worker.run)
                self.visualization_worker.finished.connect(
                    self.visualization_thread.quit)
                self.visualization_worker.finished.connect(
                    self.visualization_worker.deleteLater)
                self.visualization_thread.finished.connect(
                    self.visualization_thread.deleteLater)
                self.visualization_worker.error.connect(
                    self.handle_visualization_error)
                self.visualization_worker.finished.connect(lambda: self.finalize_process(
                    choice, method_info, output_path_en, sampling_method_description, chart_paths, file_name, sample_type, best_study))
                self.visualization_thread.start()
                return  # Exit the method to wait for visualization to finish
            else:
                self.finalize_process(choice, method_info, output_path_en,
                                      sampling_method_description, chart_paths, file_name, sample_type, None)

        except Exception as e:
            logger.exception("Error occurred while processing result")
            QtWidgets.QMessageBox.critical(
                self, self.tr("Помилка"), str(e))

    @QtCore.Slot()
    def handle_visualization_error(self, error_message):
        self.create_button.setEnabled(True)
        self.status_label.setText("")
        QtWidgets.QMessageBox.critical(
            self, self.tr("Помилка"), error_message)

    def finalize_process(self, choice, method_info, output_path_en, sampling_method_description, chart_paths, file_name, sample_type, best_study):
        self.create_button.setEnabled(True)
        self.status_label.setText("")

        # Add UMAP visualization to chart_paths
        if choice in (5, 6, 7, 8, 9):
            umap_projection_path = (
                f"{file_name}_{sample_type}_umap_projection.png"
            )
            chart_paths.append(umap_projection_path)

            if choice in (7, 9) and best_study:
                base_optuna_results_path = (
                    f"{file_name}_{sample_type}_optuna_results"
                )
                visualize_optuna_results(
                    best_study, base_optuna_results_path)
                pattern = f"{file_name}_{sample_type}_optuna_results*.png"
                optuna_result_files = glob.glob(pattern)
                chart_paths.extend(optuna_result_files)

        # Use sampling_method_description from the function if available, else use method_info
        if not sampling_method_description:
            sampling_method_description = method_info['description_en']

        # Generate English PDF
        self.generate_pdf(
            output_path_en, sampling_method_description, chart_paths, language='en')

        message = f"{self.tr('Вибірку збережено у файлі')}:\n{output_path_en}"
        self.result_label.setText(message)

    def generate_pdf(self, output_path, sampling_method_description, chart_paths, language='en'):
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        flowables = []

        def convert_newlines(text):
            return text.replace("\n", "<br/>")

        custom_style = ParagraphStyle(
            "Custom", parent=styles["Normal"], spaceAfter=12, leading=15
        )

        if self.preprocessing_method_description:
            heading = "Description of data preprocessing methods:"
            flowables.append(
                Paragraph(
                    heading,
                    styles["Heading2"],
                )
            )
            formatted_preprocess_desc = convert_newlines(
                self.preprocessing_method_description
            )
            flowables.append(
                Paragraph(formatted_preprocess_desc, custom_style)
            )
            flowables.append(Spacer(1, 12))

        if sampling_method_description:
            heading = "Description of the sampling method:"
            flowables.append(
                Paragraph(
                    heading, styles["Heading2"]
                )
            )
            formatted_sampling_desc = convert_newlines(
                sampling_method_description
            )
            flowables.append(
                Paragraph(formatted_sampling_desc, custom_style)
            )
            flowables.append(Spacer(1, 12))

        for chart_path in chart_paths:
            if os.path.exists(chart_path):
                heading = "Chart"
                flowables.append(
                    Paragraph(
                        f"{heading}: {os.path.basename(chart_path)}",
                        styles["Heading3"],
                    )
                )
                img = Image(chart_path, width=6 * inch, height=4 * inch)
                flowables.append(img)
                flowables.append(Spacer(1, 12))
            else:
                message = f"Chart {os.path.basename(chart_path)} not found."
                flowables.append(
                    Paragraph(
                        message,
                        styles["Normal"],
                    )
                )
                flowables.append(Spacer(1, 12))

        doc.build(flowables)

    def closeEvent(self, event):
        # Wait for worker threads to finish
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        if hasattr(self, 'visualization_thread') and self.visualization_thread.isRunning():
            self.visualization_thread.quit()
            self.visualization_thread.wait()
        event.accept()


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = SamplingApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

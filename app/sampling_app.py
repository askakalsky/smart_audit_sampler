import os
import random
import logging
import pandas as pd
from PySide6 import QtCore, QtWidgets
from ml_sampling.isolation_forest import isolation_forest_sampling
from ml_sampling.lof import lof_sampling
from ml_sampling.kmeans import kmeans_sampling
from ml_sampling.autoencoder import autoencoder_sampling
from ml_sampling.hdbscan import hdbscan_sampling
from statistical_sampling.random import random_sampling
from statistical_sampling.systematic import systematic_sampling
from statistical_sampling.stratified import stratified_sampling
from statistical_sampling.monetary_unit import monetary_unit_sampling
from .workers import (
    Worker, FileLoaderWorker, PreprocessingWorker,
    VisualizationWorker, PdfGenerationWorker
)
from utils.visualization import (
    create_strata_chart,
    create_cumulative_chart,
    create_umap_projection,
    visualize_optuna_results,
)
import glob

# Configure logging to output debug information with timestamps and log levels
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SamplingApp(QtWidgets.QMainWindow):
    """
    Main application window for creating audit samples using various sampling methods.

    This class handles the user interface, user interactions, data loading, preprocessing,
    sampling, visualization, and PDF report generation.
    """

    def __init__(self):
        """
        Initialize the SamplingApp.

        Sets up the user interface, translation dictionaries, sampling methods,
        and initializes the main UI components.
        """
        super().__init__()
        self.language = 'ua'  # 'ua' for Ukrainian, 'en' for English

        # Translation dictionaries for UI text
        self.translations = {
            'ua': {
                'window_title': "Створення аудиторських вибірок",
                'select_sampling_method': "Оберіть тип вибірки:",
                'file_with_population_data': "Файл з генеральною сукупністю:",
                'browse': "Огляд",
                'sample_size': "Розмір вибірки:",
                'strata_column': "Стовпець для стратифікації:",
                'value_column': "Стовпець зі значеннями грошових одиниць:",
                'use_threshold_value': "Використовувати порогове значення?",
                'threshold_value': "Порогове значення:",
                'use_stratification': "Використовувати стратифікацію?",
                'strata_column_for_mus': "Стовпець для стратифікації:",
                'define_column_types': "Вказати типи колонок",
                'data_preprocessing_completed': "Передобробка даних виконана.",
                'create_sample': "Створити вибірку",
                'loading_file': "Завантаження файлу...",
                'error': "Помилка",
                'first_load_data': "Спочатку завантажте дані.",
                'in_dataframe_no_columns_of_type': "У датафреймі немає {column_type} стовпців.",
                'no_columns_selected_of_type': "Не вибрано жодного {column_type} стовпця.",
                'data_preprocessing': "Передобробка даних...",
                'creating_sample': "Створення вибірки...",
                'file_size': "Розмір файлу",
                'sample_saved_in_file': "Вибірку збережено у файлі",
                'select_columns_of_type': "Виберіть {column_type} стовпці:",
                'warning': "Попередження",
                'data_preprocessing_not_done': "Передобробка даних не виконана. Натисніть 'Вказати типи колонок' та збережіть вибір.",
                'no_columns_of_type_selected': "Не вибрано жодного {column_type} стовпця.",
                'error_reading_file': "Помилка при читанні файлу",
                'error_processing_data': "Помилка при обробці даних",
                'error_creating_sample': "Не вдалося сформувати вибірку або вибірка порожня.",
                'hdbscan_limit_error': "HDBSCAN не підтримує вибірки більше 300,000 рядків.",
                'lof_limit_error': "Local Outlier Factor не підтримує вибірки більше 1,000,000 рядків.",
                'error_title': "Помилка",
                'file_loading_error': "Помилка при читанні файлу",
                'visualization_error': "Помилка при створенні візуалізації",
            },
            'en': {
                'window_title': "Audit Sampling Creation",
                'select_sampling_method': "Select sampling method:",
                'file_with_population_data': "File with population data:",
                'browse': "Browse",
                'sample_size': "Sample size:",
                'strata_column': "Strata column:",
                'value_column': "Value column:",
                'use_threshold_value': "Use threshold value?",
                'threshold_value': "Threshold value:",
                'use_stratification': "Use stratification?",
                'strata_column_for_mus': "Strata column:",
                'define_column_types': "Define column types",
                'data_preprocessing_completed': "Data preprocessing completed.",
                'create_sample': "Create sample",
                'loading_file': "Loading file...",
                'error': "Error",
                'first_load_data': "First, load the data.",
                'in_dataframe_no_columns_of_type': "No {column_type} columns in the dataframe.",
                'no_columns_selected_of_type': "No {column_type} columns selected.",
                'data_preprocessing': "Data preprocessing...",
                'creating_sample': "Creating sample...",
                'file_size': "File size",
                'sample_saved_in_file': "Sample saved in file",
                'select_columns_of_type': "Select {column_type} columns:",
                'warning': "Warning",
                'data_preprocessing_not_done': "Data preprocessing not done. Click 'Define column types' and save your selection.",
                'no_columns_of_type_selected': "No {column_type} columns selected.",
                'error_reading_file': "Error reading file",
                'error_processing_data': "Error processing data",
                'error_creating_sample': "Failed to create sample or sample is empty.",
                'hdbscan_limit_error': "HDBSCAN does not support datasets larger than 300,000 rows.",
                'lof_limit_error': "Local Outlier Factor does not support datasets larger than 1,000,000 rows.",
                'error_title': "Error",
                'file_loading_error': "Error reading file",
                'visualization_error': "Error creating visualization",
            },
        }

        # Column type translations for UI elements
        self.column_type_translations = {
            'numerical': {'ua': 'числових', 'en': 'numerical'},
            'categorical': {'ua': 'категоріальних', 'en': 'categorical'}
        }

        # Initialize data-related attributes
        self.data = None  # Raw data loaded from file
        self.data_preprocessed = None  # Data after preprocessing
        self.numerical_columns = []  # List of selected numerical columns
        self.categorical_columns = []  # List of selected categorical columns
        self.use_threshold = False  # Flag to use threshold in sampling
        self.use_stratify = False  # Flag to use stratification in sampling
        self.preprocessing_method_description = ""  # Description of preprocessing steps
        self.widgets = {}  # Dictionary to hold references to UI widgets

        # Define available sampling methods with their names and descriptions in both languages
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

        # Initialize the user interface
        self.init_ui()

    def t(self, key: str) -> str:
        """
        Retrieve the translated text based on the current language and key.

        Args:
            key (str): The key for the desired translation.

        Returns:
            str: Translated string corresponding to the key.
        """
        return self.translations[self.language][key]

    def translate_column_type(self, column_type: str) -> str:
        """
        Translate the column type based on the current language.

        Args:
            column_type (str): The type of column ('numerical' or 'categorical').

        Returns:
            str: Translated column type.
        """
        return self.column_type_translations[column_type][self.language]

    def init_ui(self):
        """
        Initialize and set up the user interface components.

        This includes setting up the main window, styles, layouts, buttons, labels,
        and connecting signals to their respective slots.
        """
        self.setWindowTitle(self.t('window_title'))
        self.setMinimumSize(800, 800)

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

        # Sampling Methods Group Box
        self.method_group = QtWidgets.QGroupBox(
            self.t('select_sampling_method'))
        self.method_layout = QtWidgets.QVBoxLayout()
        self.method_group.setLayout(self.method_layout)
        self.method_buttons = []
        self.method_button_group = QtWidgets.QButtonGroup()
        for key, method in self.sampling_methods.items():
            radio_button = QtWidgets.QRadioButton()
            radio_button.setChecked(key == 1)  # Set first method as default
            self.method_button_group.addButton(radio_button, key)
            self.method_buttons.append(radio_button)
            self.method_layout.addWidget(radio_button)
        self.method_button_group.buttonClicked.connect(
            self.on_method_change)
        main_layout.addWidget(self.method_group)

        # Options Group Box
        self.options_group = QtWidgets.QGroupBox()
        self.options_layout = QtWidgets.QFormLayout()
        self.options_group.setLayout(self.options_layout)

        # File Selection Components
        self.file_button = QtWidgets.QPushButton(self.t('browse'))
        self.file_button.clicked.connect(self.browse_file)
        self.file_label = QtWidgets.QLabel(self.t('file_with_population_data'))
        self.file_path = QtWidgets.QLineEdit()
        self.file_shape_label = QtWidgets.QLabel()
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_button)
        self.options_layout.addRow(file_layout)
        self.options_layout.addRow(self.file_path)
        self.options_layout.addRow(self.file_shape_label)

        # Sample Size Components
        self.sample_size_label = QtWidgets.QLabel(self.t('sample_size'))
        self.sample_size_input = QtWidgets.QLineEdit()
        self.options_layout.addRow(
            self.sample_size_label, self.sample_size_input)

        # Strata Column Components
        self.strata_label = QtWidgets.QLabel(self.t('strata_column'))
        self.strata_combo = QtWidgets.QComboBox()
        self.options_layout.addRow(self.strata_label, self.strata_combo)

        # Value Column Components
        self.value_label = QtWidgets.QLabel(self.t('value_column'))
        self.value_combo = QtWidgets.QComboBox()
        self.options_layout.addRow(self.value_label, self.value_combo)

        # Use Threshold Components
        self.use_threshold_checkbox = QtWidgets.QCheckBox(
            self.t('use_threshold_value'))
        self.use_threshold_checkbox.toggled.connect(
            self.toggle_threshold_input)
        self.threshold_label = QtWidgets.QLabel(self.t('threshold_value'))
        self.threshold_input = QtWidgets.QLineEdit()
        self.threshold_label.setVisible(False)
        self.threshold_input.setVisible(False)
        self.options_layout.addRow(self.use_threshold_checkbox)
        self.options_layout.addRow(self.threshold_label, self.threshold_input)

        # Use Stratification Components
        self.use_stratify_checkbox = QtWidgets.QCheckBox(
            self.t('use_stratification'))
        self.use_stratify_checkbox.toggled.connect(
            self.toggle_stratify_input)
        self.mus_strata_label = QtWidgets.QLabel(
            self.t('strata_column_for_mus'))
        self.mus_strata_combo = QtWidgets.QComboBox()
        self.mus_strata_label.setVisible(False)
        self.mus_strata_combo.setVisible(False)
        self.options_layout.addRow(self.use_stratify_checkbox)
        self.options_layout.addRow(
            self.mus_strata_label, self.mus_strata_combo)

        # Define Column Types Button
        self.column_types_button = QtWidgets.QPushButton(
            self.t('define_column_types'))
        self.column_types_button.clicked.connect(self.define_column_types)
        self.options_layout.addRow(self.column_types_button)

        # Preprocessing Completion Label
        self.preprocess_label = QtWidgets.QLabel(
            self.t('data_preprocessing_completed'))
        self.preprocess_label.setStyleSheet("color: green;")
        self.preprocess_label.setVisible(False)
        self.options_layout.addRow(self.preprocess_label)

        # Spacer to prevent layout shrinking
        self.options_layout.addItem(
            QtWidgets.QSpacerItem(0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding))

        main_layout.addWidget(self.options_group)

        # Create Sample Button
        self.create_button = QtWidgets.QPushButton(
            self.t('create_sample'))
        self.create_button.clicked.connect(self.create_sample)
        self.create_button.setStyleSheet(
            "background-color: #444444; color: #ffffff;")
        main_layout.addWidget(self.create_button)

        # Status Label to display messages to the user
        self.status_label = QtWidgets.QLabel()
        main_layout.addWidget(self.status_label)

        # Result Label to display the outcome of sample creation
        self.result_label = QtWidgets.QLabel()
        self.result_label.setStyleSheet("color: green;")
        main_layout.addWidget(self.result_label)

        # Progress Bar to indicate ongoing operations
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Update UI texts based on the current language
        self.update_ui_language()
        self.on_method_change()  # Adjust UI based on the default sampling method

        # Bring window to front and activate it
        self.raise_()
        self.activateWindow()

    def switch_language(self):
        """
        Toggle the application's language between Ukrainian and English.

        Updates the language button text and refreshes all UI elements to reflect the selected language.
        """
        if self.language == 'ua':
            self.language = 'en'
            self.language_button.setText("UA")
        else:
            self.language = 'ua'
            self.language_button.setText("EN")
        self.update_ui_language()

    def update_ui_language(self):
        """
        Update all UI elements with the appropriate translations based on the current language.

        This includes updating method descriptions, labels, buttons, and other textual components.
        """
        # Update method descriptions with translated text
        for idx, radio_button in enumerate(self.method_buttons):
            key = idx + 1
            method = self.sampling_methods[key]
            if self.language == 'ua':
                text = f"{method['name_ua']}: {method['description_ua']}"
            else:
                text = f"{method['name_en']}: {method['description_en']}"
            radio_button.setText(text)

        # Update labels and buttons with translated text
        self.setWindowTitle(self.t('window_title'))
        self.method_group.setTitle(self.t('select_sampling_method'))
        self.file_label.setText(self.t('file_with_population_data'))
        self.file_button.setText(self.t('browse'))
        self.sample_size_label.setText(self.t('sample_size'))
        self.strata_label.setText(self.t('strata_column'))
        self.value_label.setText(self.t('value_column'))
        self.use_threshold_checkbox.setText(self.t('use_threshold_value'))
        self.threshold_label.setText(self.t('threshold_value'))
        self.use_stratify_checkbox.setText(self.t('use_stratification'))
        self.mus_strata_label.setText(self.t('strata_column_for_mus'))
        self.column_types_button.setText(self.t('define_column_types'))
        self.preprocess_label.setText(self.t('data_preprocessing_completed'))
        self.create_button.setText(self.t('create_sample'))
        self.status_label.setText("")

    def browse_file(self):
        """
        Open a file dialog for the user to select a population data file.

        Supported file types include CSV, Excel, DBF, JSON, and Parquet.
        Once a file is selected, initiate the file loading process in a separate thread.
        """
        file_dialog = QtWidgets.QFileDialog()
        file_types = "All Files (*);;CSV Files (*.csv);;Excel Files (*.xls *.xlsx);;DBF Files (*.dbf);;JSON Files (*.json);;Parquet Files (*.parquet)"
        file_path, _ = file_dialog.getOpenFileName(
            self, self.t('browse'), "", file_types)
        if file_path:
            self.file_path.setText(file_path)
            self.status_label.setText(self.t('loading_file'))
            QtCore.QCoreApplication.processEvents()

            # Create a worker thread to load the file without blocking the UI
            self.file_loader_worker = FileLoaderWorker(file_path)
            self.file_loader_thread = QtCore.QThread()
            self.file_loader_worker.moveToThread(self.file_loader_thread)
            self.file_loader_thread.started.connect(
                self.file_loader_worker.run)
            self.file_loader_worker.finished.connect(
                self.file_loader_thread.quit)
            self.file_loader_worker.finished.connect(
                self.file_loader_worker.deleteLater)
            self.file_loader_thread.finished.connect(
                self.file_loader_thread.deleteLater)

            # Connect signals to handle the loaded data or errors
            self.file_loader_worker.result_ready.connect(
                self.handle_file_loaded)
            self.file_loader_worker.error.connect(self.handle_file_error)
            self.file_loader_thread.start()

    def populate_column_dropdowns(self):
        """
        Populate the column selection dropdowns based on the loaded data.

        Numerical columns are added to the value column dropdown, while all columns are added to the strata dropdowns.
        """
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
                self, self.t('error'), f"{self.t('error_processing_data')}: {e}")

    def on_method_change(self):
        """
        Adjust the visibility of UI components based on the selected sampling method.

        Different sampling methods require different parameters; this method ensures that only relevant UI elements are visible.
        """
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
            # Stratified Sampling requires strata column selection
            self.strata_label.setVisible(True)
            self.strata_combo.setVisible(True)
        elif choice == 4:
            # Monetary Unit Sampling requires value column and optional threshold
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
            # Advanced sampling methods require defining column types
            self.column_types_button.setVisible(True)
            if self.data_preprocessed is not None:
                self.preprocess_label.setVisible(True)

    def toggle_threshold_input(self):
        """
        Show or hide the threshold input fields based on the checkbox state.

        When the user opts to use a threshold value, the corresponding input fields become visible.
        """
        self.use_threshold = self.use_threshold_checkbox.isChecked()
        self.threshold_label.setVisible(self.use_threshold)
        self.threshold_input.setVisible(self.use_threshold)

    def toggle_stratify_input(self):
        """
        Show or hide the stratification column selection based on the checkbox state.

        When the user opts to use stratification, the corresponding dropdown becomes visible.
        """
        self.use_stratify = self.use_stratify_checkbox.isChecked()
        self.mus_strata_label.setVisible(self.use_stratify)
        self.mus_strata_combo.setVisible(self.use_stratify)

    def define_column_types(self):
        """
        Open dialogs for the user to define numerical and categorical columns.

        This method ensures that the data is properly preprocessed before sampling.
        """
        if self.data is None:
            QtWidgets.QMessageBox.critical(
                self, self.t('error'), self.t('first_load_data'))
            return

        def select_columns(column_type: str):
            """
            Open a dialog for the user to select columns of a specific type.

            Args:
                column_type (str): The type of columns to select ('numerical' or 'categorical').
            """
            columns = self.data.columns.tolist()
            if column_type == "numerical":
                # Filter columns that are numerical
                columns = [col for col in self.data.columns if pd.api.types.is_numeric_dtype(
                    self.data[col])]
            elif column_type == "categorical":
                # Exclude numerical columns to get categorical columns
                columns = [
                    col for col in self.data.columns if col not in self.numerical_columns]

            if not columns:
                QtWidgets.QMessageBox.critical(
                    self, self.t('error'), self.t('in_dataframe_no_columns_of_type').format(column_type=self.translate_column_type(column_type)))
                return

            # Create a dialog for column selection
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle(
                self.t('select_columns_of_type').format(column_type=self.translate_column_type(column_type)))
            dialog_layout = QtWidgets.QVBoxLayout()
            dialog.setLayout(dialog_layout)

            label = QtWidgets.QLabel(
                self.t('select_columns_of_type').format(column_type=self.translate_column_type(column_type)))
            dialog_layout.addWidget(label)

            list_widget = QtWidgets.QListWidget()
            list_widget.setSelectionMode(
                QtWidgets.QAbstractItemView.MultiSelection)
            list_widget.addItems(columns)
            dialog_layout.addWidget(list_widget)

            # Use 'Browse' button to confirm selection
            button = QtWidgets.QPushButton(self.t('browse'))
            button.clicked.connect(dialog.accept)
            dialog_layout.addWidget(button)

            if dialog.exec():
                # Retrieve selected columns
                selected_columns = [item.text()
                                    for item in list_widget.selectedItems()]
                if column_type == "numerical":
                    self.numerical_columns = selected_columns
                elif column_type == "categorical":
                    self.categorical_columns = selected_columns
                if not selected_columns:
                    QtWidgets.QMessageBox.warning(
                        self, self.t('warning'), self.t('no_columns_selected_of_type').format(column_type=self.translate_column_type(column_type)))
                if column_type == "numerical":
                    # After selecting numerical columns, prompt for categorical
                    select_columns("categorical")
                else:
                    # After selecting categorical columns, start preprocessing
                    self.status_label.setText(
                        self.t('data_preprocessing'))
                    self.preprocessing_worker = PreprocessingWorker(
                        self.data, self.numerical_columns, self.categorical_columns)
                    self.preprocessing_thread = QtCore.QThread()
                    self.preprocessing_worker.moveToThread(
                        self.preprocessing_thread)
                    self.preprocessing_thread.started.connect(
                        self.preprocessing_worker.run)
                    self.preprocessing_worker.finished.connect(
                        self.preprocessing_thread.quit)
                    self.preprocessing_worker.finished.connect(
                        self.preprocessing_worker.deleteLater)
                    self.preprocessing_thread.finished.connect(
                        self.preprocessing_thread.deleteLater)

                    # Connect signals to handle preprocessing results or errors
                    self.preprocessing_worker.result_ready.connect(
                        self.handle_preprocessing_result)
                    self.preprocessing_worker.error.connect(
                        self.handle_preprocessing_error)
                    self.preprocessing_thread.start()

        # Start by selecting numerical columns
        select_columns("numerical")

    def create_sample(self):
        """
        Initiate the sample creation process based on the selected sampling method and parameters.

        This method handles input validation, prepares sampling parameters, and starts the sampling worker thread.
        """
        self.create_button.setEnabled(False)
        self.status_label.setText(self.t('creating_sample'))
        self.result_label.setText("")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QtCore.QCoreApplication.processEvents()

        # Retrieve selected sampling method
        choice = self.method_button_group.checkedId()
        file_path = self.file_path.text()
        try:
            sample_size = int(self.sample_size_input.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(
                self, self.t('error'), self.t('error_processing_data'))
            self.create_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        population = self.data.copy()
        self.data = population

        dataset_size = population.shape[0]
        # Check dataset size limits for specific sampling methods
        if choice == 9 and dataset_size > 300000:
            QtWidgets.QMessageBox.warning(self, self.t('error'), self.t(
                'hdbscan_limit_error'))
            self.create_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return
        if choice == 6 and dataset_size > 1000000:
            QtWidgets.QMessageBox.warning(self, self.t('error'), self.t(
                'lof_limit_error'))
            self.create_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            return

        # Retrieve the sampling function and parameters based on the selected method
        method_info = self.sampling_methods.get(choice)
        sampling_function = self.get_sampling_function(choice)
        kwargs = self.get_sampling_parameters(choice)

        # Create a worker and thread for sampling to prevent UI blocking
        self.worker = Worker(sampling_function, kwargs, choice,
                             method_info, file_path)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Connect signals to handle sampling results or errors
        self.worker.error.connect(self.handle_worker_error)
        self.worker.progress.connect(self.update_progress_bar)
        self.worker.result_ready.connect(self.process_sampling_result)
        self.thread.start()

    def get_sampling_function(self, choice: int):
        """
        Retrieve the sampling function based on the selected method.

        Args:
            choice (int): The ID of the selected sampling method.

        Returns:
            Callable: The corresponding sampling function.
        """
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

    def get_sampling_parameters(self, choice: int) -> dict:
        """
        Prepare the parameters required for the selected sampling method.

        Args:
            choice (int): The ID of the selected sampling method.

        Returns:
            dict: A dictionary of parameters for the sampling function.
        """
        params = {}
        params["random_seed"] = random.randint(1, 10000)
        params["sample_size"] = int(self.sample_size_input.text())
        params["data"] = self.data  # Include data here

        if choice == 3:
            # Stratified Sampling requires a strata column
            params["strata_column"] = self.strata_combo.currentText()
        elif choice == 4:
            # Monetary Unit Sampling requires value column and optional threshold
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
            # Advanced sampling methods require preprocessed data and selected features
            if not self.numerical_columns and not self.categorical_columns:
                raise ValueError(
                    self.t("no_columns_of_type_selected").format(
                        column_type='')
                )
            if not hasattr(self, "data_preprocessed") or self.data_preprocessed is None:
                raise ValueError(
                    self.t('data_preprocessing_not_done')
                )
            params["data_preprocessed"] = self.data_preprocessed
            params["features"] = self.numerical_columns + \
                self.categorical_columns
        return params

    @QtCore.Slot(pd.DataFrame)
    def handle_file_loaded(self, data: pd.DataFrame):
        """
        Handle the event when the file is successfully loaded.

        Args:
            data (pd.DataFrame): The loaded data as a pandas DataFrame.
        """
        self.data = data
        self.populate_column_dropdowns()
        shape_text = f"{self.t('file_size')}: {self.data.shape}"
        self.file_shape_label.setText(shape_text)
        self.status_label.setText("")

    @QtCore.Slot(str)
    def handle_file_error(self, error_message: str):
        """
        Handle file loading errors by displaying an error message to the user.

        Args:
            error_message (str): The error message describing what went wrong.
        """
        self.status_label.setText("")
        QtWidgets.QMessageBox.critical(
            self, self.t('error'), f"{self.t('file_loading_error')}: {error_message}")

    @QtCore.Slot(object)
    def handle_preprocessing_result(self, result: tuple):
        """
        Handle the result of the data preprocessing step.

        Args:
            result (tuple): A tuple containing the preprocessed data and a description of the preprocessing method.
        """
        self.data_preprocessed, self.preprocessing_method_description = result
        self.preprocess_label.setVisible(True)
        self.status_label.setText("")

    @QtCore.Slot(str)
    def handle_preprocessing_error(self, error_message: str):
        """
        Handle errors that occur during data preprocessing.

        Args:
            error_message (str): The error message describing what went wrong.
        """
        self.status_label.setText("")
        QtWidgets.QMessageBox.critical(
            self, self.t('error'), f"{self.t('error_processing_data')}: {error_message}")

    @QtCore.Slot(int)
    def update_progress_bar(self, value: int):
        """
        Update the progress bar's value.

        Args:
            value (int): The new value for the progress bar.
        """
        self.progress_bar.setValue(value)

    @QtCore.Slot(str)
    def handle_worker_error(self, error_message: str):
        """
        Handle errors that occur during the sampling process.

        Args:
            error_message (str): The error message describing what went wrong.
        """
        self.create_button.setEnabled(True)
        self.status_label.setText("")
        self.progress_bar.setVisible(False)
        QtWidgets.QMessageBox.critical(
            self, self.t('error'), error_message)

    @QtCore.Slot(object)
    def process_sampling_result(self, result_tuple: tuple):
        """
        Process the result returned by the sampling worker.

        Args:
            result_tuple (tuple): A tuple containing the sampling results and related information.
        """
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
                    f"{self.t('error_creating_sample')}\n{sampling_method_description}"
                )

            # Prepare file paths for saving results
            file_name, file_ext = os.path.splitext(file_path)
            sample_type = method_info["name_en"].lower().replace(" ", "_")
            output_path_en = f"{file_name}_{sample_type}.pdf"

            sample_output_path = f"{file_name}_{sample_type}_sample.csv"
            population_output_path = f"{file_name}_{sample_type}_population.csv"

            # Save the population and sample data to CSV files
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

            # Create charts based on the sampling method
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
                    self.t("data_preprocessing"))
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
                # Finalize the process for sampling methods that do not require visualization
                self.finalize_process(choice, method_info, output_path_en,
                                      sampling_method_description, chart_paths, file_name, sample_type, None)

        except Exception as e:
            logger.exception("Error occurred while processing result")
            QtWidgets.QMessageBox.critical(
                self, self.t('error'), str(e))

    @QtCore.Slot()
    def handle_visualization_error(self, error_message: str):
        """
        Handle errors that occur during the visualization process.

        Args:
            error_message (str): The error message describing what went wrong.
        """
        self.create_button.setEnabled(True)
        self.status_label.setText("")
        QtWidgets.QMessageBox.critical(
            self, self.t('error'), self.t('visualization_error') + ": " + error_message)

    def finalize_process(self, choice: int, method_info: dict, output_path_en: str, sampling_method_description: str, chart_paths: list, file_name: str, sample_type: str, best_study: object):
        """
        Finalize the sampling process by generating visualizations and creating a PDF report.

        Args:
            choice (int): The ID of the selected sampling method.
            method_info (dict): Information about the selected sampling method.
            output_path_en (str): The path where the PDF report will be saved.
            sampling_method_description (str): Description of the sampling method.
            chart_paths (list): List of file paths for generated charts.
            file_name (str): Base name of the input file.
            sample_type (str): Type of sampling method in lowercase with underscores.
            best_study (object): Best study object from sampling (if applicable).
        """
        self.create_button.setEnabled(True)
        self.status_label.setText("")

        # Add UMAP visualization to chart_paths if applicable
        if choice in (5, 6, 7, 8, 9):
            umap_projection_path = (
                f"{file_name}_{sample_type}_umap_projection.png"
            )
            chart_paths.append(umap_projection_path)

            if choice in (7, 9) and best_study:
                # Generate and add Optuna results visualizations
                base_optuna_results_path = (
                    f"{file_name}_{sample_type}_optuna_results"
                )
                visualize_optuna_results(
                    best_study, base_optuna_results_path)
                pattern = f"{file_name}_{sample_type}_optuna_results*.png"
                optuna_result_files = glob.glob(pattern)
                chart_paths.extend(optuna_result_files)

        # Start PDF generation in a separate thread
        self.status_label.setText(self.t('creating_sample'))
        self.create_button.setEnabled(False)

        self.pdf_worker = PdfGenerationWorker(
            output_path_en,
            sampling_method_description,
            self.preprocessing_method_description,
            chart_paths,
            language='en'  # Assuming PDF is always in English; adjust if needed
        )
        self.pdf_thread = QtCore.QThread()
        self.pdf_worker.moveToThread(self.pdf_thread)
        self.pdf_thread.started.connect(self.pdf_worker.run)
        self.pdf_worker.finished.connect(self.pdf_thread.quit)
        self.pdf_worker.finished.connect(self.pdf_worker.deleteLater)
        self.pdf_thread.finished.connect(self.pdf_thread.deleteLater)

        # Connect signals to handle PDF generation completion or errors
        self.pdf_worker.error.connect(self.handle_pdf_error)
        self.pdf_worker.finished.connect(self.handle_pdf_finished)
        self.pdf_thread.start()

    @QtCore.Slot()
    def handle_pdf_finished(self):
        """
        Handle the completion of the PDF generation process.

        Displays a success message with the path to the generated PDF.
        """
        self.create_button.setEnabled(True)
        self.status_label.setText("")
        message = f"{self.t('sample_saved_in_file')}:\n{self.pdf_worker.output_path}"
        self.result_label.setText(message)

    @QtCore.Slot(str)
    def handle_pdf_error(self, error_message: str):
        """
        Handle errors that occur during the PDF generation process.

        Args:
            error_message (str): The error message describing what went wrong.
        """
        self.create_button.setEnabled(True)
        self.status_label.setText("")
        QtWidgets.QMessageBox.critical(self, self.t('error'), error_message)

    def closeEvent(self, event):
        """
        Handle the window close event to ensure all threads are properly terminated.

        Args:
            event (QCloseEvent): The close event.
        """
        # Wait for worker threads to finish before closing
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        if hasattr(self, 'visualization_thread') and self.visualization_thread.isRunning():
            self.visualization_thread.quit()
            self.visualization_thread.wait()
        if hasattr(self, 'pdf_thread') and self.pdf_thread.isRunning():
            self.pdf_thread.quit()
            self.pdf_thread.wait()
        event.accept()


"""
1. **Class `SamplingApp`**:
    - **Purpose**: Serves as the main window for the application, managing the user interface and orchestrating the sampling process.
    - **Attributes**:
        - `language`: Current language of the UI (`'ua'` for Ukrainian, `'en'` for English).
        - `translations`: Dictionary containing translations for UI elements.
        - `column_type_translations`: Dictionary for translating column types.
        - `data`: Loaded population data.
        - `data_preprocessed`: Data after preprocessing.
        - `numerical_columns`: Selected numerical columns for analysis.
        - `categorical_columns`: Selected categorical columns for analysis.
        - `use_threshold`: Flag indicating whether to use a threshold value.
        - `use_stratify`: Flag indicating whether to use stratification.
        - `preprocessing_method_description`: Description of the preprocessing steps applied.
        - `widgets`: Dictionary to store references to various UI widgets.
        - `sampling_methods`: Dictionary defining available sampling methods with their names and descriptions.

2. **Methods**:
    - **`__init__`**: Initializes the application, sets up translations, sampling methods, and the UI.
    - **`t`**: Helper method to retrieve translated text based on the current language.
    - **`translate_column_type`**: Translates column types (`'numerical'` or `'categorical'`) based on the current language.
    - **`init_ui`**: Constructs the user interface, including layouts, widgets, and styles.
    - **`switch_language`**: Toggles the application's language between Ukrainian and English.
    - **`update_ui_language`**: Updates all UI elements to reflect the selected language.
    - **`browse_file`**: Opens a file dialog for the user to select a data file and initiates the file loading process.
    - **`populate_column_dropdowns`**: Populates dropdown menus with column names from the loaded data.
    - **`on_method_change`**: Adjusts visible UI components based on the selected sampling method.
    - **`toggle_threshold_input`**: Shows or hides threshold input fields based on user selection.
    - **`toggle_stratify_input`**: Shows or hides stratification column selection based on user selection.
    - **`define_column_types`**: Opens dialogs for the user to select numerical and categorical columns and starts data preprocessing.
    - **`create_sample`**: Initiates the sample creation process, handling parameter validation and starting the sampling worker.
    - **`get_sampling_function`**: Retrieves the appropriate sampling function based on user selection.
    - **`get_sampling_parameters`**: Prepares parameters required for the selected sampling method.
    - **`handle_file_loaded`**: Processes the loaded data, updating UI elements accordingly.
    - **`handle_file_error`**: Displays an error message if file loading fails.
    - **`handle_preprocessing_result`**: Handles successful data preprocessing results.
    - **`handle_preprocessing_error`**: Handles errors that occur during data preprocessing.
    - **`update_progress_bar`**: Updates the progress bar's value during long-running operations.
    - **`handle_worker_error`**: Handles errors that occur during the sampling process.
    - **`process_sampling_result`**: Processes the results from the sampling worker, including saving data and generating visualizations.
    - **`handle_visualization_error`**: Handles errors that occur during the visualization process.
    - **`finalize_process`**: Completes the sampling process by generating visualizations and creating a PDF report.
    - **`handle_pdf_finished`**: Handles successful PDF generation by notifying the user.
    - **`handle_pdf_error`**: Handles errors that occur during PDF generation.
    - **`closeEvent`**: Ensures that all worker threads are properly terminated when the application window is closed.

3. **Worker Classes**:
    - **`Worker`**: Handles the execution of the sampling function in a separate thread.
    - **`FileLoaderWorker`**: Handles loading data files in a separate thread to prevent UI blocking.
    - **`PreprocessingWorker`**: Manages data preprocessing tasks in a separate thread.
    - **`VisualizationWorker`**: Handles the creation of visualizations in a separate thread.
    - **`PdfGenerationWorker`**: Manages the generation of PDF reports in a separate thread.

4. **Utility Functions**:
    - **`create_strata_chart`**: Generates a chart visualizing strata in the population data.
    - **`create_cumulative_chart`**: Creates a cumulative chart based on monetary values.
    - **`create_umap_projection`**: Generates a UMAP projection for dimensionality reduction and visualization.
    - **`visualize_optuna_results`**: Visualizes results from Optuna hyperparameter optimization studies.

### Best Practices Followed

- **Threading**: Long-running operations such as file loading, data preprocessing, sampling, visualization, and PDF generation are handled in separate threads to keep the UI responsive.
  
- **Error Handling**: Comprehensive error handling ensures that users are informed of any issues that occur during processing, enhancing the user experience and aiding in debugging.
  
- **Internationalization (i18n)**: The application supports both Ukrainian and English languages, making it accessible to a broader user base.
  
- **Modular Design**: Separation of concerns is maintained by dividing functionalities into workers and utility functions, improving code maintainability.
  
- **User Feedback**: Progress bars, status labels, and result messages provide real-time feedback to users, keeping them informed about the application's state.
"""

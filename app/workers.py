import os
import logging
import pandas as pd
from PySide6 import QtCore
from utils.preprocessing import preprocess_data
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

try:
    from dbfread import DBF
except ImportError:
    DBF = None

# Configure logging to output debug information with timestamps and log levels
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Worker(QtCore.QObject):
    """
    Worker class to handle sampling operations in a separate thread.

    This class executes the sampling function with the provided parameters
    and emits signals upon completion, error, or progress updates.
    """

    # Define signals
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)  # Emit progress percentage
    result_ready = QtCore.Signal(object)

    def __init__(self, sampling_function, params, choice, method_info, file_path, language='ua'):
        """
        Initialize the Worker instance.

        Args:
            sampling_function (Callable): The sampling function to execute.
            params (dict): Parameters for the sampling function.
            choice (int): Identifier for the selected sampling method.
            method_info (dict): Information about the sampling method.
            file_path (str): Path to the input data file.
            language (str, optional): Language for error messages ('ua' or 'en'). Defaults to 'ua'.
        """
        super().__init__()
        self.sampling_function = sampling_function
        self.params = params
        self.choice = choice
        self.method_info = method_info
        self.file_path = file_path
        self.language = language

        # Translation dictionaries for error messages
        self.translations = {
            'ua': {
                'error_in_worker_thread': 'Помилка в робочому потоці',
            },
            'en': {
                'error_in_worker_thread': 'Error occurred in worker thread',
            }
        }

    def t(self, key):
        """
        Retrieve the translated text based on the current language and key.

        Args:
            key (str): The key for the desired translation.

        Returns:
            str: Translated string corresponding to the key.
        """
        return self.translations[self.language].get(key, key)

    @QtCore.Slot()
    def run(self):
        """
        Execute the sampling function and emit appropriate signals.

        This method runs in a separate thread to prevent blocking the UI.
        It handles different sampling methods based on the choice and emits
        the results or errors accordingly.
        """
        try:
            choice = self.choice
            sampling_function = self.sampling_function
            params = self.params

            # Progress callback to emit progress updates
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

            # Emit the result ready signal with relevant data
            self.result_ready.emit((result, self.method_info,
                                    self.file_path, self.choice))
        except Exception as e:
            logger.exception(self.t('error_in_worker_thread'))
            # Emit error signal with translated error message
            self.error.emit(f"{self.t('error_in_worker_thread')}: {str(e)}")
        finally:
            # Emit finished signal regardless of success or failure
            self.finished.emit()


class FileLoaderWorker(QtCore.QObject):
    """
    Worker class to handle file loading operations in a separate thread.

    This class reads various file types, attempts to detect encodings and delimiters,
    and emits signals upon successful loading or errors. It is designed to prevent
    blocking the main UI thread by performing file operations asynchronously.

    Attributes:
        file_path (str): Path to the input data file.
        language (str): Language code for error messages ('ua' or 'en').
        translations (dict): Dictionary containing translations for error messages.

    Signals:
        finished: Emitted when the loading process is finished, regardless of success.
        error (str): Emitted when an error occurs, carrying the error message.
        result_ready (pd.DataFrame): Emitted when data is successfully loaded, carrying the DataFrame.
    """

    # Signals
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    result_ready = QtCore.Signal(pd.DataFrame)

    def __init__(self, file_path, language='ua'):
        """
        Initialize the FileLoaderWorker instance.

        Args:
            file_path (str): Path to the input data file.
            language (str, optional): Language for error messages ('ua' or 'en'). Defaults to 'ua'.
        """
        super().__init__()
        self.file_path = file_path
        self.language = language

        # Translation dictionaries for error messages
        self.translations = {
            'ua': {
                'dbf_module_not_installed': "Модуль 'dbfread' не встановлено.",
                'error_loading_file': "Помилка при завантаженні файлу",
            },
            'en': {
                'dbf_module_not_installed': "Module 'dbfread' is not installed.",
                'error_loading_file': "Error loading file",
            }
        }

    def t(self, key):
        """
        Retrieve the translated text based on the current language and key.

        Args:
            key (str): The key for the desired translation.

        Returns:
            str: Translated string corresponding to the key.
        """
        return self.translations[self.language].get(key, key)

    @QtCore.Slot()
    def run(self):
        """
        Execute the file loading process and emit appropriate signals.

        This method determines the file type based on its extension and calls
        the corresponding method to read the file. It handles exceptions and
        emits signals accordingly.

        Emits:
            result_ready (pd.DataFrame): When data is successfully loaded.
            error (str): When an error occurs during file loading.
            finished: After the loading process is completed, regardless of success.
        """
        try:
            # Determine file type by extension
            _, file_extension = os.path.splitext(self.file_path)
            file_extension = file_extension.lower()

            if file_extension == '.csv':
                data = self.read_csv_file(self.file_path)
            elif file_extension == '.xls':
                data = pd.read_excel(self.file_path, engine='xlrd')
            elif file_extension == '.xlsx':
                data = pd.read_excel(self.file_path, engine='openpyxl')
            elif file_extension == '.dbf':
                data = self.read_dbf_file(self.file_path)
            elif file_extension == '.json':
                data = pd.read_json(self.file_path)
            elif file_extension == '.parquet':
                data = pd.read_parquet(self.file_path)
            else:
                # Attempt to read as CSV with common encodings and delimiters
                data = self.read_csv_file(self.file_path)

            # Emit the loaded data
            self.result_ready.emit(data)
        except Exception as e:
            logger.exception(self.t('error_loading_file'))
            # Emit error signal with translated error message
            self.error.emit(f"{self.t('error_loading_file')}: {str(e)}")
        finally:
            # Emit finished signal regardless of success or failure
            self.finished.emit()

    def read_csv_file(self, file_path):
        """
        Read a CSV file using common encodings and delimiters.

        This method attempts to read the CSV file using combinations of common
        encodings and delimiters until it succeeds or exhausts all options.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the CSV data.

        Raises:
            Exception: If the file cannot be read with any of the common encodings and delimiters.
        """
        encodings = ['utf-8', 'cp1251', 'cp1252', 'latin1']
        delimiters = [',', ';', '\t', '|']

        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    chunk_iter = pd.read_csv(
                        file_path, encoding=encoding, delimiter=delimiter, chunksize=100000
                    )
                    df = pd.concat(chunk_iter, ignore_index=True)
                    if not df.empty and len(df.columns) > 1:
                        return df
                except Exception:
                    continue
        raise Exception(
            "Failed to read CSV file with common encodings and delimiters.")

    def read_dbf_file(self, file_path):
        """
        Read a DBF file and convert it to a pandas DataFrame.

        Args:
            file_path (str): Path to the DBF file.

        Returns:
            pd.DataFrame: DataFrame containing the DBF data.

        Raises:
            ImportError: If the 'dbfread' module is not installed.
            Exception: If an error occurs while reading the DBF file.
        """
        if DBF is None:
            raise ImportError(self.t('dbf_module_not_installed'))

        try:
            # Read DBF file using dbfread
            table = DBF(file_path, ignore_missing_memofile=True)
            df = pd.DataFrame(iter(table))
            return df
        except Exception as e:
            raise Exception(f"Error reading DBF file: {e}")


class PreprocessingWorker(QtCore.QObject):
    """
    Worker class to handle data preprocessing in a separate thread.

    This class preprocesses the data based on selected numerical and categorical columns
    and emits signals upon successful preprocessing or errors.
    """

    # Signals
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    result_ready = QtCore.Signal(object)

    def __init__(self, data, numerical_columns, categorical_columns, language='ua'):
        """
        Initialize the PreprocessingWorker instance.

        Args:
            data (pd.DataFrame): The raw data to preprocess.
            numerical_columns (list): List of selected numerical columns.
            categorical_columns (list): List of selected categorical columns.
            language (str, optional): Language for error messages ('ua' or 'en'). Defaults to 'ua'.
        """
        super().__init__()
        self.data = data
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.language = language

        # Translation dictionaries for error messages
        self.translations = {
            'ua': {
                'error_in_data_preprocessing': "Помилка при передобробці даних",
            },
            'en': {
                'error_in_data_preprocessing': "Error during data preprocessing",
            }
        }

    def t(self, key):
        """
        Retrieve the translated text based on the current language and key.

        Args:
            key (str): The key for the desired translation.

        Returns:
            str: Translated string corresponding to the key.
        """
        return self.translations[self.language].get(key, key)

    @QtCore.Slot()
    def run(self):
        """
        Execute the data preprocessing process and emit appropriate signals.

        This method runs in a separate thread to prevent blocking the UI.
        It uses the `preprocess_data` utility function to preprocess the data.
        """
        try:
            # Preprocess the data using the provided utility function
            data_preprocessed, preprocessing_method_description = preprocess_data(
                self.data, self.numerical_columns, self.categorical_columns
            )
            # Emit the preprocessed data and description
            self.result_ready.emit(
                (data_preprocessed, preprocessing_method_description))
        except Exception as e:
            logger.exception(self.t('error_in_data_preprocessing'))
            # Emit error signal with translated error message
            self.error.emit(
                f"{self.t('error_in_data_preprocessing')}: {str(e)}")
        finally:
            # Emit finished signal regardless of success or failure
            self.finished.emit()


class VisualizationWorker(QtCore.QObject):
    """
    Worker class to handle visualization tasks in a separate thread.

    This class executes the visualization function with the provided arguments
    and emits signals upon completion or errors.
    """

    # Define signals
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, visualization_function, args, kwargs, language='ua'):
        """
        Initialize the VisualizationWorker instance.

        Args:
            visualization_function (Callable): The visualization function to execute.
            args (list): Positional arguments for the visualization function.
            kwargs (dict): Keyword arguments for the visualization function.
            language (str, optional): Language for error messages ('ua' or 'en'). Defaults to 'ua'.
        """
        super().__init__()
        self.visualization_function = visualization_function
        self.args = args
        self.kwargs = kwargs
        self.language = language

        # Translation dictionaries for error messages
        self.translations = {
            'ua': {
                'error_in_visualization_worker': "Помилка в робочому потоці візуалізації",
            },
            'en': {
                'error_in_visualization_worker': "Error occurred in visualization worker thread",
            }
        }

    def t(self, key):
        """
        Retrieve the translated text based on the current language and key.

        Args:
            key (str): The key for the desired translation.

        Returns:
            str: Translated string corresponding to the key.
        """
        return self.translations[self.language].get(key, key)

    @QtCore.Slot()
    def run(self):
        """
        Execute the visualization function and emit appropriate signals.

        This method runs in a separate thread to prevent blocking the UI.
        It handles the creation of visualizations and emits signals upon success or failure.
        """
        try:
            # Execute the visualization function with provided arguments
            self.visualization_function(*self.args, **self.kwargs)
            # Emit finished signal upon successful completion
            self.finished.emit()
        except Exception as e:
            logger.exception(self.t('error_in_visualization_worker'))
            # Emit error signal with translated error message
            self.error.emit(
                f"{self.t('error_in_visualization_worker')}: {str(e)}")


class PdfGenerationWorker(QtCore.QObject):
    """
    Worker class to handle PDF generation in a separate thread.

    This class creates a PDF report containing descriptions and visualizations
    based on the sampling results and emits signals upon completion or errors.
    """

    # Define signals
    finished = QtCore.Signal()
    error = QtCore.Signal(str)
    progress = QtCore.Signal(int)  # If you want to emit progress updates

    def __init__(self, output_path, sampling_method_description,
                 preprocessing_method_description, chart_paths, language='en'):
        """
        Initialize the PdfGenerationWorker instance.

        Args:
            output_path (str): Path where the PDF report will be saved.
            sampling_method_description (str): Description of the sampling method.
            preprocessing_method_description (str): Description of the data preprocessing.
            chart_paths (list): List of file paths for the generated charts.
            language (str, optional): Language for the PDF content ('ua' or 'en'). Defaults to 'en'.
        """
        super().__init__()
        self.output_path = output_path
        self.sampling_method_description = sampling_method_description
        self.preprocessing_method_description = preprocessing_method_description
        self.chart_paths = chart_paths
        self.language = language

        # Translation dictionaries for PDF content
        self.translations = {
            'ua': {
                'description_of_data_preprocessing_methods': "Опис методів передобробки даних:",
                'description_of_sampling_method': "Опис методу вибірки:",
                'chart': "Графік",
                'chart_not_found': "Графік {chart_name} не знайдено.",
            },
            'en': {
                'description_of_data_preprocessing_methods': "Description of data preprocessing methods:",
                'description_of_sampling_method': "Description of the sampling method:",
                'chart': "Chart",
                'chart_not_found': "Chart {chart_name} not found.",
            }
        }

    def t(self, key, **kwargs):
        """
        Retrieve the translated text based on the current language and key.

        Args:
            key (str): The key for the desired translation.
            **kwargs: Additional keyword arguments for string formatting.

        Returns:
            str: Translated and formatted string corresponding to the key.
        """
        return self.translations[self.language].get(key, key).format(**kwargs)

    @QtCore.Slot()
    def run(self):
        """
        Execute the PDF generation process and emit appropriate signals.

        This method runs in a separate thread to prevent blocking the UI.
        It uses ReportLab to create a PDF report containing descriptions and charts.
        """
        try:
            # Generate the PDF report
            self.generate_pdf()
            # Emit finished signal upon successful completion
            self.finished.emit()
        except Exception as e:
            logger.exception("Error occurred in PDF generation worker")
            # Emit error signal with the exception message
            self.error.emit(str(e))

    def generate_pdf(self):
        """
        Create a PDF report with sampling and preprocessing descriptions and visualizations.

        The PDF includes paragraphs describing the preprocessing and sampling methods,
        as well as images of the generated charts.
        """
        # Initialize the PDF document
        doc = SimpleDocTemplate(self.output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        flowables = []

        def convert_newlines(text):
            """
            Convert newline characters to HTML line breaks for proper formatting in PDF.

            Args:
                text (str): The text containing newline characters.

            Returns:
                str: The formatted text with HTML line breaks.
            """
            return text.replace("\n", "<br/>")

        # Define a custom paragraph style
        custom_style = ParagraphStyle(
            "Custom", parent=styles["Normal"], spaceAfter=12, leading=15
        )

        if self.preprocessing_method_description:
            # Add a heading for data preprocessing methods
            heading = self.t('description_of_data_preprocessing_methods')
            flowables.append(
                Paragraph(
                    heading,
                    styles["Heading2"],
                )
            )
            # Add the preprocessing description
            formatted_preprocess_desc = convert_newlines(
                self.preprocessing_method_description
            )
            flowables.append(
                Paragraph(formatted_preprocess_desc, custom_style)
            )
            flowables.append(Spacer(1, 12))

        if self.sampling_method_description:
            # Add a heading for the sampling method
            heading = self.t('description_of_sampling_method')
            flowables.append(
                Paragraph(
                    heading, styles["Heading2"]
                )
            )
            # Add the sampling method description
            formatted_sampling_desc = convert_newlines(
                self.sampling_method_description
            )
            flowables.append(
                Paragraph(formatted_sampling_desc, custom_style)
            )
            flowables.append(Spacer(1, 12))

        for chart_path in self.chart_paths:
            if os.path.exists(chart_path):
                # Add a heading for each chart
                heading = self.t('chart')
                flowables.append(
                    Paragraph(
                        f"{heading}: {os.path.basename(chart_path)}",
                        styles["Heading3"],
                    )
                )
                # Add the chart image to the PDF
                img = Image(chart_path, width=6 * inch, height=4 * inch)
                flowables.append(img)
                flowables.append(Spacer(1, 12))
            else:
                # Notify if a chart image is not found
                message = self.t('chart_not_found',
                                 chart_name=os.path.basename(chart_path))
                flowables.append(
                    Paragraph(
                        message,
                        styles["Normal"],
                    )
                )
                flowables.append(Spacer(1, 12))

        # Build the PDF with all flowables
        doc.build(flowables)

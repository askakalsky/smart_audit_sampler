import os
import random
import logging
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ml_sampling.isolation_forest import isolation_forest_sampling
from ml_sampling.lof import lof_sampling
from ml_sampling.kmeans import kmeans_sampling
from ml_sampling.autoencoder import autoencoder_sampling
from ml_sampling.hdbscan import hdbscan_sampling
from statistical_sampling.random import random_sampling
from statistical_sampling.systematic import systematic_sampling
from statistical_sampling.stratified import stratified_sampling
from statistical_sampling.monetary_unit import monetary_unit_sampling
from utils.visualization import create_strata_chart, create_cumulative_chart, create_umap_projection, visualize_optuna_results
from utils.preprocessing import preprocess_data
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import glob
import threading

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SamplingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Створення аудиторських вибірок")
        self.root.configure(bg="#f0f0f0")
        self.data = None
        self.data_preprocessed = None
        self.numerical_columns = []
        self.categorical_columns = []
        self.choice_var = tk.IntVar(value=1)
        self.use_threshold_var = tk.IntVar(value=0)
        self.use_stratify_var = tk.IntVar(value=0)
        self.preprocessing_method_description = ""

        self.sampling_methods = {
            1: {"name_ua": "Випадкова вибірка", "name_en": "Random Sampling", "description": "кожен елемент генеральної сукупності має рівну ймовірність потрапити у вибірку."},
            2: {"name_ua": "Систематична вибірка", "name_en": "Systematic Sampling", "description": "елементи вибираються з генеральної сукупності через рівні інтервали."},
            3: {"name_ua": "Стратифікована вибірка", "name_en": "Stratified Sampling", "description": "генеральна сукупність ділиться на страти (групи), і з кожної страти формується випадкова вибірка."},
            4: {"name_ua": "Метод грошової одиниці", "name_en": "Monetary Unit Sampling", "description": "ймовірність вибору елемента пропорційна його грошовій величині. Використовується для оцінки сумарної величини помилок."},
            5: {"name_ua": "Isolation Forest", "name_en": "Isolation Forest", "description": "алгоритм для виявлення аномалій на основі випадкових лісів."},
            6: {"name_ua": "Local Outlier Factor", "name_en": "Local Outlier Factor", "description": "метод для виявлення локальних аномалій у даних."},
            7: {"name_ua": "Кластеризація K-Means", "name_en": "K-Means Clustering", "description": "групування даних за схожістю для виявлення незвичайних точок."},
            8: {"name_ua": "Автоенкодер", "name_en": "Autoencoder", "description": "зменшення розмірності даних для виявлення відхилень через аналіз помилки відновлення."},
            9: {"name_ua": "HDBSCAN", "name_en": "HDBSCAN", "description": "знаходження аномалій, класифікуючи точки як шум, спираючись на їхню щільність і відстань до інших точок, що дозволяє виокремлювати викиди в даних."}
        }

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", foreground="black")
        style.configure("TButton", background="#cccccc", foreground="black",
                        borderwidth=2, relief="solid", font=("Arial", 10))
        style.map("TButton", background=[
                  ('active', '#aaaaaa')], relief=[('pressed', 'sunken')])
        style.configure("TCheckbutton", background="#f0f0f0",
                        foreground="black")
        style.configure("TCombobox", background="white", foreground="black")

        self.choice_frame = ttk.Frame(self.root, padding=(10, 10))
        self.choice_frame.grid(row=0, column=0, sticky="w")

        choice_label = ttk.Label(
            self.choice_frame,
            text="Оберіть тип вибірки:",
            font=("Arial", 12, "bold")
        )
        choice_label.grid(row=0, column=0, columnspan=2,
                          sticky="w", pady=(0, 5))

        for value, method_info in self.sampling_methods.items():
            rb = tk.Radiobutton(
                self.choice_frame,
                text=f"{method_info['name_ua']}: {method_info['description']}",
                variable=self.choice_var,
                value=value,
                command=self.on_choice_change,
                bg="#f0f0f0",
                font=("Arial", 8),
                wraplength=600,
                justify="left"
            )
            rb.grid(row=value, column=0, columnspan=2, sticky="w")

        self.options_frame = ttk.Frame(self.root, padding=(10, 10))
        self.options_frame.grid(row=1, column=0, sticky="w")

        file_path_label = ttk.Label(
            self.options_frame,
            text="Файл з генеральною сукупністю:"
        )
        file_path_label.grid(row=0, column=0, sticky="w")
        self.browse_button = ttk.Button(
            self.options_frame, text="Огляд", command=self.browse_file)
        self.browse_button.grid(row=0, column=1, sticky="w")
        self.file_path_entry = ttk.Entry(self.options_frame, width=50)
        self.file_path_entry.grid(row=1, column=0, columnspan=2, sticky="w")

        self.sample_size_label = ttk.Label(
            self.options_frame, text="Розмір вибірки:")
        self.sample_size_entry = ttk.Entry(self.options_frame)

        self.strata_column_label = ttk.Label(
            self.options_frame, text="Стовпець для стратифікації:")
        self.strata_column_combobox = ttk.Combobox(self.options_frame)

        self.value_column_label = ttk.Label(
            self.options_frame, text="Стовпець зі значеннями грошових одиниць:")
        self.value_column_combobox = ttk.Combobox(self.options_frame)

        self.use_threshold_label = ttk.Label(
            self.options_frame, text="Використовувати порогове значення?")
        self.use_threshold_checkbutton = ttk.Checkbutton(
            self.options_frame,
            variable=self.use_threshold_var,
            command=self.toggle_threshold_input
        )

        self.threshold_label = ttk.Label(
            self.options_frame, text="Порогове значення:")
        self.threshold_entry = ttk.Entry(self.options_frame)

        self.use_stratify_label = ttk.Label(
            self.options_frame, text="Використовувати стратифікацію?")
        self.use_stratify_checkbutton = ttk.Checkbutton(
            self.options_frame,
            variable=self.use_stratify_var,
            command=self.toggle_stratify_input
        )

        self.mus_strata_column_label = ttk.Label(
            self.options_frame, text="Стовпець для стратифікації:")
        self.mus_strata_column_combobox = ttk.Combobox(self.options_frame)

        self.column_types_button = ttk.Button(
            self.options_frame, text="Вказати типи колонок", command=self.define_column_types)
        self.preprocess_label = ttk.Label(
            self.options_frame, text="Передобробка даних виконана.", foreground='green')

        self.ai_parameters_frame = ttk.Frame(self.root, padding=(10, 10))

        self.result_frame = ttk.Frame(self.root, padding=(10, 10))
        self.result_frame.grid(row=2, column=0, sticky="w")

        self.result_label = ttk.Label(
            self.result_frame, text="", justify=tk.LEFT)
        self.result_label.grid(row=0, column=0, sticky="w")

        self.create_button = ttk.Button(
            self.root, text="Створити вибірку", command=self.create_sample)
        self.create_button.grid(row=3, column=1, sticky="se", padx=10, pady=10)

        self.on_choice_change()

    def toggle_threshold_input(self):
        if self.use_threshold_var.get():
            self.threshold_label.grid(row=7, column=0, sticky="w")
            self.threshold_entry.grid(row=7, column=1, sticky="w")
        else:
            self.threshold_label.grid_remove()
            self.threshold_entry.grid_remove()

    def toggle_stratify_input(self):
        if self.use_stratify_var.get():
            self.mus_strata_column_label.grid(row=9, column=0, sticky="w")
            self.mus_strata_column_combobox.grid(row=9, column=1, sticky="w")
        else:
            self.mus_strata_column_label.grid_remove()
            self.mus_strata_column_combobox.grid_remove()

    def define_column_types(self):
        if self.data is None:
            messagebox.showerror("Помилка", "Спочатку завантажте дані.")
            return

        def select_numerical_columns():
            numerical_columns = [
                col for col in self.data.columns if pd.api.types.is_numeric_dtype(self.data[col])]
            if not numerical_columns:
                messagebox.showerror(
                    "Помилка", "У датафреймі немає числових колонок.")
                return

            numerical_window = tk.Toplevel(self.root)
            numerical_window.title("Вибір числових колонок")
            tk.Label(numerical_window, text="Виберіть числові колонки:").pack(
                anchor="w", padx=10, pady=5)

            numerical_listbox = tk.Listbox(
                numerical_window, selectmode=tk.MULTIPLE, width=50)
            for col in numerical_columns:
                numerical_listbox.insert(tk.END, col)
            numerical_listbox.pack(padx=10, pady=5)

            def save_numerical_columns():
                selected_indices = numerical_listbox.curselection()
                self.numerical_columns = [numerical_columns[i]
                                          for i in selected_indices]
                if not self.numerical_columns:
                    messagebox.showwarning(
                        "Попередження", "Не обрано жодної числової колонки.")
                    return
                numerical_window.destroy()
                select_categorical_columns()

            save_button = ttk.Button(
                numerical_window, text="Далі", command=save_numerical_columns)
            save_button.pack(pady=10)

        def select_categorical_columns():
            categorical_columns = [
                col for col in self.data.columns if col not in self.numerical_columns]
            if not categorical_columns:
                messagebox.showinfo(
                    "Інформація", "Усі колонки вибрано як числові. Категоріальних колонок немає.")
                self.data_preprocessed, self.preprocessing_method_description = preprocess_data(
                    self.data, self.numerical_columns, self.categorical_columns)
                self.preprocess_label.grid(row=99, column=0, sticky="w")
                return

            categorical_window = tk.Toplevel(self.root)
            categorical_window.title("Вибір категоріальних колонок")
            tk.Label(categorical_window, text="Виберіть категоріальні колонки:").pack(
                anchor="w", padx=10, pady=5)

            categorical_listbox = tk.Listbox(
                categorical_window, selectmode=tk.MULTIPLE, width=50)
            for col in categorical_columns:
                categorical_listbox.insert(tk.END, col)
            categorical_listbox.pack(padx=10, pady=5)

            def save_categorical_columns():
                selected_indices = categorical_listbox.curselection()
                self.categorical_columns = [
                    categorical_columns[i] for i in selected_indices]
                self.data_preprocessed, self.preprocessing_method_description = preprocess_data(
                    self.data, self.numerical_columns, self.categorical_columns)
                self.preprocess_label.grid(row=99, column=0, sticky="w")
                categorical_window.destroy()

            save_button = ttk.Button(
                categorical_window, text="Зберегти", command=save_categorical_columns)
            save_button.pack(pady=10)

        select_numerical_columns()

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Виберіть файл з генеральною сукупністю",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            self.populate_column_dropdowns(file_path)
            self.data = pd.read_csv(file_path)

    def populate_column_dropdowns(self, file_path):
        try:
            df = pd.read_csv(file_path)
            numerical_columns = [
                col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            self.value_column_combobox['values'] = numerical_columns
            self.strata_column_combobox['values'] = list(df.columns)
            self.mus_strata_column_combobox['values'] = list(df.columns)
        except Exception as e:
            self.result_label.config(text=f"Помилка при читанні файлу: {e}")

    def toggle_options(self, choice):
        for widget in [
            self.sample_size_label,
            self.sample_size_entry,
            self.strata_column_label,
            self.strata_column_combobox,
            self.value_column_label,
            self.value_column_combobox,
            self.use_threshold_label,
            self.use_threshold_checkbutton,
            self.threshold_label,
            self.threshold_entry,
            self.use_stratify_label,
            self.use_stratify_checkbutton,
            self.mus_strata_column_label,
            self.mus_strata_column_combobox,
        ]:
            widget.grid_remove()

        if choice in self.sampling_methods:
            self.sample_size_label.grid(row=2, column=0, sticky="w")
            self.sample_size_entry.grid(row=3, column=0, sticky="w")

        if choice == 3:
            self.strata_column_label.grid(row=4, column=0, sticky="w")
            self.strata_column_combobox.grid(row=5, column=0, sticky="w")

        if choice == 4:
            self.value_column_label.grid(row=4, column=0, sticky="w")
            self.value_column_combobox.grid(row=5, column=0, sticky="w")
            self.use_threshold_label.grid(row=6, column=0, sticky="w")
            self.use_threshold_checkbutton.grid(
                row=6, column=1, sticky="w")
            self.toggle_threshold_input()
            self.use_stratify_label.grid(row=8, column=0, sticky="w")
            self.use_stratify_checkbutton.grid(row=8, column=1, sticky="w")
            self.toggle_stratify_input()

    def on_choice_change(self):
        choice = self.choice_var.get()

        self.toggle_options(choice)

        if choice in (5, 6, 7, 8, 9):
            self.column_types_button.grid(row=10, column=0, sticky="w")
            self.ai_parameters_frame.grid(row=11, column=0, sticky="w")
        else:
            self.column_types_button.grid_remove()
            self.ai_parameters_frame.grid_remove()
            self.preprocess_label.grid_remove()

    def create_sample(self):
        threading.Thread(target=self._create_sample).start()

    def _create_sample(self):
        sample = None
        try:
            choice = self.choice_var.get()
            file_path = self.file_path_entry.get()
            sample_size = int(self.sample_size_entry.get())
            if not file_path:
                raise ValueError("Не обрано файл з генеральною сукупністю.")
            population = pd.read_csv(file_path)
            self.data = population
            logger.debug(f"Завантажено популяцію розміром {len(population)}")

            if sample_size <= 0 or sample_size > len(population):
                raise ValueError("Некоректний розмір вибірки.")

            random_seed = random.randint(1, 10000)
            logger.debug(f"Використовується випадкове зерно: {random_seed}")

            method_info = self.sampling_methods.get(choice)
            if not method_info:
                raise ValueError("Невідомий тип вибірки.")

            if choice == 1:
                population_with_results, sample, sampling_method_description = random_sampling(
                    population, sample_size, random_seed)
            elif choice == 2:
                population_with_results, sample, sampling_method_description = systematic_sampling(
                    population, sample_size, random_seed)
            elif choice == 3:
                strata_column = self.strata_column_combobox.get()
                if strata_column not in population.columns:
                    raise ValueError(
                        "Обрано неіснуючий стовпець для стратифікації.")
                population_with_results, sample, sampling_method_description = stratified_sampling(
                    population, sample_size, strata_column, random_seed)
            elif choice == 4:
                value_column = self.value_column_combobox.get()
                if value_column not in population.columns:
                    raise ValueError(
                        "Обрано неіснуючий стовпець для методу грошової одиниці.")
                if not pd.api.types.is_numeric_dtype(population[value_column]):
                    raise ValueError(
                        "Стовпець для методу грошової одиниці має бути числовим.")
                threshold = float(self.threshold_entry.get()
                                  ) if self.use_threshold_var.get() else None
                if threshold is not None and threshold < 0:
                    raise ValueError("Порогове значення має бути невід'ємним.")
                strata_column = self.mus_strata_column_combobox.get(
                ) if self.use_stratify_var.get() else None
                if strata_column is not None and strata_column not in population.columns:
                    raise ValueError(
                        "Обрано неіснуючий стовпець для стратифікації.")
                logger.debug(
                    f"Параметри методу грошової одиниці: value_column={value_column}, threshold={threshold}, strata_column={strata_column}")
                population_with_results, sample, sampling_method_description = monetary_unit_sampling(
                    population, sample_size, value_column, threshold, strata_column, random_seed)
            elif choice in (5, 6, 7, 8, 9):
                if not self.numerical_columns and not self.categorical_columns:
                    raise ValueError(
                        "Вкажіть типи колонок для передобробки даних.")
                if not hasattr(self, 'data_preprocessed'):
                    raise ValueError(
                        "Передобробка даних не виконана. Натисніть 'Вказати типи колонок' та збережіть вибір.")
                features = self.numerical_columns + self.categorical_columns
                if choice == 5:
                    population_with_results, population_for_chart, sample, sampling_method_description = isolation_forest_sampling(
                        self.data, self.data_preprocessed, sample_size, features, random_seed)
                elif choice == 6:
                    population_with_results, population_for_chart, sample, sampling_method_description = lof_sampling(
                        self.data, self.data_preprocessed, sample_size, features, random_seed)
                elif choice == 7:
                    population_with_results, population_for_chart, sample, sampling_method_description, best_study = kmeans_sampling(
                        self.data, self.data_preprocessed, sample_size, features, random_seed)
                elif choice == 8:
                    population_with_results, population_for_chart, sample, sampling_method_description = autoencoder_sampling(
                        self.data, self.data_preprocessed, sample_size, features, random_seed)
                elif choice == 9:
                    population_with_results, population_for_chart, sample, sampling_method_description, best_study = hdbscan_sampling(
                        self.data, self.data_preprocessed, sample_size, features, random_seed)
            else:
                raise ValueError("Невідомий метод вибірки.")

            if sample is None or sample.empty:
                raise ValueError(
                    f"Не вдалося сформувати вибірку або вибірка порожня. {sampling_method_description}")

            file_name, file_ext = os.path.splitext(file_path)
            sample_type = method_info['name_en'].lower().replace(' ', '_')
            # Изменение расширения на .pdf
            output_path = f"{file_name}_{sample_type}.pdf"

            # Сохранение выборки и популяции в CSV (при необходимости)
            sample_output_path = f"{file_name}_{sample_type}_sample.csv"
            population_output_path = f"{file_name}_{sample_type}_population.csv"

            population_with_results.to_csv(population_output_path, index=False)
            sample.to_csv(sample_output_path, index=False)

            # Генерация графиков
            chart_paths = []
            threshold = threshold if 'threshold' in locals() else None
            value_column = value_column if 'value_column' in locals() else None

            if choice in (3, 4) and 'strata_column' in locals() and strata_column:
                strata_chart_path = f"{file_name}_{sample_type}_strata_chart.png"
                create_strata_chart(
                    population_with_results,
                    sample,
                    strata_column,
                    strata_chart_path,
                    threshold=threshold,
                    value_column=value_column
                )
                chart_paths.append(strata_chart_path)

            if choice == 4:
                cumulative_chart_path = f"{file_name}_{sample_type}_cumulative_chart"

                base_cumulative_chart_path = f"{file_name}_{sample_type}_cumulative_chart"
                create_cumulative_chart(
                    population_with_results,
                    value_column,
                    strata_column if self.use_stratify_var.get() else None,
                    base_cumulative_chart_path,
                    threshold=threshold
                )
                pattern = f"{file_name}_{sample_type}_cumulative_chart*.png"
                cumulative_chart_files = glob.glob(pattern)
                chart_paths.extend(cumulative_chart_files)

            if choice in (5, 6, 7, 8, 9):
                umap_projection_path = f"{file_name}_{sample_type}_umap_projection.png"
                create_umap_projection(
                    population_for_chart,
                    'is_sample',
                    features,
                    umap_projection_path,
                    'cluster'
                )
                chart_paths.append(umap_projection_path)

            if choice in (7, 9):
                base_optuna_results_path = f"{file_name}_{sample_type}_optuna_results"
                visualize_optuna_results(best_study, base_optuna_results_path)
                pattern = f"{file_name}_{sample_type}_optuna_results*.png"
                optuna_result_files = glob.glob(pattern)
                chart_paths.extend(optuna_result_files)

            # Создание PDF
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            styles = getSampleStyleSheet()
            flowables = []

            # Функция для преобразования текста с \n в <br/>
            def convert_newlines(text):
                return text.replace('\n', '<br/>')

            # Пользовательский стиль для параграфов с разрывами строк
            custom_style = ParagraphStyle(
                'Custom',
                parent=styles['Normal'],
                spaceAfter=12,
                leading=15
            )

            # Добавление описаний методов
            if self.preprocessing_method_description:
                flowables.append(
                    Paragraph("Description of data preprocessing methods:", styles['Heading2']))
                formatted_preprocess_desc = convert_newlines(
                    self.preprocessing_method_description)
                flowables.append(
                    Paragraph(formatted_preprocess_desc, custom_style))
                flowables.append(Spacer(1, 12))

            if sampling_method_description:
                flowables.append(
                    Paragraph("Description of the sampling method:", styles['Heading2']))
                formatted_sampling_desc = convert_newlines(
                    sampling_method_description)
                flowables.append(
                    Paragraph(formatted_sampling_desc, custom_style))
                flowables.append(Spacer(1, 12))

            # Добавление графиков
            for chart_path in chart_paths:
                if os.path.exists(chart_path):
                    flowables.append(
                        Paragraph(f"Chart: {os.path.basename(chart_path)}", styles['Heading3']))
                    img = Image(chart_path, width=6*inch, height=4*inch)
                    flowables.append(img)
                    flowables.append(Spacer(1, 12))
                else:
                    flowables.append(Paragraph(
                        f"Chart {os.path.basename(chart_path)} not found.", styles['Normal']))
                    flowables.append(Spacer(1, 12))

            # Сохранение PDF
            doc.build(flowables)

            self.result_label.config(
                text=f"The sample is saved to the file: {output_path}\n"
            )
        except ValueError as e:
            messagebox.showerror("Помилка", str(e))
        except Exception as e:
            logger.exception("Несподівана помилка")
            messagebox.showerror(
                "Помилка", f"Сталася несподівана помилка: {e}")


def main():
    root = tk.Tk()
    app = SamplingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

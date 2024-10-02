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
import threading

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
        self.widgets = {}

        self.sampling_methods = {
            1: {
                "name_ua": "Випадкова вибірка",
                "name_en": "Random Sampling",
                "description": "кожен елемент генеральної сукупності має рівну ймовірність потрапити у вибірку.",
            },
            2: {
                "name_ua": "Систематична вибірка",
                "name_en": "Systematic Sampling",
                "description": "елементи вибираються з генеральної сукупності через рівні інтервали.",
            },
            3: {
                "name_ua": "Стратифікована вибірка",
                "name_en": "Stratified Sampling",
                "description": "генеральна сукупність ділиться на страти (групи), і з кожної страти формується випадкова вибірка.",
            },
            4: {
                "name_ua": "Метод грошової одиниці",
                "name_en": "Monetary Unit Sampling",
                "description": "ймовірність вибору елемента пропорційна його грошовій величині. Використовується для оцінки сумарної величини помилок.",
            },
            5: {
                "name_ua": "Isolation Forest",
                "name_en": "Isolation Forest",
                "description": "алгоритм для виявлення аномалій на основі випадкових лісів.",
            },
            6: {
                "name_ua": "Local Outlier Factor",
                "name_en": "Local Outlier Factor",
                "description": "метод для виявлення локальних аномалій у даних.",
            },
            7: {
                "name_ua": "Кластеризація K-Means",
                "name_en": "K-Means Clustering",
                "description": "групування даних за схожістю для виявлення незвичайних точок.",
            },
            8: {
                "name_ua": "Автоенкодер",
                "name_en": "Autoencoder",
                "description": "зменшення розмірності даних для виявлення відхилень через аналіз помилки відновлення.",
            },
            9: {
                "name_ua": "HDBSCAN",
                "name_en": "HDBSCAN",
                "description": "знаходження аномалій, класифікуючи точки як шум, спираючись на їхню щільність і відстань до інших точок, що дозволяє виокремлювати викиди в даних.",
            },
        }

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", foreground="black")
        style.configure(
            "TButton",
            background="#cccccc",
            foreground="black",
            borderwidth=2,
            relief="solid",
            font=("Arial", 10),
        )
        style.map(
            "TButton",
            background=[("active", "#aaaaaa")],
            relief=[("pressed", "sunken")],
        )
        style.configure("TCheckbutton", background="#f0f0f0",
                        foreground="black")
        style.configure("TCombobox", background="white", foreground="black")

        self.choice_frame = ttk.Frame(self.root, padding=(10, 10))
        self.choice_frame.grid(row=0, column=0, sticky="w")

        choice_label = ttk.Label(
            self.choice_frame,
            text="Оберіть тип вибірки:",
            font=("Arial", 12, "bold"),
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
                justify="left",
            )
            rb.grid(row=value, column=0, columnspan=2, sticky="w")

        self.options_frame = ttk.Frame(self.root, padding=(10, 10))
        self.options_frame.grid(row=1, column=0, sticky="w")

        self.create_label(
            self.options_frame, "Файл з генеральною сукупністю:", 0, 0
        )
        self.create_button_widget(
            self.options_frame, "Огляд", self.browse_file, 0, 1)
        self.widgets["file_path_entry"] = self.create_entry(
            self.options_frame, 1, 0, colspan=2, width=50
        )

        # Define other widgets
        self.widgets["sample_size"] = {
            "label": self.create_label(
                self.options_frame, "Розмір вибірки:", 2, 0
            ),
            "entry": self.create_entry(self.options_frame, 3, 0),
        }
        self.widgets["strata_column"] = {
            "label": self.create_label(
                self.options_frame, "Стовпець для стратифікації:", 4, 0
            ),
            "combobox": self.create_combobox(self.options_frame, 5, 0),
        }
        self.widgets["value_column"] = {
            "label": self.create_label(
                self.options_frame,
                "Стовпець зі значеннями грошових одиниць:",
                4,
                0,
            ),
            "combobox": self.create_combobox(self.options_frame, 5, 0),
        }
        self.widgets["use_threshold"] = {
            "label": self.create_label(
                self.options_frame, "Використовувати порогове значення?", 6, 0
            ),
            "checkbutton": self.create_checkbutton(
                self.options_frame,
                self.use_threshold_var,
                self.toggle_threshold_input,
                6,
                1,
            ),
        }
        self.widgets["threshold"] = {
            "label": self.create_label(
                self.options_frame, "Порогове значення:", 7, 0
            ),
            "entry": self.create_entry(self.options_frame, 7, 1),
        }
        self.widgets["use_stratify"] = {
            "label": self.create_label(
                self.options_frame, "Використовувати стратифікацію?", 8, 0
            ),
            "checkbutton": self.create_checkbutton(
                self.options_frame,
                self.use_stratify_var,
                self.toggle_stratify_input,
                8,
                1,
            ),
        }
        self.widgets["mus_strata_column"] = {
            "label": self.create_label(
                self.options_frame, "Стовпець для стратифікації:", 9, 0
            ),
            "combobox": self.create_combobox(self.options_frame, 9, 1),
        }

        self.widgets["column_types_button"] = self.create_button_widget(
            self.options_frame,
            "Вказати типи колонок",
            self.define_column_types,
            10,
            0,
        )
        self.widgets["preprocess_label"] = ttk.Label(
            self.options_frame, text="Передобробка даних виконана.", foreground="green"
        )

        self.ai_parameters_frame = ttk.Frame(self.root, padding=(10, 10))

        self.result_frame = ttk.Frame(self.root, padding=(10, 10))
        self.result_frame.grid(row=2, column=0, sticky="w")

        self.result_label = ttk.Label(
            self.result_frame, text="", justify=tk.LEFT)
        self.result_label.grid(row=0, column=0, sticky="w")

        # Assign the button widget to self.create_button
        self.create_button = self.create_button_widget(
            self.root,
            "Створити вибірку",
            self.create_sample,
            3,
            1,
            sticky="se",
            padx=10,
            pady=10,
        )

        self.status_label = ttk.Label(self.root, text="", foreground="blue")
        self.status_label.grid(row=3, column=0, sticky="w", padx=10)

        self.on_choice_change()

    def create_label(self, parent, text, row, column, **kwargs):
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=column, sticky="w", **kwargs)
        return label

    def create_entry(self, parent, row, column, colspan=1, **kwargs):
        entry = ttk.Entry(parent, **kwargs)
        entry.grid(row=row, column=column, columnspan=colspan, sticky="w")
        return entry

    def create_button_widget(self, parent, text, command, row, column, **kwargs):
        button = ttk.Button(parent, text=text, command=command)
        button.grid(row=row, column=column, **kwargs)
        return button

    def create_combobox(self, parent, row, column, **kwargs):
        combobox = ttk.Combobox(parent)
        combobox.grid(row=row, column=column, sticky="w", **kwargs)
        return combobox

    def create_checkbutton(self, parent, variable, command, row, column, **kwargs):
        checkbutton = ttk.Checkbutton(
            parent, variable=variable, command=command
        )
        checkbutton.grid(row=row, column=column, sticky="w", **kwargs)
        return checkbutton

    def toggle_widget_group(self, group_name, show=True):
        widgets = self.widgets.get(group_name, {})
        for widget in widgets.values():
            if show:
                widget.grid()
            else:
                widget.grid_remove()

    def toggle_threshold_input(self):
        self.toggle_widget_group(
            "threshold", show=self.use_threshold_var.get())

    def toggle_stratify_input(self):
        self.toggle_widget_group(
            "mus_strata_column", show=self.use_stratify_var.get())

    def define_column_types(self):
        if self.data is None:
            messagebox.showerror("Помилка", "Спочатку завантажте дані.")
            return

        def select_columns(column_type):
            columns = self.data.columns.tolist()
            if column_type == "numerical":
                columns = [
                    col
                    for col in self.data.columns
                    if pd.api.types.is_numeric_dtype(self.data[col])
                ]
            elif column_type == "categorical":
                columns = [
                    col for col in self.data.columns if col not in self.numerical_columns
                ]

            if not columns:
                messagebox.showerror(
                    "Помилка", f"У датафреймі немає {column_type} колонок."
                )
                return

            window = tk.Toplevel(self.root)
            window.title(f"Вибір {column_type} колонок")
            tk.Label(
                window, text=f"Виберіть {column_type} колонки:"
            ).pack(anchor="w", padx=10, pady=5)

            listbox = tk.Listbox(window, selectmode=tk.MULTIPLE, width=50)
            for col in columns:
                listbox.insert(tk.END, col)
            listbox.pack(padx=10, pady=5)

            def save_columns():
                selected_indices = listbox.curselection()
                selected_columns = [columns[i] for i in selected_indices]
                if column_type == "numerical":
                    self.numerical_columns = selected_columns
                elif column_type == "categorical":
                    self.categorical_columns = selected_columns
                if not selected_columns:
                    messagebox.showwarning(
                        "Попередження", f"Не обрано жодної {column_type} колонки."
                    )
                window.destroy()
                if column_type == "numerical":
                    select_columns("categorical")
                else:
                    (
                        self.data_preprocessed,
                        self.preprocessing_method_description,
                    ) = preprocess_data(
                        self.data, self.numerical_columns, self.categorical_columns
                    )
                    self.widgets["preprocess_label"].grid(
                        row=99, column=0, sticky="w"
                    )

            save_button = ttk.Button(window, text="Далі", command=save_columns)
            save_button.pack(pady=10)

        select_columns("numerical")

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Виберіть файл з генеральною сукупністю",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        if file_path:
            self.status_label.config(text="Завантаження файлу...")
            self.widgets["file_path_entry"].delete(0, tk.END)
            self.widgets["file_path_entry"].insert(0, file_path)
            try:
                self.data = pd.read_csv(file_path)
                self.populate_column_dropdowns(self.data)
            except Exception as e:
                self.result_label.config(
                    text=f"Помилка при читанні файлу: {e}")
            finally:
                self.status_label.config(text="")

    def populate_column_dropdowns(self, df):
        try:
            numerical_columns = [
                col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
            ]
            self.widgets["value_column"]["combobox"]["values"] = numerical_columns
            columns = list(df.columns)
            self.widgets["strata_column"]["combobox"]["values"] = columns
            self.widgets["mus_strata_column"]["combobox"]["values"] = columns
        except Exception as e:
            self.result_label.config(
                text=f"Помилка при заповненні списку колонок: {e}"
            )

    def toggle_options(self):
        choice = self.choice_var.get()
        # Hide all widgets
        for key in [
            "sample_size",
            "strata_column",
            "value_column",
            "use_threshold",
            "threshold",
            "use_stratify",
            "mus_strata_column",
        ]:
            self.toggle_widget_group(key, show=False)

        # Show widgets based on choice
        if choice in self.sampling_methods:
            self.toggle_widget_group("sample_size", show=True)
        if choice == 3:
            self.toggle_widget_group("strata_column", show=True)
        if choice == 4:
            self.toggle_widget_group("value_column", show=True)
            self.toggle_widget_group("use_threshold", show=True)
            self.toggle_threshold_input()
            self.toggle_widget_group("use_stratify", show=True)
            self.toggle_stratify_input()

        if choice in (5, 6, 7, 8, 9):
            self.widgets["column_types_button"].grid(
                row=10, column=0, sticky="w")
            self.ai_parameters_frame.grid(row=11, column=0, sticky="w")
        else:
            self.widgets["column_types_button"].grid_remove()
            self.ai_parameters_frame.grid_remove()
            self.widgets["preprocess_label"].grid_remove()

    def on_choice_change(self):
        self.toggle_options()

    def create_sample(self):
        self.create_button.config(state=tk.DISABLED)
        self.status_label.config(text="Створення вибірки...")
        threading.Thread(target=self._create_sample).start()

    def _create_sample(self):
        try:
            choice = self.choice_var.get()
            file_path = self.widgets["file_path_entry"].get()
            sample_size = int(self.widgets["sample_size"]["entry"].get())
            population = pd.read_csv(file_path)
            self.data = population

            # Add dataset size limits for LOF and HDBSCAN
            dataset_size = len(self.data)

            if choice == 6 and dataset_size > 1_000_000:
                raise ValueError(
                    "LOF cannot be used for datasets larger than 1 million.")
            elif choice == 9 and dataset_size > 500_000:
                raise ValueError(
                    "HDBSCAN cannot be used for datasets larger than 500 thousand.")

            method_info = self.sampling_methods.get(choice)
            sampling_function = self.get_sampling_function(choice)
            kwargs = self.get_sampling_parameters(choice)

            if choice in (5, 6, 7, 8, 9):
                # Extract parameters specific to AI methods
                features = kwargs.pop('features')
                random_seed = kwargs.pop('random_seed')
                result = sampling_function(
                    self.data, self.data_preprocessed, sample_size, features, random_seed
                )
            else:
                result = sampling_function(population, sample_size, **kwargs)

            self.process_sampling_result(
                result, method_info, file_path, choice)
        except Exception as e:
            self.handle_exception(e)
        finally:
            self.finalize_sample_creation()

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
        if choice == 3:
            params["strata_column"] = self.widgets["strata_column"][
                "combobox"
            ].get()
        elif choice == 4:
            params["value_column"] = self.widgets["value_column"][
                "combobox"
            ].get()
            params["threshold"] = (
                float(self.widgets["threshold"]["entry"].get())
                if self.use_threshold_var.get()
                else None
            )
            params["strata_column"] = (
                self.widgets["mus_strata_column"]["combobox"].get()
                if self.use_stratify_var.get()
                else None
            )
        elif choice in (5, 6, 7, 8, 9):
            if not self.numerical_columns and not self.categorical_columns:
                raise ValueError(
                    "Вкажіть типи колонок для передобробки даних."
                )
            if not hasattr(self, "data_preprocessed"):
                raise ValueError(
                    "Передобробка даних не виконана. Натисніть 'Вказати типи колонок' та збережіть вибір."
                )
            params["data_preprocessed"] = self.data_preprocessed
            params["features"] = self.numerical_columns + \
                self.categorical_columns
        return params

    def process_sampling_result(
        self, result, method_info, file_path, choice
    ):
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
                f"Не вдалося сформувати вибірку або вибірка порожня. {sampling_method_description}"
            )

        file_name, file_ext = os.path.splitext(file_path)
        sample_type = method_info["name_en"].lower().replace(" ", "_")
        output_path = f"{file_name}_{sample_type}.pdf"

        sample_output_path = f"{file_name}_{sample_type}_sample.csv"
        population_output_path = f"{file_name}_{sample_type}_population.csv"

        population_with_results.to_csv(population_output_path, index=False)
        sample.to_csv(sample_output_path, index=False)

        chart_paths = []
        threshold = (
            float(self.widgets["threshold"]["entry"].get())
            if self.use_threshold_var.get()
            else None
        )
        value_column = (
            self.widgets["value_column"]["combobox"].get()
            if "value_column" in self.widgets
            else None
        )
        strata_column = (
            self.widgets["strata_column"]["combobox"].get()
            if "strata_column" in self.widgets
            else None
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
                strata_column if self.use_stratify_var.get() else None,
                base_cumulative_chart_path,
                threshold=threshold,
            )
            pattern = f"{file_name}_{sample_type}_cumulative_chart*.png"
            cumulative_chart_files = glob.glob(pattern)
            chart_paths.extend(cumulative_chart_files)

        if choice in (5, 6, 7, 8, 9):
            umap_projection_path = (
                f"{file_name}_{sample_type}_umap_projection.png"
            )

            columns_to_drop = ['is_sample']
            if 'cluster' in population_for_chart.columns:
                columns_to_drop.append('cluster')
            columns_without_sample_and_cluster = population_for_chart.columns.drop(
                columns_to_drop)

            create_umap_projection(
                population_for_chart,
                "is_sample",
                columns_without_sample_and_cluster,
                umap_projection_path,
                cluster_column='cluster' if 'cluster' in population_for_chart.columns else None
            )
            chart_paths.append(umap_projection_path)

            if choice in (7, 9) and best_study:
                base_optuna_results_path = (
                    f"{file_name}_{sample_type}_optuna_results"
                )
                visualize_optuna_results(best_study, base_optuna_results_path)
                pattern = f"{file_name}_{sample_type}_optuna_results*.png"
                optuna_result_files = glob.glob(pattern)
                chart_paths.extend(optuna_result_files)

        self.generate_pdf(
            output_path, sampling_method_description, chart_paths)

        self.root.after(
            0,
            lambda: self.result_label.config(
                text=f"The sample is saved to the file: {output_path}\n"
            ),
        )

    def generate_pdf(self, output_path, sampling_method_description, chart_paths):
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        flowables = []

        def convert_newlines(text):
            return text.replace("\n", "<br/>")

        custom_style = ParagraphStyle(
            "Custom", parent=styles["Normal"], spaceAfter=12, leading=15
        )

        if self.preprocessing_method_description:
            flowables.append(
                Paragraph(
                    "Description of data preprocessing methods:",
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
            flowables.append(
                Paragraph(
                    "Description of the sampling method:", styles["Heading2"]
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
                flowables.append(
                    Paragraph(
                        f"Chart: {os.path.basename(chart_path)}",
                        styles["Heading3"],
                    )
                )
                img = Image(chart_path, width=6 * inch, height=4 * inch)
                flowables.append(img)
                flowables.append(Spacer(1, 12))
            else:
                flowables.append(
                    Paragraph(
                        f"Chart {os.path.basename(chart_path)} not found.",
                        styles["Normal"],
                    )
                )
                flowables.append(Spacer(1, 12))

        doc.build(flowables)

    def handle_exception(self, exception):
        logger.exception("Error occurred")
        self.root.after(
            0, lambda: messagebox.showerror("Помилка", str(exception))
        )

    def finalize_sample_creation(self):
        self.root.after(0, lambda: self.create_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.status_label.config(text=""))


def main():
    root = tk.Tk()
    app = SamplingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

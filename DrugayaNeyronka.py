import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import Button, TextBox
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Tuple, List, Optional, Dict

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU функция активации."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Производная ReLU функции активации."""
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    """Улучшенная нейронная сеть для принятия решений о посадке."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.2):
        """Инициализация нейросети с использованием He Initialization и Dropout."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # He Initialization
        he_input = np.sqrt(2.0 / self.input_size)
        he_hidden = np.sqrt(2.0 / self.hidden_size)

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * he_input
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * he_hidden

        # Dropout mask
        self.dropout_mask = None

        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Прямой проход с Dropout во время обучения."""
        self.hidden = relu(np.dot(X, self.weights_input_hidden))

        if training:
            # Применение Dropout
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.hidden.shape)
            self.hidden *= self.dropout_mask

        self.output = relu(np.dot(self.hidden, self.weights_hidden_output))
        return self.output

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000,
              lr: float = 0.001, reg_lambda: float = 0.01, threshold: float = 0.1) -> None:
        val_size = int(0.15 * len(X))
        X_train, y_train = X[:-val_size], y[:-val_size]
        X_val, y_val = X[-val_size:], y[-val_size:]

        m_wih, v_wih = np.zeros_like(self.weights_input_hidden), np.zeros_like(self.weights_input_hidden)
        m_who, v_who = np.zeros_like(self.weights_hidden_output), np.zeros_like(self.weights_hidden_output)
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8

        self.train_loss_history = []
        self.val_loss_history = []

        for epoch in range(epochs + 1):  # чтобы попасть в 5000 тоже
            output = self.forward(X_train, training=True)
            error = y_train - output

            output_delta = error * relu_derivative(output)
            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * relu_derivative(self.hidden)

            grad_who = np.dot(self.hidden.T, output_delta) - reg_lambda * self.weights_hidden_output
            grad_wih = np.dot(X_train.T, hidden_delta) - reg_lambda * self.weights_input_hidden

            m_who = beta1 * m_who + (1 - beta1) * grad_who
            v_who = beta2 * v_who + (1 - beta2) * (grad_who ** 2)
            m_wih = beta1 * m_wih + (1 - beta1) * grad_wih
            v_wih = beta2 * v_wih + (1 - beta2) * (grad_wih ** 2)

            m_who_corr = m_who / (1 - beta1 ** (epoch + 1))
            v_who_corr = v_who / (1 - beta2 ** (epoch + 1))
            m_wih_corr = m_wih / (1 - beta1 ** (epoch + 1))
            v_wih_corr = v_wih / (1 - beta2 ** (epoch + 1))

            self.weights_hidden_output += lr * m_who_corr / (np.sqrt(v_who_corr) + epsilon)
            self.weights_input_hidden += lr * m_wih_corr / (np.sqrt(v_wih_corr) + epsilon)

            # Логируем каждую 100-ю эпоху и 0-ю, а также последнюю
            if epoch % 100 == 0 or epoch == epochs:
                train_loss = np.mean(np.square(error))
                val_loss = np.mean(np.square(y_val - self.forward(X_val)))
                self.train_loss_history.append(train_loss)
                self.val_loss_history.append(val_loss)
                # (опционально: print(f"Эпоха {epoch}: Ошибка обучения = {train_loss:.4f}, Ошибка валидации = {val_loss:.4f}"))

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Оценка точности на тестовых данных."""
        predictions = np.argmax(self.forward(X_test), axis=1)
        accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
        print(f"\nТочность на тестовых данных: {accuracy * 100:.2f}%")
        return accuracy

class Simulation:
    """Симулятор полета БПЛА с системой принятия решений."""
    # Константы энергии
    # Перевод емкости батареи из Вт·ч в Джоули E (Дж) = E (Вт·ч) * 3600
    # Горизонтальный полет: v = 15 м/с P = 150 Вт t = 1 / v ≈ 0.0667 секунд E = P * t = 150 * 0.0667 ≈ 10 Дж/м
    # Подъем: vh = 2.5 м/с P = 220 Вт m = 0.9 кг g = 9.81 м/с^2 E_climb = (P / vh) + (m * g) E_climb = 88 + 8.83 ≈ 96.83 Дж/м
    # Спуск:vh = 2.5 м/с P = 100 Вт E_descent = P / vh E_descent = 100 / 2.5 = 40 Дж/м
    # Системный расход энергии включает в себя затраты на электронику и датчики
    def __init__(self, root,
                 charging_voltage=48,
                 charging_current=10,
                 charging_efficiency=0.9,
                 battery_capacity_watt_hours=80):
        """Инициализация класса Simulation."""
        self.root = root
        self.root.title("Система управления БПЛА")

        # Константы энергии (перенесены в атрибуты объекта)
        self.HORIZONTAL_ENERGY = 10  # Энергия на горизонтальный полет (Дж/м)
        self.CLIMB_ENERGY = 96.83  # Энергия на подъем (Дж/м)
        self.DESCENT_ENERGY = 40  # Энергия на спуск (Дж/м)
        self.SYSTEMS_ENERGY_WATTS = 5  # Системное энергопотребление (Вт)
        self.SYSTEMS_ENERGY = self.SYSTEMS_ENERGY_WATTS / 3600  # Системное энергопотребление (Дж/сек)
        self.OCCUPATION_PENALTY = 0.2  # Штраф за занятую станцию

        # Параметры батареи
        self.BATTERY_CAPACITY_WATT_HOURS = battery_capacity_watt_hours
        self.BATTERY_CAPACITY = self.BATTERY_CAPACITY_WATT_HOURS * 3600  # Перевод в джоули
        self.VOLTAGE_BATTERY = 15.4  # Напряжение батареи (В)
        self.INTERNAL_RESISTANCE = self.VOLTAGE_BATTERY / 4  # Внутреннее сопротивление батареи (Ом)

        # Параметры зарядки
        self.CHARGING_CURRENT = charging_current  # Ток зарядки (А)
        self.CHARGING_VOLTAGE = charging_voltage  # Напряжение во время зарядки (В)
        self.CHARGING_EFFICIENCY = charging_efficiency  # КПД зарядки

        # Состояние зарядки
        self.charge = 0.8  # Текущий заряд батареи (в долях от полной емкости)
        self.is_charging = False  # Флаг состояния зарядки

        # Параметры карты
        self.grid_width = 30  # Ширина карты в клетках
        self.grid_height = 20  # Высота карты в клетках
        self.cell_size = 100  # Размер клетки (100м x 100м)

        # Параметры дрона
        self.drone_mass = 0.9  # Масса дрона (в кг)
        self.drone_height = 500.0  # Начальная высота дрона
        self.target_height = 0.0  # Целевая высота
        self.last_logged_height = self.drone_height  # Устанавливаем начальное значение
        self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS  # Начальный заряд в Вт·ч
        self.charge = self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS
        self.flight_time = 30 * 60  # Время полета (30 минут в секундах)
        self.energy_usage = 300  # Максимальное потребление энергии (Вт·ч)
        self.range = 15000  # Радиус действия (15 км)
        self.horizontal_energy_per_meter = self.HORIZONTAL_ENERGY  # Дж/м для горизонтального перемещения
        self.climb_energy_per_meter = self.CLIMB_ENERGY  # Дж/м для подъема
        self.descent_energy_per_meter = self.DESCENT_ENERGY  # Дж/м для спуска
        self.system_energy_per_second = self.SYSTEMS_ENERGY  # Дж/сек для систем

        # Визуализация дрона
        self.drone_size = 15  # Начальный размер дрона
        self.max_drone_size = 20  # Максимальный размер (в зависимости от высоты)
        self.min_drone_size = 5  # Минимальный размер

        # Скорость дрона
        self.real_speed = 15.0  # Реальная горизонтальная скорость дрона (м/с)
        self.vertical_speed = 2.5  # Скорость подъема/спуска (м/с)
        self.simulation_speed = 1.0  # Коэффициент ускорения симуляции
        self.frame_interval = 100  # Интервал между кадрами в миллисекундах

        # Скорость в клетках/секунду
        self.speed = self.real_speed / self.cell_size  # Скорость в клетках/секунду

        # Параметры станций
        self.stations = [[5, 10], [25, 10]]  # Координаты станций (в клетках)
        self.station_heights = [100.0, 100.0]  # Высота станций (в метрах)
        self.station_statuses = [False, False]  # Статус станций (занята/свободна)

        # Состояние дрона
        self.drone_pos = np.array([15, 10], dtype=float)  # Начальная позиция дрона
        self.drone_height = 500.0  # Высота дрона
        self.mission_active = False  # Флаг активности миссии
        self.is_forced_landing = False  # Флаг принудительной посадки
        self.is_landing = False  # Флаг процесса посадки

        # Параметры маршрута
        self.start_pos = None  # Начальная позиция
        self.target_pos = None  # Целевая позиция (текущая цель)
        self.end_pos = None  # Конечная позиция маршрута

        # Создание вкладок
        self.notebook = ttk.Notebook(self.root)
        self.tab_params = ttk.Frame(self.notebook)
        self.tab_plot = ttk.Frame(self.notebook)
        self.tab_sim = ttk.Frame(self.notebook)
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=('Arial', 14, 'bold'))
        self.notebook.add(self.tab_params, text="Параметры")
        self.notebook.add(self.tab_plot, text="График обучения")
        self.notebook.add(self.tab_sim, text="Симуляция")
        self.notebook.pack(expand=True, fill='both')

        # Инициализация вкладок
        self.init_parameters_tab()
        self.init_plot_tab()
        self.init_simulation_tab()

        # Инициализация нейросети
        self.init_neural_network()

    def export_log(self):
        """Экспорт лога в текстовый файл."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Сохранить лог"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(self.log_text.get("1.0", tk.END))
                self.update_log(f"Лог успешно сохранён в файл: {file_path}")
            except Exception as e:
                self.update_log(f"Ошибка при сохранении лога: {e}")

    def generate_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация тренировочных данных с учетом простых задач."""
        print(f"\nГенерация {num_samples} примеров...")
        X = np.zeros((num_samples, 8))
        y = np.zeros((num_samples, 2))

        for i in range(num_samples):
            # Генерация параметров дрона
            charge = np.random.uniform(0.2, 1.0)
            height = np.random.uniform(100, 5000)

            # Генерация параметров станций
            station_data = []
            for _ in range(2):
                station_data.append({
                    'dist': np.random.uniform(100, 3000),
                    'height': np.random.uniform(50, 5000),
                    'status': np.random.choice([0, 1], p=[0.7, 0.3])
                })

            # Расчет разницы высот и затрат энергии
            height_diffs = [s['height'] - height for s in station_data]
            costs = [
                station_data[i]['dist'] + abs(height_diffs[i]) + (1000 if station_data[i]['status'] == 1 else 0)
                for i in range(2)
            ]

            # Выбор станции с минимальной стоимостью
            best_station = np.argmin(costs)
            y[i, best_station] = 1  # Пометить лучшую станцию как правильный выбор

            # Формирование входных данных
            X[i, :] = [
                charge,
                height / 5000,
                station_data[0]['dist'] / 3000,
                station_data[0]['status'],
                station_data[1]['dist'] / 3000,
                station_data[1]['status'],
                height_diffs[0] / 5000,
                height_diffs[1] / 5000
            ]

        return X, y

    # Расчет стоимостей
    def calculate_cost(self, distance, height_diff, status, charge_percent):
        """Расчет стоимости посадки на станцию с учетом нейросети."""
        # Подготовка входных данных для нейросети
        input_data = np.array([
            charge_percent,  # Текущий заряд в процентах
            self.drone_height / 5000,
            distance / 3000,
            status,
            self.stations[0][0] / self.grid_width,  # Координата X первой станции (нормализованная)
            self.stations[0][1] / self.grid_height,  # Координата Y первой станции (нормализованная)
            height_diff / 5000,  # Разница высот
            self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS  # Текущий заряд в долях
        ]).reshape(1, -1)  # Преобразование к форме (1, 8)

        # Предсказание нейросети
        neural_output = self.nn.forward(input_data)
        neural_factor = neural_output[0, 0]  # Берем выход для первой станции

        # Логика энергозатрат
        horizontal = distance * self.HORIZONTAL_ENERGY
        if height_diff > 0:
            vertical = abs(height_diff) * self.CLIMB_ENERGY  # Подъем
        else:
            vertical = abs(height_diff) * self.DESCENT_ENERGY  # Спуск

        # Суммируем энергозатраты
        total_energy = horizontal + vertical

        # Нормализуем энергозатраты
        normalized_energy = total_energy / (self.BATTERY_CAPACITY * 0.8)

        # Добавляем штраф за занятость станции
        penalty = self.OCCUPATION_PENALTY if status else 0

        # Итоговая стоимость: нейросеть влияет на итоговый вес
        cost = (normalized_energy * (1.5 - charge_percent) + penalty) * (1 - neural_factor)
        return cost

    def init_parameters_tab(self):
        """Вкладка с параметрами ввода."""
        frame = ttk.LabelFrame(self.tab_params, text="Начальные параметры", padding=(10, 10))
        frame.pack(padx=20, pady=20, fill='both', expand=True)

        # Увеличенные параметры шрифта
        label_font = ("Arial", 14, "bold")
        entry_font = ("Arial", 14)
        button_font = ("Arial", 16, "bold")

        params = [
            ('Высота дрона (м):', 'drone_height', 500),
            ('Высота док-станции №1 (м):', 'station1_height', 100),
            ('Высота док-станции №2 (м):', 'station2_height', 100),
            ('Ёмкость батареи дрона (Вт·ч):', 'battery_capacity_watt_hours', self.BATTERY_CAPACITY_WATT_HOURS),
        ]

        self.entries = {}
        for i, (label, name, default) in enumerate(params):
            lbl = ttk.Label(frame, text=label, font=label_font)
            lbl.grid(row=i, column=0, padx=10, pady=10, sticky='e')

            entry = ttk.Entry(frame, font=entry_font)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, padx=10, pady=10)
            self.entries[name] = entry

        btn_apply = ttk.Button(frame, text="Применить", command=self.apply_parameters, style="Apply.TButton")
        btn_apply.grid(row=len(params), column=0, columnspan=2, pady=20)

        # Настройка стиля кнопки
        style = ttk.Style()
        style.configure("Apply.TButton", font=button_font, padding=10)

    def init_plot_tab(self):
        """Создание вкладки с графиками обучения и зарядки."""
        self.figures_frame = ttk.Frame(self.tab_plot)
        self.figures_frame.pack(fill="both", expand=True)

        # График обучения нейросети
        self.fig_loss = Figure(figsize=(8, 2.5), dpi=100)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.ax_loss.set_title('Кривая обучения нейросети', fontsize=14)
        self.ax_loss.set_xlabel('Эпохи', fontsize=10)
        self.ax_loss.set_ylabel('Ошибка', fontsize=10)
        self.ax_loss.grid(True, linestyle='--', alpha=0.6)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self.figures_frame)
        self.canvas_loss.get_tk_widget().pack(side="top", fill="x", expand=False)

        # График зарядки батареи
        self.fig_charge = Figure(figsize=(8, 4), dpi=100)
        self.ax_charge_left = self.fig_charge.add_subplot(111)
        self.ax_charge_right = self.ax_charge_left.twinx()

        self.ax_charge_left.set_title('График зарядки батареи дрона', fontsize=14)
        self.ax_charge_left.set_xlabel('Время (сек)', fontsize=10)
        self.ax_charge_left.set_ylabel('Ток и напряжение', fontsize=10, color='tab:blue')
        self.ax_charge_right.set_ylabel('Заряд батареи (%)', fontsize=10, color='tab:green')
        self.ax_charge_left.grid(True, linestyle='--', alpha=0.6)

        self.canvas_charge = FigureCanvasTkAgg(self.fig_charge, master=self.figures_frame)
        self.canvas_charge.get_tk_widget().pack(side="top", fill="both", expand=True)

    def plot_charge_graph(self):
        """Отображение графика зарядки с током, напряжением и уровнем заряда."""
        if not hasattr(self, 'charge_log') or not self.charge_log["time"]:
            return

        times = self.charge_log["time"]
        percents = self.charge_log["charge_percent"]
        currents = self.charge_log.get("current", [])
        voltages = self.charge_log.get("voltage", [])

        # Привести к одинаковой длине
        min_len = min(len(times), len(percents), len(currents), len(voltages))
        if min_len == 0:
            return
        times = times[:min_len]
        percents = percents[:min_len]
        currents = currents[:min_len]
        voltages = voltages[:min_len]

        self.ax_charge_left.clear()
        self.ax_charge_right.clear()

        # Фоновые зоны CC и CV
        cc_end_time = None
        for i, p in enumerate(percents):
            if p >= 90:
                cc_end_time = times[i]
                break
        if cc_end_time is None:
            cc_end_time = times[-1]
        self.ax_charge_left.axvspan(times[0], cc_end_time, color='lightblue', alpha=0.2, label="Фаза CC")
        if cc_end_time < times[-1]:
            self.ax_charge_left.axvspan(cc_end_time, times[-1], color='lightcoral', alpha=0.2, label="Фаза CV")

        # Линии
        self.ax_charge_left.plot(times, currents, color='tab:blue', label='Ток зарядки (А)')
        self.ax_charge_left.plot(times, voltages, color='tab:orange', linestyle='--', label='Напряжение (В)')
        self.ax_charge_right.plot(times, percents, color='tab:green', label='Заряд (%)')

        self.ax_charge_left.set_ylabel('Ток (А), Напряжение (В)', fontsize=10, color='tab:blue')
        self.ax_charge_right.set_ylabel('Уровень заряда (%)', fontsize=10, color='tab:green')
        self.ax_charge_right.set_yticks(np.arange(0, 105, 5))

        lines_left, labels_left = self.ax_charge_left.get_legend_handles_labels()
        lines_right, labels_right = self.ax_charge_right.get_legend_handles_labels()
        self.ax_charge_left.legend(lines_left + lines_right, labels_left + labels_right, loc='upper center', fontsize=8)

        self.ax_charge_left.set_xlabel('Время (сек)', fontsize=10)
        self.ax_charge_left.grid(True, linestyle='--', alpha=0.6)

        self.canvas_charge.draw()

    def init_simulation_tab(self):
        """Инициализация вкладки с симуляцией."""

        # Создание стиля для увеличенного текста
        style = ttk.Style()
        style.configure("Big.TLabelframe.Label", font=("Arial", 14, "bold"))  # Для заголовков LabelFrame
        style.configure("Big.TLabel", font=("Arial", 14))  # Для обычных меток (Label)

        main_frame = ttk.Frame(self.tab_sim)
        main_frame.pack(fill="both", expand=True)

        # Лог событий
        log_frame = ttk.LabelFrame(main_frame, text="Лог событий", style="Big.TLabelframe")
        log_frame.pack(side="top", fill="x", padx=10, pady=5)

        # Текстовый виджет для лога
        self.log_text = tk.Text(log_frame, height=4, wrap="word")
        self.log_text.pack(fill="both", expand=True)

        # Кнопка для экспорта лога
        export_button = ttk.Button(log_frame, text="Экспортировать лог", command=self.export_log, style="Big.TButton")
        export_button.pack(pady=5)

        # Основной фрейм симуляции
        sim_frame = ttk.Frame(main_frame)
        sim_frame.pack(fill="both", expand=True)

        # Панель управления
        control_frame = ttk.LabelFrame(sim_frame, text="Управление", style="Big.TLabelframe")
        control_frame.pack(side="left", fill="y", padx=10, pady=5)

        # Метка таймера
        self.timer_label = ttk.Label(control_frame, text="Время: 0.0 сек", style="Big.TLabel")
        self.timer_label.pack(pady=5)

        # Создаём стиль для кнопок с увеличенным и жирным шрифтом
        style = ttk.Style()
        style.configure("Big.TButton", font=("Arial", 14, "bold"))

        # Создаём кнопки с применением стиля
        ttk.Button(control_frame, text="Начать симуляцию", command=self.start_mission, style="Big.TButton").pack(padx=5,
                                                                                                                 pady=5,
                                                                                                                 fill="x")
        ttk.Button(control_frame, text="Сброс", command=self.reset_simulation, style="Big.TButton").pack(padx=5, pady=5,
                                                                                                         fill="x")
        ttk.Button(control_frame, text="Принудительная посадка", command=self.force_decision, style="Big.TButton").pack(
            padx=5, pady=5, fill="x")

        # Параметры дрона
        params_frame = ttk.LabelFrame(control_frame, text="Параметры дрона", style="Big.TLabelframe")
        params_frame.pack(padx=5, pady=5, fill="x")
        self.charge_label = ttk.Label(params_frame, text=f"Заряд: 100.00% ({self.remaining_capacity_watt_hours:.2f} Вт·ч)", style="Big.TLabel")
        self.charge_label.pack(padx=5, pady=2)
        self.height_label = ttk.Label(params_frame, text="Высота: 500 м", style="Big.TLabel")
        self.height_label.pack(padx=5, pady=2)

        # Легенда карты (статичная, внутри панели управления)
        legend_frame = ttk.LabelFrame(control_frame, text="Легенда карты", style="Big.TLabelframe")
        legend_frame.pack(padx=5, pady=5, fill="x")

        # Используем Canvas для отображения значков
        legend_canvas = tk.Canvas(legend_frame, width=250, height=180)
        legend_canvas.pack()

        # Интервал между элементами по оси Y
        y_start = 20  # Начальная координата Y
        y_interval = 30  # Интервал между элементами

        # Линии, обозначающие дрон
        legend_canvas.create_line(20, y_start, 40, y_start + 20, fill="black", width=2)
        legend_canvas.create_line(40, y_start, 20, y_start + 20, fill="black", width=2)
        legend_canvas.create_text(50, y_start + 10, text="- Дрон", anchor="w", font=("Arial", 12))

        # Начальная точка
        legend_canvas.create_oval(22, y_start + y_interval + 2, 37, y_start + y_interval + 17, fill="blue")
        legend_canvas.create_text(50, y_start + y_interval + 10, text="- Начальная точка", anchor="w",
                                  font=("Arial", 12))

        # Конечная точка
        legend_canvas.create_oval(22, y_start + 2 * y_interval + 2, 37, y_start + 2 * y_interval + 17, fill="red")
        legend_canvas.create_text(50, y_start + 2 * y_interval + 10, text="- Конечная точка", anchor="w",
                                  font=("Arial", 12))

        # Док-станция свободна
        legend_canvas.create_polygon(
            20, y_start + 3 * y_interval + 15, 40, y_start + 3 * y_interval + 15,
            30, y_start + 3 * y_interval - 2.32, fill="green"
        )
        legend_canvas.create_text(50, y_start + 3 * y_interval + 10, text="- Док-станция свободна", anchor="w",
                                  font=("Arial", 12))

        # Док-станция занята
        legend_canvas.create_polygon(
            20, y_start + 4 * y_interval + 15, 40, y_start + 4 * y_interval + 15,
            30, y_start + 4 * y_interval - 2.32, fill="red"
        )
        legend_canvas.create_text(50, y_start + 4 * y_interval + 10, text="- Док-станция занята", anchor="w",
                                  font=("Arial", 12))

        # Карта
        map_frame = ttk.Frame(sim_frame)
        map_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self.fig_map = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig_map.add_subplot(111)
        self.setup_map_grid()

        self.canvas_map = FigureCanvasTkAgg(self.fig_map, map_frame)
        canvas_widget = self.canvas_map.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)  # Ensure it occupies all available space
        self.canvas_map.mpl_connect("button_press_event", self.on_click)

        self.toolbar = NavigationToolbar2Tk(self.canvas_map, map_frame)
        self.toolbar.update()

        # Элементы карты
        self.route_line, = self.ax.plot([], [], 'b--', linewidth=1, alpha=1)
        self.drone_icon = self.ax.scatter(
            self.drone_pos[0], self.drone_pos[1],
            s=15 ** 2, c='black', marker='x', linewidths=2, label='Дрон'
        )
        self.start_icon, = self.ax.plot([], [], 'bo', markersize=10, label='Старт')
        self.end_icon, = self.ax.plot([], [], 'ro', markersize=10, label='Финиш')
        self.add_stations()
        self.canvas_map.draw()

        speed_frame = ttk.LabelFrame(control_frame, text="Скорость симуляции", style="Big.TLabelframe")
        speed_frame.pack(padx=5, pady=5, fill="x")
        self.speed_scale = tk.Scale(
            speed_frame,
            from_=0.5,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_simulation_speed,
            label="Множитель скорости"
        )
        self.speed_scale.set(1.0)  # Установить начальное значение
        self.speed_scale.pack(fill="x")

        # Установка начального множителя скорости
        self.simulation_speed_multiplier = 1.0
        self.drone_base_speed = 15  # Базовая скорость дрона (м/с)

    def start_timer(self):
        """Запуск таймера, который отсчитывает симуляционное время."""
        if not getattr(self, "timer_running", False):
            self.start_time = time.time()
            self.simulation_time = 0  # Сбрасываем симуляционное время
            self.timer_running = True
            self.update_timer()

    def stop_timer(self):
        """Приостановка таймера."""
        if getattr(self, "timer_running", False):
            self.elapsed_time = time.time() - self.start_time
            self.timer_running = False

    def update_timer(self):
        """Обновление таймера, который отслеживает симуляционное время."""
        if self.timer_running:
            # Реальное время, прошедшее с момента запуска
            real_elapsed = time.time() - self.start_time
            # Симуляционное время с учётом множителя
            self.simulation_time = real_elapsed * self.simulation_speed_multiplier
            self.timer_label.config(text=f"Секундомер: {self.simulation_time:.1f} сек")
            self.root.after(100, self.update_timer)  # Обновляем каждые 100 мс

    def reset_timer(self):
        """Сброс таймера."""
        self.start_time = None
        self.elapsed_time = 0
        self.timer_running = False
        self.timer_label.config(text="Время: 0.0 сек")

    def start_animation(self) -> None:
        """Запуск анимации с учётом множителя скорости."""
        # Остановка предыдущей анимации, если она активна
        if hasattr(self, 'animation') and self.animation:
            if self.animation.event_source:
                self.animation.event_source.stop()

        self.start_timer()  # Реальный таймер симуляции
        self.last_update_time = time.time()  # Для расчёта real_delta_time

        # Запуск анимации
        self.animation = FuncAnimation(
            self.fig_map,
            lambda frame: self.move_drone(),
            interval=self.frame_interval,  # Интервал кадров зависит от множителя скорости
            blit=False,
            repeat=False,
            cache_frame_data=False
        )

        self.update_log("Анимация запущена.")
        self.canvas_map.draw()

    def update_simulation_speed(self, value):
        """Обновляет множитель скорости симуляции (влияет на течение симуляционного времени)."""
        self.simulation_speed_multiplier = float(value)  # Устанавливаем множитель скорости
        self.frame_interval = int(1000 / self.simulation_speed_multiplier)  # Интервал обновления кадров
        self.update_log(
            f"Множитель скорости установлен на {self.simulation_speed_multiplier:.2f}. Анимация обновляется каждые {self.frame_interval} мс.")

    def calculate_drone_speed(self):
        """Расчёт текущей скорости дрона с учётом множителя."""
        # Скорость дрона изменяется в зависимости от множителя
        return self.drone_base_speed * self.simulation_speed_multiplier

    def simulate_drone_flight(self, distance_meters):
        """Симуляция полёта дрона на заданное расстояние."""
        current_speed = self.calculate_drone_speed()  # Текущая скорость дрона (м/с)
        time_seconds = distance_meters / current_speed  # Время полёта (с)

        self.update_log(f"Дрон летит со скоростью {current_speed:.1f} м/с.")
        self.update_log(f"Ожидаемое время полёта: {time_seconds:.1f} секунд.")

    def simulation_step(self):
        # 1. Расчёт времени реального шага (например, 50 мс)
        real_dt = 0.05  # 50 ms, например, если через after вызываем каждые 50 мс

        # 2. Переводим в симуляционное время с учётом множителя:
        sim_dt = real_dt * self.simulation_speed_multiplier

        # 3. Двигаем дрона на его скорость * sim_dt (скорость в м/с, sim_dt в секундах)
        dx = self.drone_horizontal_speed * sim_dt
        dy = self.drone_vertical_speed * sim_dt

        # 4. Обновляем координаты, заряд и таймер (в симуляционных секундах!)
        self.drone_x += dx
        self.drone_y += dy

        self.simulation_time += sim_dt  # именно симуляционное время!
        self.charge -= self.battery_consumption_rate(dx, dy)  # расход на пройденный путь

        self.update_visuals()

        # 5. Запланировать следующий шаг
        self.root.after(int(real_dt * 1000), self.simulation_step)

    def setup_map_grid(self):
        """Настройка отображения карты."""
        import matplotlib.patches as patches  # Используем для добавления прямоугольников

        # Настройка сетки
        self.ax.set_xticks(np.arange(0, self.grid_width + 1, 1))
        self.ax.set_yticks(np.arange(0, self.grid_height + 1, 1))
        self.ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.set_xlim(0, self.grid_width)
        self.ax.set_ylim(0, self.grid_height)
        self.ax.set_xlabel('Расстояние (м), 1 клетка = 100м')
        self.ax.set_ylabel('Расстояние (м), 1 клетка = 100м')

        # Лес (весь фон)
        self.ax.fill_between(
            [0, self.grid_width],
            0,
            self.grid_height,
            color='green',
            alpha=0.9,
            edgecolor='darkgreen',
            linewidth=2,
            label='Лес'
        )

        # Река
        river_x = [0, 3, 5, 10, 15, 20, 25, 27, 30]
        river_y = [0, 2, 3, 4, 3.5, 3, 4, 5, 6]
        self.ax.fill_between(
            river_x,
            [y - 0.5 for y in river_y],
            [y + 0.5 for y in river_y],
            color='blue',
            alpha=1,
            edgecolor='darkblue',
            linewidth=2,
            label='Река'
        )

        # Дома
        house_positions = [
            (1, 1.5), (2, 2.5), (4, 3.5), (4.2, 3.8), (5, 3.7), (6, 4), (5.7, 3.7),
            (8, 4.3), (9, 4.5), (11, 4.7), (12, 4.5), (13, 4.8), (14, 4.6), (15, 4.9),
            (16, 4.2), (17, 4.3), (18, 4.4), (19, 4.2), (20, 4.3), (21, 4.1),
            (22, 4.2), (23, 4.4), (24, 4.5), (25, 4.6), (26, 5.3), (27, 5.9),
            (28, 6.2), (29, 6.3), (30, 6.1), (31, 6.4)
        ]
        for x, y in house_positions:
            self.ax.add_patch(
                patches.Rectangle(
                    (x, y+ 0.5),  # Левый нижний угол
                    0.15,  # Ширина (15 метров в масштабе 1 клетка = 100м)
                    0.1,  # Высота (10 метров в масштабе 1 клетка = 100м)
                    facecolor='orange',  # Цвет дома
                    edgecolor='brown',  # Цвет окантовки
                    linewidth=1,  # Толщина линии окантовки
                    alpha=1,  # Прозрачность (1 = непрозрачно)
                    label='Дом' if house_positions.index((x, y)) == 0 else None  # Добавляем в легенду только один раз
                )
            )

        # Зернохранилище
        self.ax.add_patch(
            patches.Rectangle(
                (13.80, 16),  # Левый нижний угол
                0.4,  # Ширина (40м, т.е. 0.4 в масштабе 1 клетка = 100м)
                2,  # Высота (200м, т.е. 2 в масштабе 1 клетка = 100м)
                facecolor='gray',  # Цвет зернохранилища
                edgecolor='darkgray',  # Цвет окантовки зернохранилища
                linewidth=1,  # Толщина линии окантовки
                alpha=1,  # Прозрачность (1 = непрозрачно)
                label='Зернохранилище'
            )
        )

        # Поля
        # Первое поле
        self.ax.add_patch(
            patches.Rectangle(
                (15, 11),  # Левый нижний угол
                5,  # Ширина
                7,  # Высота
                facecolor='saddlebrown',  # Цвет поля
                edgecolor='darkred',  # Окантовка поля
                linewidth=1,  # Толщина линии окантовки
                alpha=1,  # Прозрачность
                label='Поле'
            )
        )
        # Второе поле
        self.ax.add_patch(
            patches.Rectangle(
                (20.1, 11),  # Левый нижний угол
                5,  # Ширина
                7,  # Высота
                facecolor='saddlebrown',
                edgecolor='darkred',
                linewidth=1,
                alpha=1
            )
        )

        # Легенда
        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    def init_neural_network(self):
        """Инициализация и обучение нейросети."""
        print("\nИнициализация улучшенной нейросетевой модели...")
        self.nn = NeuralNetwork(8, 16, 2, dropout_rate=0.2)  # Используем улучшенную архитектуру

        # Генерация обучающих данных
        X, y = self.generate_data(5000)

        # Обучение модели
        self.nn.train(X, y, epochs=5000, lr=0.001, reg_lambda=0.01)

        # Оценка точности
        test_size = int(0.15 * len(X))
        X_test, y_test = X[-test_size:], y[-test_size:]
        self.test_accuracy = self.nn.evaluate(X_test, y_test)
        self.update_log(f"Точность нейросети на тестовых данных: {self.test_accuracy * 100:.2f}%")

        # Подготовка данных для графика
        epochs = list(range(0, 5001, 100))  # 51 точка

        # Обеспечиваем корректное соответствие размеров данных и эпох
        train_loss = self.nn.train_loss_history
        val_loss = self.nn.val_loss_history

        # Если в истории больше точек, чем в epochs, делаем сэмплирование по каждой 100-й эпохе
        if len(train_loss) > len(epochs):
            train_loss = train_loss[::100]
        if len(val_loss) > len(epochs):
            val_loss = val_loss[::100]

        # Обрезаем массивы, чтобы не было рассинхронизации длин
        min_len = min(len(epochs), len(train_loss), len(val_loss))
        epochs = epochs[:min_len]
        train_loss = train_loss[:min_len]
        val_loss = val_loss[:min_len]

        # Построение графика обучения
        self.ax_loss.plot(epochs, train_loss, label='Ошибка обучения', color='green', linestyle='-')
        self.ax_loss.plot(epochs, val_loss, label='Ошибка валидации', color='red', linestyle='-')

        # Легенда, если еще не создана
        if not self.ax_loss.get_legend():
            self.ax_loss.legend(fontsize=10, loc="upper right", frameon=True)

        # Обновляем холст для отображения изменений
        self.canvas_loss.draw()

    def update_drone_visuals(self):
        """Централизованное обновление всех визуальных параметров дрона."""
        # Определяем целевую высоту
        if self.is_landing and self.target_pos is not None:
            try:
                # Определяем высоту целевой станции
                station_idx = self.stations.index(self.target_pos.tolist())
                target_height = self.station_heights[station_idx]
            except ValueError:
                # Если станция не найдена, используем текущую высоту
                target_height = self.target_height
        else:
            # Если дрон не находится в режиме посадки, используем текущую высоту
            target_height = 0

        # Расчет соотношения высоты (чем ниже высота, тем меньше размер)
        height_ratio = max(0, min(1, self.drone_height / 5000.0))  # Высота относительно максимальной (5000 м)

        # Интерполяция размера
        target_size = self.min_drone_size + (self.max_drone_size - self.min_drone_size) * height_ratio

        # Плавное изменение размера дрона
        if self.is_landing or self.drone_height != self.target_height:
            self.drone_size += (target_size - self.drone_size) * 0.2  # Плавный переход (20% за кадр)

        self.drone_icon.set_sizes([self.drone_size ** 2])
        self.drone_icon.set_offsets(self.drone_pos)  # Обновляем позицию
        self.drone_icon.set_zorder(10)  # Устанавливаем порядок отображения

        # Принудительное обновление карты
        self.canvas_map.draw_idle()

    def update_ui(self):
        """Обновление элементов интерфейса."""
        remaining_percent = (self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS) * 100
        self.charge_label.config(
            text=f"Заряд: {remaining_percent:.2f}% ({self.remaining_capacity_watt_hours:.2f} Вт·ч)"
        )
        self.height_label.config(text=f"Высота: {self.drone_height:.0f} м")

        # Централизованное обновление визуальных параметров
        self.update_drone_visuals()

        # Проверяем, нужно ли обновлять траекторию
        if self.target_pos is not None:
            self.route_line.set_data(
                [self.drone_pos[0], self.target_pos[0]],
                [self.drone_pos[1], self.target_pos[1]]
            )
            self.route_line.set_zorder(5)  # Убедитесь, что маршрут ниже дрона

        # Принудительно обновляем карту
        self.canvas_map.draw_idle()

    def update_frame(self, frame):
        """Обновление состояния анимации на каждом кадре."""
        if self.is_landing:
            return self.perform_landing(frame)
        else:
            return self.move_drone(frame)

    def update_log(self, message: str) -> None:
        """Обновление лога сообщений."""
        self.log_text.insert('end', message + '\n')
        self.log_text.see('end')  # Автопрокрутка к новому сообщению
        self.log_text.update_idletasks()  # Принудительное обновление виджета

    def add_stations(self):
        """Добавление станций на карту."""
        self.station_plots = []
        for i, (x, y) in enumerate(self.stations):
            # Значок док-станции
            station = self.ax.plot(x, y, '^', markersize=15, color='green', zorder=3)[0]

            # Перекрестие (для отображения занятости)
            cross = self.ax.text(
                x, y, '×', fontsize=20, color='black',
                ha='center', va='center', visible=False, zorder=3
            )

            # Табличка с информацией о станции
            label = self.ax.text(
                x, y - 0.5,
                f'Станция {i + 1}\n({x * 100:.0f}м, {y * 100:.0f}м)\nВысота: {self.station_heights[i]:.0f}м',
                ha='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7), zorder=2
            )

            self.station_plots.append((station, cross, label))

    def apply_parameters(self):
        """Применение параметров из вкладки."""
        try:
            self.drone_height = float(self.entries['drone_height'].get())
            self.station_heights[0] = float(self.entries['station1_height'].get())
            self.station_heights[1] = float(self.entries['station2_height'].get())
            self.BATTERY_CAPACITY_WATT_HOURS = float(self.entries['battery_capacity_watt_hours'].get())
            self.BATTERY_CAPACITY = self.BATTERY_CAPACITY_WATT_HOURS * 3600  # Перевод в джоули
            self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS  # Начальный заряд
            self.update_stations_info()
            self.update_ui()
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные значения параметров")

    def on_hover(self, event):
        """Обработчик движения мыши."""
        if not self.lock_updates or event.inaxes != self.ax:
            return

        # Проверяем, находится ли курсор рядом с какой-либо станцией
        for i, (x, y) in enumerate(self.stations):
            if abs(event.xdata - x) < 0.5 and abs(event.ydata - y) < 0.5:
                station_info = (
                    f"Станция {i + 1}\n"
                    f"Высота: {self.station_heights[i]} м\n"
                    f"Статус: {'Занята' if self.station_statuses[i] else 'Свободна'}"
                )
                self.update_log(station_info)
                break

    def on_click(self, event):
        """Обработчик кликов мыши."""
        try:
            # Преобразование координат клика в координаты карты
            if event.inaxes != self.ax:
                self.update_log("Клик вне области карты! Игнорируем...")
                return  # Игнорируем клики вне области карты

            x, y = event.xdata, event.ydata
            self.update_log(f"Клик по карте: (x, y) = ({x:.2f}, {y:.2f})")

            # Проверка, что клик находится в пределах карты
            if x is None or y is None or not (0 <= x <= self.grid_width and 0 <= y <= self.grid_height):
                self.update_log("Координаты клика вне допустимых границ карты!")
                return

            # ЛКМ - установка точек маршрута
            if event.button == 1:  # ЛКМ
                if self.start_pos is None:  # Установка начальной точки
                    self.start_icon.set_data([x], [y])
                    self.start_pos = np.array([x, y])
                    self.drone_pos = np.array([x, y])
                    self.update_log(f"Начальная точка установлена: ({x:.2f}, {y:.2f})")
                    self.target_pos = None  # Сбрасываем предыдущую конечную точку
                    self.end_icon.set_data([], [])  # Очищаем конечную точку визуально
                elif self.target_pos is None:  # Установка конечной точки
                    self.end_icon.set_data([x], [y])
                    self.end_pos = np.array([x, y])  # Явно сохраняем конечную точку
                    self.target_pos = np.array([x, y])
                    self.update_log(f"Конечная точка установлена: ({x:.2f}, {y:.2f})")

                # Обновляем траекторию маршрута
                if self.start_pos is not None and self.target_pos is not None:
                    self.route_line.set_data(
                        [self.start_pos[0], self.target_pos[0]],
                        [self.start_pos[1], self.target_pos[1]]
                    )
                    self.update_log("Маршрут успешно обновлен!")

                # Перерисовываем интерфейс
                self.update_ui()

        except Exception as e:
            self.update_log(f"Ошибка в обработчике кликов: {str(e)}")

    def update_drone_height(self, text: str) -> None:
        """Обновление высоты дрона."""
        try:
            self.drone_height = float(text)
            self.update_ui()
        except ValueError:
            pass

    def update_station_height(self, station_idx: int, text: str) -> None:
        """Обновление высоты станции."""
        try:
            self.station_heights[station_idx] = float(text)
            self.update_stations_info()
            self.update_ui()
        except ValueError:
            pass

    def update_stations_info(self) -> None:
        """Обновление информации о станциях."""
        for i, (plot, height) in enumerate(zip(self.station_plots, self.station_heights)):
            plot[2].set_text(
                f'Станция {i + 1}\n({self.stations[i][0] * 100:.0f}м, {self.stations[i][1] * 100:.0f}м)\n'
                f'Высота: {height:.0f}м\n'
                f'Статус: {"Занята" if self.station_statuses[i] else "Свободна"}'
            )

    def set_charge(self, text: str) -> None:
        """Установка заряда батареи."""
        try:
            charge = float(text)
            self.initial_charge = np.clip(charge, 0.0, 1.0)
            self.charge = self.initial_charge
            self.update_ui()
        except ValueError:
            self.update_log("Некорректный ввод заряда")

    def reset_simulation(self, event=None):
        """Полный сброс симуляции."""
        if hasattr(self, 'animation') and self.animation:
            if self.animation.event_source:
                self.animation.event_source.stop()
            self.animation = None

        # Сброс таймера
        self.reset_timer()

        # Сброс состояния
        self.drone_pos = np.array([15, 10])
        self.drone_height = 500.0
        self.target_height = 0.0  # Сброс высоты цели
        self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS  # Сброс заряда батареи
        self.charge = self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS
        self.is_charging = False
        self.has_charged = False
        self.is_landing = False
        self.mission_started = False
        self.mission_active = False
        self.start_pos = None  # Сброс начальной точки маршрута
        self.target_pos = None  # Сброс целевой точки маршрута
        self.end_pos = None  # Сброс конечной точки маршрута

        # Очистка начальной и конечной точек на карте
        self.start_icon.set_data([], [])  # Удаляем начальную точку из интерфейса
        self.end_icon.set_data([], [])  # Удаляем конечную точку из интерфейса
        self.route_line.set_data([], [])  # Удаляем маршрут

        # Обновление интерфейса
        self.charge_label.config(
            text=f"Заряд: {self.charge * 100:.2f}% ({self.remaining_capacity_watt_hours:.2f} Вт·ч)"
        )
        self.height_label.config(text="Высота: 500 м")
        self.update_log("Симуляция сброшена.")
        self.canvas_map.draw()  # Обновляем карту

    def generate_drone_params(self) -> np.ndarray:
        """Генерация параметров для нейросети."""
        params = np.zeros(8)  # Массив из 8 параметров
        params[0] = self.charge
        params[1] = self.drone_height / 5000

        # Расчет параметров для двух станций
        for i in range(2):
            x, y = self.stations[i]
            dx = (self.drone_pos[0] - x) * self.cell_size
            dy = (self.drone_pos[1] - y) * self.cell_size
            dist = np.sqrt(dx ** 2 + dy ** 2) / 3000  # Нормализация расстояния
            params[2 + i * 2] = dist
            params[3 + i * 2] = 1.0 if self.station_statuses[i] else 0.0

        return params

    def calculate_energy_consumption(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                     start_height: float, end_height: float = 0) -> float:
        """Расчет потребления энергии для маршрута."""
        dx = (end_pos[0] - start_pos[0]) * self.cell_size
        dy = (end_pos[1] - start_pos[1]) * self.cell_size
        distance = np.sqrt(dx * dx + dy * dy)
        height_diff = start_height - end_height

        horizontal = distance * self.HORIZONTAL_ENERGY
        vertical = abs(height_diff) * (self.DESCENT_ENERGY if height_diff > 0 else self.CLIMB_ENERGY)
        systems = distance * self.SYSTEMS_ENERGY

        return horizontal + vertical + systems

    def consume_energy(self, distance: float = 0, height_change: float = 0, time_elapsed: float = 1):
        """Реалистичный расход энергии с учетом фиксированной скорости."""
        energy_joules = 0

        # Расход на горизонтальное перемещение
        if distance > 0:
            energy_joules += distance * self.horizontal_energy_per_meter

        # Расход на изменение высоты
        if height_change > 0:  # Подъем
            energy_joules += abs(height_change) * self.climb_energy_per_meter
        elif height_change < 0:  # Спуск
            energy_joules += abs(height_change) * self.descent_energy_per_meter

        # Системный расход энергии (с учетом времени)
        energy_joules += self.system_energy_per_second * time_elapsed

        # Конвертируем в Вт·ч и вычитаем из оставшегося заряда
        energy_watt_hours = energy_joules / 3600
        self.remaining_capacity_watt_hours = max(0.0, self.remaining_capacity_watt_hours - energy_watt_hours)

        # Проверка критического уровня заряда
        if self.remaining_capacity_watt_hours <= 0:
            self.update_log("Критический разряд батареи! Дрон остановился.")
            self.mission_active = False
            if hasattr(self, 'animation') and self.animation:
                self.animation.event_source.stop()

        # Обновляем интерфейс
        self.update_ui()

    def move_drone_vertically(self, *args) -> List[plt.Artist]:
        """Движение дрона вверх или вниз с учётом множителя симуляции и расхода заряда."""
        if not self.is_landing:
            return [self.drone_icon]

        # Реальное время обновления
        current_time = time.time()
        real_delta_time = current_time - self.last_update_time  # Прошедшее время в реальном мире
        self.last_update_time = current_time  # Обновляем время последнего кадра

        # Симуляционное время с учётом множителя
        sim_delta_time = real_delta_time * self.simulation_speed_multiplier

        # Расчёт изменения высоты
        height_change = self.vertical_speed * sim_delta_time  # Изменение высоты за sim_delta_time (в метрах)
        height_difference = self.target_height - self.drone_height  # Текущая разница высот (в метрах)

        if abs(height_difference) <= abs(height_change):  # Дрон достиг нужной высоты
            self.drone_height = self.target_height  # Устанавливаем точную высоту
            self.is_landing = False  # Завершаем посадку/подъем
            self.update_log(f"Дрон достиг рабочей высоты: {self.target_height:.2f} м.")
        else:
            # Двигаем дрон вверх или вниз
            self.drone_height += height_change if height_difference > 0 else -height_change

            # Расход энергии на подъем/спуск
            self.consume_energy(height_change=height_change)

            # Логирование каждые 10 метров, обновляем целевую высоту
            if abs(self.drone_height - self.last_logged_height) >= 10:
                self.last_logged_height = self.drone_height
                self.update_log(
                    f"Размер дрона: {self.drone_size:.2f}, Высота: {self.drone_height:.2f}, Целевая высота: {self.target_height:.2f}"
                )

        # Централизованное обновление визуальных параметров
        self.update_drone_visuals()

        return [self.drone_icon]

    def perform_landing(self, frame: int) -> List[plt.Artist]:
        """Обновление состояния дрона при посадке."""
        # Проверяем, завершена ли посадка
        if not self.is_landing:
            return [self.drone_icon]

        # Движение дрона вверх или вниз
        artists = self.move_drone_vertically()

        # Проверка завершения посадки
        if not self.is_landing and abs(self.drone_height - self.target_height) < 1e-2:
            # Логируем завершение посадки только один раз
            if not hasattr(self, "landing_logged") or not self.landing_logged:
                self.update_log("Дрон успешно завершил посадку.")
                self.landing_logged = True  # Устанавливаем флаг, чтобы предотвратить повторный вывод

            # Если дрон находится на высоте одной из док-станций, начинаем зарядку
            for idx, height in enumerate(self.station_heights):
                if abs(self.drone_height - height) < 1e-2:  # Проверяем совпадение высоты с док-станцией
                    self.update_log(f"Дрон находится на высоте станции {idx + 1}: {height} м. Инициируется зарядка.")
                    self.charge_at_station(idx)  # Передаем индекс станции
                    return artists

            # Если не удалось идентифицировать станцию, завершаем симуляцию
            self.update_log("Ошибка: дрон не совпадает по высоте ни с одной из станций.")
            self.complete_simulation()
            return artists

        # Обновляем визуализацию и интерфейс
        self.update_drone_visuals()
        self.update_ui()
        return artists

    def handle_arrival(self) -> None:
        """Обработка достижения цели."""
        self.drone_pos = self.target_pos.copy()

        # Проверка на достижение станции
        for i, station in enumerate(self.stations):
            if np.linalg.norm(self.drone_pos - np.array(station)) < 0.1:
                # Начало посадки/подъема
                self.is_landing = True
                self.target_height = self.station_heights[i]

                # Логирование
                self.update_log(f"Дрон достиг станции {i + 1}: {station}, инициируется посадка/подъем.")

                # Запуск анимации посадки
                if self.animation and self.animation.event_source:
                    self.animation.event_source.stop()

                self.animation = FuncAnimation(
                    self.fig_map, self.perform_landing,
                    frames=None,
                    interval=50,
                    blit=False,
                    repeat=False,
                    cache_frame_data=False
                )
                return

        # Если станция не достигнута, проверяем, достиг ли дрон конечной точки маршрута
        if np.linalg.norm(self.drone_pos - self.end_pos) < 0.1:
            self.update_log("Дрон достиг конечной точки маршрута!")
            self.complete_simulation()
        else:
            self.update_log("Дрон продолжает миссию.")

    def find_best_landing_station(self) -> Tuple[Optional[int], float]:
        """Поиск оптимальной станции для посадки."""
        min_energy = float('inf')
        best_station = None

        for i, (x, y) in enumerate(self.stations):
            if self.station_statuses[i]:
                continue

            energy = self.calculate_energy_consumption(
                self.drone_pos, [x, y],
                self.drone_height, self.station_heights[i]
            )

            if energy < min_energy:
                min_energy = energy
                best_station = i

        return best_station, min_energy if best_station is not None else (None, float('inf'))

    def check_emergency_landing(self) -> bool:
        """Проверка необходимости аварийной посадки и делегирование выбора станции."""
        try:
            # Проверка, если резервный режим уже активен
            if self.is_forced_landing:
                return False

            # Рассчитываем энергию до конечной точки
            energy_to_target = self.calculate_energy_consumption(
                self.drone_pos, self.end_pos, self.drone_height, 0
            )
            available_energy = self.remaining_capacity_watt_hours

            # Логируем доступную энергию и энергию до цели
            self.update_log(f"Энергия до цели: {energy_to_target:.2f} Дж, доступно: {available_energy:.2f} Дж")

            # Условие: достаточно энергии на полет
            if energy_to_target < available_energy * 0.8:
                self.update_log("Энергии достаточно для завершения маршрута. Аварийная посадка не требуется.")
                return False

            # Попытка выбора станции с использованием нейросети
            self.update_log("Попытка выбора станции с использованием нейросети...")
            if self.force_decision():
                self.update_log("Выбор станции завершен с использованием нейросети.")
                return True

            # Если нейросеть не справилась, активируем резервный режим
            self.update_log("Нейросеть не справилась. Активируется резервный режим.")
            if self.activate_reserve_mode():
                self.update_log("Выбор станции завершен с использованием резервного режима.")
                return True

            # Если не удалось выбрать станцию
            self.update_log("Не удалось выбрать станцию. Полет продолжается.")
            return False

        except Exception as e:
            self.update_log(f"Ошибка в check_emergency_landing: {str(e)}")
            return False

    def move_drone(self, *args) -> List[plt.Artist]:
        """Движение дрона с учётом множителя скорости."""
        if not self.mission_active or self.target_pos is None:
            return [self.drone_icon, self.route_line]

        # Преобразуем позицию дрона в float для совместимости с вычислениями
        self.drone_pos = self.drone_pos.astype(float)

        # Реальное время обновления
        current_time = time.time()
        real_delta_time = current_time - self.last_update_time  # Прошедшее время в реальном мире
        self.last_update_time = current_time  # Обновляем время последнего кадра

        # Симуляционное время с учётом множителя
        sim_delta_time = real_delta_time * self.simulation_speed_multiplier

        # Дрон движется с фиксированной скоростью 15 м/с (симуляционное время)
        distance_step = self.real_speed * sim_delta_time  # Пройденное расстояние за sim_delta_time
        direction = self.target_pos - self.drone_pos
        distance_to_target = np.linalg.norm(direction) * self.cell_size  # В метрах

        if distance_step >= distance_to_target:  # Дрон достиг цели
            self.drone_pos = self.target_pos.copy()  # Устанавливаем позицию точно на цель
            self.update_log(f"Дрон достиг цели: {self.target_pos}.")
            self.handle_arrival()
            self.mission_active = False
        else:
            # Двигаем дрон в сторону цели
            direction_normalized = direction / np.linalg.norm(direction)  # Нормализация вектора
            step_vector = direction_normalized * (distance_step / self.cell_size)  # Шаг в клетках
            self.drone_pos += step_vector

            # Расход энергии (на основе симуляционного времени)
            self.consume_energy(distance=distance_step)

        # Размер дрона не изменяется при горизонтальном движении
        self.drone_icon.set_offsets(self.drone_pos)

        # Централизованное обновление визуальных параметров
        self.update_drone_visuals()

        # Обновляем карту
        return [self.drone_icon, self.route_line]

    def start_animation(self):
        """Запуск анимации с учётом множителя скорости."""
        self.update_log("Попытка запуска анимации.")
        if hasattr(self, 'animation') and self.animation:
            if self.animation.event_source:
                self.animation.event_source.stop()
                self.update_log("Существующая анимация остановлена.")

        self.start_timer()  # Реальный таймер симуляции
        self.last_update_time = time.time()  # Для расчёта real_delta_time

        # Запуск анимации
        self.animation = FuncAnimation(
            self.fig_map,
            self.update_frame,
            interval=self.frame_interval,
            blit=False,
            repeat=False,
            cache_frame_data=False
        )

        self.update_log("Анимация успешно запущена.")
        self.canvas_map.draw()

    def start_mission(self, event=None) -> None:
        """Запуск миссии с проверкой параметров."""
        try:
            # Проверка валидности миссии
            if not self.validate_mission():
                return

            # Остановка предыдущей анимации
            if hasattr(self, 'animation') and self.animation is not None:
                if self.animation.event_source:
                    self.animation.event_source.stop()

            # Инициализация позиций
            start_data = self.start_icon.get_data()
            end_data = self.end_icon.get_data()

            # Проверка наличия данных для начальной и конечной точек
            if len(start_data[0]) == 0 or len(start_data[1]) == 0:
                raise ValueError("Начальная точка не установлена.")
            if len(end_data[0]) == 0 or len(end_data[1]) == 0:
                raise ValueError("Конечная точка не установлена.")

            self.start_pos = np.array([start_data[0][0], start_data[1][0]])
            self.target_pos = np.array([end_data[0][0], end_data[1][0]])
            self.drone_pos = self.start_pos.copy()

            # Логирование текущей и конечной позиции
            self.update_log(f"Начало миссии. Текущая позиция: {self.drone_pos}, цель: {self.target_pos}.")

            # Настройка состояния
            self.is_forced_landing = False
            self.mission_active = True
            self.mission_started = True
            self.is_landing = False

            # Настройка визуализации
            self.route_line.set_data(
                [self.start_pos[0], self.target_pos[0]],
                [self.start_pos[1], self.target_pos[1]]
            )
            self.route_line.set_linestyle('--')
            self.route_line.set_color('b')
            self.drone_icon.set_color('k')

            self.update_ui()

            # Запуск анимации
            self.start_animation()

        except ValueError as ve:
            self.update_log(f"Ошибка запуска миссии: {str(ve)}")
        except Exception as e:
            self.update_log(f"Ошибка запуска миссии: {str(e)}")
            self.mission_active = False

    def validate_mission(self) -> bool:
        """Проверка валидности миссии."""
        # Проверка стартовой точки
        start_data = self.start_icon.get_data()
        if len(start_data[0]) == 0:
            self.update_log("Ошибка: Установите стартовую точку (ЛКМ на карте)")
            return False

        # Проверка конечной точки
        end_data = self.end_icon.get_data()
        if len(end_data[0]) == 0:
            self.update_log("Ошибка: Установите конечную точку (ЛКМ на карте)")
            return False

        # Проверка станций
        if len(self.stations) < 2:
            self.update_log("Ошибка: В системе должно быть 2 станции")
            return False

        return True

    def check_mission_feasibility(self) -> bool:
        """Проверка возможности выполнения миссии."""
        dx = (self.end_pos[0] - self.start_pos[0]) * self.cell_size
        dy = (self.end_pos[1] - self.start_pos[1]) * self.cell_size
        distance = np.sqrt(dx * dx + dy * dy)

        required_energy = distance * (self.HORIZONTAL_ENERGY + self.SYSTEMS_ENERGY)
        available_energy = self.charge * self.BATTERY_CAPACITY * (1 - self.safety_margin)

        if required_energy > available_energy:
            self.update_log(
                f"НЕДОСТАТОЧНО ЗАРЯДА! Требуется {required_energy:.0f} Дж, доступно {available_energy:.0f} Дж")

            best_station, station_energy = self.find_best_landing_station()

            if best_station is not None and station_energy < self.charge * self.BATTERY_CAPACITY:
                self.update_log(
                    f"Автоматическая посадка на станцию {best_station + 1} (требуется {station_energy:.0f} Дж)")
                self.target_pos = np.array(self.stations[best_station])
                self.is_forced_landing = True

                self.route_line.set_data(
                    [self.drone_pos[0], self.target_pos[0]],
                    [self.drone_pos[1], self.target_pos[1]]
                )
                self.route_line.set_linestyle(':')
                self.route_line.set_color('r')

                self.start_animation()
                return False

        return True

    def force_decision(self) -> bool:
        """Принудительный выбор док-станции с использованием нейросети."""
        try:
            # Генерация параметров для нейросети
            X = self.generate_drone_params().reshape(1, -1)
            neural_output = self.nn.forward(X)[0]  # Результат работы нейросети
            neural_choice = np.argmax(neural_output)
            neural_confidence = neural_output[neural_choice]

            # Логируем результаты работы нейросети
            self.update_log(f"Нейросеть выбрала станцию {neural_choice + 1} с уверенностью {neural_confidence:.2f}")

            # Если уверенность нейросети ниже порога (0.5), активируем резервный режим
            confidence_threshold = 0.5
            if neural_confidence < confidence_threshold:
                self.update_log(
                    f"⚠ Уверенность нейросети ({neural_confidence:.2f}) ниже порога ({confidence_threshold}). Активируется резервный режим.")
                if self.activate_reserve_mode():
                    self.update_log("Резервный режим успешно завершил выбор станции.")
                    return True
                self.update_log("Резервный режим не смог выбрать станцию. Полет продолжается.")
                return False

            # Устанавливаем цель на основании выбора нейросети
            self.target_pos = np.array(self.stations[neural_choice])
            self.update_log(f"Принудительно выбрана станция {neural_choice + 1} для посадки.")

            # Сравнение с резервным методом
            best_station, _ = self.find_best_landing_station()
            if best_station == neural_choice:
                self.update_log(f"✅ Нейросеть сделала правильный выбор: станция {neural_choice + 1}.")
            else:
                self.update_log(
                    f"⚠ Нейросеть выбрала станцию {neural_choice + 1}, но резервный метод предпочел станцию {best_station + 1}.")

            self.route_line.set_data(
                [self.drone_pos[0], self.target_pos[0]],
                [self.drone_pos[1], self.target_pos[1]]
            )
            self.is_forced_landing = True
            return True

        except Exception as e:
            self.update_log(f"Ошибка в принудительном выборе: {str(e)}")
            return False

    def activate_reserve_mode(self) -> bool:
        """Активация резервного режима для сложных задач."""
        try:
            # Рассчитываем энергию до каждой станции и выбираем станцию с минимальными энергозатратами
            available_energy = self.remaining_capacity_watt_hours
            feasible_stations = []

            for i, station in enumerate(self.stations):
                if self.station_statuses[i]:
                    continue  # Пропускаем занятые станции

                energy = self.calculate_energy_consumption(
                    self.drone_pos, station, self.drone_height, self.station_heights[i]
                )

                feasible_stations.append((i, energy))

            # Проверяем, если нет достижимых станций
            if not feasible_stations:
                self.update_log("Нет достижимых станций! Продолжаю полет с риском...")
                return False

            # Сортируем станции по энергозатратам
            feasible_stations.sort(key=lambda x: x[1])
            best_station = feasible_stations[0][0]

            # Логируем выбор резервного режима
            self.update_log(f"⚠ Активирован резервный режим выбора док-станции ⚠")
            self.update_log(f"Станция выбрана по минимальным энергозатратам: {best_station + 1}")

            # Устанавливаем цель
            self.target_pos = np.array(self.stations[best_station])
            self.route_line.set_data(
                [self.drone_pos[0], self.target_pos[0]],
                [self.drone_pos[1], self.target_pos[1]]
            )
            self.is_forced_landing = True
            return True

        except Exception as e:
            self.update_log(f"Ошибка в резервном режиме: {str(e)}")
            return False

    def charge_at_station(self, station_idx: int):
        """Реалистичная зарядка дрона на док-станции с учетом фаз CC и CV."""
        if self.is_charging or getattr(self, 'has_charged', False):
            self.update_log("Зарядка уже активна или завершена. Повторная зарядка невозможна.")
            return

        # Проверка высоты
        if abs(self.drone_height - self.station_heights[station_idx]) >= 1e-2:
            self.update_log(f"Ошибка: Зарядка возможна только на высоте станции {station_idx + 1}.")
            return

        # Проверка полного заряда
        if self.remaining_capacity_watt_hours >= self.BATTERY_CAPACITY_WATT_HOURS:
            self.update_log("Батарея уже полностью заряжена.")
            return

        # Расчет параметров зарядки
        max_power = self.CHARGING_VOLTAGE * self.CHARGING_CURRENT  # Максимальная мощность (Вт)
        efficiency = self.CHARGING_EFFICIENCY  # КПД зарядки

        # Инициализация лога зарядки
        self.charge_log = {
            "time": [],
            "charge_percent": [],
            "power": [],
            "remaining_energy": [],
            "current": [],
            "voltage": []
        }

        self.update_log(f"Зарядка началась на станции {station_idx + 1}. Текущий заряд: {self.charge * 100:.2f}% "
                        f"({self.remaining_capacity_watt_hours:.2f} Вт·ч).")

        self.is_charging = True
        self._last_elapsed_time = 0
        start_time = time.time()

        def update_charge():
            nonlocal max_power

            elapsed_time_real = time.time() - start_time
            elapsed_time_sim = elapsed_time_real * self.simulation_speed_multiplier
            time_step = elapsed_time_sim - self._last_elapsed_time
            self._last_elapsed_time = elapsed_time_sim

            # Фаза зарядки
            if self.charge < 0.9:
                power = max_power
                current = self.CHARGING_CURRENT
                voltage = power / current if current > 0 else 0
            else:
                power = max_power * (1 - (self.charge - 0.9) / 0.1)
                power = max(0, power)
                voltage = self.CHARGING_VOLTAGE
                current = power / voltage if voltage > 0 else 0

            if power <= 0:
                self.update_log("Ошибка: мощность зарядки стала равна нулю. Завершаем зарядку.")
                self.complete_charge()
                return

            charge_increment = (power * efficiency * time_step) / 3600
            charge_increment = min(charge_increment,
                                   self.BATTERY_CAPACITY_WATT_HOURS - self.remaining_capacity_watt_hours)

            self.remaining_capacity_watt_hours += charge_increment
            self.charge = self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS

            # Лог зарядки для графика
            self.charge_log["time"].append(elapsed_time_sim)
            self.charge_log["charge_percent"].append(self.charge * 100)
            self.charge_log["power"].append(power)
            self.charge_log["remaining_energy"].append(self.remaining_capacity_watt_hours)
            self.charge_log["current"].append(current)
            self.charge_log["voltage"].append(voltage)

            self.update_log(
                f"Заряд батареи: {self.charge * 100:.2f}% ({self.remaining_capacity_watt_hours:.2f} Вт·ч). "
                f"Мощность: {power:.2f} Вт, Ток: {current:.2f} А, Напряжение: {voltage:.2f} В."
            )

            if self.charge >= 0.9999:
                self.complete_charge()
            else:
                interval = int(1000 / self.simulation_speed_multiplier)
                self.root.after(interval, update_charge)

        update_charge()

    def charge_timer(self, remaining_time):
        if remaining_time <= 0 or self.charge >= 1.0:
            self.charge = 1.0
            self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS
            self.update_log("Зарядка завершена. Батарея дрона полностью заряжена.")
            self.is_charging = False
            self.resume_mission()
            return

        # Обновление заряда
        self.charge = min(1.0, self.charge + self.charge_increment)
        self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS * self.charge
        self.charge_label.config(text=f"Заряд: {self.charge * 100:.2f}%")

        # Обновление лога каждые 5 секунд
        if int(remaining_time) % 5 == 0:
            self.update_log(
                f"Заряд батареи: {self.charge * 100:.2f}% ({self.remaining_capacity_watt_hours:.2f} Вт·ч)"
            )

        # Планируем следующий вызов
        self.root.after(100, self.charge_timer, remaining_time - 0.1)

    def complete_charge(self):
        """Завершение зарядки, обновление графика и подъем на рабочую высоту."""
        self.charge = 1.0
        self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS
        self.update_log("Зарядка завершена. Батарея дрона полностью заряжена.")
        self.is_charging = False
        self.has_charged = True

        # Обновление графика при завершении зарядки
        self.plot_charge_graph()

        # Расход энергии на подъем на рабочую высоту
        self.target_height = float(self.entries['drone_height'].get())  # Используем значение из параметров
        self.is_landing = True
        self.update_log(
            f"Дрон возвращается на рабочую высоту: {self.target_height:.2f} м. is_landing={self.is_landing}.")

        def update_height():
            """Обновление высоты при подъеме/спуске."""
            if not self.is_landing:
                self.update_log("Подъем завершен. Состояние посадки отключено (is_landing=False).")
                self.resume_mission()  # Продолжение миссии после подъема
                return

            current_time = time.time()
            elapsed_time_real = current_time - self.last_update_time
            self.last_update_time = current_time

            sim_delta_time = elapsed_time_real * self.simulation_speed_multiplier
            height_change = self.vertical_speed * sim_delta_time
            height_difference = self.target_height - self.drone_height

            if abs(height_difference) <= abs(height_change):  # Достигли рабочей высоты
                self.drone_height = self.target_height
                self.consume_energy(height_change=height_difference)
                self.is_landing = False
                self.update_log(f"Дрон достиг рабочей высоты: {self.target_height:.2f} м. "
                                f"Флаг is_landing={self.is_landing}. Начинаем продолжение миссии.")
                self.resume_mission()  # Продолжаем миссию
                return

            self.drone_height += height_change if height_difference > 0 else -height_change
            self.consume_energy(height_change=height_change)
            self.update_ui()

            self.root.after(100, update_height)

        self.last_update_time = time.time()
        update_height()

    def resume_mission(self):
        """Продолжение миссии после зарядки."""
        self.update_log("Попытка продолжить миссию.")
        self.is_landing = False  # Завершаем состояние посадки

        # Проверяем конечную точку маршрута
        if self.end_pos is None:
            self.update_log("Ошибка: конечная точка маршрута не установлена! Попытка восстановления...")
            if self.target_pos is not None:  # Используем последнюю известную цель
                self.end_pos = self.target_pos.copy()
                self.update_log(f"Конечная точка восстановлена: {self.end_pos}")
            else:
                self.update_log("Ошибка восстановления конечной точки маршрута! Миссия не может быть продолжена.")
                return

        # Устанавливаем текущую цель на конечную точку маршрута
        self.target_pos = self.end_pos.copy()
        self.update_log(f"Текущая цель установлена: {self.target_pos}. Флаг mission_active=True.")
        self.mission_active = True

        # Обновляем маршрут на карте
        self.route_line.set_data(
            [self.drone_pos[0], self.target_pos[0]],
            [self.drone_pos[1], self.target_pos[1]]
        )
        self.update_log("Маршрут до конечной точки обновлен.")

        # Принудительно перезапускаем анимацию
        if hasattr(self, 'animation') and self.animation:
            if self.animation.event_source:
                self.animation.event_source.stop()
                self.update_log("Существующая анимация остановлена.")

        self.update_log("Запуск новой анимации для продолжения миссии.")
        self.start_animation()

    def complete_simulation(self):
        """Завершение симуляции и остановка таймера."""
        # Завершаем миссию только если достигнута конечная точка маршрута
        if np.linalg.norm(self.drone_pos - self.end_pos) < 0.1:
            self.stop_timer()
            if hasattr(self, 'animation') and self.animation and self.animation.event_source:
                self.animation.event_source.stop()

            # Установка флагов завершения
            self.mission_active = False
            self.is_landing = False

            # Лог финального времени и завершения миссии
            self.update_log(f"Симуляция завершена за {self.simulation_time:.1f} секунд.")
            self.update_log("Дрон успешно завершил миссию.")
            self.update_ui()
        else:
            self.update_log("Миссия не завершена. Дрон продолжает выполнение задания.")

if __name__ == "__main__":
    root = tk.Tk()
    app = Simulation(root)
    root.mainloop()
"""
После подзарядки на док-станции дрон должен дальше лететь по заданию. Емкость, ампер часы, ватт часы.
Интерфейс
"""



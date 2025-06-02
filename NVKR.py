import numpy as np
from tkinter import filedialog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Tuple, Optional

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU функция активации."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Производная ReLU функции активации."""
    return np.where(x > 0, 1, 0)

FEATURE_ORDER = [
    "charge_norm",
    "height_norm",
    "station0_dist_norm",
    "station0_status",
    "station1_dist_norm",
    "station1_status",
    "station0_height_diff_norm",
    "station1_height_diff_norm",
]

def get_nn_features(charge_watt_hours, battery_capacity, height, station_data):
    """
    Генерация признаков для нейросети по единой структуре.
    station_data: [{'dist', 'status', 'height'} для каждой станции]
    """
    features = [
        charge_watt_hours / battery_capacity,
        height / 5000,
        station_data[0]['dist'] / 3000,
        station_data[0]['status'],
        station_data[1]['dist'] / 3000,
        station_data[1]['status'],
        (station_data[0]['height'] - height) / 5000,
        (station_data[1]['height'] - height) / 5000,
    ]
    return np.array(features)

@staticmethod
def static_calculate_energy_consumption(start_pos, end_pos, start_height, end_height,
                                        HORIZONTAL_ENERGY, CLIMB_ENERGY, DESCENT_ENERGY, SYSTEMS_ENERGY):
    dx = (end_pos[0] - start_pos[0]) * 100  # если cell_size=100m
    dy = (end_pos[1] - start_pos[1]) * 100
    distance = np.sqrt(dx * dx + dy * dy)
    height_diff = end_height - start_height
    horizontal = distance * HORIZONTAL_ENERGY
    vertical = abs(height_diff) * (CLIMB_ENERGY if height_diff > 0 else DESCENT_ENERGY)
    systems = distance * SYSTEMS_ENERGY
    total_joules = horizontal + vertical + systems
    return total_joules / 3600  # в Вт·ч

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

class DroneMotionController:
    """
    Движение дрона: horizontal — на одной высоте (по карте, макс. скорость 15 м/с),
    vertical — только для изменения высоты (подъём/спуск, макс. скорость 2.5 м/с).
    """
    def __init__(
        self,
        mass=0.9,
        max_speed_horizontal=15.0,
        max_speed_vertical=2.5,
        accel_horizontal=4.0,
        decel_horizontal=4.5,
        accel_vertical=1.5,
        decel_vertical=1.8
    ):
        self.mass = mass
        self.max_speed_horizontal = max_speed_horizontal
        self.max_speed_vertical = max_speed_vertical
        self.accel_horizontal = accel_horizontal
        self.decel_horizontal = decel_horizontal
        self.accel_vertical = accel_vertical
        self.decel_vertical = decel_vertical

        # Текущее состояние
        self.pos = np.zeros(2)    # [x, y] (для движения по карте в метрах)
        self.vel = np.zeros(2)    # [vx, vy]
        self.target = np.zeros(2) # [tx, ty]
        self.axis = 'horizontal'  # 'horizontal' (движение по карте), 'vertical' (изменение высоты)

    def set_position(self, pos):
        self.pos = np.array(pos, dtype=float)

    def set_velocity(self, vel):
        self.vel = np.array(vel, dtype=float)

    def set_target(self, target):
        self.target = np.array(target, dtype=float)

    def set_axis(self, axis):
        assert axis in ('horizontal', 'vertical')
        self.axis = axis

    def get_max_speed(self):
        # По карте (ось X/Y) — horizontal, по высоте (ось Y только высота) — vertical
        return self.max_speed_horizontal if self.axis == 'horizontal' else self.max_speed_vertical

    def get_accel(self):
        return self.accel_horizontal if self.axis == 'horizontal' else self.accel_vertical

    def get_decel(self):
        return self.decel_horizontal if self.axis == 'horizontal' else self.decel_vertical

    def update(self, dt):
        """
        Интеграция движения: плавное ускорение, торможение, ОСТАНОВКА В ЦЕЛЕВОЙ ТОЧКЕ.
        Останавливаемся точно в target, если цель пересечена или близко.
        """
        direction = self.target - self.pos
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            # Дрон уже на месте, сбрасываем скорость
            self.pos = self.target.copy()
            self.vel *= 0
            return

        dir_norm = direction / distance
        v_along = np.dot(self.vel, dir_norm)
        v_max = self.get_max_speed()
        a_accel = self.get_accel()
        a_decel = self.get_decel()
        braking_dist = (v_along ** 2) / (2 * a_decel) if a_decel > 0 else 0

        # Сохраним старую позицию для проверки пересечения цели
        prev_pos = self.pos.copy()

        # --- Логика ускорения / торможения ---
        if distance > braking_dist:
            # Фаза разгона/крейсерская скорость
            v_target = min(v_max, v_along + a_accel * dt)
        else:
            # Фаза торможения
            v_target = max(0, v_along - a_decel * dt)

        self.vel = dir_norm * v_target
        self.pos += self.vel * dt

        # --- КЛЮЧЕВАЯ ПРОВЕРКА: пересечение цели ---
        # Если скалярное произведение направлений prev->target и new->target < 0 — дрон пересёк цель!
        new_direction = self.target - self.pos
        if np.dot(direction, new_direction) <= 0 or np.linalg.norm(new_direction) < 0.05:
            # Жёстко "прилипнуть" к цели и сбросить скорость
            self.pos = self.target.copy()
            self.vel *= 0

        # --- Альтернатива: если очень близко и скорость низкая, тоже останавливаемся ---
        if np.linalg.norm(self.target - self.pos) < 0.05 and np.linalg.norm(self.vel) < 0.05:
            self.pos = self.target.copy()
            self.vel *= 0

    def get_speed(self):
        return np.linalg.norm(self.vel)

    def get_speed_vector(self):
        return self.vel.copy()

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
        """
        Инициализация класса Simulation.
        Все параметры, связанные с визуализацией (Figure, Axes, plot/scatter-объекты),
        должны создаваться только ПОСЛЕ инициализации Figure/Axes (обычно — в init_simulation_tab).
        """
        self.motion_controller = DroneMotionController(
            mass=0.9,
            max_speed_horizontal=15.0,
            max_speed_vertical=2.5,
            accel_horizontal=4.0,
            decel_horizontal=4.5,
            accel_vertical=1.5,
            decel_vertical=1.8,
        )
        self.root = root
        self.root.title("Система управления БПЛА")

        # --- Энергетические параметры ---
        self.HORIZONTAL_ENERGY = 10  # Энергия на горизонтальный полет (Дж/м)
        self.CLIMB_ENERGY = 96.83  # Энергия на подъем (Дж/м)
        self.DESCENT_ENERGY = 40  # Энергия на спуск (Дж/м)
        self.SYSTEMS_ENERGY_WATTS = 5  # Системное энергопотребление (Вт)
        self.SYSTEMS_ENERGY = self.SYSTEMS_ENERGY_WATTS / 3600  # (Дж/сек)
        self.OCCUPATION_PENALTY = 0.2  # Штраф за занятую станцию

        # --- Параметры батареи ---
        self.BATTERY_CAPACITY_WATT_HOURS = battery_capacity_watt_hours
        self.BATTERY_CAPACITY = self.BATTERY_CAPACITY_WATT_HOURS * 3600  # В джоулях
        self.VOLTAGE_BATTERY = 15.4
        self.INTERNAL_RESISTANCE = self.VOLTAGE_BATTERY / 4

        # --- Параметры зарядки ---
        self.CHARGING_CURRENT = charging_current
        self.CHARGING_VOLTAGE = charging_voltage
        self.CHARGING_EFFICIENCY = charging_efficiency

        # --- Состояние зарядки ---
        self.charge = 0.8
        self.is_charging = False

        # --- Параметры карты ---
        self.grid_width = 30
        self.grid_height = 20
        self.cell_size = 100

        # --- Параметры дрона ---
        self.drone_mass = 0.9
        self.drone_height = 500.0
        self.target_height = 0.0
        self.last_logged_height = self.drone_height
        self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS
        self.charge = self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS
        self.energy_usage = 300
        self.range = 15000
        self.horizontal_energy_per_meter = self.HORIZONTAL_ENERGY
        self.climb_energy_per_meter = self.CLIMB_ENERGY
        self.descent_energy_per_meter = self.DESCENT_ENERGY
        self.system_energy_per_second = self.SYSTEMS_ENERGY

        # --- Визуализация дрона (размеры) ---
        self.drone_size = 15
        self.max_drone_size = 20
        self.min_drone_size = 5

        # --- Скорость дрона ---
        self.real_speed = 15.0
        self.vertical_speed = 2.5
        self.simulation_speed = 1.0
        self.frame_interval = 100

        # --- Скорость в клетках/секунду ---
        self.speed = self.real_speed / self.cell_size

        # --- Параметры станций ---
        self.stations = [[5, 10], [25, 10]]
        self.station_heights = [100.0, 100.0]
        self.station_statuses = [False, False]
        self.last_charged_station_idx = None

        # --- Состояние дрона и маршрут ---
        self.drone_pos = np.array([15, 10], dtype=float)  # ВАЖНО: до drone_path!
        self.drone_path = [self.drone_pos.copy()]  # История траектории дрона (для "следа")
        self.mission_active = False
        self.is_forced_landing = False
        self.is_landing = False
        self.returning_home = False

        # --- Параметры маршрута ---
        self.start_pos = None
        self.target_pos = None
        self.end_pos = None

        # --- GUI: вкладки ---
        self.notebook = ttk.Notebook(self.root)
        self.tab_params = ttk.Frame(self.notebook)
        self.tab_plot = ttk.Frame(self.notebook)
        self.tab_sim = ttk.Frame(self.notebook)

        # Стили вкладок: обычный (не жирный) шрифт, контур при выборе
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=('Arial', 14, 'normal'))  # Обычный шрифт

        # Включить focus/highlight вокруг активной вкладки
        style.map("TNotebook.Tab",
                  foreground=[("selected", "#000"), ("active", "#000")],
                  background=[("selected", "#fff"), ("active", "#f5f5f5")],
                  bordercolor=[("selected", "#1e90ff"), ("active", "#1e90ff")],  # Голубой контур
                  lightcolor=[("selected", "#1e90ff"), ("active", "#1e90ff")],
                  borderwidth=[("selected", 2), ("!selected", 1)]
                  )
        self.notebook.add(self.tab_params, text="Начальные параметры")
        self.notebook.add(self.tab_plot, text="Графики")
        self.notebook.add(self.tab_sim, text="Симуляция")
        self.notebook.pack(expand=True, fill='both')

        # --- Инициализация вкладок (где создается Figure и self.ax!) ---
        self.init_parameters_tab()
        self.init_plot_tab()
        self.init_simulation_tab()
        # ВАЖНО: только после этого существуют self.fig_map и self.ax!

        # --- Визуализация маршрута и дрона ---
        # ВАЖНО: создавать plot-объекты только после создания self.ax!
        self.route_line, = self.ax.plot([], [], color='yellow', linewidth=3, linestyle='--', zorder=2)
        self.route_line.set_dashes([18, 12])
        # (остальные plot/scatter-объекты так же здесь, если нужно)

        # --- Инициализация нейросети ---
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

    def update_mission_log(self, x, y, h, every=10, force=False):
        """
        Добавляет новую строку в журнал задания полета с текущими координатами и высотой дрона,
        только в формате X, Y, H — раз в N шагов или по требованию (force=True).
        """
        if not hasattr(self, "_mission_log_counter"):
            self._mission_log_counter = 0
        self._mission_log_counter += 1

        if force or self._mission_log_counter % every == 0:
            self.mission_log.insert("end", f"X={x:.2f}, Y={y:.2f}, H={h:.2f}\n")
            self.mission_log.see("end")

    def generate_data(self, num_samples: int) -> tuple:
        """Генерация тренировочных данных с использованием формулы энергозатрат как в симуляции."""
        X = np.zeros((num_samples, 8))
        y = np.zeros((num_samples, 2))
        for i in range(num_samples):
            battery_capacity = self.BATTERY_CAPACITY_WATT_HOURS
            charge_watt_hours = np.random.uniform(0.2 * battery_capacity, battery_capacity)
            height = np.random.uniform(100, 5000)

            station_data = []
            for _ in range(2):
                dist = np.random.uniform(100, 3000)
                st_height = np.random.uniform(50, 5000)
                status = np.random.choice([0, 1], p=[0.7, 0.3])
                station_data.append({'dist': dist, 'height': st_height, 'status': status})

            # Используем ту же формулу энергозатрат, что и в симуляции
            costs = []
            for j in range(2):
                horizontal = station_data[j]['dist'] * self.HORIZONTAL_ENERGY
                height_diff = station_data[j]['height'] - height
                if height_diff > 0:
                    vertical = abs(height_diff) * self.CLIMB_ENERGY
                else:
                    vertical = abs(height_diff) * self.DESCENT_ENERGY
                total_energy = horizontal + vertical
                normalized_energy = total_energy / (self.BATTERY_CAPACITY_WATT_HOURS * 3600 * 0.8)
                penalty = self.OCCUPATION_PENALTY if station_data[j]['status'] else 0
                cost = (normalized_energy * (1.5 - (charge_watt_hours / battery_capacity)) + penalty)
                costs.append(cost)

            best_station = np.argmin(costs)
            y[i, best_station] = 1
            X[i, :] = get_nn_features(charge_watt_hours, battery_capacity, height, station_data)
        return X, y

    # Расчет стоимостей
    def calculate_cost(self, distance, height_diff, status, charge_percent):
        """Расчет стоимости посадки на станцию с учетом нейросети."""
        # Подготовка входных данных для нейросети
        input_data = np.array([
            self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS,
            self.drone_height / 5000,
            distance / 3000,
            status,
            self.stations[0][0] / self.grid_width,  # Координата X первой станции (нормализованная)
            self.stations[0][1] / self.grid_height,  # Координата Y первой станции (нормализованная)
            height_diff / 5000,  # Разница высот
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
        """Вкладка с параметрами ввода и скоростью симуляции."""
        frame = ttk.LabelFrame(self.tab_params, text="Начальные параметры", padding=(10, 10))
        frame.pack(padx=20, pady=20, fill='both', expand=True)

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

        # Переносим сюда управление скоростью симуляции
        speed_lbl = ttk.Label(frame, text="Скорость симуляции:", font=label_font)
        speed_lbl.grid(row=len(params), column=0, padx=10, pady=10, sticky='e')
        self.speed_scale = tk.Scale(
            frame,
            from_=0.5,
            to=20.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self.update_simulation_speed,
            label="Множитель скорости",
            length=320
        )
        self.speed_scale.set(1.0)
        self.speed_scale.grid(row=len(params), column=1, padx=10, pady=10)

        btn_apply = ttk.Button(frame, text="Применить", command=self.apply_parameters, style="Apply.TButton")
        btn_apply.grid(row=len(params) + 1, column=0, columnspan=2, pady=20)
        style = ttk.Style()
        style.configure("Apply.TButton", font=button_font, padding=10)

    def init_plot_tab(self):
        """Создание вкладки с графиками обучения и зарядки с увеличенными шрифтами и жирными линиями."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter.ttk as ttk

        self.figures_frame = ttk.Frame(self.tab_plot)
        self.figures_frame.pack(fill="both", expand=True)

        # График обучения нейросети
        self.fig_loss = Figure(figsize=(8, 2.5), dpi=100)
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.ax_loss.set_title('Кривая обучения нейросети', fontsize=20, fontweight='bold')
        self.ax_loss.set_xlabel('Эпохи', fontsize=16, fontweight='bold')
        self.ax_loss.set_ylabel('Ошибка', fontsize=16, fontweight='bold')
        self.ax_loss.tick_params(axis='x', labelsize=14)
        self.ax_loss.tick_params(axis='y', labelsize=14)
        self.ax_loss.grid(True, linestyle='--', alpha=0.6)
        # Пример жирной линии для самой кривой (потом используйте linewidth=2.5 при plot)
        # self.ax_loss.plot(epochs, losses, linewidth=2.5)

        # Легенду делаем крупнее (понадобится при plot)
        # self.ax_loss.legend(fontsize=14)

        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self.figures_frame)
        self.canvas_loss.get_tk_widget().pack(side="top", fill="x", expand=False)

        # График зарядки батареи — только Figure!
        self.fig_charge = Figure(figsize=(8, 4), dpi=100)
        self.canvas_charge = FigureCanvasTkAgg(self.fig_charge, master=self.figures_frame)
        self.canvas_charge.get_tk_widget().pack(side="top", fill="both", expand=True)

    def plot_charge_graph(self, target_ax=None, legend_below=False, for_mini=False):

        if not hasattr(self, 'charge_log') or not self.charge_log.get("time"):
            if for_mini and hasattr(self, "mini_charge_ax"):
                self.mini_charge_ax.clear()
                self.mini_charge_ax.set_title('Зарядка', fontsize=10)
                self.mini_charge_ax.set_xlabel('Время (с)', fontsize=9)
                self.mini_charge_ax.set_ylabel('Заряд (%)', fontsize=9)
                self.mini_charge_ax.set_ylim(0, 100)
                self.mini_charge_canvas.draw()
            return

        times = self.charge_log["time"]
        percents = self.charge_log.get("charge_percent", [])
        currents = self.charge_log.get("current", [])
        voltages = self.charge_log.get("voltage", [])

        min_len = min(len(times), len(percents), len(currents), len(voltages))
        if min_len == 0:
            return
        times = times[:min_len]
        percents = percents[:min_len]
        currents = currents[:min_len]
        voltages = voltages[:min_len]

        if target_ax is None and not for_mini:
            self.fig_charge.clf()
            ax_charge = self.fig_charge.add_subplot(111)
            ax_charge_right = ax_charge.twinx()
            canvas = self.canvas_charge
        elif for_mini and hasattr(self, "mini_charge_ax"):
            self.mini_charge_ax.clear()
            ax_charge = self.mini_charge_ax
            ax_charge.set_title('Зарядка', fontsize=10)
            ax_charge.set_xlabel('Время (с)', fontsize=9)
            ax_charge.set_ylabel('Заряд (%)', fontsize=9)
            ax_charge.set_ylim(0, 100)
            ax_charge.plot(times, percents, color='tab:green', label='Заряд (%)', linewidth=2.2)
            ax_charge.grid(True, linestyle='--', alpha=0.25)
            ax_charge.legend(fontsize=8, loc='lower right')
            self.mini_charge_canvas.draw()
            return
        else:
            target_ax.clear()
            ax_charge = target_ax
            ax_charge_right = ax_charge.twinx() if hasattr(target_ax, 'twinx') else None
            canvas = None

        line_current, = ax_charge.plot(times, currents, color='tab:red', label='Ток зарядки (А)', linewidth=2.5)
        line_voltage, = ax_charge.plot(times, voltages, color='tab:red', linestyle='--', label='Напряжение (В)',
                                       linewidth=2.5)

        ax_charge.set_ylabel('Ток (А)                Напряжение (В)', fontsize=16, fontweight='bold')
        ax_charge.set_xlabel('Время (сек)', fontsize=16, fontweight='bold')
        ax_charge.tick_params(axis='x', labelsize=14)
        ax_charge.tick_params(axis='y', labelsize=14)
        ax_charge.grid(True, linestyle='--', alpha=0.6)
        ax_charge.set_title('График зарядки батареи дрона', fontsize=20, fontweight='bold')

        if ax_charge_right:
            line_percent, = ax_charge_right.plot(times, percents, color='tab:green', label='Заряд (%)', linewidth=2.5)
            ax_charge_right.set_ylabel('Ёмкость аккумулятора (%)', fontsize=16, fontweight='bold')
            ax_charge_right.set_yticks(np.arange(0, 105, 5))
            ax_charge_right.set_ylim(0, 100)
            ax_charge_right.tick_params(axis='y', labelsize=14)
            ax_charge_right.axhspan(0, 90, color='lightblue', alpha=0.08, zorder=0)
            ax_charge_right.axhspan(90, 100, color='lightcoral', alpha=0.10, zorder=0)
            ax_charge_right.text(-5, 45, "CC", color="tab:blue", fontsize=14, fontweight='bold',
                                 va='center', ha='left', alpha=0.7, rotation=90, clip_on=False)
            ax_charge_right.text(-5, 95, "CV", color="tab:red", fontsize=14, fontweight='bold',
                                 va='center', ha='left', alpha=0.7, rotation=90, clip_on=False)
        else:
            line_percent, = ax_charge.plot(times, percents, color='tab:green', label='Заряд (%)', linewidth=2.5)

        if ax_charge_right:
            lines_left, labels_left = ax_charge.get_legend_handles_labels()
            lines_right, labels_right = ax_charge_right.get_legend_handles_labels()
            handles = lines_left + lines_right
            labels = labels_left + labels_right
        else:
            handles, labels = ax_charge.get_legend_handles_labels()


        ax_charge.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.88), fontsize=14)

        if canvas:
            canvas.draw()
        else:
            plt.tight_layout()

    def init_simulation_tab(self):
        # --- Стили ---
        style = ttk.Style()
        style.configure("Big.TLabelframe.Label", font=("Arial", 14, "bold"))
        style.configure("Big.TLabel", font=("Arial", 14))
        style.configure("Outline.TButton", font=("Arial", 13, "bold"))

        # --- Основной горизонтальный фрейм ---
        main_frame = ttk.Frame(self.tab_sim)
        main_frame.pack(fill="both", expand=True)

        # --- Панель управления слева ---
        control_frame = ttk.LabelFrame(main_frame, text="Управление дроном", style="Big.TLabelframe", width=255)
        control_frame.pack(side="left", fill="y", padx=10, pady=10)
        control_frame.pack_propagate(False)

        # Секундомер и кнопки
        self.timer_label = ttk.Label(control_frame, text="Время: 0.0 сек", style="Big.TLabel")
        self.timer_label.pack(pady=5)
        ttk.Button(control_frame, text="Начать полетное задание", style="Outline.TButton",
                   command=self.start_mission).pack(padx=8, pady=4, fill="x")
        ttk.Button(control_frame, text="Сброс симуляции", style="Outline.TButton",
                   command=self.reset_simulation).pack(padx=8, pady=4, fill="x")
        ttk.Button(control_frame, text="Принудительная посадка", style="Outline.TButton",
                   command=self.force_decision).pack(padx=8, pady=4, fill="x")

        # --- Параметры дрона ---
        params_frame = ttk.LabelFrame(control_frame, text="Параметры дрона", style="Big.TLabelframe")
        params_frame.pack(padx=5, pady=(10, 5), fill="x")
        self.charge_label = ttk.Label(
            params_frame,
            text=f"Заряд: 100.0% ({getattr(self, 'remaining_capacity_watt_hours', 0.0):.2f} Вт·ч)",
            style="Big.TLabel",
            anchor="w",
            font=("Arial", 12)
        )
        self.charge_label.pack(fill="x", padx=5, pady=2)
        self.height_label = ttk.Label(
            params_frame,
            text="Высота: 500 м",
            style="Big.TLabel",
            anchor="w",
            font=("Arial", 12)
        )
        self.height_label.pack(fill="x", padx=5, pady=2)

        # --- Новое: скорость дрона (горизонтальная и вертикальная) ---
        self.speed_label = ttk.Label(
            params_frame,
            text="Скорость: 0.00 м/с",
            style="Big.TLabel",
            anchor="w",
            font=("Arial", 12)
        )
        self.speed_label.pack(fill="x", padx=5, pady=2)

        # --- Задание полета ---
        mission_frame = ttk.LabelFrame(control_frame, text="Задание полета", style="Big.TLabelframe")
        mission_frame.pack(padx=5, pady=5, fill="both", expand=True)
        mission_log_scroll = tk.Scrollbar(mission_frame)
        self.mission_log = tk.Text(mission_frame, height=8, font=("Consolas", 11), wrap="none",
                                   yscrollcommand=mission_log_scroll.set)
        self.mission_log.pack(side="left", fill="both", expand=True, padx=2, pady=2)
        mission_log_scroll.pack(side="right", fill="y")
        mission_log_scroll.config(command=self.mission_log.yview)

        # --- Правая колонка: легенда карты ---
        legend_width = 235
        legend_frame = ttk.LabelFrame(main_frame, text="Легенда карты", style="Big.TLabelframe", width=legend_width)
        legend_frame.pack(side="right", fill="y", padx=2, pady=10)
        legend_frame.pack_propagate(False)
        legend_canvas = tk.Canvas(legend_frame, width=legend_width, height=210)
        legend_canvas.pack()
        y_start = 20
        y_interval = 36
        legend_canvas.create_line(20, y_start, 40, y_start + 20, fill="yellow", width=2)
        legend_canvas.create_line(40, y_start, 20, y_start + 20, fill="yellow", width=2)
        legend_canvas.create_text(50, y_start + 10, text="- Дрон", anchor="w", font=("Arial", 12))
        legend_canvas.create_oval(22, y_start + y_interval + 2, 37, y_start + y_interval + 17, fill="blue")
        legend_canvas.create_text(50, y_start + y_interval + 10, text="- Начальная точка", anchor="w",
                                  font=("Arial", 12))
        legend_canvas.create_oval(22, y_start + 2 * y_interval + 2, 37, y_start + 2 * y_interval + 17, fill="red")
        legend_canvas.create_text(50, y_start + 2 * y_interval + 10, text="- Конечная точка", anchor="w",
                                  font=("Arial", 12))
        legend_canvas.create_polygon(
            20, y_start + 3 * y_interval + 15, 40, y_start + 3 * y_interval + 15,
            30, y_start + 3 * y_interval - 2.32, fill="green"
        )
        legend_canvas.create_text(50, y_start + 3 * y_interval + 10, text="- Док-станция свободна", anchor="w",
                                  font=("Arial", 12))
        legend_canvas.create_polygon(
            20, y_start + 4 * y_interval + 15, 40, y_start + 4 * y_interval + 15,
            30, y_start + 4 * y_interval - 2.32, fill="red"
        )
        legend_canvas.create_text(50, y_start + 4 * y_interval + 10, text="- Док-станция занята", anchor="w",
                                  font=("Arial", 12))

        # --- Мини-график зарядки под легендой ---
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.mini_charge_fig = Figure(figsize=(2.2, 1.9), dpi=90)
        self.mini_charge_ax = self.mini_charge_fig.add_subplot(111)
        self.mini_charge_ax.set_title('Зарядка', fontsize=10)
        self.mini_charge_ax.set_xlabel('Время (с)', fontsize=9)
        self.mini_charge_ax.set_ylabel('Заряд (%)', fontsize=9)
        self.mini_charge_ax.tick_params(axis='both', labelsize=8)
        self.mini_charge_ax.set_ylim(0, 100)
        self.mini_charge_canvas = FigureCanvasTkAgg(self.mini_charge_fig, master=legend_frame)
        mini_widget = self.mini_charge_canvas.get_tk_widget()
        mini_widget.config(height=160, width=220)
        mini_widget.pack(fill="x", expand=False, padx=5, pady=(10, 2))
        mini_widget.bind("<Button-1>", self.show_big_charge_graph)

        # --- Центр: поле симуляции (карта) ---
        map_frame = ttk.Frame(main_frame)
        map_frame.pack(side="left", fill="both", expand=True, padx=5, pady=10)

        # ! Важно: уменьшаем высоту, чтобы лог был всегда виден!
        self.fig_map = Figure(figsize=(10, 5.5), dpi=100)
        self.fig_map.patch.set_facecolor('white')
        self.ax = self.fig_map.add_subplot(111)
        self.fig_map.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.09)
        self.route_dashed, = self.ax.plot([], [], linestyle='--', color='white', linewidth=2, alpha=0.7, zorder=1)
        if hasattr(self, "setup_map_grid"):
            self.setup_map_grid()
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, map_frame)
        self.canvas_map.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_map.mpl_connect("button_press_event", self.on_click)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas_map, map_frame)
        self.toolbar.update()
        # --- Линия пути дрона: желтая пунктирная ---
        self.route_line, = self.ax.plot([], [], color='yellow', linewidth=3, linestyle='--', zorder=2)
        self.route_line.set_dashes([18, 12])  # штрихи 18, пробелы 12 (на ваш вкус)
        # --- Дрон ---
        self.drone_icon = self.ax.scatter(
            self.drone_pos[0], self.drone_pos[1],
            s=15 ** 2, c='yellow', marker='x', linewidths=2, label='Дрон', zorder=3
        )
        # --- Начальная и конечная точки ---
        self.start_icon, = self.ax.plot([], [], 'bo', markersize=10, label='Старт', zorder=4)
        self.end_icon, = self.ax.plot([], [], 'ro', markersize=10, label='Финиш', zorder=4)
        if hasattr(self, "add_stations"):
            self.add_stations()
        self.canvas_map.draw()

        # --- Лог событий внизу ---
        log_frame = ttk.LabelFrame(self.tab_sim, text="Лог событий", style="Big.TLabelframe")
        log_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        log_inner = ttk.Frame(log_frame)
        log_inner.pack(fill="x", expand=True)
        self.log_text = tk.Text(log_inner, height=4, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)
        ttk.Button(
            log_inner,
            text="Экспортировать лог",
            command=self.export_log,
            style="Outline.TButton"
        ).pack(side="right", padx=8, pady=5)

    def show_big_charge_graph(self, event=None):
        """Открытие большого окна с графиком зарядки (с легендой ниже графика)."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import tkinter as tk

        top = tk.Toplevel(self.root)
        top.title("График зарядки — подробный просмотр")
        fig_big = plt.Figure(figsize=(8, 6), dpi=100)
        ax_big = fig_big.add_subplot(111)
        self.plot_charge_graph(target_ax=ax_big, legend_below=True)
        canvas_big = FigureCanvasTkAgg(fig_big, master=top)
        canvas_big.get_tk_widget().pack(fill="both", expand=True)
        canvas_big.draw()

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

    def update_simulation_speed(self, value):
        """Обновляет множитель скорости симуляции (влияет на течение симуляционного времени)."""
        self.simulation_speed_multiplier = float(value)  # Устанавливаем множитель скорости
        self.frame_interval = int(1000 / self.simulation_speed_multiplier)  # Интервал обновления кадров
        self.update_log(
            f"Множитель скорости установлен на {self.simulation_speed_multiplier:.2f}. Анимация обновляется каждые {self.frame_interval} мс.")

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

        img = mpimg.imread('C:\\Users\\my_ru\\Downloads\\code\\map.png')
        self.ax.imshow(
            img,
            extent=[0, self.grid_width, 0, self.grid_height],  # подгоняет под размер сетки
            aspect='auto',
            zorder=0
        )


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

    def update_drone_visuals(self, force_set_size: bool = False):
        """Централизованное обновление всех визуальных параметров дрона.
        force_set_size=True — немедленно задать размер дрона по текущей высоте (без плавного перехода).
        """
        # Определяем целевую высоту
        if self.is_landing and self.target_pos is not None:
            try:
                station_idx = self.stations.index(self.target_pos.tolist())
                target_height = self.station_heights[station_idx]
            except ValueError:
                target_height = self.target_height
        else:
            target_height = 0

        # Расчет соотношения высоты (чем ниже высота, тем меньше размер)
        height_ratio = max(0, min(1, self.drone_height / 5000.0))

        # Интерполяция размера
        target_size = self.min_drone_size + (self.max_drone_size - self.min_drone_size) * height_ratio

        if force_set_size:
            self.drone_size = target_size  # Немедленно применить размер
        elif self.is_landing or self.drone_height != self.target_height:
            self.drone_size += (target_size - self.drone_size) * 0.2  # Плавный переход

        if hasattr(self, "route_line"):
            if self.drone_path and len(self.drone_path) > 1:
                xs, ys = zip(*self.drone_path)
                self.route_line.set_data(xs, ys)
            else:
                self.route_line.set_data([], [])

        self.drone_icon.set_sizes([self.drone_size ** 2])
        self.drone_icon.set_offsets(self.drone_pos)
        self.drone_icon.set_zorder(10)
        self.canvas_map.draw_idle()

    def update_ui(self):
        """Обновление элементов интерфейса."""
        # 1. Обновление метки заряда
        remaining_percent = (self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS) * 100
        self.charge_label.config(
            text=f"Заряд: {remaining_percent:.2f}% ({self.remaining_capacity_watt_hours:.2f} Вт·ч)"
        )
        self.height_label.config(text=f"Высота: {self.drone_height:.0f} м")

        # 2. Обновление скорости дрона (один label)
        self.speed_label.config(
            text=f"Скорость: {self.motion_controller.get_speed():.2f} м/с"
        )

        # 3. Скрывать дрона если он вне маршрута (например после сброса)
        if (not self.mission_active and not self.is_landing) or self.drone_pos is None:
            self.drone_icon.set_offsets(np.empty((0, 2)))  # Скрыть дрона
        else:
            self.drone_icon.set_offsets(self.drone_pos)  # Показывать дрона, если он есть

        # 4. Обновление визуальных параметров дрона (размер, позиция)
        self.update_drone_visuals()

        if self.start_pos is not None:
            self.ax.text(self.start_pos[0], self.start_pos[1] + 0.5, "Начало", color="yellow", fontsize=12, ha="center")
        if self.end_pos is not None:
            self.ax.text(self.end_pos[0], self.end_pos[1] + 0.5, "Конец", color="yellow", fontsize=12, ha="center")

        # 5. Обновление маршрута (если есть цель)
        if self.target_pos is not None:
            self.route_line.set_zorder(5)
        else:
            self.route_line.set_data([], [])
        self.canvas_map.draw_idle()
        self.update_mission_log(self.drone_pos[0], self.drone_pos[1], self.drone_height)

    def update_frame(self, frame):
        """Обновление состояния анимации на каждом кадре."""
        if self.is_landing:
            return self.perform_landing(frame)
        else:
            return self.move_drone(frame)

    def update_log(self, message: str, level: str = "info") -> None:
        """
        Обновление лога сообщений с фильтрацией по уровню.
        Пишет только в основной лог (log_text), не дублирует ничего в mission_log!
        """
        tag = level
        self.log_text.insert('end', f"{message}\n", tag)
        self.log_text.see('end')
        self.log_text.update_idletasks()

    def add_stations(self):
        """Добавление станций на карту."""
        self.station_plots = []
        for i, (x, y) in enumerate(self.stations):
            # Значок док-станции (plot, чтобы можно было менять цвет через set_color)
            station = self.ax.plot(x, y, '^', markersize=15, color='green', zorder=3)[0]

            # Перекрестие (×) — изначально скрыто
            cross = self.ax.text(
                x, y, '×', fontsize=20, color='black',
                ha='center', va='center', visible=False, zorder=4
            )

            # Подпись станции
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
            self.BATTERY_CAPACITY = self.BATTERY_CAPACITY_WATT_HOURS * 3600
            self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS
            self.update_stations_info()
            self.update_drone_visuals(force_set_size=True)  # <--- вот здесь
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
            if event.inaxes != self.ax:
                self.update_log("Клик вне области карты! Игнорируем...")
                return

            x, y = event.xdata, event.ydata
            self.update_log(f"Клик по карте: (x, y) = ({x:.2f}, {y:.2f})")

            if x is None or y is None or not (0 <= x <= self.grid_width and 0 <= y <= self.grid_height):
                self.update_log("Координаты клика вне допустимых границ карты!")
                return

            if event.button == 1:  # ЛКМ — установка точек маршрута
                if self.start_pos is None:
                    self.start_icon.set_data([x], [y])
                    self.start_pos = np.array([x, y])
                    self.drone_pos = np.array([x, y])
                    self.drone_height = float(self.entries['drone_height'].get())  # синхронизация с параметрами
                    self.update_log(f"Начальная точка установлена: ({x:.2f}, {y:.2f})")
                    # Логируем в mission_log только если ещё не было
                    if not hasattr(self, "start_logged_in_mission_log") or not self.start_logged_in_mission_log:
                        self.mission_log.insert("end", f"Нач. точка: X={x:.2f}, Y={y:.2f}\n")
                        self.mission_log.see("end")
                        self.start_logged_in_mission_log = True
                    self.target_pos = None
                    self.end_icon.set_data([], [])
                    self.end_pos = None
                    self.drone_icon.set_color('yellow')
                    self.update_drone_visuals(force_set_size=True)
                elif self.target_pos is None:
                    self.end_icon.set_data([x], [y])
                    self.end_pos = np.array([x, y])
                    self.target_pos = np.array([x, y])
                    self.update_log(f"Конечная точка установлена: ({x:.2f}, {y:.2f})")
                    # Логируем в mission_log только если ещё не было
                    if not hasattr(self, "end_logged_in_mission_log") or not self.end_logged_in_mission_log:
                        self.mission_log.insert("end", f"Кон. точка: X={x:.2f}, Y={y:.2f}\n")
                        self.mission_log.see("end")
                        self.end_logged_in_mission_log = True

                # Обновляем белую пунктирную линию
                if self.start_pos is not None and self.end_pos is not None:
                    self.route_dashed.set_data(
                        [self.start_pos[0], self.end_pos[0]],
                        [self.start_pos[1], self.end_pos[1]]
                    )
                else:
                    self.route_dashed.set_data([], [])

            elif event.button == 3:  # ПКМ — смена статуса станции
                for i, (sx, sy) in enumerate(self.stations):
                    if abs(x - sx) < 0.5 and abs(y - sy) < 0.5:
                        self.station_statuses[i] = not self.station_statuses[i]
                        status = "Занята" if self.station_statuses[i] else "Свободна"
                        self.update_log(f"Статус станции {i + 1} изменён: {status}")
                        self.update_stations_info()
                        break

            # Обновляем маршрутную линию (если обе точки есть)
            if self.start_pos is not None and self.target_pos is not None:
                self.update_log("Маршрут успешно обновлен!")

            self.canvas_map.draw_idle()
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

    def update_stations_info(self):
        """Обновить отображение статусов станций: цвет и крестик."""
        for i, (station, cross, label) in enumerate(self.station_plots):
            if self.station_statuses[i]:
                # Занята: красный маркер, крест виден
                station.set_color('red')
                cross.set_visible(True)
            else:
                # Свободна: зеленый маркер, крест скрыт
                station.set_color('green')
                cross.set_visible(False)
        self.canvas_map.draw_idle()

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
        """Полный сброс симуляции, маршрута, индексов и карты."""
        self.stop_after_forced_landing = False
        if hasattr(self, 'animation') and self.animation:
            if self.animation.event_source:
                self.animation.event_source.stop()
            self.animation = None

        self.reset_timer()

        # Сброс всех ключевых переменных состояния
        self.drone_pos = np.array([15, 10])
        self.drone_height = 500.0
        self.target_height = 0.0
        self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS
        self.charge = self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS
        self.is_charging = False
        self.is_landing = False
        self.is_forced_landing = False
        self.mission_started = False
        self.mission_active = False
        self.start_logged_in_mission_log = False
        self.end_logged_in_mission_log = False
        self.route_points = []
        self.current_route_index = None
        self.landing_logged = False
        self._mission_log_landing_counter = 0
        self._last_energy_check_target = None
        self.energy_checked_for_current_leg = False
        self._start_pos_of_leg = None
        self._energy_start_of_leg = None
        self.last_charged_station_idx = None  # <--- Сбросим флаг повторной зарядки

        # СБРОС координат маршрута — теперь пользователь должен их выбрать заново!
        self.start_pos = None
        self.target_pos = None
        self.end_pos = None

        self.drone_path = [self.drone_pos.copy()]  # стартовая точка в траектории

        self.ax.clear()
        self.setup_map_grid()
        self.add_stations()

        self.route_line, = self.ax.plot([], [], color='yellow', linewidth=3, linestyle='--', zorder=2)
        self.route_line.set_dashes([18, 12])
        self.drone_icon = self.ax.scatter(
            self.drone_pos[0], self.drone_pos[1],
            s=15 ** 2, c='yellow', marker='x', linewidths=2, label='Дрон'
        )
        self.start_icon, = self.ax.plot([], [], 'bo', markersize=10, label='Старт')
        self.end_icon, = self.ax.plot([], [], 'ro', markersize=10, label='Финиш')
        if hasattr(self, "route_dashed"):
            self.route_dashed, = self.ax.plot([], [], linestyle='--', color='white', linewidth=2, alpha=0.7, zorder=1)

        self.canvas_map.draw_idle()

        self.charge_label.config(
            text=f"Заряд: {self.charge * 100:.2f}% ({self.remaining_capacity_watt_hours:.2f} Вт·ч)"
        )
        self.height_label.config(text="Высота: 500 м")
        self.update_log("Симуляция сброшена. Задайте новые начальную и конечную точки!", level="info")

    def generate_drone_params(self) -> np.ndarray:
        """Генерация признаков дрона для нейросети по той же структуре, что и для обучения."""
        battery_capacity = self.BATTERY_CAPACITY_WATT_HOURS
        charge_watt_hours = self.remaining_capacity_watt_hours
        height = self.drone_height
        station_data = []
        for i in range(2):
            x, y = self.stations[i]
            dx = (self.drone_pos[0] - x) * self.cell_size
            dy = (self.drone_pos[1] - y) * self.cell_size
            dist = np.sqrt(dx ** 2 + dy ** 2)
            status = 1.0 if self.station_statuses[i] else 0.0
            st_height = self.station_heights[i]
            station_data.append({'dist': dist, 'status': status, 'height': st_height})
        return get_nn_features(charge_watt_hours, battery_capacity, height, station_data)

    def calculate_energy_consumption(self, start_pos: np.ndarray, end_pos: np.ndarray,
                                     start_height: float, end_height: float = 0) -> float:
        """Расчет потребления энергии для маршрута, результат в Вт·ч."""
        dx = (end_pos[0] - start_pos[0]) * self.cell_size
        dy = (end_pos[1] - start_pos[1]) * self.cell_size
        distance = np.sqrt(dx * dx + dy * dy)
        height_diff = start_height - end_height

        horizontal = distance * self.HORIZONTAL_ENERGY
        vertical = abs(height_diff) * (self.DESCENT_ENERGY if height_diff > 0 else self.CLIMB_ENERGY)
        systems = distance * self.SYSTEMS_ENERGY

        total_joules = horizontal + vertical + systems
        return total_joules / 3600  # Теперь результат в Вт·ч!

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

    def perform_landing(self, frame: int) -> list:
        """
        Плавная посадка дрона на станцию с учетом инерции, визуализации и правильного логирования.
        После завершения посадки вызывает зарядку только если реально приземлились на станцию и на её высоте.
        Если это подъём после зарядки — вызывает handle_arrival дальше по маршруту.
        """
        if self.is_charging or not self.is_landing:
            return [self.drone_icon]

        self.motion_controller.set_axis('vertical')
        curr_height = self.drone_height
        target_height = self.target_height
        self.motion_controller.set_position([0, curr_height])
        self.motion_controller.set_target([0, target_height])

        current_time = time.time()
        real_delta_time = current_time - getattr(self, 'last_update_time', current_time)
        self.last_update_time = current_time
        sim_delta_time = real_delta_time * self.simulation_speed_multiplier

        self.motion_controller.update(sim_delta_time)
        prev_height = self.drone_height
        self.drone_height = self.motion_controller.pos[1]
        curr_speed = abs(self.motion_controller.get_speed())

        if not hasattr(self, "_mission_log_landing_counter") or not self.is_landing:
            self._mission_log_landing_counter = 0
        LANDING_MISSION_LOG_EVERY = 10

        if abs(target_height - self.drone_height) < 1e-2 and curr_speed < 0.05:
            self.drone_height = target_height
            self.is_landing = False
            if not hasattr(self, "landing_logged") or not self.landing_logged:
                self.update_log(f"Дрон завершил посадку на высоте: {self.drone_height:.2f} м.")
                self.update_mission_log(
                    self.drone_pos[0], self.drone_pos[1], self.drone_height, force=True
                )
                self.landing_logged = True
        else:
            actual_step = self.drone_height - prev_height
            self.consume_energy(
                distance=0,
                height_change=actual_step,
                time_elapsed=sim_delta_time
            )
            self.update_log(
                f"Посадка: высота дрона {self.drone_height:.2f} м (целевая {self.target_height:.2f} м)"
            )
            self._mission_log_landing_counter += 1
            if self._mission_log_landing_counter % LANDING_MISSION_LOG_EVERY == 0:
                self.update_mission_log(
                    self.drone_pos[0], self.drone_pos[1], self.drone_height
                )

        self.update_drone_visuals()
        self.update_ui()

        if not self.is_landing and abs(self.drone_height - self.target_height) < 1e-2:
            for idx, height in enumerate(self.station_heights):
                if abs(self.drone_height - height) < 1e-2 and np.allclose(self.drone_pos, self.stations[idx],
                                                                          atol=1e-2):
                    self.update_log(f"Дрон находится на высоте станции {idx + 1}: {height} м. Инициируется зарядка.")
                    self.charge_at_station(idx)
                    return [self.drone_icon]
            self.update_log("Подъём/посадка завершён: дрон не на станции, продолжаем маршрут.")
            self.handle_arrival()
            return [self.drone_icon]

        if self.is_landing:
            self.root.after(100, lambda: self.perform_landing(frame + 1 if frame is not None else 1))

        return [self.drone_icon]

    def handle_arrival(self) -> None:
        """
        Обработка достижения точки маршрута (конечной точки, стартовой точки или станции).
        После каждой точки маршрута:
          - Если это станция — инициирует посадку/зарядку.
          - Если это конечная точка или старт — просто переходит к следующей точке маршрута по индексу.
          - Если маршрут окончен — завершает миссию.
        """

        # Если дрон уже заряжается — никаких новых событий
        if self.is_charging:
            return

        # Фиксируем позицию дрона на целевой точке (во избежание накопления ошибок)
        self.drone_pos = self.target_pos.copy()

        # Проверяем — текущая цель есть в маршруте?
        if not hasattr(self, "route_points") or self.route_points is None or self.current_route_index is None:
            self.update_log("Ошибка: маршрут не задан или индекс не инициализирован.")
            self.complete_simulation()
            return

        # --- Пропуск подряд идущих одинаковых точек маршрута по координате и высоте ---
        while self.current_route_index + 1 < len(self.route_points):
            next_point = self.route_points[self.current_route_index + 1]
            # Определяем высоту следующей точки
            next_height = self.drone_height
            for i, st in enumerate(self.stations):
                if np.allclose(next_point, st, atol=1e-2):
                    next_height = self.station_heights[i]
                    break
            # Пропустить только если совпадает и координата, и высота!
            if np.allclose(self.drone_pos, next_point, atol=1e-2) and abs(self.drone_height - next_height) < 1e-2:
                self.current_route_index += 1
                continue
            else:
                break

        # Проверка: дрон на станции (по X/Y)?
        STATION_RADIUS = 0.5
        for i, station in enumerate(self.stations):
            if np.linalg.norm(self.drone_pos - np.array(station)) < STATION_RADIUS:
                # Зафиксировать положение дрона на станции (точно)
                self.drone_pos = np.array(station)
                self.motion_controller.set_position(np.array(station) * self.cell_size)
                self.motion_controller.set_velocity([0, 0])
                # Если НЕ на высоте станции — посадка
                if abs(self.drone_height - self.station_heights[i]) > 1e-2:
                    self.is_landing = True
                    self.target_height = self.station_heights[i]
                    self.update_log(f"Дрон прибыл к станции {i + 1} и начинает посадку (спуск по высоте).")
                    if hasattr(self, "animation") and self.animation and self.animation.event_source:
                        self.animation.event_source.stop()
                    from matplotlib.animation import FuncAnimation
                    self.animation = FuncAnimation(
                        self.fig_map, self.perform_landing,
                        frames=None, interval=50, blit=False, repeat=False, cache_frame_data=False
                    )
                    return
                else:
                    # Уже на высоте станции — сразу зарядка
                    self.charge_at_station(i)
                    return

        # --- Дальше: переход к следующей точке маршрута (строго по route_points) ---
        if self.current_route_index + 1 < len(self.route_points):
            self.current_route_index += 1
            self.target_pos = self.route_points[self.current_route_index].copy()
            self.update_log(f"Переход к следующей точке маршрута: {self.target_pos}")

            self.check_and_handle_feasibility()
            self.mission_active = True
            return
        else:
            self.update_log("Дрон достиг финальной точки маршрута. Миссия завершена.")
            self.complete_simulation()
            return

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
        """
        Проверка необходимости аварийной посадки для подзарядки.
        Всегда делает расчет до self.target_pos. Если не хватает — ищет ближайшую реально достижимую станцию.
        Если и это невозможно — пробует нейросеть и резервный режим.
        После аварийной посадки маршрут НЕ должен сбрасываться, а после зарядки следует вернуться к маршруту!
        """
        try:
            # Проверяем достижимость до текущей цели (target_pos)
            if self.target_pos is None:
                self.update_log("Ошибка: неизвестна текущая цель для проверки аварийной посадки!")
                return False

            # Определяем высоту цели
            end_height = self.drone_height  # По умолчанию — высота дрона
            for i, station in enumerate(self.stations):
                if np.allclose(self.target_pos, station, atol=1e-2):
                    end_height = self.station_heights[i]
                    break

            energy_to_target = self.calculate_energy_consumption(
                self.drone_pos, self.target_pos, self.drone_height, end_height
            )
            available_energy = self.remaining_capacity_watt_hours

            self.update_log(
                f"Проверка аварийной посадки: требуется до текущей цели {energy_to_target:.2f} Вт·ч, доступно: {available_energy:.2f} Вт·ч."
            )

            if energy_to_target <= available_energy:
                self.update_log("Энергии достаточно для текущей цели. Аварийная посадка не требуется.")
                self.is_forced_landing = False
                return False

            # Если не хватает — ищем ближайшую реально достижимую станцию (НЕ сбрасывая маршрут!)
            min_energy = float('inf')
            best_station = None
            for i, (x, y) in enumerate(self.stations):
                if self.station_statuses[i]:
                    continue
                energy = self.calculate_energy_consumption(
                    self.drone_pos, [x, y], self.drone_height, self.station_heights[i]
                )
                if energy <= available_energy and energy < min_energy:
                    min_energy = energy
                    best_station = i

            if best_station is not None:
                self.update_log(
                    f"Автоматическая посадка на станцию {best_station + 1} (требуется {min_energy:.2f} Вт·ч)")
                self.target_pos = np.array(self.stations[best_station])
                self.is_forced_landing = True
                self.mission_active = True
                # После зарядки handle_arrival/resume_mission_after_charge вернёт дрона на маршрут
                return True

            # Если не найдена достижимая станция — пробуем нейросеть и резервный режим
            self.update_log(
                "Нет достижимых станций для аварийной посадки! Попытка выбора станции с использованием нейросети...")
            if self.force_decision():
                self.update_log("Выбор станции завершен с использованием нейросети.")
                self.mission_active = True
                # После зарядки handle_arrival/resume_mission_after_charge вернёт дрона на маршрут
                return True

            self.update_log("Нейросеть не справилась. Активируется резервный режим.")
            if self.activate_reserve_mode():
                self.update_log("Выбор станции завершен с использованием резервного режима.")
                self.mission_active = True
                # После зарядки handle_arrival/resume_mission_after_charge вернёт дрона на маршрут
                return True

            self.update_log("Не удалось выбрать станцию. Полет продолжается с риском разряда.")
            return False

        except Exception as e:
            self.update_log(f"Ошибка в check_emergency_landing: {str(e)}")
            return False

    def move_drone(self, *args) -> list:
        """
        Движение дрона по карте (между точками на одной высоте) — ВСЕГДА с horizontal speed.
        Изменение высоты (подъём/спуск) реализуется только в методах посадки и подъёма на станцию!
        Этот метод отвечает только за движение по X/Y на фиксированной высоте.
        """

        # --- 1. Проверки статуса миссии ---
        # Если сейчас идёт зарядка, миссия не активна или нет целевой точки — ничего не делаем
        if self.is_charging or not self.mission_active or self.target_pos is None:
            return [self.drone_icon, self.route_line]

        # --- 2. Проверка на невалидную цель ---
        # Если целевая точка (0,0) и это не старт — ошибка
        if np.allclose(self.target_pos, [0, 0], atol=1e-2) and not (
                self.start_pos is not None and np.allclose(self.target_pos, self.start_pos, atol=1e-2)):
            self.update_log("Ошибка: целевая точка (0,0) невалидна. Миссия не будет продолжена.", level="error")
            self.mission_active = False
            return [self.drone_icon, self.route_line]

        # --- 3. Перевод координат в метры для контроллера движения ---
        current_pos_m = np.array(self.drone_pos, dtype=float) * self.cell_size
        target_pos_m = np.array(self.target_pos, dtype=float) * self.cell_size

        # --- 4. Инициализация нового этапа движения, если цель изменилась ---
        last_target = getattr(self, "_last_energy_check_target", None)
        # Гарантируем, что last_target и self.target_pos — np.ndarray, иначе корректно сравниваем
        need_new_leg = False
        if last_target is None:
            need_new_leg = True
        else:
            try:
                need_new_leg = not np.allclose(np.array(last_target, dtype=float),
                                               np.array(self.target_pos, dtype=float), atol=1e-3)
            except Exception:
                need_new_leg = True

        if need_new_leg:
            self._last_energy_check_target = np.array(self.target_pos, dtype=float)
            self.energy_checked_for_current_leg = False
            self._start_pos_of_leg = np.array(self.drone_pos, dtype=float)  # в клетках
            self._energy_start_of_leg = self.remaining_capacity_watt_hours
            self.motion_controller.set_position(current_pos_m)
            self.motion_controller.set_target(target_pos_m)
            # Оставляем текущую скорость (для плавного разгона/торможения)

        # --- 5. Проверка энергии на весь этап ---
        end_height = self.drone_height
        is_station = False
        for i, station in enumerate(self.stations):
            if np.allclose(self.target_pos, self.stations[i], atol=1e-2):
                end_height = self.station_heights[i]
                is_station = True
                break

        energy_to_target = self.calculate_energy_consumption(
            self._start_pos_of_leg, self.target_pos,
            self.drone_height, end_height if is_station else self.drone_height
        )
        available_energy = self._energy_start_of_leg

        if not getattr(self, "energy_checked_for_current_leg", False):
            self.update_log(
                f"Проверка энергии: требуется до текущей цели {energy_to_target:.2f} Вт·ч, доступно: {available_energy:.2f} Вт·ч."
            )
            self.energy_checked_for_current_leg = True

        if energy_to_target > available_energy:
            self.update_log("Недостаточно заряда для полета до текущей цели, поиск станции для зарядки.")
            self.is_forced_landing = False
            self.check_emergency_landing()
            return [self.drone_icon, self.route_line]

        # --- 6. ВСЕГДА движение по карте: horizontal speed (15 м/с) ---
        self.motion_controller.set_axis('horizontal')

        # --- 7. Расчёт времени для симуляционного шага ---
        current_time = time.time()
        real_delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        sim_delta_time = real_delta_time * self.simulation_speed_multiplier

        # --- 8. Сохраняем прежнюю позицию (метры) для расчёта расхода энергии ---
        prev_pos_m = self.motion_controller.pos.copy()

        # --- 9. Двигаем дрона через контроллер ---
        self.motion_controller.update(sim_delta_time)
        new_pos_m = self.motion_controller.pos.copy()

        # --- 10. Переводим новую позицию обратно в клетки для визуализации и логики ---
        self.drone_pos = new_pos_m / self.cell_size

        # --- 11. Проверка достижения цели (считаем в метрах) ---
        distance_left_m = np.linalg.norm(target_pos_m - new_pos_m)
        arrived = distance_left_m < 1e-2

        # --- 12. Добавляем точку в путь (если новая) ---
        if not self.drone_path or not np.allclose(self.drone_path[-1], self.drone_pos, atol=1e-5):
            self.drone_path.append(self.drone_pos.copy())

        # --- 13. Расход энергии за этот шаг (по реальному перемещению) ---
        prev_pos_grid = prev_pos_m / self.cell_size
        step_energy = self.calculate_energy_consumption(
            prev_pos_grid, self.drone_pos, self.drone_height, self.drone_height
        )
        self.remaining_capacity_watt_hours = max(0.0, self.remaining_capacity_watt_hours - step_energy)

        # --- 14. Если дрон прибыл — фиксируем на целевой точке, сбрасываем скорость и логируем ---
        if arrived:
            self.drone_pos = self.target_pos.copy()
            self.motion_controller.set_position(target_pos_m)
            self.motion_controller.set_velocity([0, 0])
            self.remaining_capacity_watt_hours = max(0.0, self._energy_start_of_leg - energy_to_target)
            if not self.drone_path or not np.allclose(self.drone_path[-1], self.drone_pos, atol=1e-5):
                self.drone_path.append(self.drone_pos.copy())
            self.energy_checked_for_current_leg = False
            self.handle_arrival()

        # --- 15. Обновляем визуализацию и UI ---
        self.drone_icon.set_offsets(self.drone_pos)
        self.update_drone_visuals()
        self.update_ui()
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
        """
        Запуск миссии туда-обратно с учетом промежуточных подзарядок и полной проверкой валидности маршрута.
        После сброса симуляции пользователь должен обязательно выбрать новые начальную и конечную точки!
        """
        self.stop_after_forced_landing = False
        try:
            # Валидация: обе точки должны быть выбраны
            start_data = self.start_icon.get_data() if hasattr(self, 'start_icon') else ([], [])
            end_data = self.end_icon.get_data() if hasattr(self, 'end_icon') else ([], [])
            if len(start_data[0]) == 0 or len(start_data[1]) == 0:
                self.update_log("Ошибка: Установите начальную точку (ЛКМ на карте)", level="error")
                return
            if len(end_data[0]) == 0 or len(end_data[1]) == 0:
                self.update_log("Ошибка: Установите конечную точку (ЛКМ на карте)", level="error")
                return

            # Сохраняем точки маршрута
            self.start_pos = np.array([start_data[0][0], start_data[1][0]], dtype=float)
            self.end_pos = np.array([end_data[0][0], end_data[1][0]], dtype=float)
            self.target_pos = self.end_pos.copy()
            self.drone_pos = self.start_pos.copy()
            self.drone_path = [self.start_pos.copy()]

            # Останавливаем предыдущую анимацию, если есть
            if hasattr(self, 'animation') and self.animation is not None:
                if self.animation.event_source:
                    self.animation.event_source.stop()
                self.animation = None

            # Проверка валидности маршрута (валидатор может выдавать сообщения)
            if not self.validate_mission():
                return

            # --- Построение полного маршрута (с учетом станций подзарядки) ---
            # check_and_handle_feasibility вызывает plan_full_mission_with_charging
            if not self.check_and_handle_feasibility():
                self.update_log("Маршрут невозможен!", level="error")
                self.mission_active = False
                return

            # Проверка на (0,0) по всем точкам маршрута (кроме старта)
            if hasattr(self, "route_points"):
                for idx, pt in enumerate(self.route_points):
                    if np.allclose(pt, [0, 0], atol=1e-2) and (
                            idx != 0 or not np.allclose(pt, self.start_pos, atol=1e-2)):
                        self.update_log("Ошибка: маршрут содержит невалидную точку (0,0). Миссия не будет запущена.",
                                        level="error")
                        self.mission_active = False
                        return

            self.update_log(f"Полный маршрут (route_points): {self.route_points}")
            self.update_log(f"Стартуем. Текущая позиция: {self.drone_pos}, цель: {self.target_pos}.")

            # Устанавливаем флаги состояния
            self.is_forced_landing = False
            self.mission_active = True
            self.mission_started = True
            self.is_landing = False

            self.route_line.set_linestyle('--')

            self.update_ui()
            self.start_animation()

        except ValueError as ve:
            self.update_log(f"Ошибка запуска миссии: {str(ve)}", level="error")
            self.mission_active = False
        except Exception as e:
            self.update_log(f"Ошибка запуска миссии: {str(e)}", level="error")
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

    def plan_full_mission_with_charging(self, start, end):
        import numpy as np
        from itertools import permutations

        stations = [np.array(s, dtype=float) for s in self.stations]
        station_names = [f"Док{idx + 1}" for idx in range(len(stations))]
        all_points = [start] + stations + [end]
        point_labels = ["Старт"] + station_names + ["Конец"]

        coord2height = {tuple(np.round(start, 5)): self.drone_height}
        for idx, s in enumerate(stations):
            coord2height[tuple(np.round(s, 5))] = self.station_heights[idx]
        coord2height[tuple(np.round(end, 5))] = self.drone_height

        def list_station_indices():
            return list(range(len(stations)))

        def route_to_label(route):
            return " → ".join(point_labels[i] for i in route)

        dists = [np.linalg.norm(start - s) for s in stations]
        sorted_station_indices = np.argsort(dists).tolist()

        candidate_routes = []
        N = len(stations)
        for n_tuda in range(0, N + 1):
            for n_obratno in range(0, N + 1):
                for tuda_stops in permutations(list_station_indices(), n_tuda):
                    if len(tuda_stops) > 1 and any(
                            tuda_stops[i] == tuda_stops[i + 1] for i in range(len(tuda_stops) - 1)):
                        continue
                    for obratno_stops in permutations(list_station_indices(), n_obratno):
                        if len(obratno_stops) > 1 and any(
                                obratno_stops[i] == obratno_stops[i + 1] for i in range(len(obratno_stops) - 1)):
                            continue
                        route_idx = [0] + [s + 1 for s in tuda_stops] + [N + 1] + [s + 1 for s in obratno_stops] + [0]
                        if n_tuda > 0 and route_idx[1] != sorted_station_indices[0] + 1:
                            continue
                        first_end_idx = route_idx.index(N + 1)
                        if 0 in route_idx[1:first_end_idx]:
                            continue
                        if N + 1 not in route_idx[1:-1]:
                            continue
                        candidate_routes.append(route_idx)

        valid_routes = []
        best_route = None
        best_energy = None

        def calc_energy(a, b, ha, hb):
            return self.calculate_energy_consumption(np.array(a), np.array(b), ha, hb)

        def is_station(idx):
            return 1 <= idx <= N

        for route_idx in candidate_routes:
            route_points = [all_points[i] for i in route_idx]
            route_names = [point_labels[i] for i in route_idx]
            route_str = " → ".join(route_names)
            log = f"Проверяем маршрут: {route_str}\n"
            charge = self.BATTERY_CAPACITY_WATT_HOURS
            feasible = True
            total_energy = 0.0
            prev_height = self.drone_height  # Стартовая высота

            for i in range(len(route_points) - 1):
                a = route_points[i]
                b = route_points[i + 1]
                idx_a = route_idx[i]
                idx_b = route_idx[i + 1]
                ha = coord2height[tuple(np.round(a, 5))]
                hb = coord2height[tuple(np.round(b, 5))]

                # Если сейчас на станции (и не на старте), моделируем подъём после зарядки
                if is_station(idx_a):
                    if ha != self.drone_height:
                        climb = abs(self.drone_height - ha) * (
                            self.CLIMB_ENERGY if self.drone_height > ha else self.DESCENT_ENERGY
                        )
                        climb_watt_hours = climb / 3600
                        log += f"    Подъём с высоты станции {ha:.2f} м на рабочую высоту {self.drone_height:.2f} м: {climb_watt_hours:.2f} Вт·ч (осталось {charge:.2f})\n"
                        if climb_watt_hours > charge + 1e-5:
                            log += f"    ❌ Не хватает заряда на подъём после зарядки! Нужно {climb_watt_hours:.2f}, есть {charge:.2f}\n"
                            feasible = False
                            break
                        charge -= climb_watt_hours
                        total_energy += climb_watt_hours
                    prev_height = self.drone_height
                else:
                    prev_height = ha

                # Горизонтальный участок — всегда между рабочими высотами
                e = calc_energy(a, b, self.drone_height, self.drone_height)
                log += f"  {route_names[i]} → {route_names[i + 1]}: {e:.2f} Вт·ч (осталось {charge:.2f})\n"
                if e > charge + 1e-5:
                    log += f"    ❌ Не хватает заряда! Нужно {e:.2f}, есть {charge:.2f}\n"
                    feasible = False
                    break
                charge -= e
                total_energy += e

                # Если следующая точка — станция, моделируем спуск перед зарядкой
                if is_station(idx_b):
                    if self.drone_height != hb:
                        descend = abs(self.drone_height - hb) * (
                            self.CLIMB_ENERGY if self.drone_height < hb else self.DESCENT_ENERGY
                        )
                        descend_watt_hours = descend / 3600
                        log += f"    Спуск на высоту станции {hb:.2f} м: {descend_watt_hours:.2f} Вт·ч (осталось {charge:.2f})\n"
                        if descend_watt_hours > charge + 1e-5:
                            log += f"    ❌ Не хватает заряда на спуск к станции! Нужно {descend_watt_hours:.2f}, есть {charge:.2f}\n"
                            feasible = False
                            break
                        charge -= descend_watt_hours
                        total_energy += descend_watt_hours
                    log += f"    ↺ Зарядка на {route_names[i + 1]} (заряд до {self.BATTERY_CAPACITY_WATT_HOURS:.2f})\n"
                    charge = self.BATTERY_CAPACITY_WATT_HOURS
                    prev_height = hb

            # Критическая проверка: после конечной точки хватит ли заряда до следующей точки (учитывая подъём/спуск)
            idx_K = route_idx.index(N + 1)
            if feasible and idx_K < len(route_idx) - 2:
                # После Конца идёт ещё движение (обратно)
                next_pt = route_idx[idx_K + 1]
                next_label = point_labels[next_pt]
                a = route_points[idx_K]
                b = route_points[idx_K + 1]
                ha = coord2height[tuple(np.round(a, 5))]
                hb = coord2height[tuple(np.round(b, 5))]
                # Перед стартом участка нужен подъём, если конец не на рабочей высоте
                if ha != self.drone_height:
                    climb = abs(self.drone_height - ha) * (
                        self.CLIMB_ENERGY if self.drone_height > ha else self.DESCENT_ENERGY
                    )
                    climb_watt_hours = climb / 3600
                    log += f"    Подъём после Конца с высоты {ha:.2f} м на рабочую высоту {self.drone_height:.2f} м: {climb_watt_hours:.2f} Вт·ч (осталось {charge:.2f})\n"
                    if climb_watt_hours > charge + 1e-5:
                        log += f"    ❌ После конечной точки не хватает заряда на подъём! Нужно {climb_watt_hours:.2f}, есть {charge:.2f}\n"
                        feasible = False
                # Горизонтальный участок
                if feasible:
                    e = calc_energy(a, b, self.drone_height, self.drone_height)
                    log += f"    После Конца: {point_labels[route_idx[idx_K]]} → {next_label}: {e:.2f} Вт·ч (осталось {charge:.2f})\n"
                    if e > (charge - climb_watt_hours) + 1e-5:
                        log += f"    ❌ После конечной точки не хватает заряда на участок {next_label}: нужно {e:.2f}, есть {charge - climb_watt_hours:.2f}\n"
                        feasible = False

            if feasible:
                log += f"✅ Маршрут выполним! Суммарная энергия: {total_energy:.2f} Вт·ч\n"
                valid_routes.append((route_idx, route_names, total_energy, log))
                if best_energy is None or total_energy < best_energy:
                    best_energy = total_energy
                    best_route = (route_idx, route_points, total_energy, log)
            else:
                log += f"❌ Маршрут невозможен.\n"
            self.update_log(log)

        if best_route is not None:
            idxs, route_points, total_energy, summary_log = best_route
            self.update_log(
                f"Выбран маршрут: {' → '.join(point_labels[i] for i in idxs)}\nСуммарная энергия: {total_energy:.2f} Вт·ч\n")
            return route_points
        else:
            self.update_log("❌ Не найден ни один выполнимый маршрут даже с учётом подзарядок (перебор всех сценариев).")
            return None

    def total_route_energy(self, route_points, heights=None):
        """Считает суммарную энергию по всему маршруту."""
        # Список высот для точек маршрута: по умолчанию высота дрона, если это станция — высота станции
        if heights is None:
            heights = []
            for p in route_points:
                is_station = False
                for i, st in enumerate(self.stations):
                    if np.allclose(p, st, atol=1e-2):
                        heights.append(self.station_heights[i])
                        is_station = True
                        break
                if not is_station:
                    heights.append(self.drone_height)
        total = 0.0
        for i in range(len(route_points) - 1):
            h1 = heights[i]
            h2 = heights[i + 1]
            total += self.calculate_energy_consumption(
                route_points[i], route_points[i + 1],
                h1, h2
            )
        return total

    def check_and_handle_feasibility(self):
        """
        Планировщик миссии — теперь всегда строит маршрут «от текущей позиции»!
        """
        # 1. Если маршрут не построен — строим его
        if not hasattr(self, "route_points") or not self.route_points or self.current_route_index is None:
            planned_route = self.plan_full_mission_with_charging(self.drone_pos, self.end_pos)
            if planned_route is None:
                self.update_log("❌ Невозможно построить маршрут туда-обратно даже с подзарядками!")
                self.mission_active = False
                return False
            self.route_points = planned_route
            self.current_route_index = 1  # Первая цель после старта/зарядки
            if self.current_route_index >= len(self.route_points):
                self.update_log("Маршрут построен, но нет следующей точки для полета.")
                self.mission_active = False
                return False
            self.target_pos = self.route_points[self.current_route_index].copy()
            self.update_log(
                f"Маршрут построен с учетом подзарядок. Первая цель: {self.target_pos}."
            )
            total_energy = self.total_route_energy(self.route_points)
            self.update_log(f"Суммарная энергия на весь маршрут: {total_energy:.2f} Вт·ч")

        # Защита от выхода за пределы маршрута!
        if self.current_route_index >= len(self.route_points):
            self.update_log("Маршрут окончен — больше нет точек для движения.")
            self.mission_active = False
            return False

        # 2. Проверка достижимости до следующей точки маршрута
        target_point = self.route_points[self.current_route_index]
        end_height = self.drone_height  # По умолчанию — высота дрона
        for i, station in enumerate(self.stations):
            if np.allclose(target_point, station, atol=1e-2):
                end_height = self.station_heights[i]
                break
        required_energy = self.calculate_energy_consumption(self.drone_pos, target_point, self.drone_height, end_height)
        available_energy = self.remaining_capacity_watt_hours

        self.update_log(
            f"Проверка возможности полёта до следующей точки: требуется {required_energy:.2f} Вт·ч, доступно: {available_energy:.2f} Вт·ч."
        )

        if required_energy > available_energy:
            # Здесь — строим новый маршрут к ближайшей достижимой станции!
            self.update_log("Недостаточно заряда — перестраиваем маршрут через ближайшую достижимую станцию.")
            best_station = None
            min_energy = float('inf')
            for i, st in enumerate(self.stations):
                if self.station_statuses[i]:
                    continue
                energy = self.calculate_energy_consumption(self.drone_pos, st, self.drone_height,
                                                           self.station_heights[i])
                if energy <= available_energy and energy < min_energy:
                    min_energy = energy
                    best_station = i
            if best_station is not None:
                # Строим новый маршрут: текущая позиция -> станция -> конечная цель (через plan_full_mission_with_charging)
                planned_route = self.plan_full_mission_with_charging(self.drone_pos, self.end_pos)
                if planned_route is None:
                    self.update_log(
                        "❌ Даже через станции маршрут невозможен (из-за цепочки или занятости)! Миссия завершена.")
                    self.mission_active = False
                    return False
                self.route_points = planned_route
                self.current_route_index = 1
                if self.current_route_index >= len(self.route_points):
                    self.update_log("Перестроен маршрут, но нет следующей точки для полета.")
                    self.mission_active = False
                    return False
                self.target_pos = self.route_points[self.current_route_index].copy()
                self.update_log(f"Перестроен маршрут через станцию {best_station + 1}. Новая цель: {self.target_pos}")
                return self.check_and_handle_feasibility()  # Проверим достижимость до этой станции
            else:
                self.update_log("Нет достижимых станций для аварийной посадки! Миссия завершена с аварией.")
                self.mission_active = False
                return False
        else:
            self.update_log("Энергии достаточно для перехода к следующей точке маршрута.")
            return True

    def force_decision(self) -> bool:
        """Принудительная посадка на док-станцию, выбранную нейросетью, после зарядки продолжить миссию."""
        try:
            # Генерация параметров для нейросети
            X = self.generate_drone_params().reshape(1, -1)
            neural_output = self.nn.forward(X)[0]
            neural_choice = np.argmax(neural_output)
            neural_confidence = neural_output[neural_choice]

            self.update_log(f"Нейросеть выбрала станцию {neural_choice + 1} с уверенностью {neural_confidence:.2f}")

            confidence_threshold = 0.5
            if neural_confidence < confidence_threshold:
                self.update_log(
                    f"⚠ Уверенность нейросети ({neural_confidence:.2f}) ниже порога ({confidence_threshold}). Активируется резервный режим.")
                if self.activate_reserve_mode():
                    self.update_log("Резервный режим успешно завершил выбор станции.")
                    # НЕ останавливаемся после зарядки!
                    # self.stop_after_forced_landing = True
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
            self.is_forced_landing = True
            # self.stop_after_forced_landing = True  # <-- УБРАТЬ!
            # После зарядки миссия продолжится автоматически (resume_mission_after_charge)

            # Запускаем полет к выбранной станции
            self.mission_active = True
            return True

        except Exception as e:
            self.update_log(f"Ошибка в принудительном выборе: {str(e)}")
            return False

    def activate_reserve_mode(self) -> bool:
        """Активация резервного режима для сложных задач."""
        try:
            available_energy = self.remaining_capacity_watt_hours
            feasible_stations = []

            for i, station in enumerate(self.stations):
                if self.station_statuses[i]:
                    continue  # Пропускаем занятые станции

                energy = self.calculate_energy_consumption(
                    self.drone_pos, station, self.drone_height, self.station_heights[i]
                )

                feasible_stations.append((i, energy))

            if not feasible_stations:
                self.update_log("Нет достижимых станций! Продолжаю полет с риском...")
                return False

            feasible_stations.sort(key=lambda x: x[1])
            best_station = feasible_stations[0][0]

            self.update_log(f"⚠ Активирован резервный режим выбора док-станции ⚠")
            self.update_log(f"Станция выбрана по минимальным энергозатратам: {best_station + 1}")

            self.target_pos = np.array(self.stations[best_station])
            self.is_forced_landing = True
            return True

        except Exception as e:
            self.update_log(f"Ошибка в резервном режиме: {str(e)}")
            return False

    def charge_at_station(self, station_idx: int):
        """Реалистичная зарядка дрона на док-станции с учетом фаз CC и CV и обновлением мини-графика."""
        # Защита от повторной зарядки на той же станции подряд
        if hasattr(self, "last_charged_station_idx") and self.last_charged_station_idx == station_idx:
            self.update_log(
                f"Повторная зарядка на станции {station_idx + 1} невозможна (уже заряжались на этом этапе).")
            self.resume_mission_after_charge()
            return

        if self.is_charging:
            self.update_log("Зарядка уже активна. Повторная зарядка невозможна.")
            return

        # Проверка высоты
        if abs(self.drone_height - self.station_heights[station_idx]) >= 1e-2:
            self.update_log(f"Ошибка: Зарядка возможна только на высоте станции {station_idx + 1}.")
            return

        # Проверка полного заряда
        if self.remaining_capacity_watt_hours >= self.BATTERY_CAPACITY_WATT_HOURS:
            self.update_log("Батарея уже полностью заряжена.")
            return

        # --- Инициализация charge_log для графика ---
        self.charge_log = {
            "time": [],
            "charge_percent": [],
            "power": [],
            "remaining_energy": [],
            "current": [],
            "voltage": []
        }
        # --------------------------------------------

        # Расчет параметров зарядки
        max_power = self.CHARGING_VOLTAGE * self.CHARGING_CURRENT  # 480 Вт (пример)
        cv_voltage = 42  # Фиксированное напряжение для CV-фазы
        efficiency = self.CHARGING_EFFICIENCY

        self.is_charging = True
        self._last_elapsed_time = 0
        self.last_charged_station_idx = station_idx  # <-- Запоминаем станцию, на которой была зарядка
        start_time = time.time()

        def update_charge():
            nonlocal max_power, cv_voltage

            elapsed_time_real = time.time() - start_time
            elapsed_time_sim = elapsed_time_real * self.simulation_speed_multiplier
            time_step = elapsed_time_sim - self._last_elapsed_time
            self._last_elapsed_time = elapsed_time_sim

            # CC-фаза (до 90%): 10 А, напряжение фиксированное (или растет, но для простоты 42 В)
            U_min = 36.0
            U_max = 42.0

            if self.charge < 0.9:
                current = self.CHARGING_CURRENT
                # Линейный рост напряжения в CC-фазе
                voltage = U_min + (U_max - U_min) * (self.charge / 0.9)
                voltage = min(voltage, U_max)
                power = voltage * current
            else:
                voltage = U_max
                # Пропорция оставшегося заряда в CV-фазе
                cv_progress = (self.charge - 0.9) / 0.1
                current = max(1.0, self.CHARGING_CURRENT * (1 - cv_progress))
                power = voltage * current

            if power <= 0 or current <= 0:
                self.update_log("Ошибка: мощность или ток зарядки стали равны нулю. Завершаем зарядку.")
                self.complete_charge()
                return

            charge_increment = (power * efficiency * time_step) / 3600
            charge_increment = min(charge_increment,
                                   self.BATTERY_CAPACITY_WATT_HOURS - self.remaining_capacity_watt_hours)

            self.remaining_capacity_watt_hours += charge_increment
            self.charge = self.remaining_capacity_watt_hours / self.BATTERY_CAPACITY_WATT_HOURS

            self.charge_log["time"].append(elapsed_time_sim)
            self.charge_log["charge_percent"].append(self.charge * 100)
            self.charge_log["power"].append(power)
            self.charge_log["remaining_energy"].append(self.remaining_capacity_watt_hours)
            self.charge_log["current"].append(current)
            self.charge_log["voltage"].append(voltage)

            self.update_log(
                f"Заряд: {self.charge * 100:.2f}% ({self.remaining_capacity_watt_hours:.2f} Вт·ч). "
                f"Мощность: {power:.2f} Вт, Ток: {current:.2f} А, Напряжение: {voltage:.2f} В."
            )
            self.update_ui()

            # --- ОБНОВЛЕНИЕ мини-графика зарядки ---
            if hasattr(self, "plot_charge_graph"):
                self.plot_charge_graph(for_mini=True)

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
            self.resume_mission_after_charge()
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
        """
        Завершение зарядки, обновление графика и подъем на рабочую высоту.
        После зарядки дрон должен продолжить маршрут строго по route_points, НИКАКИХ вызовов handle_arrival!
        Гарантируется переход к следующей уникальной точке маршрута (по координате и высоте).
        Если анимация была остановлена — она будет перезапущена.
        """
        self.charge = 1.0
        self.remaining_capacity_watt_hours = self.BATTERY_CAPACITY_WATT_HOURS
        self.update_log("Зарядка завершена. Батарея дрона полностью заряжена.", level="info")
        self.is_charging = False
        self.is_forced_landing = False

        # Обновление графика
        self.plot_charge_graph()

        # Подъем на рабочую высоту
        self.target_height = float(self.entries['drone_height'].get())
        self.is_landing = True
        self.update_log(
            f"Дрон возвращается на рабочую высоту: {self.target_height:.2f} м. is_landing={self.is_landing}.",
            level="info"
        )

        def update_height():
            if not self.is_landing:
                self.update_log("Подъем завершен. Состояние посадки отключено (is_landing=False).", level="info")
                # --- Пропуск подряд идущих точек маршрута с такими же координатами и высотой! ---
                if hasattr(self, "route_points") and self.route_points and self.current_route_index is not None:
                    # !!! ВАЖНО: НЕ увеличивать индекс дважды после зарядки !!!
                    # После подъема переходим к следующей уникальной точке маршрута (по XY и высоте)
                    next_index = self.current_route_index
                    while next_index + 1 < len(self.route_points):
                        next_pt = self.route_points[next_index + 1]
                        next_height = self.target_height
                        for i, st in enumerate(self.stations):
                            if np.allclose(next_pt, st, atol=1e-2):
                                next_height = self.station_heights[i]
                                break
                        # Пропустить только если совпадает и координата, и высота!
                        if np.allclose(self.drone_pos, next_pt, atol=1e-2) and abs(
                                self.target_height - next_height) < 1e-2:
                            next_index += 1
                            continue
                        else:
                            break
                    # Теперь выставляем следующую цель, если есть
                    if next_index + 1 < len(self.route_points):
                        self.current_route_index = next_index + 1
                        self.target_pos = self.route_points[self.current_route_index].copy()
                        self.update_log(f"После зарядки и подъёма: следующая цель по маршруту: {self.target_pos}")
                        self.mission_active = True
                        self.last_charged_station_idx = None  # <--- сбрасываем после подъема
                        # --- ГАРАНТИЯ запуска анимации, если она была остановлена ---
                        if not hasattr(self, "animation") or self.animation is None or not getattr(self.animation,
                                                                                                   "event_source",
                                                                                                   None) or not getattr(
                            self.animation.event_source, "_job", None):
                            self.start_animation()
                    else:
                        self.update_log("Все точки маршрута пройдены после зарядки. Миссия завершена.")
                        self.complete_simulation()
                else:
                    self.update_log("Ошибка: нет актуального маршрута после зарядки!")
                    self.complete_simulation()
                return

            current_time = time.time()
            elapsed_time_real = current_time - self.last_update_time
            self.last_update_time = current_time

            sim_delta_time = elapsed_time_real * self.simulation_speed_multiplier
            height_change = self.vertical_speed * sim_delta_time
            height_difference = self.target_height - self.drone_height

            if abs(height_difference) <= abs(height_change):
                self.drone_height = self.target_height
                self.consume_energy(height_change=height_difference)
                self.is_landing = False
                self.update_log(
                    f"Дрон достиг рабочей высоты: {self.target_height:.2f} м. "
                    f"Флаг is_landing={self.is_landing}. Продолжаем маршрут.",
                    level="info"
                )
                # Повтор логики перехода к следующей точке (с пропуском по координате и высоте)
                if hasattr(self, "route_points") and self.route_points and self.current_route_index is not None:
                    next_index = self.current_route_index
                    while next_index + 1 < len(self.route_points):
                        next_pt = self.route_points[next_index + 1]
                        next_height = self.target_height
                        for i, st in enumerate(self.stations):
                            if np.allclose(next_pt, st, atol=1e-2):
                                next_height = self.station_heights[i]
                                break
                        if np.allclose(self.drone_pos, next_pt, atol=1e-2) and abs(
                                self.target_height - next_height) < 1e-2:
                            next_index += 1
                            continue
                        else:
                            break
                    if next_index + 1 < len(self.route_points):
                        self.current_route_index = next_index + 1
                        self.target_pos = self.route_points[self.current_route_index].copy()
                        self.update_log(f"После зарядки и подъёма: следующая цель по маршруту: {self.target_pos}")
                        self.mission_active = True
                        self.last_charged_station_idx = None  # <--- сбрасываем после подъема
                        if not hasattr(self, "animation") or self.animation is None or not getattr(self.animation,
                                                                                                   "event_source",
                                                                                                   None) or not getattr(
                            self.animation.event_source, "_job", None):
                            self.start_animation()
                    else:
                        self.update_log("Все точки маршрута пройдены после зарядки. Миссия завершена.")
                        self.complete_simulation()
                else:
                    self.update_log("Ошибка: нет актуального маршрута после зарядки!")
                    self.complete_simulation()
                return

            self.drone_height += height_change if height_difference > 0 else -height_change
            self.consume_energy(height_change=height_change)
            self.update_ui()
            self.root.after(100, update_height)

        self.last_update_time = time.time()
        update_height()

    def resume_mission_after_charge(self):
        """
        Продолжение маршрута после зарядки: дрон должен продолжать движение строго по route_points.
        НЕ инкрементировать current_route_index! Цель уже выставлена в complete_charge.
        """
        self.update_log("Продолжаем маршрут после зарядки.")
        self.is_landing = False

        # Проверяем маршрут и индекс
        if not hasattr(self, "route_points") or self.route_points is None or self.current_route_index is None:
            self.update_log("Ошибка: маршрут не задан или индекс не инициализирован при продолжении после зарядки!")
            self.complete_simulation()
            return

        # Проверяем хватит ли заряда до текущей цели (target_pos уже выставлен!)
        end_height = self.drone_height
        for i, station in enumerate(self.stations):
            if np.allclose(self.target_pos, station, atol=1e-2):
                end_height = self.station_heights[i]
                break

        energy_to_target = self.calculate_energy_consumption(
            self.drone_pos, self.target_pos, self.drone_height, end_height
        )
        available_energy = self.remaining_capacity_watt_hours

        self.update_log(
            f"Проверка после зарядки: требуется до текущей цели {energy_to_target:.2f} Вт·ч, доступно: {available_energy:.2f} Вт·ч."
        )

        if energy_to_target > available_energy:
            self.update_log(
                "Сразу после зарядки: заряда всё ещё недостаточно для продолжения маршрута. Запуск повторной зарядки."
            )
            # Определяем индекс станции, на которой стоим
            current_station_idx = None
            for i, station in enumerate(self.stations):
                if np.allclose(self.drone_pos, station, atol=1e-2):
                    current_station_idx = i
                    break
            if current_station_idx is not None:
                self.root.after(100, lambda: self.charge_at_station(current_station_idx))
            else:
                self.update_log("Ошибка: не удалось определить индекс станции для повторной зарядки!")
            return

        self.mission_active = True
        # (НЕ запускать start_animation, просто продолжить)

    def complete_simulation(self):
        """Завершение симуляции и остановка таймера."""

        # 1. Получаем позицию "домой" (стартовая точка) и "конечной" точки
        at_start = self.start_pos is not None and np.linalg.norm(self.drone_pos - self.start_pos) < 0.1
        at_end = self.end_pos is not None and np.linalg.norm(self.drone_pos - self.end_pos) < 0.1

        # 2. Проверяем последний маршрут (если есть)
        at_route_last = (
                hasattr(self, "route_points")
                and isinstance(self.route_points, list)
                and len(self.route_points) > 0
                and np.linalg.norm(self.drone_pos - self.route_points[-1]) < 0.1
        )

        # 3. Миссия завершена, если:
        #   - Дрон вернулся на старт (для туда-обратно)
        #   - Или достиг конечной точки (если только туда)
        #   - Или совпал с последней точкой маршрута
        mission_done = at_start or at_end or at_route_last

        if mission_done:
            self.mission_active = False
            self.is_landing = False
            self.stop_timer()
            if hasattr(self, 'animation') and self.animation and self.animation.event_source:
                self.animation.event_source.stop()

            self.update_log(f"Симуляция завершена за {self.simulation_time:.1f} секунд.")
            self.update_log("Дрон успешно завершил миссию.")
            self.update_ui()
        else:
            # Если не финальная точка — не продолжаем вечный цикл!
            self.mission_active = False
            if hasattr(self, 'animation') and self.animation and self.animation.event_source:
                self.animation.event_source.stop()
            self.update_log("Миссия не завершена, но маршрут неактуален. Анимация остановлена.")

if __name__ == "__main__":
    root = tk.Tk()
    app = Simulation(root)
    root.mainloop()
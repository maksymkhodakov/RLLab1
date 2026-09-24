"""
Дискретизація середовища MountainCar-v0 (Gymnasium) та побудова точної
табличної MDP-моделі на основі відомої (детермінованої) фізики середовища.
Це дозволяє застосувати класичні методи динамічного програмування —
Policy Iteration та Value Iteration — які вимагають повної моделі переходів
P(s'|s,a) та функції винагороди R(s,a).

Фізика MountainCar (з gymnasium.envs.classic_control.mountain_car):
    velocity += (action - 1) * force + cos(3 * position) * (-gravity)
    velocity  = clip(velocity, -max_speed, max_speed)
    position += velocity
    position  = clip(position, min_position, max_position)
    if position == min_position and velocity < 0: velocity = 0
    terminated = position >= goal_position and velocity >= goal_velocity
    reward = -1.0 кожен крок

===========================================================================
Інтерпретація оптимальної політики: чому "штовхай у напрямку швидкості"
===========================================================================
Двигун машинки (force = 0.001) занадто слабкий, щоб заїхати на пагорб
прямим штовханням вправо з дна долини — сили тяжіння (яка гальмує підйом)
завжди більше за силу двигуна. Тому пряма стратегія "завжди push right"
НЕ розв'язує задачу: машинка застрягне десь на схилі, не досягнувши цілі.

Єдиний спосіб дістатись вершини — накопичити достатньо МЕХАНІЧНОЇ ЕНЕРГІЇ
(кінетичної + потенціальної), розгойдуючись між двома схилами, перш ніж
здійснити фінальний ривок. Це і дає оптимальна політика, яку виводять і
Value/Policy Iteration (mountain_car_dp.py + dp_algorithms.py), і Monte
Carlo control (mountain_car_mc.py) — обидва підходи, попри принципову
різницю (планування за точною моделлю проти навчання з досвіду), незалежно
приходять до ОДНІЄЇ Й ТІЄЇ Ж якісної стратегії:

    push right (2), якщо velocity > 0   — машинка вже рухається вправо
                                           (вгору правим схилом або вниз
                                           лівим після повороту) — штовхай
                                           у той самий бік, щоб збільшити
                                           швидкість і, відповідно, висоту,
                                           на яку заїде інерція.
    push left  (0), якщо velocity < 0   — дзеркально: машинка рухається
                                           вліво (вгору лівим схилом) —
                                           штовхай вліво, щоб піднятись
                                           якомога вище, перш ніж сила
                                           тяжіння розверне її назад.
    (перемикання відбувається в момент, коли velocity змінює знак, тобто
     в нижній точці долини або на вершині одного зі схилів, де швидкість
     миттєво дорівнює нулю)

Інакше кажучи: політика "штовхає в напрямку поточного руху" замість
"штовхає в напрямку цілі" — контрінтуїтивно, але саме так система набирає
енергію коливань (кожен цикл "вліво-вправо" підвищує максимальну висоту,
до якої дістається машинка, аж поки цього не вистачить, щоб перевалити
через правий пагорб). На тепловій карті політики (results/policy_heatmap_*.png,
results/mc_policy_heatmap.png) це виглядає як дві зони, розділені кривою,
що майже збігається з лінією velocity=0: червоне (push right) зверху
(velocity>0), синє (push left) знизу (velocity<0).

Точні (DP, VI/PI) та наближені (Monte Carlo) реалізації дають ЯКІСНО
однакову межу перемикання: у DP вона гладка й точна (аналітичний
розв'язок рівняння Белмана на сітці 300x300); у MC — та сама форма, але
статистично зашумлена (окремі "острівці" неправильного кольору), бо
кожна комірка (s,a) оцінена лише за скінченну кількість спостережених
епізодів, а не обчислена аналітично.
"""
import math
import numpy as np

# Параметри середовища (ідентичні gymnasium MountainCarEnv)
MIN_POSITION = -1.2
MAX_POSITION = 0.6
MAX_SPEED = 0.07
GOAL_POSITION = 0.5
GOAL_VELOCITY = 0.0
FORCE = 0.001
GRAVITY = 0.0025

N_ACTIONS = 3  # 0: push left, 1: no push, 2: push right


def mountain_car_step(position: float, velocity: float, action: int):
    """Один крок точної (неперервної) динаміки MountainCar."""
    velocity += (action - 1) * FORCE + math.cos(3 * position) * (-GRAVITY)
    velocity = np.clip(velocity, -MAX_SPEED, MAX_SPEED)
    position += velocity
    position = np.clip(position, MIN_POSITION, MAX_POSITION)
    if position == MIN_POSITION and velocity < 0:
        velocity = 0.0
    terminated = bool(position >= GOAL_POSITION and velocity >= GOAL_VELOCITY)
    return position, velocity, terminated


class DiscretizedMountainCar:
    """
    Дискретизує простір станів (position, velocity) на регулярну сітку
    n_pos x n_vel комірок та будує детерміновану табличну модель переходів,
    використовуючи справжню фізику середовища, застосовану до центру кожної
    комірки. Термінальні (цільові) стани — поглинальні, з нульовою винагородою.
    """

    def __init__(self, n_pos: int = 60, n_vel: int = 60):
        self.n_pos = n_pos
        self.n_vel = n_vel
        self.n_states = n_pos * n_vel

        self.pos_bins = np.linspace(MIN_POSITION, MAX_POSITION, n_pos)
        self.vel_bins = np.linspace(-MAX_SPEED, MAX_SPEED, n_vel)

        # P[s, a] -> s'  (int),  R[s, a] -> reward (float),  terminal[s] -> bool
        self.P = np.zeros((self.n_states, N_ACTIONS), dtype=np.int64)
        self.R = np.full((self.n_states, N_ACTIONS), -1.0, dtype=np.float64)
        self.terminal = np.zeros(self.n_states, dtype=bool)

        self._build_model()

    # --- індексація -------------------------------------------------
    def state_index(self, i_pos: int, i_vel: int) -> int:
        return i_pos * self.n_vel + i_vel

    def index_to_cell(self, s: int):
        return divmod(s, self.n_vel)

    def continuous_to_index(self, position: float, velocity: float) -> int:
        i_pos = int(np.argmin(np.abs(self.pos_bins - position)))
        i_vel = int(np.argmin(np.abs(self.vel_bins - velocity)))
        return self.state_index(i_pos, i_vel)

    def index_to_continuous(self, s: int):
        i_pos, i_vel = self.index_to_cell(s)
        return self.pos_bins[i_pos], self.vel_bins[i_vel]

    # --- побудова моделі ---------------------------------------------
    def _build_model(self):
        for i_pos in range(self.n_pos):
            for i_vel in range(self.n_vel):
                s = self.state_index(i_pos, i_vel)
                pos = self.pos_bins[i_pos]
                vel = self.vel_bins[i_vel]

                is_goal = bool(pos >= GOAL_POSITION and vel >= GOAL_VELOCITY)
                self.terminal[s] = is_goal

                if is_goal:
                    # поглинальний стан: залишаємось у собі, винагорода 0
                    for a in range(N_ACTIONS):
                        self.P[s, a] = s
                        self.R[s, a] = 0.0
                    continue

                for a in range(N_ACTIONS):
                    n_pos_, n_vel_, terminated = mountain_car_step(pos, vel, a)
                    s_next = self.continuous_to_index(n_pos_, n_vel_)
                    self.P[s, a] = s_next
                    self.R[s, a] = -1.0

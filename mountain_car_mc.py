"""
On-policy, first-visit Monte Carlo control для MountainCar-v0 (Gymnasium),
з потенціал-базованим формуванням винагороди (potential-based reward
shaping, Ng, Harada & Russell, 1999) для подолання проблеми розрідженої
винагороди.

Принципова відмінність від Policy/Value Iteration (mountain_car_dp.py,
dp_algorithms.py):

    DP-методи (PI, VI)                      Monte Carlo control
    -------------------------------------   -------------------------------------
    Потребують ПОВНОЇ моделі MDP:           Model-free: агент НІЧОГО не знає
    P(s'|s,a) та R(s,a) відомі заздалегідь  про P і R. Q(s,a) оцінюється
    (тут — з точної фізики середовища).     виключно за середньою емпіричною
                                             віддачею (return), спостереженою
                                             під час реальних епізодів взаємодії.
    "Планування" (planning): один прохід    "Навчання" (learning): потрібно
    по всіх станах одразу дає V*.           програти багато епізодів і
                                             поступово усереднювати Q(s,a).
    Дискретизація потрібна лише щоб          Та сама дискретизація потрібна ще
    застосувати табличні рівняння Беллмана. й тому, що табличний MC не може
                                             оцінити Q для неперервного простору
                                             станів без апроксимації.

===========================================================================
Чому "наївний" Monte Carlo не працює на MountainCar (і як це виправлено)
===========================================================================

MountainCar — хрестоматійний приклад задачі зі "складним дослідженням"
(hard-exploration problem): винагорода дорівнює -1 на КОЖНОМУ кроці, і
0 лише в момент досягнення прапорця. Це означає, що:

  1. Оптимальна поведінка вимагає спершу від'їхати ВІД цілі (щоб накопичити
     потенціальну + кінетичну енергію), перш ніж штовхнути машинку вгору —
     тобто потрібна довга (~100+ кроків) послідовність "правильних" дій
     без ЖОДНОГО проміжного заохочення.
  2. Випадкова (або майже випадкова, epsilon≈1) політика практично ніколи
     не потрапляє в ціль за розумний бюджет кроків: емпірично, повністю
     випадкова політика зі стандартного старту вирішує епізод лише у
     ~1.5% випадків НАВІТЬ за горизонт 2000 кроків (виміряно окремо).
  3. Тому чистий Monte Carlo control (агент навчається лише з повних
     епізодів, підсумовуючи -1 за кожен крок) отримує ОДНАКОВО погану
     віддачу (≈ -max_steps) для майже всіх траєкторій — жодного сигналу,
     що відрізняв би "гарну" дію від "поганої", і Q(s,a) не сходиться до
     чогось кращого за випадкову політику в межах практичного бюджету
     епізодів. Це підтверджено експериментально (див. нижче) двома
     класичними виправленнями "дослідження", жодне з яких саме по собі
     не розв'язало проблему в розумний час:
       - Exploring Starts (Sutton & Barto §5.3): старт кожного епізоду з
         випадкової точки простору станів. Дає непогане покриття (s,a) і
         високий success rate, УСЕРЕДНЕНИЙ по всьому простору станів
         (~97%) — але це оманливо, бо велика частка випадкових стартів
         тривіальна (вже поруч із ціллю). У вузькій "важкій" зоні
         стандартного старту Gymnasium (position∈[-0.6,-0.4], velocity=0)
         політика лишається невивченою: 0% успіху при чесному
         оцінюванні зі стандартного старту навіть після 150 000+150 000
         епізодів (ES-фаза + додаткова фаза "донавчання" зі стандартного
         старту) на сітці 150x150 (розмір, що усуває аліасинг
         дискретизації швидкості — див. README).
       - Просто epsilon-greedy зі стандартного старту: без жодного
         випадкового успіху немає диференційного сигналу — Q(s,a)
         лишається на рівні ініціалізації (перевірено).

  Висновок: сама лише зміна СХЕМИ ДОСЛІДЖЕННЯ недостатня — потрібно дати
  агенту сигнал ПРО ПРОГРЕС на кожному кроці, а не лише в момент успіху.

  Рішення — **potential-based reward shaping** (Ng, Harada & Russell,
  1999): до кожної винагороди додається член
        F(s, s') = gamma * Phi(s') - Phi(s'),
  де Phi(s) — потенціал стану (тут: Phi(position) = C * sin(3 * position),
  що пропорційне до висоти машинки на схилі — тій самій формулі, що
  формує рельєф MountainCar). Теорема Ng et al. гарантує: додавання
  ТАКОГО (потенціал-базованого) члена НЕ змінює множину оптимальних
  політик вихідної MDP — це не "підказка", яка змінює задачу, а лише
  щільніший, інформативніший сигнал навчання, який спрямовує агента до
  того самого оптимуму. Практично: тепер кожен крок "вгору схилом"
  винагороджується відносно кроку "вниз схилом", і Monte Carlo отримує
  осмислений диференційний сигнал з першого ж епізоду, навіть зі
  стандартного (не exploring-starts) старту та звичайного
  epsilon-greedy дослідження.

  Результат (виміряно): з формуванням винагороди (C=100) на сітці
  50x50 та стандартним стартом Gymnasium, MC control досягає ~98.5%
  success rate (100 епізодів, ліміт 200 кроків) вже за ~100 000
  епізодів навчання (~370с), без exploring starts.

===========================================================================
Дискретизація
===========================================================================
Сітка тут суттєво грубіша (за замовчуванням 50x50 = 2500 станів, 7500 пар
(s,a)), ніж у DP (300x300 = 90000 станів). Причина — вибіркова складність:
DP отримує модель миттєво аналітично, а MC повинен НАВІДВІДУВАТИ кожну
пару (s,a) багато разів, щоб середнє зійшлося; емпірично грубіша сітка
(менше унікальних (s,a) на той самий бюджет епізодів) дала КРАЩИЙ
результат (50x50: 98.5%) ніж дрібніша (100x100: 43%, 80x80: 65% при
порівнянному бюджеті) — типовий для model-free методів компроміс
"роздільна здатність проти вибіркової ефективності".
"""
import math
import time
import numpy as np
import gymnasium as gym

N_ACTIONS = 3
MIN_POSITION, MAX_POSITION = -1.2, 0.6
MAX_SPEED = 0.07


class MCDiscretizer:
    """Дискретизація (position, velocity) на регулярну сітку n_pos x n_vel.

    На відміну від DiscretizedMountainCar (mountain_car_dp.py), тут НЕ
    будується модель переходів P(s'|s,a) — вона Monte Carlo не потрібна.
    Дискретизація використовується лише для індексації табличного Q(s,a).
    """

    def __init__(self, n_pos: int = 50, n_vel: int = 50):
        self.n_pos = n_pos
        self.n_vel = n_vel
        self.n_states = n_pos * n_vel
        self.pos_bins = np.linspace(MIN_POSITION, MAX_POSITION, n_pos)
        self.vel_bins = np.linspace(-MAX_SPEED, MAX_SPEED, n_vel)

    def state_index(self, obs) -> int:
        i_pos = int(np.argmin(np.abs(self.pos_bins - obs[0])))
        i_vel = int(np.argmin(np.abs(self.vel_bins - obs[1])))
        return i_pos * self.n_vel + i_vel

    def continuous_to_index(self, position: float, velocity: float) -> int:
        return self.state_index((position, velocity))


def potential(position: float, scale: float) -> float:
    """Phi(s) — потенціал стану для reward shaping, пропорційний до
    висоти машинки на схилі (та сама форма рельєфу sin(3x), що й фізика
    середовища; знак/масштаб не мають значення для теореми інваріантності
    оптимальної політики — обраний scale лише покращує співвідношення
    сигнал/шум навчання)."""
    return scale * math.sin(3.0 * position)


def _epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(Q[s]))


def _generate_episode(env, disc, Q, epsilon, rng, max_steps, gamma, shaping_scale):
    obs, _ = env.reset(seed=int(rng.integers(1_000_000_000)))
    s = disc.state_index(obs)
    trajectory = []
    terminated = False
    for _ in range(max_steps):
        a = _epsilon_greedy(Q, s, epsilon, rng)
        obs_next, reward, terminated, truncated, _ = env.step(a)
        if shaping_scale:
            # F(s,a,s') = gamma*Phi(s') - Phi(s); Phi = висота на схилі.
            reward = reward + gamma * potential(obs_next[0], shaping_scale) - potential(obs[0], shaping_scale)
        s_next = disc.state_index(obs_next)
        trajectory.append((s, a, reward))
        s, obs = s_next, obs_next
        if terminated or truncated:
            break
    return trajectory, terminated


def mc_control(n_episodes: int = 100_000, n_pos: int = 50, n_vel: int = 50,
                gamma: float = 0.99, epsilon_start: float = 1.0, epsilon_end: float = 0.03,
                max_steps: int = 500, log_every: int = 2_000, seed: int = 0,
                shaping_scale: float = 100.0):
    """
    On-policy, first-visit Monte Carlo control з potential-based reward
    shaping (обидва пояснені в докстрінгу модуля).

    Алгоритм (Sutton & Barto, §5.3, on-policy first-visit MC control,
    адаптований шейпінгом винагороди для щільнішого сигналу навчання):
        1. Згенерувати епізод зі СТАНДАРТНОГО старту Gymnasium, слідуючи
           поточній epsilon-greedy політиці щодо Q; кожна винагорода
           доповнюється потенціал-базованим членом F(s,a,s').
        2. Пройти епізод у зворотному напрямку, накопичуючи дисконтовану
           віддачу G_t = r_t + gamma*G_{t+1} (r_t тут — вже "сформована"
           винагорода).
        3. Для ПЕРШОГО входження кожної пари (s,a) у епізоді: оновити
           Q(s,a) інкрементним середнім по всіх спостережених G.
        4. epsilon лінійно згасає від epsilon_start до epsilon_end
           (GLIE-подібна схема): на старті майже повне дослідження, потім
           поступовий перехід до експлуатації вивченої політики.

    Повертає Q, жадібну policy, дискретизатор та статистику навчання
    (включно з кривою навчання — ковзний success rate за log_every
    епізодів — для побудови графіків).
    """
    disc = MCDiscretizer(n_pos, n_vel)
    Q = np.zeros((disc.n_states, N_ACTIONS))
    N = np.zeros((disc.n_states, N_ACTIONS), dtype=np.int64)

    env = gym.make("MountainCar-v0", max_episode_steps=max_steps)
    rng = np.random.default_rng(seed)

    successes_so_far = 0
    learning_curve = []   # (episode, rolling_success_rate, epsilon)
    window = []

    start = time.perf_counter()
    for ep in range(1, n_episodes + 1):
        frac = ep / n_episodes
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac

        trajectory, terminated = _generate_episode(env, disc, Q, epsilon, rng, max_steps, gamma, shaping_scale)
        successes_so_far += int(terminated)
        window.append(int(terminated))
        if len(window) > log_every:
            window.pop(0)

        G = 0.0
        visited = set()
        for s, a, r in reversed(trajectory):
            G = gamma * G + r
            key = (s, a)
            if key not in visited:
                visited.add(key)
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]

        if ep % log_every == 0 or ep == n_episodes:
            learning_curve.append((ep, float(np.mean(window)), epsilon))

    env.close()
    time_sec = time.perf_counter() - start

    policy = np.argmax(Q, axis=1)
    # Незважаючи на те, що Q тут отримано вибірковим усередненням (а не
    # аналітично, як у VI/PI), жадібна policy сходиться до ЯКІСНО ТІЄЇ Ж
    # стратегії "розгойдування" (push right при velocity>0, push left при
    # velocity<0) — детальне пояснення чому це оптимально для MountainCar
    # див. у докстрінгу модуля mountain_car_dp.py. Різниця лише в тому, що
    # межа перемикання дій тут статистично зашумлена (кожна комірка (s,a)
    # оцінена за скінченну к-сть епізодів), а не гладка, як у точному
    # DP-розв'язку.
    stats = {
        "time_sec": time_sec,
        "n_episodes": n_episodes,
        "shaping_scale": shaping_scale,
        "learning_curve": learning_curve,
        "visited_state_actions": int(np.sum(N > 0)),
        "total_state_actions": int(Q.size),
        "training_successes": successes_so_far,
        "training_success_rate": successes_so_far / n_episodes,
    }
    return Q, policy, disc, stats

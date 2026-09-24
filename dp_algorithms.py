"""
Класичні методи динамічного програмування — Policy Iteration та
Value Iteration — для детермінованої табличної MDP, побудованої в
mountain_car_dp.DiscretizedMountainCar (P[s, a] -> s', R[s, a] -> r).
"""
import time
import numpy as np


def policy_evaluation(policy, P, R, terminal, gamma, theta=1e-6, max_iter=10_000,
                       V0=None):
    """Ітеративна оцінка політики (policy evaluation) до збіжності.

    Векторизовано по станах (Gauss-Jacobi замість Gauss-Seidel), щоб не
    ганяти повільний Python-цикл по ~3600 станах на кожній розгортці.
    """
    n_states = P.shape[0]
    V = np.zeros(n_states) if V0 is None else V0.copy()
    P_a = P[np.arange(n_states), policy]
    R_a = R[np.arange(n_states), policy]
    for it in range(1, max_iter + 1):
        V_new = R_a + gamma * V[P_a]
        V_new[terminal] = 0.0
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            return V, it
    return V, max_iter


def policy_iteration(P, R, terminal, gamma=0.99, theta=1e-6, max_eval_iter=10_000):
    """
    Policy Iteration: почергово (1) повна оцінка поточної політики,
    (2) жадібне покращення політики, поки політика не перестане змінюватись.
    Повертає V, policy та статистику збіжності.
    """
    n_states, n_actions = P.shape
    policy = np.zeros(n_states, dtype=np.int64)
    V = np.zeros(n_states)

    start = time.perf_counter()
    stats = {"policy_eval_sweeps": [], "policy_improve_steps": 0, "value_change_history": []}

    while True:
        V, sweeps = policy_evaluation(policy, P, R, terminal, gamma, theta, max_eval_iter, V0=V)
        stats["policy_eval_sweeps"].append(sweeps)

        Q = R + gamma * V[P]
        best_a = np.argmax(Q, axis=1)
        policy_stable = bool(np.all(best_a[~terminal] == policy[~terminal]))
        stats["value_change_history"].append(float(np.max(np.abs(best_a != policy))))
        policy = best_a

        stats["policy_improve_steps"] += 1
        if policy_stable:
            break

    stats["time_sec"] = time.perf_counter() - start
    stats["total_eval_sweeps"] = int(sum(stats["policy_eval_sweeps"]))
    # policy тут — та сама π* "розгойдування" (energy pumping), що й у
    # value_iteration (обидва методи доводять збіжність до єдиного
    # оптимуму — див. mountain_car_dp.py для детального пояснення стратегії).
    return V, policy, stats


def value_iteration(P, R, terminal, gamma=0.99, theta=1e-6, max_iter=100_000):
    """
    Value Iteration: одна комбінована операція оцінки+покращення за крок,
    повторюється до збіжності функції цінності, потім витягується жадібна
    політика.
    """
    n_states, n_actions = P.shape
    V = np.zeros(n_states)

    delta_history = []
    start = time.perf_counter()
    for it in range(1, max_iter + 1):
        Q = R + gamma * V[P]  # (n_states, n_actions)
        Q[terminal, :] = 0.0
        V_new = np.max(Q, axis=1)
        V_new[terminal] = 0.0
        delta = np.max(np.abs(V_new - V))
        delta_history.append(float(delta))
        V = V_new
        if delta < theta:
            break
    time_sec = time.perf_counter() - start

    Q = R + gamma * V[P]
    policy = np.argmax(Q, axis=1)
    # Отримана π*(s) — стратегія "розгойдування" (energy pumping): push
    # right при velocity>0, push left при velocity<0 (детальне пояснення
    # чому це оптимально — докстрінг модуля mountain_car_dp.py).

    stats = {"sweeps": it, "time_sec": time_sec, "delta_history": delta_history}
    return V, policy, stats

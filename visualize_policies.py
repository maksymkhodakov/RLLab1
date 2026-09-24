"""
Візуалізація роботи політик, отриманих Value Iteration, Policy Iteration
та Monte Carlo control, у справжньому середовищі MountainCar-v0
(Gymnasium, render_mode="human").

Запуск:
    python visualize_policies.py                # усі методи, 1 епізод кожен
    python visualize_policies.py --episodes 3    # 3 епізоди кожен
    python visualize_policies.py --method vi     # лише Value Iteration
    python visualize_policies.py --method pi     # лише Policy Iteration
    python visualize_policies.py --method mc     # лише Monte Carlo control
"""
import argparse
import time

import gymnasium as gym

from mountain_car_dp import DiscretizedMountainCar
from dp_algorithms import policy_iteration, value_iteration
from mountain_car_mc import mc_control
from run_experiment import N_POS, N_VEL, GAMMA, THETA
from run_mc_experiment import MC_N_POS, MC_N_VEL, MC_N_EPISODES, MC_MAX_STEPS_TRAIN, MC_SHAPING_SCALE, \
    MC_EPSILON_START, MC_EPSILON_END

MAX_STEPS = 1000


def run_visual_episode(env, state_index_fn, policy, label, episode_idx, seed):
    obs, _ = env.reset(seed=seed)
    print(f"\n[{label}] Епізод {episode_idx + 1}: старт = {obs}")
    for t in range(1, MAX_STEPS + 1):
        s = state_index_fn(obs)
        a = int(policy[s])
        obs, reward, terminated, truncated, _ = env.step(a)
        env.render()
        if terminated:
            print(f"[{label}] Успіх за {t} кроків!")
            break
        if truncated:
            print(f"[{label}] Не досягнуто цілі за {t} кроків (ліміт).")
            break
    time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1, help="К-сть епізодів на метод")
    parser.add_argument("--method", choices=["vi", "pi", "mc", "both", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    method = "all" if args.method == "both" else args.method  # "both" — старий синонім для "all"

    # policies: label -> (policy_array, state_index_fn)
    policies = {}

    if method in ("vi", "pi", "all"):
        print(f"Побудова дискретизованої MDP-моделі ({N_POS}x{N_VEL})...")
        model = DiscretizedMountainCar(n_pos=N_POS, n_vel=N_VEL)
        dp_state_index = lambda obs: model.continuous_to_index(float(obs[0]), float(obs[1]))

        if method in ("vi", "all"):
            print("Обчислення Value Iteration...")
            _, policy_vi, _ = value_iteration(model.P, model.R, model.terminal, gamma=GAMMA, theta=THETA)
            policies["Value Iteration"] = (policy_vi, dp_state_index)
        if method in ("pi", "all"):
            print("Обчислення Policy Iteration...")
            _, policy_pi, _ = policy_iteration(model.P, model.R, model.terminal, gamma=GAMMA, theta=THETA)
            policies["Policy Iteration"] = (policy_pi, dp_state_index)

    if method in ("mc", "all"):
        print(f"Навчання Monte Carlo control (сітка {MC_N_POS}x{MC_N_VEL}, {MC_N_EPISODES} епізодів, "
              f"це займе кілька хвилин)...")
        _, policy_mc, disc_mc, _ = mc_control(
            n_episodes=MC_N_EPISODES, n_pos=MC_N_POS, n_vel=MC_N_VEL, gamma=GAMMA,
            epsilon_start=MC_EPSILON_START, epsilon_end=MC_EPSILON_END,
            max_steps=MC_MAX_STEPS_TRAIN, shaping_scale=MC_SHAPING_SCALE, seed=0,
        )
        policies["Monte Carlo"] = (policy_mc, disc_mc.state_index)

    env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=MAX_STEPS)
    try:
        for label, (policy, state_index_fn) in policies.items():
            for ep in range(args.episodes):
                run_visual_episode(env, state_index_fn, policy, label, ep, seed=args.seed + ep)
    finally:
        env.close()


if __name__ == "__main__":
    main()

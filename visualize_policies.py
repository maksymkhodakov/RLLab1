"""
Візуалізація роботи політик, отриманих Policy Iteration та Value Iteration,
у справжньому середовищі MountainCar-v0 (Gymnasium, render_mode="human").

Запуск:
    python visualize_policies.py                # обидва методи, 1 епізод кожен
    python visualize_policies.py --episodes 3    # 3 епізоди кожен
    python visualize_policies.py --method vi     # лише Value Iteration
    python visualize_policies.py --method pi     # лише Policy Iteration
"""
import argparse
import time

import gymnasium as gym

from mountain_car_dp import DiscretizedMountainCar
from dp_algorithms import policy_iteration, value_iteration
from run_experiment import N_POS, N_VEL, GAMMA, THETA

MAX_STEPS = 1000


def run_visual_episode(env, model, policy, label, episode_idx, seed):
    obs, _ = env.reset(seed=seed)
    print(f"\n[{label}] Епізод {episode_idx + 1}: старт = {obs}")
    for t in range(1, MAX_STEPS + 1):
        s = model.continuous_to_index(float(obs[0]), float(obs[1]))
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
    parser.add_argument("--method", choices=["vi", "pi", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Побудова дискретизованої MDP-моделі ({N_POS}x{N_VEL})...")
    model = DiscretizedMountainCar(n_pos=N_POS, n_vel=N_VEL)

    policies = {}
    if args.method in ("vi", "both"):
        print("Обчислення Value Iteration...")
        _, policy_vi, _ = value_iteration(model.P, model.R, model.terminal, gamma=GAMMA, theta=THETA)
        policies["Value Iteration"] = policy_vi
    if args.method in ("pi", "both"):
        print("Обчислення Policy Iteration...")
        _, policy_pi, _ = policy_iteration(model.P, model.R, model.terminal, gamma=GAMMA, theta=THETA)
        policies["Policy Iteration"] = policy_pi

    env = gym.make("MountainCar-v0", render_mode="human", max_episode_steps=MAX_STEPS)
    try:
        for label, policy in policies.items():
            for ep in range(args.episodes):
                run_visual_episode(env, model, policy, label, ep, seed=args.seed + ep)
    finally:
        env.close()


if __name__ == "__main__":
    main()

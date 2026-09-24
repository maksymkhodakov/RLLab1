"""
Лабораторна робота: розв'язання MountainCar-v0 (Gymnasium) методами
Policy Iteration та Value Iteration, з порівнянням результатів.

Запуск:
    python run_experiment.py

Результат:
    - консольний звіт з метриками збіжності та якості політик
    - results/policy_heatmap.png   — теплові карти отриманих політик
    - results/value_function.png   — функції цінності V*(s)
    - results/convergence.png      — швидкість збіжності VI та PI
    - results/comparison.png       — підсумкове порівняння метрик
"""
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from mountain_car_dp import DiscretizedMountainCar
from dp_algorithms import policy_iteration, value_iteration

N_POS, N_VEL = 200, 200
# Примітка: крок сили в MountainCar дуже малий (force=0.001) відносно
# діапазону швидкості (0.14), тому груба сітка (наприклад 60x60, ширина
# комірки швидкості ~0.0024) призводить до аліасингу при округленні до
# найближчої комірки — похибка дискретизації маскує реальний вплив дії,
# і отримана політика виявляється майже випадковою в неперервному
# середовищі. Сітка 200x200 (ширина комірки швидкості ~0.0007 < force)
# усуває цей ефект і дає стабільну, майже 100% успішну політику.
GAMMA = 0.99
THETA = 1e-6
EVAL_EPISODES = 100
MAX_STEPS = 1000  # більше за стандартний ліміт (200), щоб не занижувати політику під час оцінки

ACTION_NAMES = {0: "← push left", 1: "no push", 2: "push right →"}


def evaluate_policy(model: DiscretizedMountainCar, policy: np.ndarray,
                     n_episodes: int = EVAL_EPISODES, max_steps: int = MAX_STEPS, seed: int = 0):
    """Прогін отриманої (дискретизованої) політики у справжньому
    неперервному середовищі Gymnasium; повертає статистику успішності."""
    env = gym.make("MountainCar-v0", max_episode_steps=max_steps)
    lengths = []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(1, max_steps + 1):
            s = model.continuous_to_index(float(obs[0]), float(obs[1]))
            a = int(policy[s])
            obs, reward, terminated, truncated, _ = env.step(a)
            if terminated:
                successes += 1
                lengths.append(t)
                break
            if truncated:
                lengths.append(t)
                break
    env.close()
    success_rate = successes / n_episodes
    avg_len = float(np.mean(lengths))
    return {
        "success_rate": success_rate,
        "avg_episode_len": avg_len,
        "successes": successes,
        "n_episodes": n_episodes,
    }


def plot_policy_heatmap(model, policy, title, fname):
    grid = policy.reshape(model.n_pos, model.n_vel)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="coolwarm",
                    extent=[model.pos_bins[0], model.pos_bins[-1], model.vel_bins[0], model.vel_bins[-1]])
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([ACTION_NAMES[0], ACTION_NAMES[1], ACTION_NAMES[2]])
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_value_function(model, V, title, fname):
    grid = V.reshape(model.n_pos, model.n_vel)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis",
                    extent=[model.pos_bins[0], model.pos_bins[-1], model.vel_bins[0], model.vel_bins[-1]])
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="V*(s)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_convergence(vi_stats, pi_stats, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(vi_stats["delta_history"], label="Value Iteration (Δ на розгортку)")
    # для PI: показуємо к-сть sweeps policy evaluation на кожній ітерації покращення
    pi_sweeps = pi_stats["policy_eval_sweeps"]
    ax.axvline(len(vi_stats["delta_history"]), color="gray", linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("ітерація (розгортка) Value Iteration")
    ax.set_ylabel("max |V_new - V| (log scale)")
    ax.set_title("Збіжність Value Iteration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_comparison(vi_stats, pi_stats, vi_eval, pi_eval, fname):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].bar(["Value Iteration", "Policy Iteration"],
                [vi_stats["time_sec"], pi_stats["time_sec"]], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Час обчислення (с)")
    axes[0].set_ylabel("секунди")

    axes[1].bar(["Value Iteration\n(sweeps)", "Policy Iteration\n(improve. steps)"],
                [vi_stats["sweeps"], pi_stats["policy_improve_steps"]], color=["#4C72B0", "#DD8452"])
    axes[1].set_title("К-сть ітерацій до збіжності")

    axes[2].bar(["Value Iteration", "Policy Iteration"],
                [vi_eval["success_rate"] * 100, pi_eval["success_rate"] * 100], color=["#4C72B0", "#DD8452"])
    axes[2].set_title("Успішність політики (%)")
    axes[2].set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    print(f"Побудова дискретизованої MDP-моделі MountainCar ({N_POS}x{N_VEL} = {N_POS*N_VEL} станів)...")
    t0 = time.perf_counter()
    model = DiscretizedMountainCar(n_pos=N_POS, n_vel=N_VEL)
    print(f"  модель побудована за {time.perf_counter()-t0:.3f} с, "
          f"термінальних станів: {int(model.terminal.sum())}")

    print("\n=== Value Iteration ===")
    V_vi, policy_vi, vi_stats = value_iteration(model.P, model.R, model.terminal, gamma=GAMMA, theta=THETA)
    print(f"  збіжність за {vi_stats['sweeps']} розгорток, час {vi_stats['time_sec']:.3f} с")

    print("\n=== Policy Iteration ===")
    V_pi, policy_pi, pi_stats = policy_iteration(model.P, model.R, model.terminal, gamma=GAMMA, theta=THETA)
    print(f"  збіжність за {pi_stats['policy_improve_steps']} кроків покращення політики, "
          f"сумарно {pi_stats['total_eval_sweeps']} розгорток оцінки, час {pi_stats['time_sec']:.3f} с")
    print(f"  розгорток policy evaluation на кожному кроці: {pi_stats['policy_eval_sweeps']}")

    same_policy = np.array_equal(policy_vi, policy_pi)
    max_v_diff = float(np.max(np.abs(V_vi - V_pi)))
    print(f"\nПолітики VI та PI ідентичні: {same_policy}")
    print(f"Максимальна різниця V*(s) між VI та PI: {max_v_diff:.6f}")

    print("\nОцінювання політик у справжньому (неперервному) середовищі Gymnasium "
          f"({EVAL_EPISODES} епізодів кожна)...")
    vi_eval = evaluate_policy(model, policy_vi)
    pi_eval = evaluate_policy(model, policy_pi)

    print(f"  Value Iteration : success_rate={vi_eval['success_rate']*100:.1f}%  "
          f"avg_episode_len={vi_eval['avg_episode_len']:.1f}")
    print(f"  Policy Iteration: success_rate={pi_eval['success_rate']*100:.1f}%  "
          f"avg_episode_len={pi_eval['avg_episode_len']:.1f}")

    print("\nПобудова графіків у results/ ...")
    plot_policy_heatmap(model, policy_vi, "Політика: Value Iteration", "results/policy_heatmap_vi.png")
    plot_policy_heatmap(model, policy_pi, "Політика: Policy Iteration", "results/policy_heatmap_pi.png")
    plot_value_function(model, V_vi, "V*(s): Value Iteration", "results/value_function_vi.png")
    plot_value_function(model, V_pi, "V*(s): Policy Iteration", "results/value_function_pi.png")
    plot_convergence(vi_stats, pi_stats, "results/convergence.png")
    plot_comparison(vi_stats, pi_stats, vi_eval, pi_eval, "results/comparison.png")

    print("\n=== Підсумкова таблиця ===")
    header = f"{'Метрика':35s} {'Value Iteration':>18s} {'Policy Iteration':>18s}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Час обчислення, с", f"{vi_stats['time_sec']:.3f}", f"{pi_stats['time_sec']:.3f}"),
        ("К-сть ітерацій/розгорток", f"{vi_stats['sweeps']}", f"{pi_stats['policy_improve_steps']} (покращень)"),
        ("Сумарних розгорток оцінки", f"{vi_stats['sweeps']}", f"{pi_stats['total_eval_sweeps']}"),
        ("Success rate, %", f"{vi_eval['success_rate']*100:.1f}", f"{pi_eval['success_rate']*100:.1f}"),
        ("Середня довжина епізоду", f"{vi_eval['avg_episode_len']:.1f}", f"{pi_eval['avg_episode_len']:.1f}"),
        ("Політики ідентичні", str(same_policy), str(same_policy)),
    ]
    for name, v1, v2 in rows:
        print(f"{name:35s} {v1:>18s} {v2:>18s}")


if __name__ == "__main__":
    main()

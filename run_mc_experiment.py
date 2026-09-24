"""
Лабораторна робота: розв'язання MountainCar-v0 (Gymnasium) методом
Monte Carlo control (model-free) з порівнянням до Policy Iteration та
Value Iteration (класичне динамічне програмування, run_experiment.py).

Запуск:
    python run_mc_experiment.py

Результат:
    - консольний звіт з метриками навчання, збіжності та якості політик
      усіх трьох методів (MC, VI, PI)
    - results/mc_learning_curve.png   — крива навчання MC (success rate
      та epsilon у часі)
    - results/mc_policy_heatmap.png   — теплова карта політики MC
    - results/mc_value_function.png   — max_a Q(s,a), вивчена MC
    - results/three_way_comparison.png — підсумкове порівняння MC/VI/PI
      за часом, "зусиллям" (епізоди/розгортки) та якістю політики
"""
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from mountain_car_dp import DiscretizedMountainCar
from dp_algorithms import policy_iteration, value_iteration
from mountain_car_mc import mc_control, MCDiscretizer

# --- DP-налаштування (ідентичні run_experiment.py, для чесного порівняння) --
DP_N_POS, DP_N_VEL = 300, 300
GAMMA = 0.99
THETA = 1e-6

# --- MC-налаштування (підібрані експериментально, див. mountain_car_mc.py) --
MC_N_POS, MC_N_VEL = 50, 50
MC_N_EPISODES = 100_000
MC_MAX_STEPS_TRAIN = 500
MC_SHAPING_SCALE = 100.0
MC_EPSILON_START, MC_EPSILON_END = 1.0, 0.03

EVAL_EPISODES = 100
MAX_STEPS = 200  # стандартний ліміт епізоду MountainCar-v0

ACTION_NAMES = {0: "← push left", 1: "no push", 2: "push right →"}


def evaluate_policy(state_index_fn, policy: np.ndarray,
                     n_episodes: int = EVAL_EPISODES, max_steps: int = MAX_STEPS, seed: int = 0):
    """Прогін політики у справжньому (неперервному) середовищі Gymnasium.
    state_index_fn(obs) -> int дозволяє використати цю саму функцію як для
    DP-моделі (сітка 300x300), так і для MC-дискретизатора (сітка 50x50)."""
    env = gym.make("MountainCar-v0", max_episode_steps=max_steps)
    lengths = []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(1, max_steps + 1):
            s = state_index_fn(obs)
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
    mean_reward = -avg_len
    return {
        "success_rate": success_rate,
        "avg_episode_len": avg_len,
        "mean_reward": mean_reward,
        "successes": successes,
        "n_episodes": n_episodes,
    }


def plot_policy_heatmap(disc, policy, title, fname):
    grid = policy.reshape(disc.n_pos, disc.n_vel)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="coolwarm",
                    extent=[disc.pos_bins[0], disc.pos_bins[-1], disc.vel_bins[0], disc.vel_bins[-1]])
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([ACTION_NAMES[0], ACTION_NAMES[1], ACTION_NAMES[2]])
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_value_function(disc, V, title, fname):
    grid = V.reshape(disc.n_pos, disc.n_vel)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis",
                    extent=[disc.pos_bins[0], disc.pos_bins[-1], disc.vel_bins[0], disc.vel_bins[-1]])
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="V(s) = max_a Q(s,a)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_mc_learning_curve(mc_stats, fname):
    episodes, success_rates, epsilons = zip(*mc_stats["learning_curve"])
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(episodes, success_rates, color="#4C72B0", label="success rate (ковзне вікно)")
    ax1.set_xlabel("епізод навчання")
    ax1.set_ylabel("success rate", color="#4C72B0")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(episodes, epsilons, color="#DD8452", linestyle="--", label="epsilon")
    ax2.set_ylabel("epsilon", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    ax1.set_title("Крива навчання Monte Carlo control (з reward shaping)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_three_way_comparison(mc_stats, mc_eval, vi_stats, vi_eval, pi_stats, pi_eval, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    labels = ["Monte Carlo", "Value Iteration", "Policy Iteration"]
    colors = ["#55A868", "#4C72B0", "#DD8452"]

    axes[0].bar(labels, [mc_stats["time_sec"], vi_stats["time_sec"], pi_stats["time_sec"]], color=colors)
    axes[0].set_title("Час обчислення (с)")
    axes[0].set_ylabel("секунди")
    axes[0].set_yscale("log")

    axes[1].bar(labels,
                [mc_stats["n_episodes"], vi_stats["sweeps"], pi_stats["policy_improve_steps"]],
                color=colors)
    axes[1].set_title("К-сть епізодів (MC) / розгорток (VI) /\nкроків покращення (PI)")
    axes[1].set_yscale("log")

    axes[2].bar(labels,
                [mc_eval["success_rate"] * 100, vi_eval["success_rate"] * 100, pi_eval["success_rate"] * 100],
                color=colors)
    axes[2].set_title("Успішність політики (%, 100 епізодів, ліміт 200)")
    axes[2].set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    # === Monte Carlo control (model-free) ===================================
    print(f"=== Monte Carlo control ===")
    print(f"Сітка {MC_N_POS}x{MC_N_VEL} = {MC_N_POS*MC_N_VEL} станів, "
          f"{MC_N_EPISODES} епізодів навчання, reward shaping scale={MC_SHAPING_SCALE}")
    Q_mc, policy_mc, disc_mc, mc_stats = mc_control(
        n_episodes=MC_N_EPISODES, n_pos=MC_N_POS, n_vel=MC_N_VEL, gamma=GAMMA,
        epsilon_start=MC_EPSILON_START, epsilon_end=MC_EPSILON_END,
        max_steps=MC_MAX_STEPS_TRAIN, shaping_scale=MC_SHAPING_SCALE, seed=0,
    )
    V_mc = Q_mc.max(axis=1)
    print(f"  навчання завершено за {mc_stats['time_sec']:.1f} с")
    print(f"  покриття (s,a): {mc_stats['visited_state_actions']}/{mc_stats['total_state_actions']} "
          f"({100*mc_stats['visited_state_actions']/mc_stats['total_state_actions']:.1f}%)")
    print(f"  success rate під час навчання (сумарно): {mc_stats['training_success_rate']*100:.1f}%")

    # === Value Iteration та Policy Iteration (табличне DP, повна модель) ====
    print(f"\n=== Побудова точної DP-моделі ({DP_N_POS}x{DP_N_VEL}) для VI/PI ===")
    t0 = time.perf_counter()
    dp_model = DiscretizedMountainCar(n_pos=DP_N_POS, n_vel=DP_N_VEL)
    print(f"  модель побудована за {time.perf_counter()-t0:.3f} с")

    print("\n=== Value Iteration ===")
    V_vi, policy_vi, vi_stats = value_iteration(dp_model.P, dp_model.R, dp_model.terminal, gamma=GAMMA, theta=THETA)
    print(f"  збіжність за {vi_stats['sweeps']} розгорток, час {vi_stats['time_sec']:.3f} с")

    print("\n=== Policy Iteration ===")
    V_pi, policy_pi, pi_stats = policy_iteration(dp_model.P, dp_model.R, dp_model.terminal, gamma=GAMMA, theta=THETA)
    print(f"  збіжність за {pi_stats['policy_improve_steps']} кроків покращення політики, "
          f"час {pi_stats['time_sec']:.3f} с")

    # === Оцінювання всіх трьох політик у справжньому середовищі ==============
    print(f"\nОцінювання політик у справжньому середовищі Gymnasium "
          f"(ліміт {MAX_STEPS} кроків/епізод, {EVAL_EPISODES} епізодів)...")
    dp_state_index = lambda obs: dp_model.continuous_to_index(float(obs[0]), float(obs[1]))
    mc_eval = evaluate_policy(disc_mc.state_index, policy_mc)
    vi_eval = evaluate_policy(dp_state_index, policy_vi)
    pi_eval = evaluate_policy(dp_state_index, policy_pi)

    for name, ev in [("Monte Carlo", mc_eval), ("Value Iteration", vi_eval), ("Policy Iteration", pi_eval)]:
        print(f"  {name:18s}: success_rate={ev['success_rate']*100:5.1f}%  "
              f"avg_len={ev['avg_episode_len']:6.1f}  mean_reward={ev['mean_reward']:7.2f}")
    print("  (класичний поріг Gym \"розв'язано\": середня винагорода за 100 епізодів >= -110)")

    # === Графіки ==============================================================
    print("\nПобудова графіків у results/ ...")
    plot_mc_learning_curve(mc_stats, "results/mc_learning_curve.png")
    plot_policy_heatmap(disc_mc, policy_mc, "Політика: Monte Carlo control", "results/mc_policy_heatmap.png")
    plot_value_function(disc_mc, V_mc, "V(s)=max_a Q(s,a): Monte Carlo control", "results/mc_value_function.png")
    plot_three_way_comparison(mc_stats, mc_eval, vi_stats, vi_eval, pi_stats, pi_eval,
                               "results/three_way_comparison.png")

    # === Підсумкова таблиця ===================================================
    print("\n=== Підсумкова таблиця: Monte Carlo vs Value Iteration vs Policy Iteration ===")
    header = f"{'Метрика':38s} {'Monte Carlo':>15s} {'Value Iteration':>18s} {'Policy Iteration':>18s}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Тип методу", "model-free", "model-based (DP)", "model-based (DP)"),
        ("Розмір сітки станів", f"{MC_N_POS}x{MC_N_VEL}", f"{DP_N_POS}x{DP_N_VEL}", f"{DP_N_POS}x{DP_N_VEL}"),
        ("Час обчислення, с", f"{mc_stats['time_sec']:.2f}", f"{vi_stats['time_sec']:.3f}", f"{pi_stats['time_sec']:.3f}"),
        ("К-сть епізодів/розгорток", f"{mc_stats['n_episodes']}", f"{vi_stats['sweeps']}",
         f"{pi_stats['policy_improve_steps']} покращень"),
        ("Success rate, %", f"{mc_eval['success_rate']*100:.1f}", f"{vi_eval['success_rate']*100:.1f}",
         f"{pi_eval['success_rate']*100:.1f}"),
        ("Середня довжина епізоду", f"{mc_eval['avg_episode_len']:.1f}", f"{vi_eval['avg_episode_len']:.1f}",
         f"{pi_eval['avg_episode_len']:.1f}"),
        ("Середня винагорода (solved: >=-110)", f"{mc_eval['mean_reward']:.2f}", f"{vi_eval['mean_reward']:.2f}",
         f"{pi_eval['mean_reward']:.2f}"),
    ]
    for name, v1, v2, v3 in rows:
        print(f"{name:38s} {v1:>15s} {v2:>18s} {v3:>18s}")


if __name__ == "__main__":
    main()

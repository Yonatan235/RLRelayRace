import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from rlrelayrace import RLRelayRaceEnvContest, ContestEnvConfig

def animate():
    env = RLRelayRaceEnvContest(ContestEnvConfig(M=4, lanes=4, track_length=400, max_steps=400, exchange_zone_width=0.12, seed=3))
    obs, info = env.reset()

    xs = [[] for _ in range(env.cfg.lanes)]
    runner_lane = [[] for _ in range(env.cfg.lanes)]
    leg = [[] for _ in range(env.cfg.lanes)]
    last_runner = env.runner_lane.copy()
    handoff_frames = [[] for _ in range(env.cfg.lanes)]
    t = 0
    while not (env.terminated or env.truncated):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        for K in range(env.cfg.lanes):
            xs[K].append(env.baton_pos[K])
            runner_lane[K].append(int(env.runner_lane[K]))
            leg[K].append(int(env.leg_index[K]))
            if env.runner_lane[K] != last_runner[K]:
                handoff_frames[K].append(t)
        last_runner = env.runner_lane.copy()
        t += 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, env.cfg.track_length)
    ax.set_ylim(-1, env.cfg.lanes)
    ax.set_yticks(range(env.cfg.lanes))
    ax.set_yticklabels([f"Lane {K}" for K in range(env.cfg.lanes)])
    ax.set_xlabel("Track Position")
    ax.set_title("Contested Relay — Dot color = Runner’s lane; Row = Baton lane; Pulse = Handoff")

    # exchange zones
    for j in range(env.cfg.M - 1):
        L = env.cfg.track_length
        half = (env.cfg.exchange_zone_width * L) / 2
        c = (j + 1) / env.cfg.M * L
        ax.axvspan(c - half, c + half, alpha=0.12, color="gray")

    colors = plt.cm.tab10(np.arange(env.cfg.lanes))
    scatters = [ax.scatter([], [], s=150) for _ in range(env.cfg.lanes)]

    def update(f):
        for K in range(env.cfg.lanes):
            i = min(f, len(xs[K]) - 1)
            scatters[K].set_offsets([xs[K][i], K])
            scatters[K].set_color(colors[runner_lane[K][i]])
            size = 260 if f in handoff_frames[K] else 150
            scatters[K].set_sizes([size])
        return scatters

    ani = animation.FuncAnimation(fig, update, frames=len(xs[0]), interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    animate()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from rlrelayrace import RLRelayRaceEnvContest, ContestEnvConfig

def animate():
    env = RLRelayRaceEnvContest(ContestEnvConfig(M=4, lanes=4, track_length=400, max_steps=400, exchange_zone_width=0.12, seed=3))
    obs, info = env.reset()

    xs = [[] for _ in range(env.cfg.lanes)]
    holder = [[] for _ in range(env.cfg.lanes)]
    leg = [[] for _ in range(env.cfg.lanes)]

    last_holder = env.holder_team.copy()
    handoff_frames = [[] for _ in range(env.cfg.lanes)]
    t = 0
    while not (env.terminated or env.truncated):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        for k in range(env.cfg.lanes):
            xs[k].append(env.baton_pos[k])
            holder[k].append(int(env.holder_team[k]))
            leg[k].append(int(env.leg_index[k]))
            if env.holder_team[k] != last_holder[k]:
                handoff_frames[k].append(t)
        last_holder = env.holder_team.copy()
        t += 1

    fig, ax = plt.subplots(figsize=(9,4))
    ax.set_xlim(0, env.cfg.track_length)
    ax.set_ylim(-1, env.cfg.lanes)
    ax.set_yticks(range(env.cfg.lanes))
    ax.set_yticklabels([f"Lane {k}" for k in range(env.cfg.lanes)])
    ax.set_xlabel("Track Position")
    ax.set_title("Contested Relay (Color = Current Team, pulse = handoff)")

    # draw exchange zones as vertical bands
    for j in range(env.cfg.M-1):
        lo = (j+1)/env.cfg.M*env.cfg.track_length - (env.cfg.exchange_zone_width*env.cfg.track_length)/2
        hi = (j+1)/env.cfg.M*env.cfg.track_length + (env.cfg.exchange_zone_width*env.cfg.track_length)/2
        ax.axvspan(lo, hi, alpha=0.12, color="gray")

    colors = plt.cm.tab10(np.arange(env.cfg.teams))
    scatters = [ax.scatter([], [], s=150) for _ in range(env.cfg.lanes)]

    def update(f):
        for k in range(env.cfg.lanes):
            i = min(f, len(xs[k])-1)
            scatters[k].set_offsets([xs[k][i], k])
            scatters[k].set_color(colors[holder[k][i]])
            size = 150 if f not in handoff_frames[k] else 260
            scatters[k].set_sizes([size])
        return scatters

    ani = animation.FuncAnimation(fig, update, frames=len(xs[0]), interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    animate()

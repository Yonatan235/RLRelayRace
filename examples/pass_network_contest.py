"""
Pass graph per lane (contested exchange zones).
Node label format:
   Lane{K} Leg{J} (Runner from Lane{R})
Meaning: The baton for Lane K was carried during Leg J by the runner from Lane R.
Node color = Runner's lane (R). Each subplot = fixed baton lane (K).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from rlrelayrace import RLRelayRaceEnvContest, ContestEnvConfig

def main():
    env = RLRelayRaceEnvContest(ContestEnvConfig(M=4, track_length=800, exchange_zone_width=0.1, seed=7))
    obs, info = env.reset()

    G_list = [nx.DiGraph() for _ in range(env.cfg.lanes)]
    last_runner_lane = env.runner_lane.copy()
    last_leg = env.leg_index.copy()

    while not (env.terminated or env.truncated):
        # random joint action (replace with your policy)
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)

        for K in range(env.cfg.lanes):
            if env.leg_index[K] != last_leg[K] or env.runner_lane[K] != last_runner_lane[K]:
                # build labels using A2
                src = f"Lane{K} Leg{int(last_leg[K])} (Runner from Lane{int(last_runner_lane[K])})"
                dst = f"Lane{K} Leg{int(env.leg_index[K])} (Runner from Lane{int(env.runner_lane[K])})"
                G_list[K].add_edge(src, dst)

        last_runner_lane = env.runner_lane.copy()
        last_leg = env.leg_index.copy()

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    cmap = plt.cm.tab10
    lane_colors = {r: cmap(r) for r in range(env.cfg.lanes)}
    legend_handles = [mpatches.Patch(color=lane_colors[r], label=f"Runner from Lane {r}") for r in range(env.cfg.lanes)]

    for K in range(env.cfg.lanes):
        G = G_list[K]
        ax = axes[K]
        ax.set_title(f"Lane {K} Handoffs (Contested)")

        node_colors = []
        for node in G.nodes:
            # parse R from "... (Runner from Lane{R})"
            r_str = node.split("Runner from Lane")[-1].rstrip(")")
            r = int(r_str)
            node_colors.append(lane_colors[r])

        pos = nx.circular_layout(G) if len(G.nodes) <= 5 else nx.spring_layout(G, seed=2)
        nx.draw_networkx(
            G, pos, ax=ax, with_labels=True,
            node_color=node_colors, node_size=950,
            font_size=8, font_weight='bold',
            edge_color="black", arrows=True, arrowsize=14, width=2
        )

    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, fontsize=10)
    fig.text(
        0.5, 0.02,
        "Node label: LaneK LegJ (Runner from LaneR).  Color = R (runner's lane).  Subplot = Lane K.",
        ha="center", va="center", fontsize=11
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()

if __name__ == "__main__":
    main()

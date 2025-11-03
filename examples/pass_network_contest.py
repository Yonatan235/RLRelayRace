"""
Pass graph per lane for contested exchange zones.
Nodes = T{team}-L{leg}, meaning:
  T = Team currently carrying the baton
  L = Leg index (0 = first segment, 1 = second, ..., M-1 = final segment)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from rlrelayrace import RLRelayRaceEnvContest, ContestEnvConfig

def team_from_node(node_label: str) -> int:
    # Node label format: "T{team}-L{leg}"
    return int(node_label.split("T")[1].split("-")[0])

def main():
    env = RLRelayRaceEnvContest(
        ContestEnvConfig(M=4, track_length=800, exchange_zone_width=0.1, seed=7)
    )
    obs, info = env.reset()

    G_list = [nx.DiGraph() for _ in range(env.cfg.lanes)]
    last_holder = env.holder_team.copy()
    last_leg = env.leg_index.copy()

    while not (env.terminated or env.truncated):
        a = env.action_space.sample()  # replace with your policy later
        obs, r, term, trunc, info = env.step(a)

        for k in range(env.cfg.lanes):
            # Detect leg or team change → baton switched carriers
            if env.leg_index[k] != last_leg[k] or env.holder_team[k] != last_holder[k]:
                src = f"T{int(last_holder[k])}-L{int(last_leg[k])}"
                dst = f"T{int(env.holder_team[k])}-L{int(env.leg_index[k])}"
                G_list[k].add_edge(src, dst)

        last_holder = env.holder_team.copy()
        last_leg = env.leg_index.copy()

    # ----- Plotting -----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    cmap = plt.cm.tab10

    team_colors = {t: cmap(t) for t in range(env.cfg.teams)}
    legend_handles = [
        mpatches.Patch(color=team_colors[t], label=f"Team {t}") for t in range(env.cfg.teams)
    ]

    for k in range(env.cfg.lanes):
        G = G_list[k]
        ax = axes[k]
        ax.set_title(f"Lane {k} Handoffs (Contested)")

        node_colors = [team_colors[team_from_node(node)] for node in G.nodes]

        # Layout: circular if small, spring if more complex
        pos = nx.circular_layout(G) if len(G.nodes) <= 5 else nx.spring_layout(G, seed=2)

        nx.draw_networkx(
            G, pos, ax=ax,
            with_labels=True,
            node_color=node_colors,
            node_size=950,
            font_size=9,
            font_weight='bold',
            edge_color="black",
            arrows=True,
            arrowsize=14,
            width=2
        )

    # TEAM COLOR LEGEND
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False, fontsize=10)

    # TEXT EXPLANATION OF LABEL MEANING
    fig.text(
        0.5, 0.02,
        "Node label format:  T{team}-L{leg}",
        ha="center", va="center", fontsize=11
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()

if __name__ == "__main__":
    main()
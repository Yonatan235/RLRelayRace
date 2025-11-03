# RLRelayRace — Contested Exchange Zones

RLRelayRace is a multi-agent reinforcement learning environment where runners participate in a long relay race with a twist. Instead of always handing the baton to a teammate, runners can strategically choose any other runner in the race to receive the baton, based on prior performance data. They still want their team to win, and that ultimately depends on the final runners. The interesting question is whether the slowest team can win by outsmarting and co-opting from the faster teams. The environment will be custom-built in Python using OpenAI Gym interfaces and will feature fully RL-trained AI agents. 🎮 Project Overview 📏 4 Teams (4 Lanes) 🧍 M Runners per Team (Total: (4M-4) agents) 🧠 Each runner is an RL agent trained to decide whom to pass the baton to (based on race history and team strategy) with a stationary (untrained) capacity to run. This creates a strategic and emergent decision-making problem where cooperation and betrayal can both be optimal.

## Install and run
```
pip install -e .
python examples/track_animation_contest.py
python examples/pass_network_contest.py
```

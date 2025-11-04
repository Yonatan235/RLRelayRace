## RLRelayRace

**RLRelayRace** is a multi-agent reinforcement learning environment modeling a relay race where baton passing is strategic. Instead of always passing to the next teammate, each runner may pass the baton to any runner available in that leg of the run, using learned decision policies based on race history and performance outcomes. Teams still want their own lane to win, but the passing choices can lead to cooperation, interference, or opportunistic play. 

A Gymnasium environment for a strategic relay:
- Each lane has M runners, one for each leg.
- A runner is permanently tied to their lane and never runs in any other lane.
- The baton may be passed to the next runner of any lane when entering a handoff zone.
- The reward for all runners in a lane (and thus the RL agent for that lane) is based on the finish time of that lane, and optionally its finishing rank.

## Install and run
```
pip install -e .
python examples/track_animation_contest.py
python examples/pass_network_contest.py
```

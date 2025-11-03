## RLRelayRace

**RLRelayRace** is a multi-agent reinforcement learning environment modeling a relay race where baton passing is strategic. Instead of always passing to the next teammate, each runner may pass the baton to any runner available in that leg of the run, using learned decision policies based on race history and performance outcomes. Teams still want their own lane to win, but the passing choices can lead to cooperation, interference, or opportunistic play. 

A Gymnasium environment for a strategic relay:
- 4 lanes (teams), M runners per team
- 4 batons, each moves in a fixed lane (finish depends on lane)
- At exchange zone *j*, only 4 fresh runners are available (one per team, leg *j*)
- The arriving baton may pass to one of those runners (or keep current holder)
- A selected runner retires after their leg

## Install and run
```
pip install -e .
python examples/track_animation_contest.py
python examples/pass_network_contest.py
```

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

@dataclass
class ContestEnvConfig:
    """
    Contested Relay Race (final model)

    - There are T lanes and T teams (T=4 by default), exactly one baton per lane.
    - Each lane k has M runners, one per relay leg j=0..M-1. Runners are permanently tied to their lane.
    - The baton always moves along its current lane; the lane’s finish time determines placement.
    - At exchange zone j (between legs j and j+1), the baton MUST hand off to the next runner of exactly one lane:
        candidates = { (Lane r, Leg j) | r in 0..T-1 AND that runner unused }
      (4 or fewer available if some are already used.)
    - After a runner (Lane r, Leg j) carries a baton, that runner retires (cannot be reused).

    Action (joint Discrete over lanes):
      For each lane k:
        a_k ∈ {1..T}  → choose which lane’s next runner (Lane r = a_k-1) receives the baton if the baton is in the zone.
        If the baton for lane k is NOT inside its zone this step, a_k is ignored.
      The vector (a_0 .. a_{T-1}) is encoded as one integer in base T.

    Speed model:
      Each lane r has a fixed base speed per leg j:
          base_speed[r, j] ~ Uniform(speed_low, speed_high)
      Baton advance each step:
          speed = base_speed[current_runner_lane, current_leg] + Normal(0, sigma)

    Rewards (R2):
      - Primary:  reward_lane = -finish_time(lane)
      - Bonus:    + rank_bonus(rank_of_lane), e.g. {1: +1.0, 2: +0.3, 3: -0.3, 4: -1.0}
      Rewards are given at episode termination (sparse). Per-step reward is 0.
      The env returns scalar sum(reward_lane); per-lane vector is in info['final_lane_rewards'].

    Must-handoff enforcement:
      If a baton is inside its current leg’s exchange zone and no valid handoff is selected,
      the baton does NOT advance (it is “blocked in zone”) until a valid handoff occurs.

    Tie-breaking at zones:
      If multiple batons are in their zones the same step, lanes are processed in ascending lane id (0..T-1).
    """
    M: int = 4
    lanes: int = 4
    teams: int = 4
    track_length: float = 1000.0
    max_steps: int = 5000
    speed_low: float = 4.0
    speed_high: float = 8.0
    speed_noise_std: float = 0.1
    exchange_zone_width: float = 0.10  # fraction of track length
    seed: Optional[int] = None

    def base(self) -> int:
        return self.teams  # choices are 1..T in the UI, but we encode 0..T-1 internally, see decode.

    def action_size(self) -> int:
        return (self.base()) ** self.lanes


class RLRelayRaceEnvContest(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, config: ContestEnvConfig = ContestEnvConfig()):
        super().__init__()
        self.cfg = config
        assert self.cfg.teams == self.cfg.lanes, "Number of lanes must equal number of teams."

        self.rng = np.random.default_rng(self.cfg.seed)
        self.T = self.cfg.teams
        self.L = self.cfg.lanes
        self.M = self.cfg.M
        self.Ltrack = self.cfg.track_length

        # Base speed: which lane’s runner is carrying, and which leg determines speed
        self.base_speed = self.rng.uniform(self.cfg.speed_low, self.cfg.speed_high, size=(self.T, self.M))

        # Observation/Action spaces
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.cfg.action_size())

        self.reset(seed=self.cfg.seed)

    # ---------- encode/decode joint action ----------
    def _decode_joint(self, a: int) -> np.ndarray:
        """
        Decode base-T integer to vector of length L with entries in {0..T-1}.
        We map UI choice {1..T} → internal {0..T-1}. (No 'keep'; must hand off when in zone.)
        """
        out = np.zeros(self.L, dtype=int)
        x = int(a)
        base = self.cfg.base()
        for k in range(self.L):
            out[k] = x % base
            x //= base
        return out  # each entry is a chosen lane r in 0..T-1

    def _encode_joint(self, vec: List[int]) -> int:
        base = self.cfg.base()
        acc, mul = 0, 1
        for v in vec:
            acc += int(v) * mul
            mul *= base
        return acc

    # ---------- exchange zone geometry ----------
    def _zone_bounds(self, j: int) -> Tuple[float, float]:
        center = (j + 1) / self.M * self.Ltrack
        half = (self.cfg.exchange_zone_width * self.Ltrack) / 2.0
        return center - half, center + half

    def _in_zone(self, pos: float, j: int) -> bool:
        if j >= self.M - 1:
            return False
        lo, hi = self._zone_bounds(j)
        return (pos >= lo) and (pos <= hi)

    # ---------- core API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        # Lane identity == baton identity; start: baton k in Lane k, Leg 0, carried by Lane k’s runner.
        self.baton_lane = np.arange(self.L, dtype=int)          # fixed identity per subplot
        self.runner_lane = np.arange(self.L, dtype=int)         # which lane’s runner is currently carrying each baton
        self.leg_index = np.zeros(self.L, dtype=int)            # current leg per baton
        self.baton_pos = np.zeros(self.L, dtype=float)

        # used[j, r] → whether (Lane r, Leg j) has been consumed
        self.used = np.zeros((self.M, self.T), dtype=bool)

        # finishing bookkeeping
        self.finish_times = np.full(self.L, np.inf, dtype=float)
        self.finished = np.zeros(self.L, dtype=bool)
        self.finish_order = None

        self.terminated = False
        self.truncated = False
        return self._get_obs(), self._info()

    def step(self, action: int):
        if self.terminated or self.truncated:
            raise RuntimeError("Call reset before stepping a finished episode.")

        self.step_count += 1
        chosen_lanes = self._decode_joint(action)  # each entry in 0..T-1

        prev_pos = self.baton_pos.copy()

        # 1) MUST-HANDOFF phase: if baton is in zone for its current leg, we must select a valid receiver.
        blocked = np.zeros(self.L, dtype=bool)
        for k in range(self.L):
            j = int(self.leg_index[k])
            if j >= self.M - 1:
                continue  # last leg: no handoff
            if not self._in_zone(self.baton_pos[k], j):
                continue

            r = int(chosen_lanes[k])  # chosen receiver runner’s lane (0..T-1)
            # must be unused for leg j
            if not self.used[j, r]:
                # consume the chosen runner for leg j
                self.used[j, r] = True
                # set new runner identity and advance to next leg
                self.runner_lane[k] = r
                self.leg_index[k] = j + 1
            else:
                # invalid choice: baton is blocked in zone until a valid handoff is made
                blocked[k] = True

        # 2) RUN phase: advance all batons that are not blocked
        for k in range(self.L):
            if blocked[k]:
                continue  # cannot move until valid handoff occurs
            j = int(self.leg_index[k])
            r_lane = int(self.runner_lane[k])
            mu = self.base_speed[r_lane, j]
            spd = max(0.0, mu + self.rng.normal(0.0, self.cfg.speed_noise_std))
            self.baton_pos[k] += spd

        # 3) record new finishers & termination
        for k in range(self.L):
            if (not self.finished[k]) and self.baton_pos[k] >= self.Ltrack:
                self.finished[k] = True
                # first time crossing; record finish "time" as step_count (or use physical time if you model dt)
                self.finish_times[k] = float(self.step_count)

        if np.all(self.finished):
            self.terminated = True
            # rank: smaller finish_times are better
            self.finish_order = np.argsort(self.finish_times).tolist()

        if self.step_count >= self.cfg.max_steps and not self.terminated:
            self.truncated = True

        # 4) Rewards (R2): only at termination
        if self.terminated:
            # rank bonus table
            rank_bonus = {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0}
            # compute lane ranks (1..T)
            order = np.argsort(self.finish_times)
            ranks = np.empty(self.L, dtype=int)
            for rank_idx, lane_id in enumerate(order):
                ranks[lane_id] = rank_idx + 1

            per_lane = -self.finish_times.copy()
            for k in range(self.L):
                per_lane[k] += rank_bonus.get(int(ranks[k]), 0.0)

            reward = float(per_lane.sum())
            info = self._info()
            info["finish_times"] = self.finish_times.copy()
            info["finish_order"] = self.finish_order
            info["final_lane_rewards"] = per_lane
            obs = self._get_obs()
            return obs, reward, True, self.truncated, info
        else:
            # per-step reward = 0 (sparse)
            obs = self._get_obs()
            info = self._info()
            return obs, 0.0, False, self.truncated, info

    # ---------- observations ----------
    def _obs_dim(self) -> int:
        # pos_norm[L] + time_norm[1] + onehot current runner lane [L*T] + leg_norm[L] + zone_flags[L] + finished_flags[L]
        return self.L + 1 + self.L*self.T + self.L + self.L + self.L

    def _get_obs(self) -> np.ndarray:
        pos_norm = (self.baton_pos / self.Ltrack).astype(np.float32)
        time_norm = np.array([self.step_count / max(1, self.cfg.max_steps)], dtype=np.float32)

        holder_oh = np.zeros((self.L, self.T), dtype=np.float32)
        for k in range(self.L):
            holder_oh[k, int(self.runner_lane[k])] = 1.0

        leg_norm = (self.leg_index / max(1, self.M - 1)).astype(np.float32)

        zone_flags = np.zeros(self.L, dtype=np.float32)
        for k in range(self.L):
            j = int(self.leg_index[k])
            zone_flags[k] = 1.0 if self._in_zone(self.baton_pos[k], j) else 0.0

        finished_flags = self.finished.astype(np.float32)

        return np.concatenate([
            pos_norm, time_norm, holder_oh.flatten(), leg_norm, zone_flags, finished_flags
        ]).astype(np.float32)

    # ---------- info ----------
    def _info(self) -> Dict:
        return dict(
            step=self.step_count,
            baton_pos=self.baton_pos.copy(),
            runner_lane=self.runner_lane.copy(),
            leg_index=self.leg_index.copy(),
            finished=self.finished.copy(),
            terminated=self.terminated,
            truncated=self.truncated,
        )

    def render(self) -> str:
        bars = []
        for k in range(self.L):
            bar = int(30 * (self.baton_pos[k] / self.Ltrack))
            bar = max(0, min(30, bar))
            bars.append(f"Lane{k}[" + "#" * bar + "-" * (30 - bar) + f"] runnerLane={self.runner_lane[k]} leg={self.leg_index[k]}")
        return f"t={self.step_count} | " + " | ".join(bars)

    def close(self):
        pass

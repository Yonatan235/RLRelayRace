from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

@dataclass
class ContestEnvConfig:
    """
    Contested handoff relay:
      - 4 lanes (teams), M runners per team (legs 0..M-1).
      - 4 batons (one per lane). Batons move along *their lane*; lane finish decides winner.
      - At exchange zone j, ONLY the 4 fresh runners (one per team at leg j) are available.
      - The baton arriving to zone j may pass to exactly ONE of the available teams (or keep running).
      - A selected runner (team t, leg j) retires after that leg; cannot be reused later.

    Action (joint Discrete for all lanes):
      - For each lane k: a_k ∈ {0..T}, where T=teams (default 4).
          0    = keep current holder (no handoff even if inside zone)
          1..T = select team (a_k-1) at current zone (if inside zone and team available)
      - We encode the vector (a_0..a_{L-1}) in base (T+1).

    Speed model:
      - Each team t has a base speed per leg j: base_speed[t, j] ~ U(low, high) (fixed).
      - Current baton speed = base_speed[holder_team[k], leg_index[k]] + N(0, sigma).

    Rewards:
      - info['per_lane_reward'] = normalized progress delta per lane.
      - scalar reward = sum over lanes (use the per-lane vector for credit assignment).

    Deterministic tie-break at zones:
      - If multiple batons enter a zone the same step, lower lane id resolves first.
    """
    M: int = 4
    lanes: int = 4
    teams: int = 4
    track_length: float = 1000.0
    max_steps: int = 5000
    speed_low: float = 4.0
    speed_high: float = 8.0
    speed_noise_std: float = 0.1
    pass_cooldown: int = 0           # optional cool-down between decisions (not essential here)
    exchange_zone_width: float = 0.10  # fraction of track length
    seed: Optional[int] = None

    def base(self) -> int:
        return 1 + self.teams         # 0 keep, 1..teams choose team

    def action_size(self) -> int:
        return self.base() ** self.lanes


class RLRelayRaceEnvContest(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, config: ContestEnvConfig = ContestEnvConfig()):
        super().__init__()
        self.cfg = config
        assert self.cfg.teams == self.cfg.lanes == 4, "This build assumes 4 teams == 4 lanes."

        self.rng = np.random.default_rng(self.cfg.seed)
        self.L = self.cfg.lanes
        self.T = self.cfg.teams
        self.M = self.cfg.M
        self.track_length = self.cfg.track_length

        # Base speeds per (team, leg)
        self.base_speed = self.rng.uniform(
            self.cfg.speed_low, self.cfg.speed_high, size=(self.T, self.M)
        )

        # State:
        # - holder_team[k] ∈ {0..T-1}
        # - leg_index[k]   ∈ {0..M-1}
        # - used[j, t] whether team t's runner at leg j has been used (retired) already
        # - baton_pos[k]
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.cfg.action_size())

        self.reset(seed=self.cfg.seed)

    # ---------- encoding helpers ----------
    def _decode_joint(self, a: int) -> np.ndarray:
        base = self.cfg.base()
        vec = np.zeros(self.L, dtype=int)
        x = int(a)
        for k in range(self.L):
            vec[k] = x % base
            x //= base
        return vec

    def _encode_joint(self, vec: List[int]) -> int:
        base = self.cfg.base()
        acc, mul = 0, 1
        for v in vec:
            acc += int(v) * mul
            mul *= base
        return acc

    # ---------- exchange zone geometry ----------
    def _zone_bounds(self, j: int) -> Tuple[float, float]:
        # center at (j+1)/M * L, width = w*L
        L = self.track_length
        center = (j + 1) / self.M * L
        half = (self.cfg.exchange_zone_width * L) / 2.0
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

        # Start: baton k in lane k, held by team k, leg 0
        self.holder_team = np.arange(self.T, dtype=int)
        self.leg_index = np.zeros(self.L, dtype=int)
        self.baton_pos = np.zeros(self.L, dtype=float)

        # used[j, t] indicates team t's runner for leg j is already consumed
        self.used = np.zeros((self.M, self.T), dtype=bool)

        self.terminated = False
        self.truncated = False
        return self._get_obs(), self._info()

    def step(self, action: int):
        if self.terminated or self.truncated:
            raise RuntimeError("Call reset before stepping a finished episode.")

        self.step_count += 1
        joint = self._decode_joint(action)

        prev_pos = self.baton_pos.copy()

        # 1) Resolve passes for batons that are inside their current leg's zone.
        # If multiple are in zone this step, resolve in lane-id order (0..L-1).
        for k in range(self.L):
            j = int(self.leg_index[k])
            if j >= self.M - 1:
                continue  # last leg: no passing
            if not self._in_zone(self.baton_pos[k], j):
                continue

            a_k = joint[k]  # 0 keep, 1..T choose team
            if a_k == 0:
                pass  # keep current holder
            else:
                chosen_team = a_k - 1
                # candidate must be unused at this leg
                if not self.used[j, chosen_team]:
                    # consume the chosen runner at leg j
                    self.used[j, chosen_team] = True
                    # set new holder team immediately for next segment
                    self.holder_team[k] = chosen_team
                    # advance baton to next leg index (handoff completed within zone)
                    self.leg_index[k] = j + 1
                # else: chosen team unavailable; do nothing (keep current holder)

        # 2) Run phase: all batons advance by holder's leg speed + noise
        for k in range(self.L):
            j = int(self.leg_index[k])
            t = int(self.holder_team[k])
            mu = self.base_speed[t, j]
            spd = max(0.0, mu + self.rng.normal(0.0, self.cfg.speed_noise_std))
            self.baton_pos[k] += spd

        # 3) Termination
        winners = np.where(self.baton_pos >= self.track_length)[0]
        if len(winners) > 0:
            self.terminated = True
        if self.step_count >= self.cfg.max_steps:
            self.truncated = True

        # 4) Rewards
        delta = (self.baton_pos - prev_pos) / self.track_length
        reward = float(delta.sum())
        obs = self._get_obs()
        info = self._info()
        info["per_lane_reward"] = delta
        if self.terminated:
            info["winner_lanes"] = winners.tolist()

        return obs, reward, self.terminated, self.truncated, info

    # ---------- observations ----------
    def _obs_dim(self) -> int:
        # baton_pos_norm[L] + time[1] + holder_team_onehot[L*T] + leg_index_norm[L] + zone_flags[L]
        return self.L + 1 + self.L*self.T + self.L + self.L

    def _get_obs(self) -> np.ndarray:
        pos_norm = (self.baton_pos / self.track_length).astype(np.float32)
        time_norm = np.array([self.step_count / max(1, self.cfg.max_steps)], dtype=np.float32)

        holder_oh = np.zeros((self.L, self.T), dtype=np.float32)
        for k in range(self.L):
            holder_oh[k, int(self.holder_team[k])] = 1.0

        leg_norm = (self.leg_index / max(1, self.M - 1)).astype(np.float32)

        zone_flags = np.zeros(self.L, dtype=np.float32)
        for k in range(self.L):
            j = int(self.leg_index[k])
            zone_flags[k] = 1.0 if self._in_zone(self.baton_pos[k], j) else 0.0

        return np.concatenate([
            pos_norm, time_norm, holder_oh.flatten(), leg_norm, zone_flags
        ]).astype(np.float32)

    # ---------- info ----------
    def _info(self) -> Dict:
        return dict(
            step=self.step_count,
            baton_pos=self.baton_pos.copy(),
            holder_team=self.holder_team.copy(),
            leg_index=self.leg_index.copy(),
            terminated=self.terminated,
            truncated=self.truncated,
        )

    def render(self) -> str:
        bars = []
        for k in range(self.L):
            bar = int(30 * (self.baton_pos[k] / self.track_length))
            bar = max(0, min(30, bar))
            bars.append(f"L{k}[" + "#" * bar + "-" * (30 - bar) + f"] team={self.holder_team[k]} leg={self.leg_index[k]}")
        return f"t={self.step_count} | " + " | ".join(bars)

    def close(self):
        pass

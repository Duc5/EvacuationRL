import gymnasium as gym
from gymnasium import spaces
import numpy as np

from simulation.jupedsim_backend import JuPedSimBackend


class EvacuationEnv(gym.Env):
    """
    Gymnasium interface for the evacuation simulation.

    Gymnasium knows about:
        - observations
        - actions
        - rewards
        - episode termination

    It does NOT know how JuPedSim works internally.
    """

    def __init__(
        self,
        scenario,
        control_interval=0.5,
        max_time=30.0,
        record=False,
        trajectory_path="integratedgym_evac.sqlite",
    ):
        super().__init__()

        self.scenario = scenario

        self.control_interval = control_interval
        self.max_time = max_time

        # The physical simulator sits underneath Gymnasium.
        self.backend = JuPedSimBackend(
            scenario=scenario,
            record=record,
            trajectory_path=trajectory_path,
        )

        # --------------------------------------------------------------
        # RL action space
        # --------------------------------------------------------------

        # 0 = LL
        # 1 = LR
        # 2 = RL
        # 3 = RR
        self.action_space = spaces.Discrete(4)

        # --------------------------------------------------------------
        # RL observation space
        # --------------------------------------------------------------

        # Same six normalized crowd measurements used in v1.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        state = self.backend.reset(seed=seed)

        observation = self._make_observation(state)
        info = self._make_info(state)

        return observation, info

    def step(self, action):

        # Convert PPO/Gymnasium action into a simulator-level
        # guidance command.
        guidance = self._decode_action(action)

        time_before = self.backend.elapsed_time

        self.backend.apply_guidance(guidance)

        # Do not simulate beyond max_time.
        remaining_time = self.max_time - time_before

        seconds_to_advance = min(
            self.control_interval,
            remaining_time,
        )

        state = self.backend.advance(
            seconds_to_advance
        )

        time_after = state.elapsed_time

        # Exact amount of simulated time consumed.
        reward = -(time_after - time_before)

        terminated = state.is_evacuated

        truncated = (
            not terminated
            and time_after >= self.max_time
        )

        observation = self._make_observation(state)
        info = self._make_info(state)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # ------------------------------------------------------------------
    # RL action encoding
    # ------------------------------------------------------------------

    def _decode_action(self, action):

        action_map = {
            0: {
                "left": "left",
                "right": "left",
            },

            1: {
                "left": "left",
                "right": "right",
            },

            2: {
                "left": "right",
                "right": "left",
            },

            3: {
                "left": "right",
                "right": "right",
            },
        }

        return action_map[int(action)]

    # ------------------------------------------------------------------
    # RL observation encoding
    # ------------------------------------------------------------------

    def _make_observation(self, state):

        left_half = 0
        right_half = 0

        near_left_exit = 0
        near_right_exit = 0

        assigned_left = 0
        assigned_right = 0

        for agent in state.agents:

            x, y = agent.position

            if x < self.scenario.guidance_split_x:
                left_half += 1
            else:
                right_half += 1

            if x < 2:
                near_left_exit += 1

            if x > 8:
                near_right_exit += 1

            if agent.assignment == "left":
                assigned_left += 1

            elif agent.assignment == "right":
                assigned_right += 1

        n = state.initial_population

        return np.array([
            left_half / n,
            right_half / n,
            near_left_exit / n,
            near_right_exit / n,
            assigned_left / n,
            assigned_right / n,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Additional episode information
    # ------------------------------------------------------------------

    def _make_info(self, state):

        return {
            "elapsed_time": state.elapsed_time,
            "remaining_agents": state.remaining_agents,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):

        self.backend.close()
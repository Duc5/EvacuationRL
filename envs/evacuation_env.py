import gymnasium as gym
from gymnasium import spaces
import numpy as np

from shapely.geometry import Point

from simulation.jupedsim_backend import JuPedSimBackend
from envs.static_plan_env import generate_acyclic_policies

class EvacuationEnv(gym.Env):
    """
    Gymnasium interface for the Building V2 evacuation environment.

    The RL agent controls the guidance signs at J1, J2 and J3.

    Gymnasium owns:
        - observation encoding
        - action encoding
        - reward
        - termination / truncation

    JuPedSimBackend owns the physical simulation.
    """

    def __init__(
        self,
        scenario,
        control_interval=1.0,
        max_time=180.0,
        record=False,
        trajectory_path="v2_evac.sqlite",
        population_scenarios=None,
        exit_proximity=3.0,
    ):
        super().__init__()

        if scenario.routing_mode != "network":
            raise ValueError(
                "EvacuationEnv now supports network-routing scenarios only."
            )

        self.scenario = scenario
        self.control_interval = control_interval
        self.max_time = max_time

        self.population_scenarios = population_scenarios
        self.current_population_name = None
        self.congestion_region_names = list(
            self.scenario.congestion_regions.keys()
        )
        self.exit_proximity = exit_proximity

        self.backend = JuPedSimBackend(
            scenario=scenario,
            record=record,
            trajectory_path=trajectory_path,
        )

        # Keep a fixed ordering for observations/actions.
        self.room_names = tuple(self.scenario.rooms.keys())
        self.exit_names = tuple(self.scenario.exits.keys())
        self.junction_names = tuple(self.scenario.guidance_choices.keys())

        # --------------------------------------------------------------
        # Action space
        # --------------------------------------------------------------

        # One discrete choice for each junction.
        #
        # Building V2:
        # J1 -> 4 choices
        # J2 -> 3 choices
        # J3 -> 4 choices
        #
        # Therefore MultiDiscrete([4, 3, 4]).
        self.guidance_plans = generate_acyclic_policies(self.scenario)          

        self.action_space = spaces.Discrete(len(self.guidance_plans))
        # --------------------------------------------------------------
        # Observation space
        # --------------------------------------------------------------

        # 4 room populations
        # 3 agents targeting exits
        # 3 agents targeting junctions
        # 3 exit congestion measurements
        # 1 elapsed-time fraction
        #
        # Total = 14 observations.
        self.base_observation_size = (
            len(self.room_names)
            + len(self.exit_names)
            + len(self.junction_names)
            + len(self.exit_names)
            + 1
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                self.base_observation_size
                + len(self.congestion_region_names),
            ),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.population_scenarios:
            population_names = list(self.population_scenarios.keys())
            index = int(self.np_random.integers(len(population_names)))

            self.current_population_name = population_names[index]
            self.scenario.room_counts = dict(
                self.population_scenarios[self.current_population_name]
            )

            # Random but reproducible pedestrian placement.
            backend_seed = int(self.np_random.integers(0, 2**31 - 1))

        else:
            self.current_population_name = None
            backend_seed = seed

        state = self.backend.reset(seed=backend_seed)

        observation = self._make_observation(state)
        info = self._make_info(state)

        return observation, info

    def step(self, action):
        guidance = self._decode_action(action)

        time_before = self.backend.elapsed_time

        self.backend.apply_guidance(guidance)

        remaining_time = self.max_time - time_before
        seconds_to_advance = min(self.control_interval, remaining_time)

        state = self.backend.advance(seconds_to_advance)

        time_after = state.elapsed_time
        delta_time = time_after - time_before

        reward = self._calculate_reward(state, delta_time)

        terminated = state.is_evacuated
        truncated = not terminated and time_after >= self.max_time

        observation = self._make_observation(state)
        info = self._make_info(state)
        info["guidance"] = guidance

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action encoding
    # ------------------------------------------------------------------
    def _decode_action(self, action):
        action = int(action)

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected 0-{len(self.guidance_plans) - 1}."
            )

        return dict(self.guidance_plans[action])

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------
    def _make_observation(self, state):
        room_counts = {room: 0 for room in self.room_names}
        exit_assignments = {exit_name: 0 for exit_name in self.exit_names}
        junction_assignments = {junction: 0 for junction in self.junction_names}
        near_exits = {exit_name: 0 for exit_name in self.exit_names}
        congestion_counts = {region: 0 for region in self.congestion_region_names}

        for agent in state.agents:
            point = Point(agent.position)

            # Physical room occupancy.
            for room_name in self.room_names:
                if self.scenario.rooms[room_name].covers(point):
                    room_counts[room_name] += 1
                    break

            # Current navigation target.
            if agent.assignment in exit_assignments:
                exit_assignments[agent.assignment] += 1
            elif agent.assignment in junction_assignments:
                junction_assignments[agent.assignment] += 1

            # Local congestion around exits.
            for exit_name in self.exit_names:
                exit_area = self.scenario.exits[exit_name]
                if exit_area.distance(point) <= self.exit_proximity:
                    near_exits[exit_name] += 1

            # Physical occupancy of important routing regions.
            for region_name in self.congestion_region_names:
                region = self.scenario.congestion_regions[region_name]
                if region.covers(point):
                    congestion_counts[region_name] += 1

        n = state.initial_population

        observation = []
        observation.extend(room_counts[name] / n for name in self.room_names)
        observation.extend(exit_assignments[name] / n for name in self.exit_names)
        observation.extend(junction_assignments[name] / n for name in self.junction_names)
        observation.extend(near_exits[name] / n for name in self.exit_names)

        observation.append(min(state.elapsed_time / self.max_time, 1.0))

        observation.extend(
            congestion_counts[name] / n for name in self.congestion_region_names
        )

        return np.array(observation, dtype=np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _calculate_reward(self, state, delta_time):
        remaining_fraction = (state.remaining_agents/ state.initial_population)

        return -delta_time * (1.0 + remaining_fraction) /self.max_time

    # ------------------------------------------------------------------
    # Episode information
    # ------------------------------------------------------------------

    def _make_info(self, state):
        return {
            "elapsed_time": state.elapsed_time,
            "remaining_agents": state.remaining_agents,
            "population_name": self.current_population_name,
            
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        self.backend.close()
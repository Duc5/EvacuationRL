import itertools
import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def policy_key(policy, junction_names):
    return "|".join(
        f"{junction}={policy[junction]}"
        for junction in junction_names
    )


def is_cyclic(policy, junction_names):
    junction_set = set(junction_names)

    for start in junction_names:
        current = start
        visited = set()

        while current in junction_set:
            if current in visited:
                return True

            visited.add(current)
            current = policy[current]

    return False


def generate_acyclic_policies(scenario):
    junction_names = list(scenario.junctions.keys())

    choice_lists = [
        scenario.guidance_choices[junction]
        for junction in junction_names
    ]

    policies = []

    for targets in itertools.product(*choice_lists):
        policy = dict(zip(junction_names, targets))

        if not is_cyclic(policy, junction_names):
            policies.append(policy)

    return policies


class StaticPlanEnv(gym.Env):
    """
    One-step contextual-bandit version of Building V2.

    Observation:
        Initial room population fractions.

    Action:
        One complete acyclic junction-sign configuration.

    Reward:
        Cached result from a JuPedSim static-policy simulation.

    One action = one complete episode.
    """

    def __init__(
        self,
        scenario,
        population_scenarios,
        results_path,
        max_time=180.0,
    ):
        super().__init__()

        self.scenario = scenario
        self.population_scenarios = population_scenarios
        self.max_time = max_time

        self.room_names = list(scenario.rooms.keys())
        self.junction_names = list(scenario.junctions.keys())

        self.policies = generate_acyclic_policies(scenario)

        self.action_space = spaces.Discrete(
            len(self.policies)
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.room_names),),
            dtype=np.float32,
        )

        with open(results_path) as file:
            self.results = json.load(file)

        self.current_population_name = None
        self._done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "population_name" in options:
            population_name = options["population_name"]

            if population_name not in self.population_scenarios:
                raise ValueError(
                    f"Unknown population: {population_name}"
                )

        else:
            names = list(
                self.population_scenarios.keys()
            )

            index = int(
                self.np_random.integers(len(names))
            )

            population_name = names[index]

        self.current_population_name = population_name
        self._done = False

        observation = self._make_observation()

        info = {
            "population_name": population_name,
        }

        return observation, info

    def step(self, action):
        if self._done:
            raise RuntimeError(
                "Episode is finished. Call reset()."
            )

        action = int(action)

        policy = self.policies[action]

        key = policy_key(
            policy,
            self.junction_names,
        )

        result = self.results[
            self.current_population_name
        ][key]

        elapsed_time = result["elapsed_time"]
        remaining = result["remaining_agents"]

        total_population = sum(
            self.population_scenarios[
                self.current_population_name
            ].values()
        )

        # Successful policies preserve evacuation-time ranking.
        # Failures receive an additional remaining-population penalty.
        if remaining == 0:
            reward = -elapsed_time / self.max_time
        else:
            reward = (
                -1.0
                - remaining / total_population
            )

        self._done = True

        observation = self._make_observation()

        info = {
            "population_name": self.current_population_name,
            "policy": policy,
            "policy_key": key,
            "elapsed_time": elapsed_time,
            "remaining_agents": remaining,
            "evacuated": remaining == 0,
        }

        # This environment is deliberately a one-step episode.
        return observation, reward, True, False, info

    def _make_observation(self):
        counts = self.population_scenarios[
            self.current_population_name
        ]

        total = sum(counts.values())

        return np.array(
            [
                counts[room] / total
                for room in self.room_names
            ],
            dtype=np.float32,
        )
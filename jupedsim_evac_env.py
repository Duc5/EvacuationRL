import gymnasium as gym
from gymnasium import spaces
import numpy as np

import jupedsim as jps
from shapely import Polygon, Point
import pathlib

class JuPedSimEvacEnv(gym.Env):

    def __init__(
        self,
        control_interval=0.5,
        max_time=30.0,
        record=False
    ):
        super().__init__()

        self.control_interval = control_interval
        self.max_time = max_time
        self.record = record
        # ---------- Static geometry ----------

        self.room = Polygon([
            (0, 0),
            (10, 0),
            (10, 8),
            (0, 8)
        ])

        self.exit_left = Polygon([
            (0, 3),
            (0.5, 3),
            (0.5, 5),
            (0, 5)
        ])

        self.exit_right = Polygon([
            (9.5, 3),
            (10, 3),
            (10, 5),
            (9.5, 5)
        ])

        self.guidance_area = Polygon([
            (2.5, 2.5),
            (7.5, 2.5),
            (7.5, 7.5),
            (2.5, 7.5)
        ])

        self.start_positions = [
            (x, y)
            for x in range(3, 8)
            for y in range(3, 8)
        ]

        self.initial_population = len(self.start_positions)

        # 0 = LL
        # 1 = LR
        # 2 = RL
        # 3 = RR
        self.action_space = spaces.Discrete(4)

        # Six normalized crowd measurements
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        # Created fresh in reset()
        self.simulation = None
        self.routes = None

        # Per-episode bookkeeping
        self.agent_groups = {}
        self.assignments = {}
        self.committed_agents = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.record:
            writer = jps.SqliteTrajectoryWriter(
                output_file=pathlib.Path("integratedgym_evac.sqlite"),
                every_nth_frame=5)
        else:
            writer = None
        self.simulation = jps.Simulation(
            model=jps.CollisionFreeSpeedModel(),
            geometry=self.room,
            trajectory_writer= writer
        )

        left_exit_id = self.simulation.add_exit_stage(
            self.exit_left
        )

        right_exit_id = self.simulation.add_exit_stage(
            self.exit_right
        )

        left_journey_id = self.simulation.add_journey(
            jps.JourneyDescription([left_exit_id])
        )

        right_journey_id = self.simulation.add_journey(
            jps.JourneyDescription([right_exit_id])
        )

        self.routes = {
            "left": (
                left_journey_id,
                left_exit_id
            ),
            "right": (
                right_journey_id,
                right_exit_id
            )
        }
        # which group(Left side/Right side)
        self.agent_groups = {}
        # which exit is agent actually assigned
        self.assignments = {}
        self.committed_agents = set()

        for position in self.start_positions:

            # Which sign controls this pedestrian?
            if position[0] < 5:
                group = "left"

            elif position[0] > 5:
                group = "right"

            else:
                group = (
                    "left"
                    if self.np_random.random() < 0.5
                    else "right"
                )

            # Placeholder journey.
            # No simulation movement happens before step(),
            # so the first RL action will overwrite it.
            journey_id, stage_id = self.routes[group]

            parameters = (
                jps.CollisionFreeSpeedModelAgentParameters(
                    position=position,
                    journey_id=journey_id,
                    stage_id=stage_id
                )
            )

            agent_id = self.simulation.add_agent(parameters)

            self.agent_groups[agent_id] = group
            self.assignments[agent_id] = None

        observation = self._get_obs()

        info = self._get_info()

        return observation, info

    def step(self, action):

        self._apply_action(action)

        time_before = self.simulation.elapsed_time()

        iterations = round(
            self.control_interval /
            self.simulation.delta_time()
        )

        for _ in range(iterations):

            if self.simulation.agent_count() == 0:
                break

            if self.simulation.elapsed_time() >= self.max_time:
                break

            self.simulation.iterate()

        self._update_committed_agents()

        terminated = (
            self.simulation.agent_count() == 0
        )

        truncated = (
            not terminated
            and self.simulation.elapsed_time() >= self.max_time
        )

        time_after = self.simulation.elapsed_time()

        # Exact amount of simulated time consumed
        reward = -(time_after - time_before)

        observation = self._get_obs()
        info = self._get_info()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
    def _apply_action(self, action):

        action_map = {
            0: ("left", "left"),
            1: ("left", "right"),
            2: ("right", "left"),
            3: ("right", "right"),
        }

        left_sign, right_sign = action_map[int(action)]

        for agent in self.simulation.agents():

            if agent.id in self.committed_agents:
                continue

            position = Point(agent.position)
            
            # Only agents still within the sign-controlled
            # region respond to guidance.
            if not self.guidance_area.covers(position):
                continue

            x, y = agent.position

            if x <= 5:
                group = "left"
            elif x > 5:
                group = "right"

            if group == "left":
                target = left_sign
            else:
                target = right_sign

            # Don't issue an identical reroute unnecessarily.
            if self.assignments[agent.id] == target:
                continue

            journey_id, stage_id = self.routes[target]

            self.simulation.switch_agent_journey(
                agent.id,
                journey_id,
                stage_id
            )

            self.assignments[agent.id] = target
    def _update_committed_agents(self):

        for agent in self.simulation.agents():

            position = Point(agent.position)

            if not self.guidance_area.covers(position):
                self.committed_agents.add(agent.id)
    def _get_obs(self):

        agents = list(self.simulation.agents())

        left_half = 0
        right_half = 0

        near_left_exit = 0
        near_right_exit = 0

        assigned_left = 0
        assigned_right = 0

        for agent in agents:

            x, y = agent.position

            if x < 5:
                left_half += 1
            else:
                right_half += 1

            if x < 2:
                near_left_exit += 1

            if x > 8:
                near_right_exit += 1

            assignment = self.assignments.get(agent.id)

            if assignment == "left":
                assigned_left += 1

            elif assignment == "right":
                assigned_right += 1

        n = self.initial_population

        return np.array([
            left_half / n,
            right_half / n,
            near_left_exit / n,
            near_right_exit / n,
            assigned_left / n,
            assigned_right / n,
        ], dtype=np.float32)
    def _get_info(self):

        return {
            "elapsed_time": self.simulation.elapsed_time(),
            "remaining_agents": self.simulation.agent_count(),
        }

    def close(self):
        if self.simulation is not None and self.simulation._writer is not None:
            self.simulation._writer.close()
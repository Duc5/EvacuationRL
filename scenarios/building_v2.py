import numpy as np

from shapely import Point
from shapely.geometry import box
from shapely.ops import unary_union

class BuildingV2Scenario:
    """
    Multi-room, three-exit, three-junction evacuation scenario.

    Geometry only for now.

    Exit naming follows the current design:

        A = west exit  (narrow)
        B = south exit (medium)
        C = east exit  (wide)
    """

    def __init__(self,room_counts=None,spawn_spacing=0.65):
        self.routing_mode = "network"
        self.room_counts = room_counts
        self.spawn_spacing = spawn_spacing
        # ==============================================================
        # ROOMS
        # ==============================================================

        self.rooms = {
            "A": box(
                9.0, 30.0,
                15.0, 36.0,
            ),

            "B": box(
                2.0, 10.0,
                8.0, 15.0,
            ),

            "C": box(
                10.0, 10.0,
                16.0, 15.0,
            ),

            "D": box(
                3.0, 3.0,
                8.0, 8.0,
            ),
        }

        # ==============================================================
        # MAIN CORRIDOR NETWORK
        # ==============================================================

        # Upper horizontal corridor:
        #
        # Exit A ---------------- J1 ---------------- Exit C
        #
        self.upper_corridor = box(
            0.0, 23.0,
            24.0, 27.0,
        )

        # Lower horizontal part of the loop:
        #
        #          J3 ----------- J2
        #
        self.lower_corridor = box(
            4.0, 17.0,
            20.0, 19.0,
        )

        # Left side of the loop
        self.left_loop_connector = box(
            4.0, 19.0,
            6.0, 23.0,
        )

        # Right side of the loop
        self.right_loop_connector = box(
            18.0, 19.0,
            20.0, 23.0,
        )

        # ==============================================================
        # ROOM → NETWORK CONNECTIONS
        # ==============================================================

        # Room A → J1
        self.room_A_connector = box(
            11.0, 27.0,
            13.0, 30.0,
        )

        # Room B → J3
        self.room_B_connector = box(
            4.0, 15.0,
            6.0, 17.0,
        )

        # Room C → J2
        self.room_C_connector = box(
            12.0, 15.0,
            14.0, 17.0,
        )

        # Room D → Room B
        self.room_D_connector = box(
            4.5, 8.0,
            6.5, 10.0,
        )

        # ==============================================================
        # SOUTH EXIT CONNECTION
        # ==============================================================

        # Room D → Exit B
        self.south_exit_corridor = box(
            4.6, 1.0,
            6.4, 3.0,
        )

        # ==============================================================
        # EXIT STAGES
        # ==============================================================

        # Exit A — narrow: 1.2 m
        self.exit_A_neck = box(
           -2.0, 24.4,
            0.0, 25.6,
        )
        self.exit_A = box(
            -2.0, 24.4,
            -1.5, 25.6,
        )

        # Exit B — medium: 1.8 m
        self.exit_B = box(
            4.6, 1.0,
            6.4, 1.5,
        )

        # Exit C — wide: 2.4 m
        self.exit_C_neck = box(
            24.0, 23.8,
            26.0, 26.2,
        )
        self.exit_C = box(
            25.5, 23.8,
            26.0, 26.2,
        )

        self.exits = {
            "A": self.exit_A,
            "B": self.exit_B,
            "C": self.exit_C,
        }

        # ==============================================================
        # JUNCTION DATA
        # ==============================================================

        # 1. JUNCTION AREA
        # They identify the physical regions around each decision point and will later be
        # used by the guidance controller.
        self.junctions = {
            # Room A meets upper corridor
            "J1": box(
                11.0, 23.0,
                13.0, 27.0,
            ),

            # Room C meets lower corridor
            "J2": box(
                12.0, 17.0,
                14.0, 19.0,
            ),

            # Room B meets lower corridor
            "J3": box(
                4.0, 17.0,
                6.0, 19.0,
            ),
        }

        # JUNCTION WAYPOINTS
        # Central target point inside each controlled junction.
        # These will become JuPedSim WaypointStages to be added to journey of agents
        self.junction_waypoints = {
            "J1": (12.0, 25.0),
            "J2": (13.0, 18.0),
            "J3": (5.0, 18.0),
        }
        self.junction_waypoint_radius = 0.35

        # ==============================================================
        # GUIDANCE GRAPH
        # ==============================================================        

        # Each controlled junction can direct a pedestrian toward another
        # controlled junction or directly toward an exit.
        #
        # Exit names correspond to keys in self.exits.      

        self.guidance_choices = {       

            "J1": (
                "A",     # Exit A
                "C",     # Exit C
                "J2",
                "J3",
            ),      

            "J2": (
                "C",     # Exit C
                "J1",
                "J3",
            ),      

            "J3": (
                "B",     # Exit B
                "J1",
                "J2",
                "A"
            ),
        }
        
        # ==============================================================
        # SPAWN REGIONS
        # ==============================================================

        # Slightly inset from room walls.
        # We will later sample pedestrian positions from these regions.

        self.spawn_regions = {
            "A": box(
                9.75, 30.75,
                14.25, 35.25,
            ),

            "B": box(
                2.75, 10.75,
                7.25, 14.25,
            ),

            "C": box(
                10.75, 10.75,
                15.25, 14.25,
            ),

            "D": box(
                3.75, 3.75,
                7.25, 7.25,
            ),
        }
        # Which room initially heads where
        self.initial_targets = {
           "A": "J1",
           "B": "J3",
           "C": "J2",
           "D": "B",     # Directly to Exit B
        }       

        # test position, one agent per room
        self.test_positions = {
            "A": (12.0, 33.0),
            "B": (5.0, 12.5),
            "C": (13.0, 12.5),
            "D": (5.5, 5.5),
        }


        # ==============================================================
        # COMPLETE WALKABLE GEOMETRY
        # ==============================================================

        walkable_parts = [
            # Rooms
            *self.rooms.values(),

            # Main network
            self.upper_corridor,
            self.lower_corridor,
            self.left_loop_connector,
            self.right_loop_connector,

            # Room connections
            self.room_A_connector,
            self.room_B_connector,
            self.room_C_connector,
            self.room_D_connector,
            self.exit_A_neck,
            self.exit_C_neck,

            # South exit corridor
            self.south_exit_corridor,
        ]

        self.geometry = unary_union(walkable_parts)

        if not self.geometry.is_valid:
            raise ValueError(
                "BuildingV2 geometry is invalid."
            )

    def generate_start_positions(self, seed=None):
        """
        Generate reproducible pedestrian starting positions for each room.

        Positions are selected without replacement from a grid of valid
        candidate points inside each room's spawn region.
        """

        if self.room_counts is None:
            raise ValueError(
                "room_counts must be provided to generate a crowd."
            )

        self._validate_room_counts()

        rng = np.random.default_rng(seed)

        positions_by_room = {}

        for room_name, count in self.room_counts.items():

            region = self.spawn_regions[room_name]

            candidates = self._generate_spawn_candidates(region)

            if count > len(candidates):
                raise ValueError(
                    f"Room {room_name} requests {count} pedestrians, "
                    f"but only {len(candidates)} valid spawn positions "
                    f"are available with spacing "
                    f"{self.spawn_spacing} m."
                )

            selected_indices = rng.choice(len(candidates),size=count,replace=False)

            positions_by_room[room_name] = [candidates[index] for index in selected_indices]

        return positions_by_room


    def _generate_spawn_candidates(self, region):
        """
        Generate a regular grid of candidate positions inside a spawn
        region.

        Candidate spacing guarantees pedestrians are not initially placed
        on top of one another.
        """

        min_x, min_y, max_x, max_y = region.bounds

        x_values = np.arange(min_x,max_x + 1e-9,self.spawn_spacing)

        y_values = np.arange(min_y,max_y + 1e-9,self.spawn_spacing)

        candidates = []

        for x in x_values:
            for y in y_values:

                point = Point(float(x),float(y))
                if region.covers(point):
                    candidates.append((float(x),float(y)))
        return candidates


    def _validate_room_counts(self):
        unknown_rooms = (set(self.room_counts)- set(self.rooms)
        )

        if unknown_rooms:
            raise ValueError(
                f"Unknown rooms in room_counts: "
                f"{unknown_rooms}"
            )

        for room_name, count in self.room_counts.items():

            if not isinstance(count, int):
                raise TypeError(
                    f"Population for Room {room_name} "
                    f"must be an integer."
                )

            if count < 0:
                raise ValueError(
                    f"Population for Room {room_name} "
                    f"cannot be negative."
                )
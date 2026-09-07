from shapely import Polygon


class TwoExitV1Scenario:
    """
    Physical definition of the original v1 two-exit environment.

    This class contains no Gymnasium logic and no JuPedSim simulation logic.
    It only describes what exists in the scenario.
    """

    def __init__(self):

        # Walkable geometry (the room) 
        self.geometry = Polygon([
            (0, 0),
            (10, 0),
            (10, 8),
            (0, 8),
        ])

        # Thw two Exits 
        self.exits = {
            "left": Polygon([
                (0, 3),
                (0.5, 3),
                (0.5, 5),
                (0, 5),
            ]),

            "right": Polygon([
                (9.5, 3),
                (10, 3),
                (10, 5),
                (9.5, 5),
            ]),
        }

        # Guidance-controlled area, the area that RL signs will dictate directions  
        self.guidance_area = Polygon([
            (2.5, 2.5),
            (7.5, 2.5),
            (7.5, 7.5),
            (2.5, 7.5),
        ])

        # Boundary separating the two guidance regions
        self.guidance_split_x = 5.0

        # Initial pedestrian positions 
        self.start_positions = [
            (x, y)
            for x in range(3, 8)
            for y in range(3, 8)
        ]

    @property
    def initial_population(self):
        return len(self.start_positions)
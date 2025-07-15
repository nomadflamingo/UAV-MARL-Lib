"""Handler for flag stations in the environment"""

import math
import os

import numpy as np
from pybullet_utils import bullet_client

from PyFlyt.CL2_envs.utils.utils import generate_circle_points

class FlagHandler:
    """Manages flag stations"""

    def __init__(self,
                 enable_render: bool,
                 num_flags: int,
                 flag_reach_distance: float,
                 flag_spread_raduis: float,
                 flight_dome_size: float,
                 height: float,
                #  np_random: np.random.Generator,
                 ):
        """__init__
        
        Args:
            enable_render (bool): enable_render
            num_targets (int): num_targets
            goal_reach_distance (float): goal_reach_distance
            goal_spread_radius (float): goal_spread_radius
            flight_dome_size (float): flight_dome_size
            height (float): height
            np_random (np.random.Generator): np_random
        
        """
        # Constants
        self.enable_render = enable_render
        self.num_flags = num_flags
        self.flag_reach_distance = flag_reach_distance
        self.flag_spread_radius = flag_spread_raduis
        self.flight_dome_size = flight_dome_size
        self.height = height
        # self.np_random = np_random

        # the flag visual
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, "../../models/target.urdf")

        self.team_colors = {
            "red": (1.0, 0.0, 0.0, 1.0),   # RGBA
            "blue": (0.0, 0.0, 1.0, 1.0),
        }
        self.ownership = [None] * self.num_flags  # track team ownership

    def reset(
            self,
            p: bullet_client.BulletClient,
            np_random: None | np.random.Generator = None,
    ):
        """Resets the flags"""
        # Store pybullet client
        self.p = p

        # reset error
        self.new_distance = np.inf
        self.old_distance = np.inf

        # generate flag locations
        xy_list = generate_circle_points(radius=self.flag_spread_radius, n=self.num_flags)
        start_pos0 = [[float(x), float(y), self.height] for (x, y) in xy_list]
        self.targets = np.array(start_pos0, dtype=np.float32)

        # if we are rendering, load in the targets
        if self.enable_render:
            self.target_visual = []
            for target in self.targets:
                self.target_visual.append(
                    self.p.loadURDF(
                        self.targ_obj_dir,
                        basePosition=target,
                        useFixedBase=True,
                        globalScaling=self.flag_reach_distance,
                    )
                )

            for i, visual in enumerate(self.target_visual):
                self.p.changeVisualShape(
                    visual,
                    linkIndex=-1,
                    rgbaColor=(0, 1 - (i / len(self.target_visual)), 0, 0.25),
                )


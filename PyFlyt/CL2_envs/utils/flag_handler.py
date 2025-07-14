"""Handler for flag stations in the environment"""

import math
import os

import numpy as np
from pybullet_utils import bullet_client

class FlagHandler:
    """Manages flag stations"""

    def __init__(self,
                 enable_render: bool,
                 num_flags: int,
                 goal_reach_distance: float,
                 goal_reach_angle: float,
                 flight_dome_size: float,
                 min_height: float,
                 np_random: np.random.Generator,
                 ):
        """__init__
        
        Args:
            enable_render (bool): enable_render
            num_targets (int): num_targets
            goal_reach_distance (float): goal_reach_distance
            goal_reach_angle (float): goal_reach_angle
            flight_dome_size (float): flight_dome_size
            min_height (float): min_height
            np_random (np.random.Generator): np_random
        
        """
        # Constants
        self.enable_render = enable_render
        self.num_flags = num_flags
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.flight_dome_size = flight_dome_size
        self.min_height = min_height
        self.np_random = np_random

        # the flag visual
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, "../../models/target.urdf")

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

        # we sample form polar coordinates to generate linear targets
        self.targets = np.zeros(shape=(self.num_flags, 3)) 
        thetas = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_flags))


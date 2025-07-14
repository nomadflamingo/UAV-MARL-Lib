
from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
# Import Flag handler

class QuadXCaptureFlagEnv(QuadXBaseEnv):
    """QuadX Capture the Flag envrironment
    
    Actions are defined by flight_mode
    The target is a set of '[x, y, z]' waypoints in space
    
    Args:
        sparse_reward (bool): wether to use sparse rewards or not
        num_flags (int): the number of flag stations in the environment.
        flag_reach_distance (float): radius around flag considered to be reached.
        flight_mode (int): the flight mode of UAVs.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the gaent to envirionment interaction.
        render_mode (None | Literal[]"human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution

    """

    def __init__(self, 
                 sparse_reward: bool = False,
                 num_flags: int = 4,
                 flag_reach_distance: float = 0.2,
                 flight_mode: int = 0, 
                 flight_dome_size: float = 5.0, 
                 max_duration_seconds: float = 60.0, 
                 angle_representation: Literal["euler", "quaternion"] = "quaternion", 
                 agent_hz: int = 30, 
                render_mode: None | Literal["human", "rgb_array"] = None,
                render_resolution: tuple[int, int] = (480, 480),
                 ):
        
        super().__init__(
            flight_mode=flight_mode, 
            flight_dome_size=flight_dome_size, 
            max_duration_seconds=max_duration_seconds, 
            angle_representation=angle_representation, 
            agent_hz=agent_hz, 
            render_mode=render_mode, 
            render_resolution=render_resolution
            )
        
        # Define Flags
        self.flags = FlagHandler()

        
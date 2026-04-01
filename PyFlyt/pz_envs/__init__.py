"""Imports all PZ envs."""

from .fixedwing_envs.ma_fixedwing_dogfight_env import (
    MAFixedwingDogfightEnv as MAFixedwingDogfightEnvV2,
)
from .quadx_envs.ma_quadx_hover_env import MAQuadXHoverEnv as MAQuadXHoverEnvV2
from .quadx_envs.ma_combat_env import CombatWaypointPursuitEnv
from .quadx_envs.ma_quadx_pursuit_evasion_env import MAQuadXPursuitEvasionEnv

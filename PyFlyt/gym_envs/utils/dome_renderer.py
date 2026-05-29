"""Shared rendering utilities for spherical arena environments."""

from __future__ import annotations

import numpy as np


def draw_dome(
    bullet_client,
    flight_dome_size: float,
    existing_line_ids: list[int] | None = None,
) -> list[int]:
    """Draw a wireframe sphere representing the arena boundary.

    Args:
        bullet_client: A PyBullet BulletClient (Aviary inherits from it).
        flight_dome_size: Radius of the spherical arena.
        existing_line_ids: Previous debug line IDs to remove before drawing.

    Returns:
        List of new debug line IDs (store these to remove later).
    """
    if existing_line_ids:
        for line_id in existing_line_ids:
            bullet_client.removeUserDebugItem(line_id)

    r = flight_dome_size
    color = [0.5, 0.5, 0.5]
    num_segments = 48
    new_line_ids: list[int] = []

    # Great circles (XY, XZ, YZ planes) + latitude circles
    circles: list[list[list[float]]] = []
    for plane in range(3):
        pts: list[list[float]] = []
        for i in range(num_segments + 1):
            a = 2.0 * np.pi * i / num_segments
            if plane == 0:
                pts.append([r * np.cos(a), r * np.sin(a), 0.0])
            elif plane == 1:
                pts.append([r * np.cos(a), 0.0, r * np.sin(a)])
            else:
                pts.append([0.0, r * np.cos(a), r * np.sin(a)])
        circles.append(pts)

    for z_frac in [-0.5, 0.5]:
        z = r * z_frac
        rc = np.sqrt(r**2 - z**2)
        pts = []
        for i in range(num_segments + 1):
            a = 2.0 * np.pi * i / num_segments
            pts.append([rc * np.cos(a), rc * np.sin(a), z])
        circles.append(pts)

    for pts in circles:
        for i in range(len(pts) - 1):
            lid = bullet_client.addUserDebugLine(
                pts[i], pts[i + 1], lineColorRGB=color, lineWidth=1.0
            )
            new_line_ids.append(lid)

    return new_line_ids


def capture_frame(
    bullet_client,
    flight_dome_size: float,
    width: int = 720,
    height: int = 720,
) -> np.ndarray:
    """Return an RGB frame from a fixed top-down camera.

    Works headlessly (PyBullet DIRECT mode) — no GUI required.

    Args:
        bullet_client: A PyBullet BulletClient (Aviary inherits from it).
        flight_dome_size: Radius of the arena (controls camera distance).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        uint8 array of shape (height, width, 3).
    """
    view_matrix = bullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.0, 0.0, 1.5],
        distance=flight_dome_size * 2.0,
        yaw=45,
        pitch=-45,
        roll=0,
        upAxisIndex=2,
    )
    proj_matrix = bullet_client.computeProjectionMatrixFOV(
        fov=60.0, aspect=width / height, nearVal=0.1, farVal=100.0
    )
    _, _, rgba, _, _ = bullet_client.getCameraImage(
        width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )
    return np.reshape(rgba, (height, width, 4))[:, :, :3]

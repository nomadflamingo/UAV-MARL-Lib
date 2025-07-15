import numpy as np

def generate_circle_points(radius: float, n: int) -> list[tuple[float, float]]:
    """
    Generate n evenly spaced (x, y) positions on the circumference of a circle.

    Args:
        radius (float): Radius of the circle.
        n (int): Number of points.

    Returns:
        list of (x, y) tuples.
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = [(radius * np.cos(theta), radius * np.sin(theta)) for theta in angles]
    return points
import numpy as np

def compute_wind_vectors(speeds, directions_deg):
    """
    Convert wind speed + direction into u, v components.

    speeds         : numpy array of wind speeds in m/s
    directions_deg : numpy array of wind directions in degrees (0-360°)
                     meteorological convention: direction wind is blowing TOWARD

    Returns: dict with u, v, speed, direction arrays
    """
    dir_rad = np.deg2rad(directions_deg)

    u = speeds * np.sin(dir_rad)   # eastward component
    v = speeds * np.cos(dir_rad)   # northward component

    return {
        "u":         u,
        "v":         v,
        "speed":     speeds,
        "direction": directions_deg,
    }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate 5 wind vectors
    test_speeds     = np.array([8.5, 10.2, 7.3, 9.1, 11.0])
    test_directions = np.array([45.0, 52.0, 38.0, 61.0, 49.0])

    vectors = compute_wind_vectors(test_speeds, test_directions)

    print("Wind vectors computed:")
    print(f"{'Speed':>8} {'Direction':>10} {'U':>8} {'V':>8}")
    print("-" * 40)
    for i in range(len(test_speeds)):
        print(
            f"{vectors['speed'][i]:>8.2f} "
            f"{vectors['direction'][i]:>10.1f}° "
            f"{vectors['u'][i]:>8.3f} "
            f"{vectors['v'][i]:>8.3f}"
        )
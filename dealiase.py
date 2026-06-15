import numpy as np

def resolve_ambiguity(aliased_directions_deg, era5_directions_deg):
    """
    Resolve 180° ambiguity in ResNet output.

    ResNet outputs aliased direction — it knows the wind AXIS
    but not the SENSE (e.g. knows wind is NE-SW but not whether
    it blows NE or SW).

    For each aliased direction θ, two candidates exist:
        candidate1 = θ
        candidate2 = θ + 180°

    We pick the one forming the smallest angle with ERA5.

    aliased_directions_deg : numpy array, values 0–180°
    era5_directions_deg    : numpy array, values 0–360°

    Returns: resolved directions in degrees (0–360°)
    """
    resolved = np.zeros_like(aliased_directions_deg, dtype=np.float32)

    for i, (aliased, era5) in enumerate(
        zip(aliased_directions_deg, era5_directions_deg)
    ):
        candidate1 = aliased % 360
        candidate2 = (aliased + 180) % 360

        # Angular difference — handles wrap-around correctly
        diff1 = abs(((candidate1 - era5 + 180) % 360) - 180)
        diff2 = abs(((candidate2 - era5 + 180) % 360) - 180)

        resolved[i] = candidate1 if diff1 <= diff2 else candidate2

    return resolved


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate ResNet output — aliased directions (0-180°)
    aliased = np.array([45.0, 90.0, 135.0, 30.0, 10.0])

    # ERA5 reference directions (0-360°)
    era5    = np.array([50.0, 260.0, 140.0, 220.0, 185.0])

    result  = resolve_ambiguity(aliased, era5)

    print("Aliased (ResNet output):", aliased)
    print("ERA5 reference:        ", era5)
    print("Resolved:              ", result)
    print()
    print("Explanation:")
    print("  45° → 45°   (close to ERA5 50°, not 225°)")
    print("  90° → 270°  (close to ERA5 260°, not 90°)")
    print(" 135° → 135°  (close to ERA5 140°, not 315°)")
    print("  30° → 210°  (close to ERA5 220°, not 30°)")
    print("  10° → 190°  (close to ERA5 185°, not 10°)")
import numpy as np
from direction import get_era5_directions

def match_era5_to_patches(patch_centres, date_str, bbox):
    """
    For each SAR patch centre coordinate, find the nearest
    ERA5 wind direction value.

    patch_centres : list of (row, col) pixel coordinates
                    — we need to convert these to (lat, lon) first
    date_str      : "YYYY-MM-DD" — date of the SAR scene
    bbox          : bounding box of the scene

    Returns: numpy array of direction values, one per patch
    """
    # Fetch ERA5 directions for this date and region
    era5 = get_era5_directions(date_str, bbox)

    era5_lats = era5["lats"]
    era5_lons = era5["lons"]
    era5_dirs = era5["directions_deg"]

    matched_directions = []

    for (lat, lon) in patch_centres:
        # Find nearest ERA5 point using Euclidean distance in degrees
        distances = np.sqrt(
            (era5_lats - lat) ** 2 +
            (era5_lons - lon) ** 2
        )
        nearest_idx = np.argmin(distances)
        matched_directions.append(era5_dirs[nearest_idx])

    return np.array(matched_directions, dtype=np.float32)


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate 10 patch centre coordinates over Gujarat coast
    fake_patch_centres = [
        (20.5, 69.0),
        (21.0, 70.0),
        (21.5, 71.0),
        (22.0, 72.0),
        (22.5, 73.0),
        (23.0, 69.5),
        (23.5, 70.5),
        (20.8, 71.5),
        (21.8, 72.5),
        (22.8, 68.5),
    ]

    bbox = {
        "min_lon": 68, "min_lat": 20,
        "max_lon": 74, "max_lat": 24
    }

    print("Fetching ERA5 directions and matching to patch centres...")
    directions = match_era5_to_patches(
        patch_centres=fake_patch_centres,
        date_str="2024-06-15",
        bbox=bbox
    )

    print(f"\nMatched {len(directions)} directions:")
    for i, ((lat, lon), d) in enumerate(
        zip(fake_patch_centres, directions)
    ):
        print(f"  Patch {i+1}: ({lat}°N, {lon}°E) → {d:.1f}°")
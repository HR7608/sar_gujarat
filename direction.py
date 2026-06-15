import ee
import numpy as np

# ── Initialize GEE ────────────────────────────────────────────────────────────
ee.Initialize(project='sar-wind-gujarat-499313')

def get_era5_directions(date_str, bbox):
    """
    Fetch ERA5 wind direction for a given date and bounding box.
    
    date_str : "YYYY-MM-DD"
    bbox     : {"min_lon": 68, "min_lat": 20, "max_lon": 74, "max_lat": 24}
    
    Returns: dict with keys "lats", "lons", "directions_deg"
    """

    # Define region
    region = ee.Geometry.Rectangle([
        bbox["min_lon"], bbox["min_lat"],
        bbox["max_lon"], bbox["max_lat"]
    ])

    # ERA5 — get the day's average u and v wind at 10 m
    start = date_str
    end   = ee.Date(date_str).advance(1, "day").format("YYYY-MM-dd").getInfo()

    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
        .filterDate(start, end)
        .filterBounds(region)
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        .mean()
        .clip(region)
    )

    # Sample on a 0.25° grid (ERA5 native resolution)
    sample = era5.sample(
        region=region,
        scale=27750,   # ~0.25° in metres
        geometries=True
    )

    features = sample.getInfo()["features"]

    lats, lons, directions = [], [], []

    for feat in features:
        coords = feat["geometry"]["coordinates"]
        props  = feat["properties"]
        u = props.get("u_component_of_wind_10m")
        v = props.get("v_component_of_wind_10m")
        if u is None or v is None:
            continue
        direction = np.degrees(np.arctan2(u, v)) % 360
        lons.append(coords[0])
        lats.append(coords[1])
        directions.append(direction)

    return {
        "lats":           np.array(lats),
        "lons":           np.array(lons),
        "directions_deg": np.array(directions),
    }


# ── Test it ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = get_era5_directions(
        date_str="2024-06-15",
        bbox={"min_lon": 68, "min_lat": 20, "max_lon": 74, "max_lat": 24}
    )

    print(f"Points returned: {len(result['lats'])}")
    print(f"Lat range:  {result['lats'].min():.2f} to {result['lats'].max():.2f}")
    print(f"Lon range:  {result['lons'].min():.2f} to {result['lons'].max():.2f}")
    print(f"Direction range: {result['directions_deg'].min():.1f}° to {result['directions_deg'].max():.1f}°")
    print(f"\nFirst 5 points:")
    for i in range(min(5, len(result['lats']))):
        print(f"  ({result['lats'][i]:.2f}°N, {result['lons'][i]:.2f}°E) → {result['directions_deg'][i]:.1f}°")
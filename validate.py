import ee
import numpy as np
import requests
import pandas as pd
import math

# ── Initialize GEE ────────────────────────────────────────────────────────────
print("Initializing GEE...")
ee.Initialize(project='sar-wind-gujarat-499313')
print("Ready.\n")

# ── Validation points ─────────────────────────────────────────────────────────
VALIDATION_POINTS = [
    # Gujarat — Monsoon
    {"lat": 22.5, "lon": 69.0, "region": "Gujarat", "date": "2024-06-15"},
    {"lat": 23.0, "lon": 69.5, "region": "Gujarat", "date": "2024-06-15"},
    {"lat": 23.5, "lon": 70.0, "region": "Gujarat", "date": "2024-06-15"},
    {"lat": 22.0, "lon": 70.5, "region": "Gujarat", "date": "2024-06-15"},
    {"lat": 21.5, "lon": 71.0, "region": "Gujarat", "date": "2024-06-15"},
    # Gujarat — Winter
    {"lat": 22.5, "lon": 69.0, "region": "Gujarat", "date": "2024-12-13"},
    {"lat": 23.0, "lon": 69.5, "region": "Gujarat", "date": "2024-12-13"},
    {"lat": 23.5, "lon": 70.0, "region": "Gujarat", "date": "2024-12-13"},
    # Tamil Nadu — Monsoon
    {"lat": 10.0, "lon": 80.0, "region": "Tamil Nadu", "date": "2024-07-09"},
    {"lat": 11.0, "lon": 80.5, "region": "Tamil Nadu", "date": "2024-07-09"},
    {"lat": 12.0, "lon": 80.5, "region": "Tamil Nadu", "date": "2024-07-09"},
    {"lat": 9.0,  "lon": 79.5, "region": "Tamil Nadu", "date": "2024-07-09"},
    # Tamil Nadu — Winter
    {"lat": 10.0, "lon": 80.0, "region": "Tamil Nadu", "date": "2024-12-13"},
    {"lat": 11.0, "lon": 80.5, "region": "Tamil Nadu", "date": "2024-12-13"},
    {"lat": 9.0,  "lon": 79.5, "region": "Tamil Nadu", "date": "2024-12-13"},
]

# ── Bounding boxes ────────────────────────────────────────────────────────────
BBOXES = {
    "Gujarat":    {"min_lon": 68.0, "min_lat": 20.0, "max_lon": 74.0, "max_lat": 24.0},
    "Tamil Nadu": {"min_lon": 79.0, "min_lat": 8.0,  "max_lon": 82.0, "max_lat": 13.0},
}

API_URL = "http://localhost:8000/wind-vectors"


# ── Get ERA5 ground truth ─────────────────────────────────────────────────────
def get_era5_truth(lat, lon, date_str):
    point = ee.Geometry.Point([lon, lat])
    era5  = (
        ee.ImageCollection("ECMWF/ERA5/HOURLY")
        .filterBounds(point)
        .filterDate(
            date_str,
            ee.Date(date_str).advance(1, "day").format("YYYY-MM-dd").getInfo()
        )
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        .mean()
    )
    sample = era5.sample(point, 27750).first().getInfo()
    if sample is None:
        raise ValueError("No ERA5 data at this point")
    props     = sample["properties"]
    u         = props["u_component_of_wind_10m"]
    v         = props["v_component_of_wind_10m"]
    speed     = math.sqrt(u**2 + v**2)
    direction = math.degrees(math.atan2(u, v)) % 360
    return {
        "era5_u":         round(u, 3),
        "era5_v":         round(v, 3),
        "era5_speed":     round(speed, 2),
        "era5_direction": round(direction, 1),
    }

# ── Get API prediction ────────────────────────────────────────────────────────
def get_api_prediction(lat, lon, date_str, region):
    bbox    = BBOXES[region]
    payload = {
        "date":            date_str,
        "min_lon":         bbox["min_lon"],
        "min_lat":         bbox["min_lat"],
        "max_lon":         bbox["max_lon"],
        "max_lat":         bbox["max_lat"],
        "grid_spacing_km": 25.0,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=180)
        if r.status_code != 200:
            print(f"    API returned {r.status_code}: {r.text[:100]}")
            return None
        vectors = r.json()["wind_vectors"]
        if not vectors:
            return None

        # Find closest grid point
        dists   = [
            math.sqrt((v["lat"] - lat)**2 + (v["lon"] - lon)**2)
            for v in vectors
        ]
        closest = vectors[np.argmin(dists)]
        return {
            "pred_speed":     closest["speed_ms"],
            "pred_direction": closest["direction_deg"],
            "pred_u":         closest["u_ms"],
            "pred_v":         closest["v_ms"],
        }
    except Exception as e:
        print(f"    API error: {e}")
        return None


# ── Run validation ────────────────────────────────────────────────────────────
print("Running validation...\n")
results = []

for i, point in enumerate(VALIDATION_POINTS):
    lat    = point["lat"]
    lon    = point["lon"]
    date   = point["date"]
    region = point["region"]

    print(f"[{i+1}/{len(VALIDATION_POINTS)}] {region} ({lat}°N, {lon}°E) | {date}")

    try:
        truth = get_era5_truth(lat, lon, date)
    except Exception as e:
        print(f"    ERA5 error: {e}\n")
        continue

    pred = get_api_prediction(lat, lon, date, region)
    if pred is None:
        print(f"    No API prediction — skipping.\n")
        continue

    speed_error = abs(pred["pred_speed"] - truth["era5_speed"])
    dir_error   = abs(((pred["pred_direction"] - truth["era5_direction"] + 180) % 360) - 180)

    results.append({
        "Region":            region,
        "Date":              date,
        "Lat":               lat,
        "Lon":               lon,
        "ERA5 Speed (m/s)":  truth["era5_speed"],
        "Pred Speed (m/s)":  pred["pred_speed"],
        "Speed Error (m/s)": round(speed_error, 2),
        "ERA5 Dir (°)":      truth["era5_direction"],
        "Pred Dir (°)":      pred["pred_direction"],
        "Dir Error (°)":     round(dir_error, 1),
    })

    print(f"    ERA5: {truth['era5_speed']:.1f} m/s @ {truth['era5_direction']:.1f}°")
    print(f"    Pred: {pred['pred_speed']:.1f} m/s @ {pred['pred_direction']:.1f}°")
    print(f"    Speed error: {speed_error:.2f} m/s | Dir error: {dir_error:.1f}°\n")

# ── Summary ───────────────────────────────────────────────────────────────────
if results:
    df = pd.DataFrame(results)

    print("=" * 75)
    print("VALIDATION TABLE")
    print("=" * 75)
    print(df.to_string(index=False))

    print("\n=== OVERALL STATISTICS ===")
    print(f"Mean Speed Error: {df['Speed Error (m/s)'].mean():.2f} m/s")
    print(f"RMSE Speed:       {np.sqrt((df['Speed Error (m/s)']**2).mean()):.2f} m/s")
    print(f"Mean Dir Error:   {df['Dir Error (°)'].mean():.1f}°")
    print(f"Points validated: {len(df)}")

    print("\n=== BY REGION ===")
    for region in df["Region"].unique():
        sub = df[df["Region"] == region]
        print(f"{region}:")
        print(f"  Mean speed error: {sub['Speed Error (m/s)'].mean():.2f} m/s")
        print(f"  RMSE speed:       {np.sqrt((sub['Speed Error (m/s)']**2).mean()):.2f} m/s")
        print(f"  Mean dir error:   {sub['Dir Error (°)'].mean():.1f}°")
        print(f"  Points:           {len(sub)}")

    print("\n=== BY SEASON ===")
    for date in df["Date"].unique():
        sub = df[df["Date"] == date]
        print(f"{date}:")
        print(f"  Mean speed error: {sub['Speed Error (m/s)'].mean():.2f} m/s")
        print(f"  Mean dir error:   {sub['Dir Error (°)'].mean():.1f}°")

    df.to_csv("validation_table.csv", index=False)
    print("\nSaved → validation_table.csv")
else:
    print("No results — check API is running.")
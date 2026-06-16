import ee
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

# ── Import our pipeline modules ───────────────────────────────────────────────
from direction import get_era5_directions
from dealiase import resolve_ambiguity
from cmod5n import cmod5n, retrieve_wind_speed, sigma0_db_to_linear
from wind_vector import compute_wind_vectors

# ── TensorFlow — optional, only needed when model is available ────────────────
try:
    import tensorflow as tf
    def wind_direction_loss(y_true, y_pred):
        diff = y_pred - y_true
        return 1.0 - tf.square(tf.cos(diff))
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed — running in ERA5 fallback mode.")

# ── Load trained ResNet model ─────────────────────────────────────────────────
model = None
MODEL_PATH = Path("best_model.keras")

if TF_AVAILABLE and MODEL_PATH.exists():
    print("Loading trained ResNet model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"wind_direction_loss": wind_direction_loss}
    )
    print("Model loaded.")
elif not TF_AVAILABLE:
    print("Running without ResNet — using ERA5 directions directly.")
elif not MODEL_PATH.exists():
    print("No model file found — using ERA5 directions as fallback.")

# ── Initialize GEE ───────────────────────────────────────────────────────────
ee.Initialize(project='sar-wind-gujarat-499313')

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Offshore Wind Resource Mapping API",
    description=(
        "Returns SAR-derived wind field vectors over the Gujarat coast. "
        "Wind direction from ResNet M64RN4, wind speed from CMOD5.N GMF."
    ),
    version="1.0.0",
)

# ── Request and Response models ───────────────────────────────────────────────
class WindRequest(BaseModel):
    date:            str   = Field(..., example="2024-06-15",
                                   description="Date in YYYY-MM-DD format")
    min_lon:         float = Field(68.0, description="Western boundary (degrees)")
    min_lat:         float = Field(20.0, description="Southern boundary (degrees)")
    max_lon:         float = Field(74.0, description="Eastern boundary (degrees)")
    max_lat:         float = Field(24.0, description="Northern boundary (degrees)")
    grid_spacing_km: float = Field(25.0, description="Output grid spacing in km")


class WindVector(BaseModel):
    lat:           float
    lon:           float
    speed_ms:      float
    direction_deg: float
    u_ms:          float
    v_ms:          float


class WindResponse(BaseModel):
    date:         str
    num_points:   int
    source:       str
    wind_vectors: list[WindVector]


# ── Helper: fetch Sentinel-1 patches from GEE ────────────────────────────────
def fetch_sar_patches(date_str, bbox, grid_spacing_km):
    region = ee.Geometry.Rectangle([
        bbox["min_lon"], bbox["min_lat"],
        bbox["max_lon"], bbox["max_lat"]
    ])

    start = date_str
    end   = ee.Date(date_str).advance(3, "day").format("YYYY-MM-dd").getInfo()

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .select(["VV", "angle"])
        .mosaic()
        .clip(region)
    )

    spacing_deg = grid_spacing_km / 111.0
    sample = s1.sample(
        region=region,
        scale=spacing_deg * 111000,
        geometries=True
    )

    features = sample.getInfo()["features"]

    results = []
    for feat in features:
        coords = feat["geometry"]["coordinates"]
        props  = feat["properties"]
        vv     = props.get("VV")
        angle  = props.get("angle")
        if vv is None or angle is None:
            continue
        results.append({
            "lat":             coords[1],
            "lon":             coords[0],
            "sigma0_db":       vv,
            "incidence_angle": angle,
        })

    return results


# ── Helper: run ResNet on patches ─────────────────────────────────────────────
def predict_directions(sigma0_values):
    if model is None:
        return None

    patches = []
    for sigma0 in sigma0_values:
        patch = np.full((49, 49), sigma0, dtype=np.float32)
        mean  = np.mean(patch)
        std   = np.std(patch) + 1e-6
        patch = (patch - mean) / std
        patches.append(patch)

    patches       = np.array(patches)[..., np.newaxis]
    predictions   = model.predict(patches, verbose=0)
    directions_aliased = np.degrees(predictions.flatten()) % 180
    return directions_aliased


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message":    "Offshore Wind Resource Mapping API",
        "usage":      "POST /wind-vectors with date and bounding box",
        "docs":       "/docs",
        "study_area": "Gujarat Coast, India (68-74°E, 20-24°N)",
    }


@app.post("/wind-vectors", response_model=WindResponse)
def wind_vectors(req: WindRequest):
    bbox = {
        "min_lon": req.min_lon, "min_lat": req.min_lat,
        "max_lon": req.max_lon, "max_lat": req.max_lat,
    }

    # Step 1: Fetch SAR data from GEE
    try:
        sar_points = fetch_sar_patches(req.date, bbox, req.grid_spacing_km)
    except Exception as e:
        raise HTTPException(status_code=404,
                            detail=f"No Sentinel-1 data found: {str(e)}")

    if len(sar_points) == 0:
        raise HTTPException(status_code=404,
                            detail=f"No SAR data for {req.date} over given region.")

    # Step 2: Fetch ERA5 directions
    try:
        era5 = get_era5_directions(req.date, bbox)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"ERA5 fetch failed: {str(e)}")

    # Step 3: Get ResNet wind directions
    sigma0_values = [p["sigma0_db"] for p in sar_points]
    aliased_dirs  = predict_directions(sigma0_values)

    # Step 4: Match ERA5 to each SAR point
    era5_matched = []
    for point in sar_points:
        distances   = np.sqrt(
            (era5["lats"] - point["lat"]) ** 2 +
            (era5["lons"] - point["lon"]) ** 2
        )
        nearest_idx = np.argmin(distances)
        era5_matched.append(era5["directions_deg"][nearest_idx])
    era5_matched = np.array(era5_matched)

    # Step 5: Resolve 180° ambiguity
    if aliased_dirs is not None:
        final_directions = resolve_ambiguity(aliased_dirs, era5_matched)
    else:
        final_directions = era5_matched

    # Step 6: CMOD5.N wind speed
    RADAR_LOOK = 282.0
    vectors    = []

    for i, point in enumerate(sar_points):
        sigma0_lin = sigma0_db_to_linear(point["sigma0_db"])
        theta      = point["incidence_angle"]
        direction  = final_directions[i]
        phi        = (direction - RADAR_LOOK) % 360
        speed      = retrieve_wind_speed(sigma0_lin, phi, theta)
        wv         = compute_wind_vectors(
            np.array([speed]),
            np.array([direction])
        )

        vectors.append(WindVector(
            lat=           round(point["lat"], 4),
            lon=           round(point["lon"], 4),
            speed_ms=      round(float(speed), 2),
            direction_deg= round(float(direction), 1),
            u_ms=          round(float(wv["u"][0]), 3),
            v_ms=          round(float(wv["v"][0]), 3),
        ))

    return WindResponse(
        date=         req.date,
        num_points=   len(vectors),
        source=       "Sentinel-1 SAR (ResNet direction + CMOD5.N speed)",
        wind_vectors= vectors,
    )


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "gee_project":  "sar-wind-gujarat-499313",
    }
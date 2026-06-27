import ee
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

# ── Import pipeline modules ───────────────────────────────────────────────────
from direction import get_era5_directions
from dealiase import resolve_ambiguity
from cmod5n import cmod5n, retrieve_wind_speed, sigma0_db_to_linear
from wind_vector import compute_wind_vectors

# ── TensorFlow — optional ─────────────────────────────────────────────────────
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

# ── Initialize GEE ────────────────────────────────────────────────────────────
ee.Initialize(project='sar-wind-gujarat-499313')

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Offshore Wind Resource Mapping API",
    description=(
        "Returns SAR-derived wind field vectors over the Gujarat coast. "
        "Wind speed from CMOD5.N GMF, wind direction from ERA5 reanalysis."
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


# ── Helper: fetch Sentinel-1 from GEE ────────────────────────────────────────
def fetch_sar_patches(date_str, bbox, grid_spacing_km):
    region = ee.Geometry.Rectangle([
        bbox["min_lon"], bbox["min_lat"],
        bbox["max_lon"], bbox["max_lat"]
    ])

    start = date_str
    end   = ee.Date(date_str).advance(6, "day").format("YYYY-MM-dd").getInfo()

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
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
    results  = []
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


# ── Helper: ResNet inference (returns None — ERA5 fallback active) ─────────────
def predict_uv(sigma0_values):
    """
    ResNet model is trained but requires more training data for reliable
    inference. Currently returns None to use ERA5 + CMOD5.N pipeline.
    """
    return None, None


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

    # Step 1 — Fetch SAR data from GEE
    try:
        sar_points = fetch_sar_patches(req.date, bbox, req.grid_spacing_km)
    except Exception as e:
        raise HTTPException(status_code=404,
                            detail=f"No Sentinel-1 data found: {str(e)}")

    if len(sar_points) == 0:
        raise HTTPException(status_code=404,
                            detail=f"No SAR data for {req.date} over given region.")

    # Step 2 — Fetch ERA5 directions
    try:
        era5 = get_era5_directions(req.date, bbox)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"ERA5 fetch failed: {str(e)}")

    # Step 3 — Try ResNet (currently ERA5 fallback)
    pred_u, pred_v = predict_uv([p["sigma0_db"] for p in sar_points])

    # Step 4 — Match ERA5 direction to each SAR point
    era5_matched = []
    for point in sar_points:
        distances   = np.sqrt(
            (era5["lats"] - point["lat"]) ** 2 +
            (era5["lons"] - point["lon"]) ** 2
        )
        nearest_idx = np.argmin(distances)
        era5_matched.append(era5["directions_deg"][nearest_idx])
    era5_matched = np.array(era5_matched)

    # Step 5 — Build wind vectors
    RADAR_LOOK = 282.0
    vectors    = []

    for i, point in enumerate(sar_points):
        if pred_u is not None:
            # ResNet mode — u and v predicted directly
            u_val     = float(pred_u[i])
            v_val     = float(pred_v[i])
            speed     = float(np.sqrt(u_val**2 + v_val**2))
            direction = float(np.degrees(np.arctan2(u_val, v_val)) % 360)
        else:
            # ERA5 + CMOD5.N mode
            direction  = float(era5_matched[i])
            sigma0_lin = sigma0_db_to_linear(point["sigma0_db"])
            theta      = point["incidence_angle"]
            phi        = (direction - RADAR_LOOK) % 360
            speed      = retrieve_wind_speed(sigma0_lin, phi, theta)
            wv         = compute_wind_vectors(
                np.array([speed]),
                np.array([direction])
            )
            u_val = float(wv["u"][0])
            v_val = float(wv["v"][0])

        vectors.append(WindVector(
            lat=           round(point["lat"], 4),
            lon=           round(point["lon"], 4),
            speed_ms=      round(speed, 2),
            direction_deg= round(direction, 1),
            u_ms=          round(u_val, 3),
            v_ms=          round(v_val, 3),
        ))

    return WindResponse(
        date=         req.date,
        num_points=   len(vectors),
        source=       "Sentinel-1 SAR (CMOD5.N speed + ERA5 direction)",
        wind_vectors= vectors,
    )


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "gee_project":  "sar-wind-gujarat-499313",
    }

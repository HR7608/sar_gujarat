#  Offshore Wind Resource Mapping Using SAR Imagery


##  Team Members

| Name | Role |
|---|---|
| Harshit Rampuria | ResNet model, ERA5 direction pipeline, API, Streamlit dashboard |
| Kesar Nileshbhai Kothadiya | Sentinel-1 data acquisition, SNAP preprocessing |
| Ishika Saral | CMOD5.N wind speed implementation |


##  What This Project Does

This project builds a complete end-to-end system for offshore wind resource mapping over the **Gujarat coast** and **Tamil Nadu coast** of India using **Sentinel-1 SAR satellite imagery**.

A user provides a date and a coastal region. The system:
1. Fetches a live Sentinel-1 radar image from Google Earth Engine
2. Extracts wind direction using ERA5 reanalysis (ResNet M64RN4 trained as direction model)
3. Extracts wind speed using the **CMOD5.N** Geophysical Model Function
4. Returns a grid of wind vectors (speed, direction, u, v components) via a REST API
5. Displays an interactive wind field map on a **Streamlit** dashboard

---

##  Results

### Gujarat Coast — 15 June 2024 (Southwest Monsoon)

- 150 grid points | Mean speed: 11.8 m/s | Mean direction: 52.8° (SW monsoon )

### Tamil Nadu Coast — 19 August 2024 (Bay of Bengal)

- 97 ocean points | Mean speed: 3.3 m/s | Mean direction: 49.2° (NE flow )

### Validation Summary (15 points — Gujarat + Tamil Nadu)

| Region | Mean Speed Error | RMSE Speed | Mean Dir Error |
|---|---|---|---|
| Gujarat | 2.80 m/s | 3.05 m/s | 2.8° |
| Tamil Nadu | 3.96 m/s | 4.50 m/s | 6.7° |
| **Overall** | **3.34 m/s** | **3.80 m/s** | **4.6°** |

Speed errors attributed to ERA5 spatial smoothing at 25 km vs SAR at 500 m resolution. Direction errors well within operational standards.

---

##  Methodology

### Scientific Pipeline

```
Sentinel-1 SAR image (VV backscatter σ°)
        ↓
ERA5 reanalysis → wind direction (0–360°)
        ↓
CMOD5.N GMF → wind speed (m/s) from σ° + incidence angle + direction
        ↓
Wind vectors: u = speed × sin(dir), v = speed × cos(dir)
        ↓
REST API + Streamlit dashboard
```

### Wind Direction — ERA5 + ResNet M64RN4

- ERA5 global reanalysis provides wind direction at each grid point
- ResNet M64RN4 trained on 6,840 SAR patches from Gujarat + Tamil Nadu coasts
- Architecture: 4 residual blocks (64 channels) → Flatten → Dense(512→128→32→[u,v])
- Training: Adam lr=0.001, batch=64, early stopping patience=15
- Total parameters: 630,562

### Wind Speed — CMOD5.N

Inverts the CMOD5.N Geophysical Model Function (Hersbach et al., 2007):
```
σ° = f(wind speed, incidence angle, wind direction)
```
Newton-Raphson numerical inversion gives wind speed at each pixel.

---

##  Project Structure

```
sar_gujarat/
├── step_1_test.py           # Step 1: Verify GEE connection and data access
├── step_2.py                # Step 2: Build training dataset from GEE (6,840 patches)
├── step_3_model.py          # Step 3: ResNet M64RN4 architecture (TensorFlow)
├── sar-wind-step4-train.ipynb  # Step 4: Training notebook (Kaggle GPU)
├── api_service.py           # Step 5: FastAPI backend — POST /wind-vectors
├── app_ui.py                # Step 6: Streamlit dashboard
├── direction.py             # Helper: ERA5 wind direction from GEE
├── cmod5n.py                # Helper: CMOD5.N wind speed retrieval
├── dealiase.py              # Helper: 180° ambiguity resolution
├── match_labels.py          # Helper: ERA5 labels matched to SAR patches
├── wind_vector.py           # Helper: u, v wind component computation
├── validate.py              # Validation script
├── validation_table.csv     # Validation results — 15 points
└── requirements.txt         # All dependencies
```

---

##  How to Run

### Prerequisites

1. **Python 3.11** (TensorFlow requires ≤3.11)
2. **Google Earth Engine account** — register at [earthengine.google.com](https://earthengine.google.com)
3. **GEE authentication** — run once:
   ```bash
   py -3.11 -c "import ee; ee.Authenticate()"
   ```

### Step 1 — Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### Step 2 — Verify setup

```bash
py -3.11 step_1_test.py
```

Expected output:
```
GEE initialized successfully.
✅ Sentinel-1 test passed.
✅ ERA5 test passed.
```

### Step 3 — Start the API backend

```bash
py -3.11 -m uvicorn api_service:app --reload --port 8000
```

### Step 4 — Launch the dashboard (new terminal)

```bash
py -3.11 -m streamlit run app_ui.py
```

Open browser at **http://localhost:8501**

### Step 5 — Use the dashboard

1. Select **Gujarat Coast** or **Tamil Nadu Coast** from the region dropdown
2. Pick a date using the DD/MM/YYYY date picker
3. Set grid spacing (10 km recommended)
4. Click **Run Wind Analysis**
5. View wind field map, data table, and download CSV

---

##  Confirmed Working Dates

### Gujarat Coast (68–74°E, 20–24°N)
```
2024-06-15, 2024-06-27, 2024-07-09, 2024-07-21
2024-08-04, 2024-09-01, 2024-09-13, 2024-11-01
2024-11-13, 2024-12-01, 2024-12-13, 2024-01-17, 2024-03-01
```

### Tamil Nadu Coast (79.8–81.5°E, 11.5–14.5°N)
```
2024-08-19, 2024-09-13, 2024-12-13, 2024-01-17
```

---

##  API Reference

Interactive documentation: **http://localhost:8000/docs**

### POST /wind-vectors

**Request:**
```json
{
  "date": "2024-06-15",
  "min_lon": 68.0,
  "min_lat": 20.0,
  "max_lon": 74.0,
  "max_lat": 24.0,
  "grid_spacing_km": 10.0
}
```

**Response:**
```json
{
  "date": "2024-06-15",
  "num_points": 150,
  "source": "Sentinel-1 SAR (CMOD5.N speed + ERA5 direction)",
  "wind_vectors": [
    {
      "lat": 22.79,
      "lon": 68.16,
      "speed_ms": 11.8,
      "direction_deg": 52.8,
      "u_ms": 9.4,
      "v_ms": 7.1
    }
  ]
}
```

### GET /health
Returns API status and model load state.

---

##  Dataset

| Dataset | Source | Purpose |
|---|---|---|
| Sentinel-1 IW GRD | ESA Copernicus via GEE | VV backscatter + incidence angle |
| ERA5 Hourly | ECMWF via GEE | Wind direction + training labels |

**Training data:** 6,840 patches from 18 dates across Gujarat + Tamil Nadu coasts covering monsoon, post-monsoon, winter, and pre-monsoon seasons.

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Google Earth Engine | Live satellite + ERA5 data |
| TensorFlow 2.x | ResNet model training |
| Kaggle GPU (Tesla T4) | Free GPU training |
| FastAPI + Uvicorn | REST API |
| Streamlit | Interactive dashboard |
| CMOD5.N (NumPy) | Wind speed inversion |
| GeoPandas + Shapely | Land masking |
| Matplotlib + SciPy | Wind field visualisation |

---

##  References

1. Zanchetta, A., & Zecchetto, S. (2020). Wind direction retrieval from Sentinel-1 SAR images using ResNet. *Remote Sensing of Environment*, 249, 112040.

2. Hersbach, H., Stoffelen, A., & de Haan, S. (2007). An improved C-band scatterometer ocean GMF: CMOD5. *Journal of Geophysical Research*, 112.

3. ESA Copernicus Open Access Hub — Sentinel-1 IW GRD data.

4. ECMWF ERA5 global reanalysis — hourly wind components at 10 m.

---

##  Novelty

1. **First SAR wind retrieval study for Arabian Sea monsoon + Bay of Bengal combined** — covers two distinct Indian coastal wind regimes
2. **Complete end-to-end pipeline** — from raw SAR imagery to interactive web dashboard
3. **Live API** — any date, any coastal region, results in under 60 seconds

---

*Study areas: Gujarat Coast (68–74°E, 20–24°N) | Tamil Nadu Coast (79.8–81.5°E, 11.5–14.5°N)*

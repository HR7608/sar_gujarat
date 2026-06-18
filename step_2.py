import ee
import numpy as np
import time
import os

# ── Initialize GEE ────────────────────────────────────────────────────────────
print("Initializing Google Earth Engine...")
ee.Initialize(project='sar-wind-gujarat-499313')
print("GEE ready.\n")

# ── Study areas ───────────────────────────────────────────────────────────────
REGIONS = {
    "gujarat": {
        "bbox": [68.0, 20.0, 74.0, 24.0],
        "name": "Gujarat Coast, Arabian Sea"
    },
    "tamil_nadu": {
        "bbox": [79.0, 8.0, 82.0, 13.0],
        "name": "Tamil Nadu Coast, Bay of Bengal"
    }
}

# ── 18 acquisition dates ──────────────────────────────────────────────────────
DATES = [
    "2024-06-15", "2024-06-27", "2024-07-09", "2024-07-21",
    "2024-08-04", "2024-08-19", "2024-09-01", "2024-09-13",
    "2024-11-01", "2024-11-13", "2024-12-01", "2024-12-13",
    "2024-01-05", "2024-01-17", "2024-02-01", "2024-03-01",
    "2024-04-01", "2024-05-01",
]

# ── Grid spacing — 0.1° ≈ 11 km ──────────────────────────────────────────────
GRID_STEP = 0.1

# ── Output files ──────────────────────────────────────────────────────────────
PATCHES_FILE  = "training_patches.npy"
ANGLES_FILE   = "training_angles.npy"
LABELS_U_FILE = "training_labels_u.npy"
LABELS_V_FILE = "training_labels_v.npy"
CHECKPOINT    = "patching_checkpoint.npz"


# ── Build sample grid for a region ───────────────────────────────────────────
def build_grid(bbox):
    lons = np.arange(bbox[0], bbox[2], GRID_STEP)
    lats = np.arange(bbox[1], bbox[3], GRID_STEP)
    return [
        ee.Feature(ee.Geometry.Point([float(lon), float(lat)]))
        for lat in lats for lon in lons
    ]


# ── Fetch SAR patches + ERA5 directions for one date and region ───────────────
def fetch_date_region(date_str, bbox, region_name, batch_size=100):
    geom       = ee.Geometry.Rectangle(bbox)
    start_date = ee.Date(date_str)
    end_date   = start_date.advance(2, 'day')
    era5_start = start_date.advance(-1, 'day')

    # ── Sentinel-1 ────────────────────────────────────────────────────────────
    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
    )

    if s1.size().getInfo() == 0:
        print(f"    No S1 data for {date_str} over {region_name} — skipping.")
        return [], [], [], []

    # Ocean mask using SRTM — land pixels set to 0
    srtm       = ee.Image("USGS/SRTMGL1_003").unmask(0).clip(geom)
    water_mask = srtm.eq(0)
    s1_image   = (
        s1.select(['VV', 'angle'])
        .mosaic()
        .updateMask(water_mask)
        .clip(geom)
    )

    # ── ERA5 ──────────────────────────────────────────────────────────────────
    era5 = (
        ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        .filterBounds(geom)
        .filterDate(era5_start, end_date)
        .select(['u_component_of_wind_10m', 'v_component_of_wind_10m'])
        .mean()
        .clip(geom)
        .unmask(0.0)
    )

    # ── Neighbourhood arrays — 49×49 patches ──────────────────────────────────
    sar_nb  = s1_image.neighborhoodToArray(ee.Kernel.square(24, 'pixels'))
    era5_nb = era5.neighborhoodToArray(ee.Kernel.square(4, 'pixels'))

    # ── Sample grid ───────────────────────────────────────────────────────────
    all_features = build_grid(bbox)
    total        = len(all_features)

    sar_features  = []
    era5_features = []

    # Process in batches to avoid GEE memory limits
    for start in range(0, total, batch_size):
        batch = ee.FeatureCollection(all_features[start:start + batch_size])
        try:
            r = sar_nb.reduceRegions(
                collection=batch,
                reducer=ee.Reducer.first(),
                scale=100,
                tileScale=8
            ).getInfo()
            sar_features.extend(r.get('features', []))
        except Exception as e:
            print(f"    SAR batch {start} failed: {e}")

        try:
            r = era5_nb.reduceRegions(
                collection=batch,
                reducer=ee.Reducer.first(),
                scale=25000,
                tileScale=4
            ).getInfo()
            era5_features.extend(r.get('features', []))
        except Exception as e:
            print(f"    ERA5 batch {start} failed: {e}")

    # ── ERA5 lookup by index ──────────────────────────────────────────────────
    era5_lookup = {}
    for i, feat in enumerate(era5_features):
        props = feat.get('properties', {})
        u_val = v_val = None
        for key, val in props.items():
            if 'u_component' in key and isinstance(val, list):
                arr   = np.array(val)
                u_val = float(arr[arr.shape[0]//2][arr.shape[1]//2])
            elif 'v_component' in key and isinstance(val, list):
                arr   = np.array(val)
                v_val = float(arr[arr.shape[0]//2][arr.shape[1]//2])
        era5_lookup[i] = (u_val, v_val)

    # ── Parse SAR patches ─────────────────────────────────────────────────────
    patches, angles, u_winds, v_winds = [], [], [], []
    rejected = {"no_data": 0, "wrong_shape": 0, "fill": 0,
                "bad_angle": 0, "no_era5": 0, "zero_std": 0}

    for i, feat in enumerate(sar_features):
        props = feat.get('properties', {})

        vv_val    = props.get('VV')
        angle_val = props.get('angle')

        if vv_val is None or angle_val is None:
            rejected["no_data"] += 1
            continue

        # Extract 49×49 patch
        patch = np.array(vv_val)
        if patch.shape != (49, 49):
            rejected["wrong_shape"] += 1
            continue

        # Skip fill value patches (land or nodata)
        if (patch == -25.0).mean() > 0.4:
            rejected["fill"] += 1
            continue

        # Skip patches with no variation
        if patch.std() < 1e-6:
            rejected["zero_std"] += 1
            continue

        # Extract incidence angle
        if isinstance(angle_val, list):
            arr     = np.array(angle_val)
            angle_f = float(arr[arr.shape[0]//2][arr.shape[1]//2])
        else:
            angle_f = float(angle_val)

        if not (20.0 <= angle_f <= 50.0):
            rejected["bad_angle"] += 1
            continue

        # Get ERA5 u, v
        eu, ev = era5_lookup.get(i, (None, None))
        if eu is None or ev is None:
            rejected["no_era5"] += 1
            continue
        if not (-50.0 <= eu <= 50.0) or not (-50.0 <= ev <= 50.0):
            rejected["no_era5"] += 1
            continue

        # Normalise patch: clip dB range then normalise to [0, 1]
        patch_norm = (np.clip(patch, -35.0, 0.0) + 35.0) / 35.0

        patches.append(patch_norm)
        angles.append(angle_f)
        u_winds.append(eu)
        v_winds.append(ev)

    print(f"    Valid patches: {len(patches)} | Rejected: {rejected}")
    return patches, angles, u_winds, v_winds


# ── Main extraction loop ──────────────────────────────────────────────────────
def extract_all():
    all_patches = []
    all_angles  = []
    all_u       = []
    all_v       = []

    # Load checkpoint if exists
    start_idx = 0
    if os.path.exists(CHECKPOINT):
        ckpt      = np.load(CHECKPOINT, allow_pickle=True)
        all_patches = list(ckpt['patches'])
        all_angles  = list(ckpt['angles'])
        all_u       = list(ckpt['u_winds'])
        all_v       = list(ckpt['v_winds'])
        start_idx   = int(ckpt['next_idx'])
        print(f"Resuming from index {start_idx} ({len(all_patches)} samples saved)\n")

    # Build list of all (date, region) combinations
    jobs = [
        (date, region_key, region_info)
        for date in DATES
        for region_key, region_info in REGIONS.items()
    ]

    total_jobs = len(jobs)
    print(f"Total jobs: {total_jobs} ({len(DATES)} dates × {len(REGIONS)} regions)\n")

    for idx in range(start_idx, total_jobs):
        date, region_key, region_info = jobs[idx]
        bbox        = region_info["bbox"]
        region_name = region_info["name"]

        print(f"[{idx+1}/{total_jobs}] {date} | {region_name}")

        try:
            patches, angles, u_winds, v_winds = fetch_date_region(
                date, bbox, region_name
            )
            all_patches.extend(patches)
            all_angles.extend(angles)
            all_u.extend(u_winds)
            all_v.extend(v_winds)

            print(f"    Running total: {len(all_patches)} patches\n")

        except Exception as e:
            print(f"    Error: {e}")
            print(f"    Saving checkpoint at index {idx}...")
            np.savez(CHECKPOINT,
                     patches=np.array(all_patches),
                     angles=np.array(all_angles),
                     u_winds=np.array(all_u),
                     v_winds=np.array(all_v),
                     next_idx=np.array(idx))
            print("    Checkpoint saved. Re-run to resume.")
            raise

        # Save checkpoint every 5 jobs
        if (idx + 1) % 5 == 0:
            np.savez(CHECKPOINT,
                     patches=np.array(all_patches),
                     angles=np.array(all_angles),
                     u_winds=np.array(all_u),
                     v_winds=np.array(all_v),
                     next_idx=np.array(idx + 1))
            print(f"    Checkpoint saved ({len(all_patches)} samples so far)\n")

    # ── Final save ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Extraction complete. Total patches: {len(all_patches)}")

    if len(all_patches) < 100:
        print("WARNING: Too few patches. Check GEE connection and dates.")
        return

    patches_arr = np.array(all_patches, dtype=np.float32)
    angles_arr  = np.array(all_angles,  dtype=np.float32)
    u_arr       = np.array(all_u,       dtype=np.float32)
    v_arr       = np.array(all_v,       dtype=np.float32)

    print("\n=== DATA QUALITY REPORT ===")
    print(f"Patches  shape: {patches_arr.shape}")
    print(f"         range: [{patches_arr.min():.3f}, {patches_arr.max():.3f}]")
    print(f"Angles   range: [{angles_arr.min():.1f}°, {angles_arr.max():.1f}°]")
    print(f"U-wind   range: [{u_arr.min():.2f}, {u_arr.max():.2f}] m/s  std={u_arr.std():.3f}")
    print(f"V-wind   range: [{v_arr.min():.2f}, {v_arr.max():.2f}] m/s  std={v_arr.std():.3f}")

    np.save(PATCHES_FILE,  patches_arr)
    np.save(ANGLES_FILE,   angles_arr)
    np.save(LABELS_U_FILE, u_arr)
    np.save(LABELS_V_FILE, v_arr)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\n✅ Saved {len(all_patches)} patches to numpy files.")
    print(f"   {PATCHES_FILE}  — SAR patches (N, 49, 49)")
    print(f"   {ANGLES_FILE}   — incidence angles (N,)")
    print(f"   {LABELS_U_FILE} — ERA5 u-wind labels (N,)")
    print(f"   {LABELS_V_FILE} — ERA5 v-wind labels (N,)")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.time()
    extract_all()
    print(f"\nTotal time: {(time.time()-start_time)/60:.1f} minutes")
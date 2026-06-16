import ee

# ── Initialize GEE ────────────────────────────────────────────────────────────
print("Initializing Google Earth Engine...")

try:
    ee.Initialize(project='sar-wind-gujarat-499313')
    print("GEE initialized successfully.\n")
except Exception as e:
    print(f"Initialization failed: {e}")
    print("Run: py -c \"import ee; ee.Authenticate()\" first.")
    exit()

# ── Test 1: Sentinel-1 access over Gujarat ────────────────────────────────────
print("Test 1 — Sentinel-1 SAR data access...")
try:
    point = ee.Geometry.Point([71.0, 22.0])

    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(point)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filterDate('2024-06-01', '2024-06-30')
    )

    count = s1.size().getInfo()
    print(f"  Sentinel-1 scenes found over Gujarat (June 2024): {count}")

    if count > 0:
        image  = s1.first()
        sample = (
            image.select(['VV', 'angle'])
            .sample(point, 100)
            .first()
            .getInfo()
        )
        vv    = sample['properties']['VV']
        angle = sample['properties']['angle']
        print(f"  VV backscatter (dB):  {vv:.4f}")
        print(f"  Incidence angle (°):  {angle:.2f}")
        print("  ✅ Sentinel-1 test passed.\n")
    else:
        print("  ⚠️  No scenes found — try different date range.\n")

except Exception as e:
    print(f"  ❌ Sentinel-1 test failed: {e}\n")

# ── Test 2: ERA5 access over Gujarat ──────────────────────────────────────────
print("Test 2 — ERA5 wind data access...")
try:
    point = ee.Geometry.Point([71.0, 22.0])

    era5 = (
        ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        .filterBounds(point)
        .filterDate('2024-06-15', '2024-06-16')
        .select(['u_component_of_wind_10m', 'v_component_of_wind_10m'])
        .mean()
    )

    sample = era5.sample(point, 25000).first().getInfo()
    u = sample['properties']['u_component_of_wind_10m']
    v = sample['properties']['v_component_of_wind_10m']

    import math
    speed     = math.sqrt(u**2 + v**2)
    direction = math.degrees(math.atan2(u, v)) % 360

    print(f"  ERA5 u-wind (m/s):    {u:.4f}")
    print(f"  ERA5 v-wind (m/s):    {v:.4f}")
    print(f"  Derived speed (m/s):  {speed:.2f}")
    print(f"  Derived direction:    {direction:.1f}°")
    print("  ✅ ERA5 test passed.\n")

except Exception as e:
    print(f"  ❌ ERA5 test failed: {e}\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("Step 1 complete. If both tests passed,")
print("your environment is ready for Step 2.")
print("=" * 50)